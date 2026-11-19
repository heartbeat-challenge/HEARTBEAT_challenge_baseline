# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""OmniPro matching/content rules with retained HEARTBEAT episode/timing policies."""
import re
from pathlib import Path

from .common import digest_json, load_rows, reference_count
from .omnipro_rules import TASK_CONTENT_KIND, _score_content, match_emits_to_gt
from .metrics import aggregate

PROTOCOL = "video_omnipro_dev_reference_v3"
DEV_TARGETS = Path(__file__).with_name("data") / "video_dev_targets.jsonl"
AGGREGATION = "omnipro_task_macro_v1"
REGIONS = {"top-left", "top-center", "top-right", "center-left", "center",
           "center-right", "bottom-left", "bottom-center", "bottom-right"}

TARGET_FIELDS = {"count": "count", "position": "position", "state": "state_to"}
REGION_PATTERN = "|".join(sorted(REGIONS, key=lambda region: (-len(region), region)))


# Video judge rubric adapted from RuixiangZhao/OmniPro (MIT). See the repository root NOTICE.
PROMPT = """You are evaluating the quality of a streaming video assistant's response.

Task context: The user is watching a video stream and asked the assistant to monitor for specific events/conditions. When triggered, the assistant should provide an accurate and relevant response.

User's original question: {question}

Ground truth (what actually happened): {gt_response}

Model's prediction: {pred_response}

Rate the model's response on a scale of 1-5:
- 5: Perfect match — captures the same event/information accurately
- 4: Good — mostly correct, minor differences in detail
- 3: Acceptable — captures the right event but with notable inaccuracies
- 2: Poor — partially relevant but significantly wrong or vague
- 1: Wrong — completely irrelevant or describes wrong event

Respond with ONLY a JSON object: {{"score": <int>, "explanation": "<brief reason>"}}"""


def summarize(cases, windows):
    """Micro within each Video task, then equal task weights, as in OmniPro.

    Official code rounds each task F1 to four decimals before averaging.
    Incomplete content cannot silently remove a task from the denominator.
    A subset mean is reported explicitly, never as a complete nine-task score.
    """
    cases = [c for c in cases if c["subtrack"] == "video"]
    if not cases:
        return None
    by_task = {task: [c for c in cases if c["task_type"] == task]
               for task in TASK_CONTENT_KIND if any(c["task_type"] == task for c in cases)}
    missing = [task for task in TASK_CONTENT_KIND if task not in by_task]
    complete = all(c["status"] == "complete" and c["content_complete"] for c in cases)
    result = dict(aggregation=AGGREGATION, task_types=list(by_task), num_task_types=len(by_task),
                  missing_task_types=missing, scope="subset" if missing else "all_nine_tasks",
                  status="complete" if complete else "incomplete", windows={})
    for window in windows:
        key = str(float(window))
        per_task = {}
        for task, rows in by_task.items():
            per_task[task] = {}
            for metric in ("timing", "joint", "joint_no_early"):
                items = [c["windows"][key][metric] for c in rows]
                per_task[task][metric] = round(aggregate(items)["f1"], 4) if all(i is not None for i in items) else None
        totals = {}
        for metric in ("timing", "joint", "joint_no_early"):
            values = [row[metric] for row in per_task.values()]
            totals[metric] = round(sum(values)/len(values), 4) if all(v is not None for v in values) else None
        result["windows"][key] = dict(
            per_task_f1=per_task, macro_time_f1=totals["timing"], macro_joint_f1=totals["joint"],
            macro_joint_no_early_f1=totals["joint_no_early"],
            headline_joint_f1=totals["joint"] if complete and not missing else None)
    return result


def kind(reference):
    task = reference.get("task_type")
    if task not in TASK_CONTENT_KIND:
        raise ValueError("Video requires a canonical OmniPro task_type")
    return TASK_CONTENT_KIND[task]


def attach_dev_targets(references):
    """Restore published Dev targets only for exact public question/answer timelines."""
    catalog = {row["id"]: row for row in load_rows(DEV_TARGETS)}
    for ref in references:
        entry = catalog.get(ref["id"])
        if entry is None:
            continue
        events = ref["response"]
        identity = {k: ref[k] for k in ("id", "task_type", "question")}
        identity["response"] = [{"time": event["time"], "content": event["content"]} for event in events]
        if entry["reference_sha256"] != digest_json(identity) or len(entry["targets"]) != len(events):
            raise ValueError(f"Reference differs from the published Dev row for {ref['id']}")
        field = TARGET_FIELDS[kind(ref)]
        for event, target in zip(events, entry["targets"]):
            if target is None:
                continue
            if set(target) != {field} or (event.get(field) is not None and event[field] != target[field]):
                raise ValueError(f"Conflicting Dev target for {ref['id']}")
            event[field] = target[field]


def resolve_targets(reference):
    """Use typed GT first, recover explicit answers, then tag unresolved events for semantic judging."""
    content_kind = kind(reference)
    for index in missing_targets(reference):
        event = reference["response"][index]
        field = TARGET_FIELDS[content_kind]
        if event.get(field) is not None:
            raise ValueError(f"Invalid Video {field} for {reference['id']} event {index}")
        answer = event["content"].strip().lower()
        target = None
        if content_kind == "count":
            target = reference_count(answer, running=reference["task_type"] != "snapshot_counting")
        elif content_kind == "position":
            locations = set(re.findall(rf"\b({REGION_PATTERN})\b", answer))
            if len(locations) == 1:
                region = locations.pop()
                if (answer.rstrip(".!") == region
                        or re.search(rf"\bposition\s*:\s*{region}\b", answer)
                        or re.search(rf"\b(?:at|in|on) the {region} of (?:the|her|his|their) ", answer)):
                    target = region
        elif content_kind == "state":
            match = re.fullmatch(r"(?:state_to|state)\s*:\s*([^.!?\n]+)[.!]?", answer)
            if match and match[1].strip():
                target = match[1].strip()
        if target is None:
            event["scoring_fallback"] = "llm_judge"
        else:
            event[field] = target


def missing_targets(reference):
    """Return events without a valid target or a declared semantic fallback."""
    content_kind = kind(reference)
    missing = []
    for j, event in enumerate(reference["response"]):
        if event.get("scoring_fallback") == "llm_judge" and content_kind in TARGET_FIELDS:
            continue
        if content_kind == "count":
            valid = type(event.get("count")) is int and event["count"] >= 0
        elif content_kind == "position":
            valid = event.get("position") in REGIONS
        elif content_kind == "state":
            valid = isinstance(event.get("state_to"), str) and bool(event["state_to"].strip())
        else:
            valid = True
        if not valid:
            missing.append(j)
    return missing


def temporal_pairs(reference, episodes, window):
    """Freeze the official nearest-first assignment BEFORE evaluating content."""
    emits = [dict(t_sec=e["time"], raw=e["prefixes"][-1]["content"]) for e in episodes]
    gts = [dict(trigger_time_sec=g["time"]) for g in reference["response"]]
    matches, _, _ = match_emits_to_gt(emits, gts, window)
    # Retain the shared no-credit-before-instruction guard without re-matching.
    return [(i, j, dt) for i, j, dt in matches
            if episodes[i]["time"] >= reference["question"][0]["time"]]


def requested_pairs(reference, episodes, windows):
    free_text_task = kind(reference) == "gpt_judge"
    judged = {j for j, event in enumerate(reference["response"])
              if free_text_task or event.get("scoring_fallback") == "llm_judge"}
    if not judged:
        return set()
    return {(i, j) for w in windows for i, j, _ in temporal_pairs(reference, episodes, w) if j in judged}


def validate_judgments(pairs, expected, episodes):
    if not isinstance(pairs, list) or len(pairs) != len(expected):
        raise ValueError("Video judge must cover exactly the fixed temporal pairs")
    seen = set()
    for pair in pairs:
        i, j = pair.get("episode_index"), pair.get("reference_index")
        if type(i) is not int or type(j) is not int or (i, j) not in expected or (i, j) in seen:
            raise ValueError("Invalid or duplicate video judge pair")
        seen.add((i, j))
        scores = pair.get("prefix_scores")
        if (not isinstance(scores, list) or len(scores) != len(episodes[i]["prefixes"])
                or any(type(s) is not int or not 1 <= s <= 5 for s in scores)):
            raise ValueError("Video judge requires a 1–5 score for every prefix")


def first_correct(reference, episode, j, judgment=None):
    content_kind = kind(reference)
    if content_kind == "time_only":
        return 0
    if content_kind == "gpt_judge" or reference["response"][j].get("scoring_fallback") == "llm_judge":
        scores = judgment["prefix_scores"]
        # Official content decision uses the full merged response. Prefix scores
        # locate its first sufficient output, retaining the current time policy.
        if scores[-1] < 3:
            return None
        return next(k for k, score in enumerate(scores) if score >= 3)
    gts = reference["response"]
    correct = [_score_content(reference["task_type"], dict(raw=p["content"]), gts[j], gts)
               for p in episode["prefixes"]]
    if not correct[-1]:
        return None
    return next(k for k, ok in enumerate(correct) if ok)


def score_window(reference, episodes, window, judgments, content_complete):
    matches = temporal_pairs(reference, episodes, window)
    def metric(deltas):
        return dict(tp=len(deltas), fp=len(episodes)-len(deltas),
                    fn=len(reference["response"])-len(deltas), deltas=deltas)
    timing = metric([episodes[i]["time"]-reference["response"][j]["time"] for i, j, _ in matches])
    if not content_complete:
        return dict(timing=timing, joint=None, joint_no_early=None)
    lookup = {(p["episode_index"], p["reference_index"]): p for p in judgments}
    deltas, no_early = [], []
    for i, j, _ in matches:
        k = first_correct(reference, episodes[i], j, lookup.get((i, j)))
        if k is None:
            continue
        delta = episodes[i]["prefixes"][k]["time"]-reference["response"][j]["time"]
        if abs(delta) <= window:
            deltas.append(delta)
            if delta >= 0:
                no_early.append(delta)
    return dict(timing=timing, joint=metric(deltas), joint_no_early=metric(no_early))


def attach_targets(references, target_rows):
    """Attach explicitly curated fields to reference-only data, never model input."""
    targets = {row["id"]: row for row in target_rows}
    ids = {r["id"] for r in references if r["subtrack"] == "video"}
    if len(targets) != len(target_rows) or set(targets)-ids:
        raise ValueError("Duplicate or unknown video target IDs")
    for ref in references:
        if ref["id"] not in targets:
            continue
        events = targets[ref["id"]]["response"]
        if len(events) != len(ref["response"]):
            raise ValueError("Video target event count mismatch")
        for original, extra in zip(ref["response"], events):
            if set(extra)-{"time", "content", "count", "position", "state_to"}:
                raise ValueError("Unexpected target fields")
            if extra.get("time") != original["time"] or extra.get("content") != original["content"]:
                raise ValueError("Video targets must exactly match reference time and text")
            for field in ("count", "position", "state_to"):
                if field in extra:
                    original[field] = extra[field]
                    if field == TARGET_FIELDS.get(kind(ref)):
                        original.pop("scoring_fallback", None)
    return references
