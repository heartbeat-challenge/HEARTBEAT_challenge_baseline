# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Score all expected Dev tasks; missing predictions remain in the denominator."""
import argparse
import math
from pathlib import Path

from .common import atomic_json, digest_json, finite, load_rows, validate_prediction
from .metrics import aggregate, candidate_pairs, match, validate_pairs
from . import video, audio


def score(references, predictions, windows=None, judgments=None, clock="stream", audio_protocol="routed"):
    if audio_protocol not in ("routed", "legacy"):
        raise ValueError("Unknown Audio protocol")
    if windows is None:
        windows = (3.,) if audio_protocol == "routed" or any(r.get("subtrack") == "video" for r in references) else (1., 2., 3., 5., 10.)
    if not references or not windows or any(not math.isfinite(w) or w < 0 for w in windows):
        raise ValueError("Nonempty references and finite nonnegative windows required")
    ref_ids = {r["id"] for r in references}
    pred = {p["id"]: p for p in predictions}
    if len(ref_ids) != len(references) or len(pred) != len(predictions) or set(pred) - ref_ids:
        raise ValueError("Duplicate or unexpected prediction/reference IDs")
    judges = {j["id"]: j for j in (judgments or [])}
    if len(judges) != len(judgments or []) or set(judges) - ref_ids:
        raise ValueError("Duplicate or unexpected judgment IDs")
    cases, missing, failed, pending, missing_targets, missing_audio = [], [], [], [], [], []
    settings = {p.get("settings_sha256") for p in predictions}
    protocols = {p.get("protocol_sha256") for p in predictions}
    judge_settings = {j.get("judge_signature") for j in (judgments or [])}
    if len(settings) > 1 or len(protocols) > 1 or len(judge_settings) > 1:
        raise ValueError("Mixed inference or judge protocols in one report")
    for ref in references:
        if ref.get("subtrack") not in ("audio", "video"):
            raise ValueError("Unknown subtrack")
        duration = finite(ref["duration"])
        previous = finite(ref["question"][0]["time"])
        if duration <= 0 or not 0 <= previous <= duration or not ref["response"]:
            raise ValueError("Invalid reference task")
        for event in ref["response"]:
            timestamp = finite(event["time"])
            if not previous <= timestamp <= duration or not isinstance(event["content"], str) or not event["content"].strip():
                raise ValueError("Invalid reference event")
            previous = timestamp
        uid = ref["id"]
        result = pred.get(uid)
        if result is None:
            missing.append(uid)
            episodes = []
        else:
            episodes = validate_prediction(result, ref, clock)
            if result["status"] != "complete":
                failed.append(uid)
        is_video = ref.get("subtrack") == "video"
        is_routed_audio = not is_video and audio_protocol == "routed"
        scoring_mode = None
        pairs, content_complete = [], judgments is not None
        if is_video:
            content_kind = video.kind(ref)
            scoring_mode = {"time_only": "time_only", "gpt_judge": "llm_judge"}.get(content_kind, "rule_based")
            absent = video.missing_targets(ref)
            content_complete = not absent
            if absent:
                missing_targets.append(dict(id=uid, event_indices=absent))
            expected = video.requested_pairs(ref, episodes, windows)
            if expected:
                judge = judges.get(uid, {})
                valid = (judge.get("status") == "complete" and judge.get("clock") == clock
                         and judge.get("protocol") == video.PROTOCOL
                         and judge.get("reference_sha256") == digest_json(ref)
                         and judge.get("prediction_sha256") == digest_json(result)
                         and set(windows).issubset(set(judge.get("windows", []))))
                if valid:
                    video.validate_judgments(judge["pairs"], video.requested_pairs(ref, episodes, judge["windows"]), episodes)
                    pairs = judge["pairs"]
                else:
                    content_complete = False
                    pending.append(uid)
        elif is_routed_audio:
            issues = audio.metadata_issues(ref)
            content_complete = not issues
            if issues:
                missing_audio.append(dict(id=uid, issues=issues))
            else:
                scoring_mode = audio.configuration(ref)["mode"]
            expected = audio.requested_pairs(ref, episodes, windows)
            if expected:
                judge = judges.get(uid, {})
                valid = (judge.get("status") == "complete" and judge.get("clock") == clock
                         and judge.get("protocol") == audio.PROTOCOL
                         and judge.get("reference_sha256") == digest_json(ref)
                         and judge.get("prediction_sha256") == digest_json(result)
                         and set(windows).issubset(set(judge.get("windows", []))))
                if valid:
                    audio.validate_judgments(judge["pairs"], audio.requested_pairs(ref, episodes, judge["windows"]), episodes)
                    pairs = judge["pairs"]
                else:
                    content_complete = False
                    pending.append(uid)
        elif judgments is not None:
            expected = candidate_pairs(ref["response"], episodes, max(windows))
            judge = judges.get(uid, {})
            if expected:
                valid = (judge.get("status") == "complete" and judge.get("clock") == clock
                         and judge.get("reference_sha256") == digest_json(ref)
                         and judge.get("prediction_sha256") == digest_json(result)
                         and judge.get("max_window", -1) >= max(windows))
                if valid:
                    # A larger judge window may contain additional pairs; validate all of them.
                    validate_pairs(judge["pairs"], candidate_pairs(ref["response"], episodes, judge["max_window"]), episodes)
                    pairs = judge["pairs"]
                else:
                    content_complete = False
                    pending.append(uid)
        item = dict(id=uid, level=ref.get("level"), audio_type=ref.get("audio_type"),
                    subtrack=ref.get("subtrack"), task_type=ref.get("task_type"),
                    status=result["status"] if result else "missing", episodes=len(episodes),
                    content_complete=content_complete, scoring_mode=scoring_mode, windows={})
        if is_video:
            fallback = [i for i, event in enumerate(ref["response"]) if event.get("scoring_fallback") == "llm_judge"]
            if fallback:
                item["semantic_fallback_event_indices"] = fallback
                item["scoring_mode"] = "llm_judge" if len(fallback) == len(ref["response"]) else "mixed_rule_and_judge"
        for window in windows:
            question_time = ref["question"][0]["time"]
            if is_video or is_routed_audio:
                policy = video if is_video else audio
                item["windows"][str(float(window))] = policy.score_window(ref, episodes, window, pairs, content_complete)
                continue
            item["windows"][str(float(window))] = dict(
                timing=match(ref["response"], episodes, window, not_before=question_time),
                joint=match(ref["response"], episodes, window, pairs, not_before=question_time) if content_complete else None,
                joint_no_early=match(ref["response"], episodes, window, pairs, True, question_time) if content_complete else None)
        cases.append(item)
    groups = {"all": cases}
    for field in ("subtrack", "level", "audio_type", "task_type", "scoring_mode"):
        for value in sorted({c[field] for c in cases if c[field] is not None}):
            groups[f"{field}/{value}"] = [c for c in cases if c[field] == value]
    for modality, level in sorted({(c["audio_type"], c["level"]) for c in cases if c["audio_type"] and c["level"]}):
        groups[f"audio_type_level/{modality}/{level}"] = [c for c in cases if c["audio_type"] == modality and c["level"] == level]
    report = dict(schema_version="heartbeat_eval_v1", protocol="diagnostic", clock=clock,
                  video_protocol=video.PROTOCOL if any(r.get("subtrack") == "video" for r in references) else None,
                  references_sha256=digest_json(references),
                  settings_sha256=next(iter(settings), None), judge_signature=next(iter(judge_settings), None),
                  status=("not_evaluated" if not cases else "complete" if not (missing or failed or pending or missing_targets or missing_audio) else "incomplete"),
                  expected_tasks=len(cases), input_tasks=len(references), excluded_audio_samples=[],
                  missing_predictions=missing, failed_predictions=failed,
                  pending_judgments=pending, missing_video_targets=missing_targets,
                  missing_audio_scoring=missing_audio,
                  content_judged=judgments is not None, groups={})
    has_audio = any(r.get("subtrack") == "audio" for r in references)
    report["audio_protocol"] = (audio.PROTOCOL if audio_protocol == "routed" else "legacy") if has_audio else None
    report["audio_summary"] = audio.summarize(cases, windows) if has_audio and audio_protocol == "routed" else None
    report["audio_aggregation"] = audio.AGGREGATION if report["audio_summary"] else None
    report["video_summary"] = video.summarize(cases, windows)
    if report["video_summary"]:
        fallback_events = sum(len(c.get("semantic_fallback_event_indices", [])) for c in cases)
        report["video_summary"]["semantic_fallback_events"] = fallback_events
        report["video_summary"]["content_policy"] = (
            "dev_reference_text_fallback" if fallback_events else "omnipro_task_rules"
        )
    report["video_aggregation"] = video.AGGREGATION if report["video_summary"] else None
    for name, values in groups.items():
        report["groups"][name] = dict(tasks=len(values), aggregation="event_micro", windows={})
        for w in windows:
            key = str(float(w))
            report["groups"][name]["windows"][key] = {
                metric: aggregate([c["windows"][key][metric] for c in values])
                if all(c["windows"][key][metric] is not None for c in values) else None
                for metric in ("timing", "joint", "joint_no_early")}
    return report, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=float, nargs="+", help="Default ±3s; legacy Audio ±1/2/3/5/10s")
    parser.add_argument("--audio-protocol", choices=["routed", "legacy"], default="routed")
    parser.add_argument("--clock", choices=["stream", "wall"], default="stream")
    args = parser.parse_args()
    report, cases = score(load_rows(args.references), load_rows(args.predictions), args.windows,
                          load_rows(args.judgments) if args.judgments else None, args.clock, args.audio_protocol)
    atomic_json(args.output, report)
    atomic_json(args.output.with_name(args.output.stem + ".cases.json"), cases)
    summaries = {name: report[name] for name in ("audio_summary", "video_summary") if report[name] is not None}
    print(report["status"], report["expected_tasks"], "samples;", summaries or report["groups"]["all"])
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
