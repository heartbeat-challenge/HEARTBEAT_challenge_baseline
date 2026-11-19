"""Reference-chat acoustic/contextual aggregation; preserves conditional denominators."""

import argparse
import collections
import json
import statistics

from .common import MODEL_NAME, ROOT, TYPES, save
from .point_reference import point

CONTINUATION_TYPES = {"backchannel", "background_speech"}
EVENT_TYPES = {"cut", *CONTINUATION_TYPES}


def static_trigger(row, run, acoustic):
    """Recover the event's active assistant turn from aligned playback."""
    span = row["ground_truth"].get("event_span_s")
    event_words = (acoustic.get("event") or {}).get("words")
    if not span or not event_words:
        return None

    index = min(len(run["steps"]) - 1, max(0, int(event_words[0]["start_s"])))
    delivery = run["steps"][index]["sent_wall_s"]
    by_turn = collections.defaultdict(list)
    for turn in acoustic["turns"]:
        alignment = turn["alignment"]
        if alignment["valid"]:
            by_turn[turn["turn_id"]].extend(alignment["words"])
    for turn_id, words in by_turn.items():
        if words and min(w["wall_start_s"] for w in words) < delivery < max(
            w["wall_end_s"] for w in words
        ):
            return {
                "turn_id": turn_id,
                "media_start_s": span[0],
                "delivery_wall_s": delivery,
            }
    return None


def score_one(row, run, acoustic, judge):
    """Combine timing evidence and judge output for one sample."""
    measured = run
    if row["subset"] == "static" and run["type"] in EVENT_TYPES:
        measured = {**run, "trigger": static_trigger(row, run, acoustic)}
    z = point(measured, acoustic)
    z.update(
        subset=row["subset"],
        triggered=(run["trigger"] is not None)
        if row["subset"] == "dynamic" and run["type"] in EVENT_TYPES
        else None,
    )
    if judge is not None and z["eligible"] and z["type"] in CONTINUATION_TYPES:
        z["behavior"] = {"pass": True, "fail": False}.get(
            judge["result"]["continuation_behavior"]
        )
        z["judge_behavior_reason"] = judge["result"]["continuation_reason"]
        if z["behavior"] is not None:
            z.pop("uncertain", None)
    z["semantic_score"] = judge["result"]["score"] if judge else None
    z["semantic_uncertain"] = judge["result"]["uncertain"] if judge else False
    return z


def summarize_category(ds):
    """Aggregate behavior, timing diagnostics and content for one category."""
    eligible = [z for z in ds if z["eligible"] is True]
    known = [z for z in eligible if z["behavior"] is not None]
    semantic = [z["semantic_score"] for z in ds if z["semantic_score"] is not None]
    passed = sum(z["behavior"] is True for z in known)
    c = {
        "total": len(ds),
        "eligible": len(eligible),
        "determinate": len(known),
        "passed": passed,
        "conditional_pass_rate": passed / len(known) if known else None,
        "coverage": len(known) / len(ds) if ds else None,
        "semantic_mean_0_4": statistics.mean(semantic) if semantic else None,
        "semantic_determinate": len(semantic),
        "semantic_uncertain": sum(z["semantic_uncertain"] for z in ds),
        "uncertain_reasons": dict(
            collections.Counter(
                z.get("uncertain", "unknown")
                for z in ds
                if z["eligible"] is not True or z["behavior"] is None
            )
        ),
        "triggered": sum(z["triggered"] is True for z in ds),
        "trigger_required": sum(z["triggered"] is not None for z in ds),
    }
    for key in [
        "playback_after_last_input_packet_s",
        "playback_lexical_estimate_s",
        "stop_after_event_packet_s",
        "block_latency_s",
    ]:
        vs = [z[key] for z in ds if z.get(key) is not None]
        c[key] = (
            {"n": len(vs), "mean": statistics.mean(vs), "median": statistics.median(vs)}
            if vs
            else None
        )
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--acoustic-only", action="store_true")
    a = p.parse_args()
    with (ROOT / "data" / f"{a.subset}.jsonl").open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest]
    if a.limit:
        rows = rows[: a.limit]
    details = []
    judge_calls = 0
    judge_models = set()
    for row in rows:
        d = ROOT / "runs" / MODEL_NAME / row["sample_id"]
        r = json.loads((d / "run.json").read_text())
        ac = json.loads((d / "acoustic.json").read_text())
        j = None if a.acoustic_only else json.loads((d / "judge.json").read_text())
        assert (
            not r["error"]
            and ac["status"] == "complete"
            and (j is None or j["status"] == "complete")
        ), row["sample_id"]
        if j is not None:
            assert j["audio_sha256"] == ac["audio_sha256"]
            judge_calls += j.get("judge_called", True)
            if j.get("judge_model"):
                judge_models.add(j["judge_model"])
        z = score_one(row, r, ac, j)
        save(d / "point_score.json", z)
        details.append(z)
    by_type = collections.defaultdict(list)
    for detail in details:
        by_type[detail["type"]].append(detail)
    cats = {event_type: summarize_category(by_type[event_type]) for event_type in TYPES}
    macro = [c["conditional_pass_rate"] for c in cats.values()]
    sem = [z["semantic_score"] for z in details if z["semantic_score"] is not None]
    summary = {
        "status": "complete",
        "subset": a.subset,
        "n": len(details),
        "expected": len(rows),
        "scope": "smoke" if a.limit else "provided_partition",
        "inference_completed": len(details),
        "asr_completed": len(details),
        "judge_completed": 0 if a.acoustic_only else len(details),
        "judge_calls": judge_calls,
        "judge_models": sorted(judge_models),
        "processing_errors": 0,
        "categories": cats,
        "semantic_mean_0_4": statistics.mean(sem) if sem else None,
        "semantic_determinate": len(sem),
        "joint_reference_macro": statistics.mean(macro)
        if all(v is not None for v in macro)
        else None,
        "aggregation": "five-category mean of conditional behavior pass rates",
        "official_weighted_competition_score": None,
        "official_weighted_competition_score_status": "baseline does not define a competition weight",
    }
    save(ROOT / f"{a.subset}_summary.json", summary)
    save(ROOT / f"{a.subset}_point_details.json", details)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
