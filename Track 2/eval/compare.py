# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Write a concise original-versus-finetuned Dev comparison from score reports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=float, default=3.)
    args = parser.parse_args()
    base, tuned = (json.loads(p.read_text()) for p in (args.base, args.finetuned))
    if base.get("video_protocol") != tuned.get("video_protocol"):
        raise ValueError("Reports use different Video scoring protocols")
    if base.get("video_aggregation") != tuned.get("video_aggregation"):
        raise ValueError("Reports use different Video aggregation policies")
    for field in ("audio_protocol", "audio_aggregation", "excluded_audio_samples"):
        if base.get(field) != tuned.get(field):
            raise ValueError(f"Reports use different Audio policies: {field}")
    for field in ("clock", "expected_tasks", "references_sha256", "content_judged", "settings_sha256", "judge_signature"):
        if base[field] != tuned[field]:
            raise ValueError(f"Reports are not comparable: {field}")
    window = str(float(args.window))
    lines = ["# Dev baseline comparison", "", "Diagnostic metrics; not an official leaderboard score.", "",
             f"Clock: {base['clock']}; tolerance: ±{args.window:g}s; expected tasks: {base['expected_tasks']}.",
             f"Completeness: base={base['status']}, finetuned={tuned['status']}.", ""]
    if base.get("video_summary"):
        a, b = base["video_summary"], tuned["video_summary"]
        if a["task_types"] != b["task_types"]:
            raise ValueError("Video task subsets differ")
        lines += [f"Video task-macro F1: {a['num_task_types']}/9 task types ({a['scope']}).",
                  "Each task has equal weight; the event-micro diagnostics below are not the Video total.", "",
                  "| Video total | Base F1 | Finetuned F1 | Change |", "|---|---:|---:|---:|"]
        for metric in ("macro_time_f1", "macro_joint_f1"):
            x, y = a["windows"][window][metric], b["windows"][window][metric]
            if x is not None and y is not None:
                lines.append(f"| {metric} | {100*x:.2f}% | {100*y:.2f}% | {100*(y-x):+.2f} pp |")
        if a["windows"][window]["headline_joint_f1"] is None or b["windows"][window]["headline_joint_f1"] is None:
            lines += ["", "Full nine-task Video total unavailable: subset coverage or incomplete evaluation. Any mean above is provisional."]
        lines += [""]
    if base.get("audio_summary"):
        a, b = base["audio_summary"], tuned["audio_summary"]
        if a["levels"] != b["levels"]:
            raise ValueError("Audio level subsets differ")
        lines += [f"Audio level-macro F1: {len(a['levels'])}/3 levels ({a['scope']}).",
                  "L1, L2 and L3 have equal weight.", "",
                  "| Audio total | Base F1 | Finetuned F1 | Change |", "|---|---:|---:|---:|"]
        for metric in ("macro_time_f1", "macro_joint_f1"):
            x, y = a["windows"][window][metric], b["windows"][window][metric]
            if x is not None and y is not None:
                lines.append(f"| {metric} | {100*x:.2f}% | {100*y:.2f}% | {100*(y-x):+.2f} pp |")
        if a["windows"][window]["headline_joint_f1"] is None or b["windows"][window]["headline_joint_f1"] is None:
            lines += ["", "Full three-level Audio total unavailable: subset coverage or incomplete evaluation."]
        lines += [""]
    lines += ["| Group (event-micro diagnostics) | Metric | Base F1 | Finetuned F1 | Change |", "|---|---|---:|---:|---:|"]
    for group in sorted(set(base["groups"]) & set(tuned["groups"])):
        for metric in ("timing", "joint"):
            a = base["groups"][group]["windows"][window][metric]
            b = tuned["groups"][group]["windows"][window][metric]
            if a is None or b is None:
                continue
            lines.append(f"| {group} | {metric} | {100*a['f1']:.2f}% | {100*b['f1']:.2f}% | {100*(b['f1']-a['f1']):+.2f} pp |")
    if base["status"] != "complete" or tuned["status"] != "complete":
        lines += ["", "Incomplete run: missing/error predictions remain in timing denominators; pending judgments suppress joint metrics."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
