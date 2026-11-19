# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Download or read Track 2 Dev, run Venus and compute the Dev score."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from train.prepare import parquet_files
from .common import atomic_json, digest_file, load_rows, write_rows
from .judge import GEMINI_MODEL
from .prepare import prepare_evaluation
from . import audio, video

TRACK = Path(__file__).resolve().parents[1]


def fetch_dev(task, limit, output):
    partition = "task2_audio" if task == "audio" else "task1_video"
    root = TRACK.parent
    download = output / "download"
    download.mkdir(exist_ok=True)
    folder = download / partition / "dev"
    if folder.is_dir():
        manifest = json.loads((folder / "export-manifest.json").read_text())
        inventory = json.loads((root / "data_assets/packs.json").read_text())
        if (manifest["data_commit"] != inventory["data_commit"] or not manifest["round_trip_verified"]
                or manifest["sample_rows_per_partition"] != (limit or None)):
            raise ValueError("Cached public Dev selection differs. Choose a new output directory.")
        return folder
    with TemporaryDirectory(dir=download) as temp:
        stage = Path(temp)
        command = [sys.executable, str(root / "data.py"), "--track", "2", "--task", partition,
                   "--split", "dev", "--format", "parquet", "--output-dir", str(stage)]
        if limit:
            command.extend(["--sample-rows", str(limit), "--max-download-mib", str(128 * limit)])
        else:
            # Keep verified packs and partial HTTP downloads if a Parquet export is interrupted.
            packs = download / "packed"
            packs.mkdir(exist_ok=True)
            (stage / "packed").symlink_to(packs.resolve(), target_is_directory=True)
        subprocess.run(command, check=True)
        where = "sample" if limit else "export"
        exported = stage / where / "parquet" / "track2" / partition / "dev"
        (stage / "export-manifest.json").rename(exported / "export-manifest.json")
        folder.parent.mkdir(exist_ok=True)
        exported.rename(folder)
    return folder


def prepare(source, output, args):
    """Publish the prepared inputs and media together after successful validation."""
    output = output.resolve()
    if output.is_dir():
        manifest = json.loads((output / "manifest.json").read_text())
        for kind in ("inputs", "references"):
            if digest_file(output / f"{kind}.jsonl") != manifest[f"{kind}_sha256"]:
                raise ValueError(f"Prepared {kind} have changed. Use a new output directory.")
        return
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "dev"
        manifest = prepare_evaluation(source, stage, args.task, args.limit)
        inputs = load_rows(stage / "inputs.jsonl")
        for row in inputs:
            row["media_path"] = str(output / Path(row["media_path"]).relative_to(stage))
        write_rows(stage / "inputs.jsonl", inputs)
        manifest["inputs_sha256"] = digest_file(stage / "inputs.jsonl")
        atomic_json(stage / "manifest.json", manifest)
        stage.rename(output)


def run(args):
    output = args.output.resolve()
    if not args.resume and any(p.name != ".dev.lock" for p in output.iterdir()):
        raise ValueError("Choose an empty output directory, or pass --resume for this run.")
    if args.resume and not (output / "run.json").exists():
        if any(p.name not in {".dev.lock", "download", "status.json"} for p in output.iterdir()):
            raise ValueError("This directory has no matching evaluation run. Choose a new output directory.")
    source = args.parquet_dir.resolve() if args.parquet_dir else fetch_dev(args.task, args.limit, output)
    inputs = parquet_files(source)
    plan = {"task": args.task, "limit": args.limit, "model": str(args.model.resolve()),
            "adapter": str(args.adapter.resolve()) if args.adapter else None,
            "shards": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in inputs]}
    plan["scoring_protocol"] = (audio if args.task == "audio" else video).PROTOCOL
    if args.task == "video":
        plan["targets_sha256"] = digest_file(video.DEV_TARGETS)
    plan_path = output / "run.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        previous.pop("judge_model", None)  # Older runs stored the local judge path with inference settings.
        if previous != plan:
            raise ValueError("Dev data or model selection differs from this run. Choose a new output directory.")
    atomic_json(plan_path, plan)
    status_path = output / "status.json"
    dev, inference, judging = output / "dev", output / "inference", output / "judge"
    stage = "prepare"
    try:
        atomic_json(status_path, {"status": "running", "stage": stage})
        prepare(source, dev, args)
        stage = "inference"
        atomic_json(status_path, {"status": "running", "stage": stage})
        command = [sys.executable, "-m", "eval.infer", "--inputs", str(dev / "inputs.jsonl"),
                   "--model", str(args.model.resolve()), "--device", args.device, "--output", str(inference)]
        if args.adapter:
            command += ["--adapter", str(args.adapter.resolve())]
        if args.resume and (inference / "protocol.json").is_file():
            command += ["--resume", "--retry-errors"]
        subprocess.run(command, check=True, cwd=TRACK)

        files = ["--references", str(dev / "references.jsonl"), "--predictions", str(inference / "predictions.jsonl")]
        stage = "judge"
        atomic_json(status_path, {"status": "running", "stage": stage})
        subprocess.run([sys.executable, "-m", "eval.judge", *files, "--output", str(judging)], check=True, cwd=TRACK)
        stage = "score"
        atomic_json(status_path, {"status": "running", "stage": stage})
        metrics = output / "metrics.json"
        subprocess.run([sys.executable, "-m", "eval.score", *files, "--judgments", str(judging / "judgments.jsonl"),
                        "--output", str(metrics)], check=True, cwd=TRACK)
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(status_path, {"status": "incomplete", "stage": stage, "error": type(exc).__name__})
        raise
    report = json.loads(metrics.read_text())
    judgments = load_rows(judging / "judgments.jsonl")
    report["dev_judge"] = {"backend": "gemini", "model": GEMINI_MODEL,
                           "rated_prefixes": sum(len(pair["prefix_scores"]) for row in judgments for pair in row["pairs"])}
    atomic_json(metrics, report)
    summary = report[f"{args.task}_summary"]
    scores = summary["windows"]["3.0"]
    field = "headline_joint_f1" if scores["headline_joint_f1"] is not None else "macro_joint_f1"
    atomic_json(status_path, {"status": "complete", "samples": report["expected_tasks"], "scope": summary["scope"],
                              "metric": field, "f1": scores[field]})
    print(f"{args.task.title()} Dev · {report['expected_tasks']} samples · {summary['scope']} · {field} = {scores[field]}")
    print(f"Metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    parser.add_argument("--task", choices=("audio", "video"), required=True)
    parser.add_argument("--adapter", type=Path, help="Trained LoRA adapter. Omit to evaluate the base model.")
    parser.add_argument("--model", type=Path, help="Base Venus checkpoint. Defaults to the corresponding models/ directory.")
    parser.add_argument("--parquet-dir", type=Path, help="Existing Dev Parquet partition. Defaults to the public Dev download.")
    parser.add_argument("--output", type=Path, help="Run directory. Defaults to eval_outputs/<task>.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate the first N samples. Default: entire Dev split.")
    parser.add_argument("--resume", action="store_true", help="Resume this run, including interrupted inference and judging.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    suffix = "Audio" if args.task == "audio" else "Omni"
    args.model = args.model or TRACK / "models" / f"Realtime-Venus-{suffix}"
    if not (args.model / "config.json").is_file():
        parser.error(f"Base model not found at {args.model}. Download the Venus checkpoint or pass --model.")
    if args.adapter and not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"LoRA adapter not found at {args.adapter}")
    args.output = args.output or Path("eval_outputs") / args.task
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".dev.lock"), timeout=0):
        run(args)


if __name__ == "__main__":
    main()
