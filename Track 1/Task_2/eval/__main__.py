"""Evaluate HEARTBEAT Track I Task 2 on Dev or internal Test."""

import argparse
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from data_io import positive_int


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def duplex(args, parser):
    from .prepare import prepare, prepare_test

    if args.test_manifest and args.stage != "infer" and not args.test_labels:
        parser.error(
            "Task 2 Test scoring requires --test-labels. Use --stage infer for unlabeled Test."
        )
    if args.test_labels and not args.test_manifest:
        parser.error("--test-labels requires --test-manifest")
    required = {
        "HEARTBEAT_DUPLEX_REPO": args.duplex_repo,
        "HEARTBEAT_REFERENCE_AUDIO": args.reference_audio
        or args.model / "assets" / "HT_ref_audio.wav",
    }
    if args.stage != "infer":
        required.update(
            HEARTBEAT_ASR_MODEL=args.asr_model, HEARTBEAT_ALIGNER=args.aligner
        )
    if args.stage == "all":
        for name in (
            "HEARTBEAT_JUDGE_BASE_URL",
            "HEARTBEAT_JUDGE_API_KEY",
            "HEARTBEAT_JUDGE_MODEL",
        ):
            if not os.environ.get(name):
                parser.error(f"Set {name} for Task 2 semantic judging")
    for name, path in required.items():
        if path is None or not path.exists():
            parser.error(f"Missing path for {name}: {path}")
    if not (args.duplex_repo / "core" / "processors" / "unified.py").is_file():
        parser.error("--duplex-repo must contain core/processors/unified.py")
    root = args.output / args.subset
    code_root = Path(__file__).resolve().parents[1]
    with run_lock(root):
        if (root / "data").exists():
            parser.error(f"Output already used: {root}. Choose a new run directory.")
        if args.test_manifest:
            manifest = prepare_test(
                args.test_manifest,
                root / "data",
                args.subset,
                args.limit,
                args.test_labels,
            )
        else:
            manifest = prepare(args.parquet_dir, root / "data", args.subset, args.limit)
        env = dict(
            os.environ, **{key: str(path.resolve()) for key, path in required.items()}
        )
        env.update(
            HEARTBEAT_EVAL_ROOT=str(root),
            HEARTBEAT_BASE_MODEL=str(args.model.resolve()),
            HEARTBEAT_ADAPTER=str(args.adapter.resolve()) if args.adapter else "",
            HEARTBEAT_MODEL="venus",
            HEARTBEAT_END_PROTOCOL="legacy",
        )
        subset = ["--subset", args.subset]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.infer",
                "--model",
                "venus",
                "--manifest",
                str(manifest),
            ],
            env=env,
            cwd=code_root,
            check=True,
        )
        if args.stage != "infer":
            asr_env = env.copy()
            if args.asr_python:
                asr_env.pop("PYTHONPATH", None)
            subprocess.run(
                [args.asr_python or sys.executable, "-m", "eval.asr", *subset],
                env=asr_env,
                cwd=code_root,
                check=True,
            )
            if args.stage == "all":
                subprocess.run(
                    [sys.executable, "-m", "eval.judge", *subset],
                    env=env,
                    cwd=code_root,
                    check=True,
                )
            command = [sys.executable, "-m", "eval.score", *subset]
            if args.stage == "acoustic":
                command.append("--acoustic-only")
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            subprocess.run(command, env=env, cwd=code_root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet-dir", type=Path)
    source.add_argument(
        "--test-manifest",
        type=Path,
        help="Test audio manifest or its containing directory",
    )
    parser.add_argument(
        "--test-labels", type=Path, help="Directory containing per-case Test labels"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--subset", choices=("static", "dynamic"), required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "acoustic", "infer"),
        default="all",
    )
    parser.add_argument("--duplex-repo", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--asr-model", type=Path)
    parser.add_argument("--aligner", type=Path)
    parser.add_argument(
        "--asr-python", help="Python executable for a separate Qwen3-ASR environment"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.asr_python and "/" in args.asr_python:
        args.asr_python = str(Path(args.asr_python).absolute())
    if not (args.model / "config.json").is_file():
        parser.error(f"Missing model config in {args.model}")
    if args.adapter and not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"Missing adapter config in {args.adapter}")
    duplex(args, parser)


if __name__ == "__main__":
    main()
