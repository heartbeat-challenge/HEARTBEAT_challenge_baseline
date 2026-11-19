"""Evaluate HEARTBEAT Track I Task 1 on Dev or internal Test."""

import argparse
import fcntl
from contextlib import contextmanager
from pathlib import Path

from data_io import positive_int

from .infer import evaluate


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


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
        "--questions", type=Path, help="Task 1 Test questions JSON joined by scene ID"
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        help="Task 1 labeled records.jsonl or Task 2 per-case label directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not (args.model / "config.json").is_file():
        parser.error(f"Missing model config in {args.model}")
    if args.adapter and not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"Missing adapter config in {args.adapter}")
    if args.test_manifest and not args.questions:
        parser.error("Task 1 --test-manifest requires --questions")
    if (args.questions or args.test_labels) and not args.test_manifest:
        parser.error("--questions and --test-labels require --test-manifest")
    with run_lock(args.output):
        evaluate(args)


if __name__ == "__main__":
    main()
