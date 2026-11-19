# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Merge independently generated GPU shards without dropping missing tasks."""
import argparse
from pathlib import Path
from .common import load_rows, validate_prediction, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = load_rows(args.inputs)
    predictions = [p for f in args.shards for p in load_rows(f)]
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["id"] for r in expected}:
        raise ValueError("Missing, duplicated or unexpected shard IDs")
    protocols = {p["protocol_sha256"] for p in predictions}
    if len(protocols) != 1:
        raise ValueError("Shards used different models/data/decoding settings")
    for row in expected:
        validate_prediction(by_id[row["id"]], row)
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, [by_id[r["id"]] for r in expected])
    print(len(predictions), "tasks merged; errors retained:", sum(p["status"] != "complete" for p in predictions))


if __name__ == "__main__":
    main()
