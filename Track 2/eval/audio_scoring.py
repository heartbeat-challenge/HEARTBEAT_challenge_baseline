# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Attach curated Audio scoring metadata to existing references without re-extracting media."""
import argparse
from pathlib import Path

from .audio import attach_scoring, metadata_issues
from .common import load_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--audio-scoring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New reference JSONL; originals are preserved")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; references are never overwritten")
    rows = attach_scoring(load_rows(args.references), load_rows(args.audio_scoring))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    missing = [{"id": r["id"], "issues": metadata_issues(r)} for r in rows
               if r.get("subtrack") == "audio" and metadata_issues(r)]
    print(len(rows), "references;", len(missing), "invalid scoring annotations")
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
