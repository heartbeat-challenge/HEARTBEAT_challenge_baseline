# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Extract released Parquet media and build one-second Venus supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

SYSTEM = "Streaming Omni Conversation."
END = "<|turn_eos|><|chunk_eos|>"


def parquet_files(source):
    """Accept a shard, flat shards, or data.sh's partition/<shard>/data.parquet."""
    source = Path(source).expanduser().resolve()
    files = [source] if source.is_file() and source.suffix == ".parquet" else (
        sorted(source.rglob("*.parquet")) if source.is_dir() else [])
    if not files:
        raise ValueError(f"No Parquet shards found: {source}")
    return files


def validate_partition(files, subtrack, split):
    """Check all shard identities before extracting media, even for --limit runs."""
    import pyarrow.parquet as pq

    seen = set()
    for file in files:
        parquet = pq.ParquetFile(file)
        required = {"id", "track_id", "subtrack", "split", "media", "question", "response"}
        if not required.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Not a released Track II Parquet shard: {file}")
        for batch in parquet.iter_batches(batch_size=4096, columns=["id", "track_id", "subtrack", "split"]):
            for row in batch.to_pylist():
                if (row["track_id"], row["subtrack"], row["split"]) != ("track_2", subtrack, split):
                    raise ValueError("Wrong track/subtrack/split; pass one partition directory, "
                                     f"not its parent: {file} ({row['id']})")
                if not isinstance(row["id"], str) or not row["id"]:
                    raise ValueError("A nonempty public task ID is required")
                if row["id"] in seen:
                    raise ValueError(f"Duplicate task ID: {row['id']}")
                seen.add(row["id"])
    if not seen:
        raise ValueError("No tasks found")
    return len(seen)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return float(value)


def validate_row(row):
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("A nonempty public task ID is required")
    if row.get("subtrack") not in ("audio", "video"):
        raise ValueError("subtrack must be audio or video")
    duration = number(row["media"]["duration"])
    if duration <= 0:
        raise ValueError("Media duration must be positive")
    questions, responses = row["question"], row["response"]
    if len(questions) != 1 or not responses:
        raise ValueError("Expected one question and at least one response")
    previous = number(questions[0]["time"])
    for entry in questions + responses:
        timestamp = number(entry["time"])
        if not 0 <= previous <= timestamp <= duration:
            raise ValueError("Timestamps must be ordered, after the question, and within the clip")
        content = entry["content"]
        if not isinstance(content, str) or not content.strip() or "<|" in content:
            raise ValueError("Expected nonempty plain text without model control tokens")
        previous = timestamp
    return duration


def convert(row, media):
    duration = validate_row(row)
    question = row["question"][0]
    q_unit = max(0, math.ceil(question["time"]) - 1)
    events = {}
    for response in row["response"]:
        unit = max(q_unit, math.ceil(response["time"]) - 1, 0)
        events.setdefault(unit, []).append(response["content"])
    messages = []
    for i in range(math.ceil(duration)):
        messages.append({"role": "user", "content": question["content"] if i == q_unit else ""})
        target = "<|listen|>"
        if i in events:
            target = "<|speak|>" + "\n".join(events[i]) + END
        messages.append({"role": "assistant", "content": target})
    video = row["subtrack"] == "video"
    return dict(id=row["id"], raw_prompt=messages[:-1], raw_response=messages[-1:],
                raw_system=SYSTEM, raw_videos=[str(media)] if video else [],
                raw_audios=[] if video else [str(media)], training_mode="duplex_streaming",
                instruction_mode="text", use_vision=video, use_audio=True, duration=duration,
                dataset_name="heartbeat_" + row["subtrack"])


def probe_media(path, row, decode_check=False):
    """Check the actual extracted file without rewriting or resampling it."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                             "-of", "json", str(path)], capture_output=True, text=True,
                            timeout=120, check=True)
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    kind = "audio" if row["subtrack"] == "audio" else "video"
    if not any(s.get("codec_type") == kind for s in streams):
        raise ValueError(f"Missing {kind} stream: {row['id']}")
    actual = float(info.get("format", {}).get("duration", 0))
    if not math.isfinite(actual) or abs(actual - row["media"]["duration"]) > .25:
        raise ValueError(f"Media/annotation duration mismatch: {row['id']} ({actual}s)")
    if decode_check:
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                        "-f", "null", "-"], capture_output=True, timeout=1200, check=True)


def prepare(parquet_dir, output, subtrack, split, limit=0, decode_check=False):
    import pyarrow.parquet as pq

    files = parquet_files(parquet_dir)
    if limit < 0:
        raise ValueError("limit cannot be negative")
    # A separate destination per split avoids cross-process manifest overwrites.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; failed conversions must be rerun into a new directory")
    validate_partition(files, subtrack, split)
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    media_dir = output / "media"
    media_dir.mkdir()
    name = f"heartbeat_{subtrack}_{split}"
    partial = output / (name + ".partial.jsonl")
    seen, checked, entries = set(), set(), []
    with partial.open("w", encoding="utf-8") as handle:
        for file in files:
            # Read one embedded media row at a time, rather than loading a whole shard.
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                row = batch.to_pylist()[0]
                validate_row(row)
                if row.get("track_id") != "track_2" or row.get("subtrack") != subtrack or row.get("split") != split:
                    raise ValueError(f"Wrong track/subtrack/split in {row['id']}")
                if row["id"] in seen:
                    raise ValueError(f"Duplicate task ID: {row['id']}")
                seen.add(row["id"])
                data = row["media"].pop("bytes")
                if not isinstance(data, bytes) or not data:
                    raise ValueError(f"Missing embedded media bytes: {row['id']}")
                digest = hashlib.sha256(data).hexdigest()
                media = media_dir / (digest + (".wav" if subtrack == "audio" else ".mp4"))
                if not media.exists():
                    media.write_bytes(data)
                # Check duration for every distinct annotation/file combination.
                check_key = (digest, row["media"]["duration"])
                if check_key not in checked:
                    probe_media(media, row, decode_check)
                    checked.add(check_key)
                handle.write(json.dumps(convert(row, media), ensure_ascii=False) + "\n")
                entries.append(dict(id=row["id"], duration=row["media"]["duration"],
                                    responses=len(row["response"]), media_sha256=digest))
                if limit and len(entries) >= limit:
                    break
            if limit and len(entries) >= limit:
                break
    if not entries:
        raise ValueError("No tasks found")
    final = output / (name + ".jsonl")
    partial.replace(final)
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    manifest = dict(subtrack=subtrack, split=split, tasks=len(entries),
                    responses=sum(e["responses"] for e in entries), unique_files=len({e["media_sha256"] for e in entries}),
                    chunk_seconds=1, limit=limit, full_decode_checked=decode_check,
                    jsonl_sha256=digest.hexdigest(), samples=entries)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    info = {name: dict(file_name=final.name, formatting="minicpmo_runtime",
                       training_mode="duplex_streaming", instruction_mode="text",
                       use_vision=subtrack == "video", use_audio=True)}
    # Registration is the last operation: a failed conversion cannot be trained accidentally.
    (output / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="One train/dev partition (recursive shards), or one .parquet file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtrack", choices=["audio", "video"], required=True)
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; 0 reads all rows")
    parser.add_argument("--decode-check", action="store_true", help="Also fully decode each unique file with FFmpeg")
    args = parser.parse_args()
    prepare(args.parquet_dir, args.output, args.subtrack, args.split, args.limit, args.decode_check)


if __name__ == "__main__":
    main()
