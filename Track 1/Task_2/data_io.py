"""Read either release Parquet shards or the baseline repository's exports."""

import json
from pathlib import Path


def parquet_rows(source, limit=None):
    import pyarrow.parquet as pq

    source = Path(source)
    shards = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    if not shards:
        raise ValueError(f"No Parquet shards found in {source}")
    seen = set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen.add(sample_id)
            yield row
            if limit is not None and len(seen) >= limit:
                return
    if not seen:
        raise ValueError(f"No records found in {source}")


def write_media(media, directory, name):
    payload = media["bytes"]
    if not payload:
        raise ValueError(f"Empty audio: {name}")
    suffix = ".flac" if payload.startswith(b"fLaC") else ".wav"
    path = directory / (name + suffix)
    path.write_bytes(payload)
    return str(path.resolve())


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def positive_int(value):
    import argparse

    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value
