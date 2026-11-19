"""Convert Track I train partitions into the Venus runtime dataset format."""

import argparse
import json
import math
from pathlib import Path

from data_io import parquet_rows, positive_int, write_json, write_media

DATASET_NAME = "task2"


def assistant_units(row):
    """Build one-second supervision targets for full-duplex training."""
    count = math.ceil(row["duration"])
    if count < 1:
        raise ValueError(f"Invalid duration for {row['sample_id']}")
    targets = [""] * count
    for message in row["messages"]:
        if message["role"] != "assistant":
            continue
        start = max(0, min(count - 1, int(message["start_time"])))
        end = max(start, min(count - 1, math.ceil(message["end_time"]) - 1))
        for index in range(start, end + 1):
            text = "".join(
                word["text"]
                for word in message["word_timestamps"]
                if index <= word["start_time"] < index + 1
            )
            terminal = "<|turn_eos|>" if index == end else "<|chunk_eos|>"
            targets[index] += "<|speak|>" + text + terminal
    for target in targets:
        yield {"role": "user", "content": "<audio>"}
        yield {"role": "assistant", "content": target or "<|listen|>"}


def prepare(parquet_dir, output, limit=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    options = dict(
        training_mode="duplex_streaming",
        instruction_mode="audio",
        use_vision=False,
        use_audio=True,
    )
    runtime = output / "runtime.jsonl"
    temporary = runtime.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(parquet_rows(parquet_dir, limit)):
            paths = [write_media(row["audio"]["user"], audio_dir, f"{index}_user")]
            units = list(assistant_units(row))
            duration = row["duration"]
            record = dict(
                options,
                sample_id=row["sample_id"],
                dataset_name=DATASET_NAME,
                raw_system="You are a helpful voice assistant.",
                raw_audios=paths,
                raw_prompt=units[:-1],
                raw_response=units[-1:],
                duration=duration,
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(runtime)
    write_json(
        output / "dataset_info.json",
        {
            DATASET_NAME: dict(
                options, file_name=runtime.name, formatting="minicpmo_runtime"
            )
        },
    )
    return runtime


def add_arguments(parser):
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--limit", type=positive_int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    print(prepare(args.parquet_dir, args.prepared_dir, args.limit))


if __name__ == "__main__":
    main()
