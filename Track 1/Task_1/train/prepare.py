"""Convert Track I train partitions into the Venus runtime dataset format."""

import argparse
import json
from pathlib import Path

from data_io import mcq_prompt, parquet_rows, positive_int, write_json, write_media

DATASET_NAME = "task1"


def prepare(parquet_dir, output, limit=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    options = dict(
        training_mode="offline_chat",
        instruction_mode="text",
        use_vision=False,
        use_audio=True,
    )
    runtime = output / "runtime.jsonl"
    temporary = runtime.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(parquet_rows(parquet_dir, limit)):
            audio = row["audio"]
            paths = [write_media(audio, audio_dir, f"{index}_input")]
            prompt = mcq_prompt(row)
            if row.get("assistant_reference_audio"):
                paths.insert(
                    0,
                    write_media(
                        row["assistant_reference_audio"],
                        audio_dir,
                        f"{index}_reference",
                    ),
                )
                prompt = (
                    "第一段音频用于识别助手声音，第二段音频是待判断的对话。\n" + prompt
                )
            if row["answer"] not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid answer for {row['sample_id']}")
            units = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["answer"]},
            ]
            duration = audio["num_frames"] / audio["sampling_rate"]
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
