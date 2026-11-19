# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Extract Dev media; write reference-free inference inputs and separate GT."""
import argparse
import hashlib
from pathlib import Path

from train.prepare import parquet_files, probe_media, validate_partition, validate_row
from .common import atomic_json, digest_file, load_rows, write_rows
from .video import attach_dev_targets, attach_targets, resolve_targets
from . import audio


def prepare_evaluation(source, output, task, limit=0, decode_check=False, video_targets=None, audio_scoring=None):
    import pyarrow.parquet as pq

    files = parquet_files(source)
    if limit < 0:
        raise ValueError("Invalid limit")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty evaluation data directory")
    validate_partition(files, task, "dev")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    (output / "media").mkdir()
    inputs, references, seen, checked = [], [], set(), set()
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            validate_row(row)
            if row.get("track_id") != "track_2" or row.get("subtrack") != task or row.get("split") != "dev":
                raise ValueError("This entrypoint requires released Track 2 Dev rows for the selected task")
            if row["id"] in seen:
                raise ValueError("Duplicate task ID")
            seen.add(row["id"])
            blob = row["media"].pop("bytes")
            if not isinstance(blob, bytes) or not blob:
                raise ValueError("Missing embedded media")
            sha = hashlib.sha256(blob).hexdigest()
            media = output / "media" / (sha + (".wav" if task == "audio" else ".mp4"))
            if not media.exists():
                media.write_bytes(blob)
            key = (sha, row["media"]["duration"])
            if key not in checked:
                probe_media(media, row, decode_check)
                checked.add(key)
            level = row.get("level")
            modality = row.get("audio_type")
            category = row.get("task_type")
            metadata = dict(id=row["id"], subtrack=task, level=level.upper() if isinstance(level, str) else level,
                            audio_type=modality.lower() if isinstance(modality, str) else modality,
                            task_type=category.rsplit("/", 1)[-1].lower() if isinstance(category, str) else category,
                            duration=row["media"]["duration"], question=row["question"])
            inputs.append(dict(metadata, media_path=str(media), media_sha256=sha))
            reference = dict(metadata, response=row["response"])
            if task == "audio" and row.get("scoring") is not None:
                # Arrow structs carry null slots for fields used by other rule types.
                reference["scoring"] = {k: v for k, v in row["scoring"].items() if v is not None}
            references.append(reference)
            if limit and len(inputs) >= limit:
                break
        if limit and len(inputs) >= limit:
            break
    if not inputs:
        raise ValueError("Empty Dev set")
    if task == "video":
        attach_dev_targets(references)
    if video_targets:
        if task != "video":
            raise ValueError("--video-targets is only for Video")
        selected = {r["id"] for r in references}
        target_rows = load_rows(video_targets)
        if limit:
            target_rows = [r for r in target_rows if r["id"] in selected]
        attach_targets(references, target_rows)
    if audio_scoring:
        if task != "audio":
            raise ValueError("--audio-scoring is only for Audio")
        selected = {r["id"] for r in references}
        target_rows = load_rows(audio_scoring)
        if limit:
            target_rows = [r for r in target_rows if r["id"] in selected]
        references = audio.attach_scoring(references, target_rows)
    if task == "audio":
        for inp, ref in zip(inputs, references):
            audio.resolve_scoring(ref)
            issues = audio.metadata_issues(ref)
            if issues:
                raise ValueError(f"Invalid Audio scoring metadata for {ref['id']}: {issues}")
            contract = audio.answer_format(ref)
            if contract:
                inp["answer_format"] = contract
    else:
        for ref in references:
            resolve_targets(ref)
    write_rows(output / "inputs.jsonl", inputs)
    write_rows(output / "references.jsonl", references)
    manifest = dict(schema_version="heartbeat_eval_v1", tasks=len(inputs), task=task, limit=limit,
                    inputs_sha256=digest_file(output / "inputs.jsonl"),
                    references_sha256=digest_file(output / "references.jsonl"))
    if task == "video":
        manifest["semantic_fallback_events"] = sum(
            e.get("scoring_fallback") == "llm_judge" for ref in references for e in ref["response"]
        )
    else:
        manifest["scoring_modes"] = {
            mode: sum(ref["scoring"]["mode"] == mode for ref in references) for mode in audio.MODES
        }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["audio", "video"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--decode-check", action="store_true")
    parser.add_argument("--video-targets", type=Path, help="Reference-only JSONL with authoritative count/position/state_to fields")
    parser.add_argument("--audio-scoring", type=Path, help="Reference-only mode/rule/target metadata JSONL")
    args = parser.parse_args()
    print(prepare_evaluation(args.parquet_dir, args.output, args.task, args.limit, args.decode_check, args.video_targets, args.audio_scoring))


if __name__ == "__main__":
    main()
