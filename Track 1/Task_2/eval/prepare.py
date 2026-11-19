"""Materialize release Task 2 media and build the native duplex manifest."""

import json
from pathlib import Path

from data_io import parquet_rows, write_media


def prepare(source, output, subset, limit):
    import soundfile as sf

    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    media.mkdir(exist_ok=True)
    manifest = output / f"{subset}.jsonl"
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(parquet_rows(source, limit)):
            if row.get("subset", subset) != subset:
                raise ValueError(
                    f"Unexpected subset for {row['sample_id']}: {row['subset']}"
                )
            sample_id = row["sample_id"]
            if Path(sample_id).name != sample_id or sample_id in {"", ".", ".."}:
                raise ValueError(f"Invalid sample ID: {sample_id!r}")
            annotations = row["annotations"]
            gt = annotations.get("gt") or annotations.get("ground_truth")
            event_type = gt.get("event_type") or gt["behavior"]
            input_audio = write_media(row["input_audio"], media, f"{index}_input")
            history = annotations["history"] if subset == "dynamic" else []
            if subset == "dynamic":
                spec = dict(row["source_manifest"]["dynamic_spec"])
                needs_event = event_type in {"cut", "backchannel", "background_speech"}
                spec["clip_source"] = (
                    write_media(row["event_audio"], media, f"{index}_event")
                    if needs_event
                    else input_audio
                )
                request_end = spec["prefix_end_s"]
            else:
                spec = dict(
                    context_start_s=0.0,
                    prefix_end_s=sf.info(input_audio).duration,
                    clip_source=input_audio,
                    clip_start_s=0.0,
                    clip_end_s=0.0,
                    arm_timeout_s=30,
                    post_event_s=45,
                )
                request_end = gt["request_span_s"][1]
            item = dict(
                sample_id=sample_id,
                language=row["language"],
                subset=subset,
                input_audio=input_audio,
                history=history,
                ground_truth=gt,
                dynamic_spec=spec,
                request_alignment_end_s=request_end,
                events=[
                    dict(
                        event_type=event_type,
                        pause_span=gt.get("pause_span") or gt.get("pause_span_s"),
                    )
                ],
            )
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    return manifest


def prepare_test(source, output, subset, limit, labels=None):
    """Use the internal Test layout without copying the original audio."""
    from itertools import islice

    source = source.resolve()
    if source.is_dir():
        source /= "manifest.jsonl"
    base = source.parent
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{subset}.jsonl"
    temporary = manifest.with_suffix(".tmp")
    seen = set()

    def media_path(relative):
        path = (base / relative).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Missing Test media or sidecar: {path}")
        return str(path)

    with (
        source.open(encoding="utf-8") as rows,
        temporary.open("w", encoding="utf-8") as stream,
    ):
        for line in islice((line for line in rows if line.strip()), limit):
            row = json.loads(line)
            sample_id = row["sample_id"]
            if (
                Path(sample_id).name != sample_id
                or sample_id in {"", ".", ".."}
                or sample_id in seen
                or not sample_id.lower().startswith(subset + "_")
            ):
                raise ValueError(
                    f"Invalid or duplicate {subset} sample ID: {sample_id}"
                )
            seen.add(sample_id)
            input_audio = media_path(row["input_audio"])
            gt = {}
            if labels:
                name = "gt.json" if subset == "dynamic" else "ground_truth.json"
                gt = json.loads((labels / sample_id / name).read_text(encoding="utf-8"))
            if subset == "dynamic":
                history = json.loads(
                    Path(media_path(row["history_file"])).read_text(encoding="utf-8")
                )
                spec = dict(row["dynamic_spec"])
                if row.get("event_audio"):
                    spec["clip_source"] = media_path(row["event_audio"])
                request_end = spec["prefix_end_s"]
            else:
                history = []
                spec = dict(
                    context_start_s=0.0,
                    prefix_end_s=row["duration_s"],
                    clip_source=input_audio,
                    clip_start_s=0.0,
                    clip_end_s=0.0,
                    arm_timeout_s=30,
                    post_event_s=45,
                )
                request_end = gt.get("request_span_s", [0.0, row["duration_s"]])[1]
            event_type = gt.get("event_type") or gt.get("behavior")
            item = dict(
                sample_id=sample_id,
                language=row["language"],
                subset=subset,
                input_audio=input_audio,
                history=history,
                ground_truth=gt,
                dynamic_spec=spec,
                request_alignment_end_s=request_end,
                events=[
                    dict(
                        event_type=event_type or "unlabeled",
                        pause_span=gt.get("pause_span") or gt.get("pause_span_s"),
                    )
                ],
            )
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    if not seen:
        raise ValueError(f"No Test records found in {source}")
    temporary.replace(manifest)
    return manifest
