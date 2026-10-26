<div align="center">

# ICASSP 2027 HEARTBEAT Challenge

**Data access and preparation**

<a href="https://heartbeat-challenge.github.io/"><img src="https://img.shields.io/badge/HEARTBEAT-Challenge_Website-2563EB?style=for-the-badge" alt="HEARTBEAT Challenge Website"></a>
<a href="https://huggingface.co/inclusionAI/Realtime-Venus/tree/main/data/HEARTBEAT_Challenge"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Dataset"></a>
<a href="https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README.md"><img src="https://img.shields.io/badge/Data-Field_Reference-0F766E?style=for-the-badge" alt="Data Field Reference"></a>

[English](README.md) · [简体中文](README_zh.md)

</div>

HEARTBEAT studies a central question in real-time interaction: **when should a system respond, and what should it say?** The challenge evaluates systems that listen to continuous audio or observe continuous audio-visual streams, understand the evolving context, and respond at the appropriate moment.

This repository helps participants prepare the challenge data for their own systems. Select a track, task and split, then use a single command to download the corresponding data and export it as **Parquet** or **JSON with original audio/video files**. A small-sample mode lets you inspect the data before downloading a full partition.

## Challenge tracks

### Track I · Reciprocal Audio Turn-Taking

Track I focuses on conversational participation: deciding when to start responding, keep listening, stop speaking, or continue speaking. Scenarios include clarification, corrections, interruptions, information updates, and task suspension or resumption.

- **Offline evaluation:** select one of four candidate actions given an audio prefix and assistant instructions. Predictions are evaluated using Top-1 accuracy.
- **Online evaluation:** process audio incrementally while generating speech. Output audio and interaction logs capture response content, speech timing, and intervals without a response.

The Offline and Online settings are evaluated and ranked separately.

### Track II · Streaming Proactive Understanding

Track II focuses on responding to relevant events in a continuous stream after receiving a user instruction. Tasks span **L1 Perception**, **L2 Comprehension**, and **L3 Reasoning**.

- **Audio-visual understanding:** recognize and respond to events using both visual and audio information.
- **Audio-only understanding:** recognize and respond to events in continuous audio.

Both settings use Online evaluation, with separate rankings. Systems process chunks sequentially and produce a JSON output for every chunk, including chunks without a response. A response is timestamped at the end of the chunk that produces it.

## Get started

Clone this repository and install the data dependencies. Python 3.10–3.12 is supported; the local validation used Python 3.12.

```bash
git clone https://github.com/heartbeat-challenge/HEARTBEAT_challenge_baseline.git
cd HEARTBEAT_challenge_baseline

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start with **two samples** from the Track I / Task 2 / static development subset:

```bash
bash data.sh --track 1 --task 2 --split dev --subset static \
  --format parquet --sample-rows 2 --output-dir downloads/quickstart
```

This retrieves the required remote blocks and writes a small Parquet file. It does not download the complete partition. The resulting file is:

```text
downloads/quickstart/sample/parquet/track1/task2/dev/static/00000/data.parquet
```

Read it with PyArrow:

```python
import pyarrow.parquet as pq

path = "downloads/quickstart/sample/parquet/track1/task2/dev/static/00000/data.parquet"
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
print(row.keys())
```

The Bash entry point works on macOS, Linux and Windows with WSL/Git Bash. Without Bash, use `python data.py` with the same arguments. To choose an interpreter explicitly, set `PYTHON=/path/to/python`.

## Choose your data

The release contains **132,571 records across nine partitions**, approximately **599.83 GB** in packed form. Download only the partitions required for your experiments.

| Track | Task identifier | Split | Subset | Records | Packed size |
|---|---|---|---|---:|---:|
| I | `task1` | `train` | — | 18,000 | 11.12 GB |
| I | `task1` | `dev` | — | 1,000 | 0.59 GB |
| I | `task2` | `train` | — | 50,000 | 196.39 GB |
| I | `task2` | `dev` | `dynamic` | 1,000 | 0.24 GB |
| I | `task2` | `dev` | `static` | 1,000 | 0.24 GB |
| II | `task1_video` | `train` | — | 30,000 | 294.31 GB |
| II | `task1_video` | `dev` | — | 1,000 | 9.28 GB |
| II | `task2_audio` | `train` | — | 29,371 | 84.12 GB |
| II | `task2_audio` | `dev` | — | 1,200 | 3.53 GB |

Sizes are rounded decimal GB. In command-line filters, `--task 1` selects `task1` in Track I and `task1_video` in Track II; `--task 2` selects `task2` and `task2_audio`, respectively.

Preview a selection before downloading:

```bash
bash data.sh --track 2 --split train --dry-run
```

| Option | Usage |
|---|---|
| `--track` | `1`, `2`, or multiple values such as `1 2` |
| `--task` | `1`, `2`, or a task identifier such as `task1_video` |
| `--split` | `train`, `dev`, or both |
| `--subset` | `dynamic` or `static`; only applies to Track I / Task 2 / dev |
| `--format` | `parquet` (default), `json`, or `packed` |
| `--output-dir` | Directory for downloaded and exported data |
| `--sample-rows` | Export only the first N rows **per selected partition** |
| `--dry-run` | Show partitions, files and total download size without downloading |
| `--list` | List all available partitions |
| `--all` | Select the entire release; use without other selection filters |

Omitted filters include all matching values. For example, `--task 2` includes both tracks and both splits; Track I / Task 2 / dev includes both subsets unless `--subset` is specified. Running without a selection does not start a download.

## Download and export

### Parquet · for training and data processing

Download a complete development partition and export its records:

```bash
bash data.sh --track 1 --task 1 --split dev \
  --format parquet --output-dir downloads/track1-task1-dev
```

Each source shard is exported separately. Parquet preserves the original Arrow field types, nested structures, null values and binary media fields. Audio and video remain encoded bytes inside their original fields.

Read one record without loading the full shard into memory:

```python
import pyarrow.parquet as pq

path = "downloads/track1-task1-dev/export/parquet/track1/task1/dev/00000/data.parquet"
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
print(row["sample_id"])
audio_bytes = row["audio"]["bytes"]
```

The example fields above belong to Track I / Task 1. Consult the [field reference](https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README.md) for other partitions.

### JSON + media · for file-based pipelines

Export records as JSON Lines and store binary content in separate files:

```bash
bash data.sh --track 1 --task 2 --split dev --subset static \
  --format json --output-dir downloads/track1-task2-static
```

Each line in `records.jsonl` is a record. Binary fields are replaced with a relative file reference:

```json
{
  "file": "media/<sha256>.wav",
  "bytes": 123456,
  "sha256": "<sha256>"
}
```

Audio/video is saved **without transcoding**. Files are named by their SHA256 to avoid filename collisions; identical content within a shard is reused. Extensions are identified from content signatures, with `.bin` used for unrecognized formats. Other fields retain their nested structure. Decimal and special scalar values use schema-aware representations for lossless restoration.

Keep `records.jsonl`, `schema.arrow` and `media/` together. To convert an exported JSON shard back to Parquet:

```bash
python json_to_parquet.py \
  --input-dir downloads/track1-task2-static/export/json/track1/task2/dev/static/00000 \
  --output downloads/static-restored.parquet
```

### Packed data · for offline Zarr access

Use `--format packed` to download the selected packs and create a local ReferenceFS index without exporting another format:

```bash
bash data.sh --track 1 --task 1 --split dev \
  --format packed --output-dir downloads/track1-task1-packed
```

```python
from examples.heartbeat_access import generate_rows

row = next(generate_rows(
    "downloads/track1-task1-packed/packed/config.json",
    track="track1", task="task1", split="dev", max_rows=1,
))
print(row["sample_id"])
```

The reader also provides `as_pandas` and `as_datasets`. For Track II, use `task1_video` or `task2_audio` in Python; for Track I / Task 2 / dev, pass `subset="dynamic"` or `subset="static"`.

## Output layout

```text
<output-dir>/
├── packed/                         # Complete downloads and offline index
├── export/                         # Full-partition exports
│   ├── parquet/<partition>/<shard>/
│   │   ├── data.parquet
│   │   ├── schema.arrow
│   │   └── export.json
│   └── json/<partition>/<shard>/
│       ├── records.jsonl
│       ├── schema.arrow
│       ├── media/
│       └── export.json
├── sample/                         # Sample exports use this instead of export/
└── export-manifest.json
```

Only the requested format is generated. Full exports retain the downloaded packs, so allow space for both. Parquet exports preserve logical content and schema, but their compression, row-group layout and physical file hashes can differ from the original source files.
