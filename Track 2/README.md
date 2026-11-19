# HEARTBEAT Track 2 Baseline

English | [简体中文](README_zh.md)

Training and development-set evaluation for HEARTBEAT Challenge Track 2, with LoRA fine-tuning of [Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus) for audio and video tasks.

| Task | Model | Training config |
|---|---|---|
| Audio | Venus Audio | [audio.yaml](configs/audio.yaml) |
| Video | Venus Omni | [video.yaml](configs/video.yaml) |

The [training code](train/) prepares Parquet data and launches fine-tuning. The [evaluation code](eval/) runs streaming inference and scores Dev predictions.

## Installation

From the repository root, enter the baseline directory and install the dependencies for training and evaluation.

```bash
cd "Track 2"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install [FFmpeg](https://ffmpeg.org/download.html) so both `ffmpeg` and `ffprobe` are on your PATH.

## Model Preparation

Download the model with the Venus download script:

```bash
VENUS_REVISION=a1d95d3ac61416c8db34736b152241396850a6ed
hf download inclusionAI/Realtime-Venus download_models.py \
  --revision "$VENUS_REVISION" --local-dir models
python models/download_models.py --model audio --local-dir models \
  --revision "$VENUS_REVISION"
```

Use `--model audio` for Venus Audio, `--model omni` for Venus Omni, or `--model all` for both. The downloader saves the models and their shared `config.yaml` under `models/`.

## Data Preparation

From `Track 2/`, run the command for your task to download the training split and export it to Parquet. The training commands below use these export directories.

**Audio**

```bash
python ../data.py --track 2 --task task2_audio --split train --format parquet --output-dir downloads/audio
```

**Video**

```bash
python ../data.py --track 2 --task task1_video --split train --format parquet --output-dir downloads/video
```

The [dataset field reference](https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README.md) describes the Parquet columns for each task.

## Training

### Venus Audio

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m train \
  --task audio \
  --model models/Realtime-Venus-Audio \
  --parquet-dir downloads/audio/export/parquet/track2/task2_audio/train \
  --prepared-dir prepared/audio/train \
  --output outputs/venus_audio
```

### Venus Omni

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m train \
  --task video \
  --model models/Realtime-Venus-Omni \
  --parquet-dir downloads/video/export/parquet/track2/task1_video/train \
  --prepared-dir prepared/video/train \
  --output outputs/venus_omni
```

The default recipe trains for one epoch on 8 GPUs with LoRA rank 128. Results are saved to `--output`, including the LoRA adapter, checkpoints and training metrics.

Use `--config` to select a custom recipe. For a short run, add `--limit 16 --max-steps 2` and use new preparation and output directories. To resume, add `--resume-from-checkpoint` with the path to a checkpoint in the same output directory.

## Evaluation

From `Track 2/`, run the command for your task. It downloads the public Dev split, runs streaming inference with your trained LoRA adapter and scores the predictions.

The content judge uses Gemini 3.8 Flash (`gemini-3.8-flash`). Set your own [Gemini API key](https://aistudio.google.com/apikey) before evaluation. Only time-matched responses that require content judging are sent to the API.

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

**Audio · Venus Audio**

```bash
python -m eval --task audio --adapter outputs/venus_audio
```

**Video · Venus Omni**

```bash
python -m eval --task video --adapter outputs/venus_omni
```

Pass `--parquet-dir PATH` to use an existing Dev partition, `--limit 5` for a short run, or `--output DIR` to select a run directory. To resume an interrupted run, repeat the command with `--resume`.

Audio Dev contains 1,200 samples across Sound, Speech, Music and Mixed, including 91 Music samples. L1 measures trigger timing. L2/L3 use timing, structured rules or content judging according to the instruction and reference. Video Dev contains 1,000 samples. Alerts use timing. Count and position tasks use the original targets bundled with the evaluator. For state changes, the public Dev annotations describe the new state in free text. These responses, narration and step instructions are judged against the reference text. The default timing window is ±3 seconds.

Read `eval_outputs/<task>/metrics.json` for scores and `metrics.cases.json` for per-sample results. Audio averages F1 over the three levels. Video averages F1 over the nine task types.

To try the scorer on its own, use the [example references](examples/references.jsonl) and [predictions](examples/predictions.jsonl).

```bash
python -m eval.score --references examples/references.jsonl --predictions examples/predictions.jsonl --output eval_outputs/example/metrics.json
```

## License

The code is licensed under [Apache-2.0](../LICENSE). Model weights and datasets retain their respective licenses. Third-party attribution is provided in [NOTICE](../NOTICE).
