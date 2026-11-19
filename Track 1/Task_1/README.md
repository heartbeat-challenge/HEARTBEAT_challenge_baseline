# HEARTBEAT Track I Task 1 · Offline audio multiple choice

[English](README.md) · [简体中文](README_zh.md)

## Installation

Use Python 3.12 and a CUDA PyTorch environment.

```bash
cd "Track 1/Task_1"
python -m pip install -r requirements.txt
```

## Model preparation

```bash
bash download_model.sh models
MODEL="$PWD/models/Realtime-Venus-Audio"
```

## Data preparation

After downloading the [Track 1 data](../README.md#data-preparation), set the Parquet root from this task directory.

```bash
DATA="$PWD/../../downloads/track1/task1/export/parquet/track1"
```

## Training

```bash
python -m train --model "$MODEL" --parquet-dir "$DATA/task1/train" \
  --prepared-dir prepared/train --output outputs/task1
```

The default configuration is [configs/task1.yaml](configs/task1.yaml). For a smoke run, add `--limit 16 --max-steps 2`. Resume with `--resume-from-checkpoint PATH`.

## Evaluation

```bash
python -m eval --model "$MODEL" --adapter outputs/task1 \
  --parquet-dir "$DATA/task1/dev" --output eval_outputs/dev
```

With a separate labeled Test set, set `TEST_MANIFEST`, `TEST_QUESTIONS`, and `TEST_LABELS` to your local files.

```bash
python -m eval --model "$MODEL" --adapter outputs/task1 \
  --test-manifest "$TEST_MANIFEST" \
  --questions "$TEST_QUESTIONS" \
  --test-labels "$TEST_LABELS" --output eval_outputs/test
```

The evaluator writes per-question predictions and Top-1 accuracy. The internal Test evaluator validates the question and audio hash before loading each answer.

## PPU environment

Keep the image's PPU PyTorch, TorchVision and TorchAudio. Add the pinned Python overlay.

```bash
OVERLAY=/tmp/heartbeat-track1-task1-ppu
python -m pip install --target "$OVERLAY" --no-deps -r requirements-ppu.txt
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
```
