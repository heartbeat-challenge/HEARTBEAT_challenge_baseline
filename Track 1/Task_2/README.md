# HEARTBEAT Track I Task 2 · Full-duplex spoken interaction

[English](README.md) · [简体中文](README_zh.md)

## Installation

Use Python 3.12 and a CUDA PyTorch environment.

```bash
cd "Track 1/Task_2"
python -m pip install -r requirements.txt
```

## Model preparation

```bash
bash download_model.sh models
MODEL="$PWD/models/Realtime-Venus-Audio"
```

```bash
ASR_ROOT="$PWD/models"
hf download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 --local-dir "$ASR_ROOT/asr"
hf download Qwen/Qwen3-ForcedAligner-0.6B --revision c7cbfc2048c462b0d63a45797104fc9db3ad62b7 --local-dir "$ASR_ROOT/aligner"
```

Set `DEMO` to the matching MiniCPM-o-Demo checkout containing `core/processors/unified.py`. Install `requirements-asr.txt` in the ASR environment for acoustic or full evaluation.

## Data preparation

After downloading the [Track 1 data](../README.md#data-preparation), set the Parquet root from this task directory.

```bash
DATA="$PWD/../../downloads/track1/task2/export/parquet/track1"
```

## Training

```bash
python -m train --model "$MODEL" --parquet-dir "$DATA/task2/train" \
  --prepared-dir prepared/train --output outputs/task2
```

The default configuration is [configs/task2.yaml](configs/task2.yaml). For a smoke run, add `--limit 16 --max-steps 2`. Resume with `--resume-from-checkpoint PATH`.

## Evaluation

```bash
for subset in static dynamic; do
  python -m eval --subset "$subset" --model "$MODEL" --adapter outputs/task2 \
    --parquet-dir "$DATA/task2/dev/$subset" --output eval_outputs/dev \
    --duplex-repo "$DEMO" --asr-model "$ASR_ROOT/asr" --aligner "$ASR_ROOT/aligner"
done
```

For a separate labeled Test set, set `TEST` and `TEST_LABELS` to the local manifest and label roots.

```bash
for subset in static dynamic; do
  python -m eval --subset "$subset" --model "$MODEL" --adapter outputs/task2 \
    --test-manifest "$TEST/task2/test/$subset/manifest.jsonl" --test-labels "$TEST_LABELS/$subset" \
    --output eval_outputs/test --duplex-repo "$DEMO" \
    --asr-model "$ASR_ROOT/asr" --aligner "$ASR_ROOT/aligner"
done
```

Full evaluation uses `HEARTBEAT_JUDGE_BASE_URL`, `HEARTBEAT_JUDGE_API_KEY` and `HEARTBEAT_JUDGE_MODEL`. Use `--stage infer` for replay or `--stage acoustic` for replay, ASR and behavior scoring. Response start and stopping after interruption both use 2-second deadlines. Behavior and content are reported independently.

## PPU environment

Keep the image's PPU PyTorch, TorchVision and TorchAudio. Add the pinned Python overlay and use the existing ASR environment.

```bash
OVERLAY=/tmp/heartbeat-track1-task2-ppu
python -m pip install --target "$OVERLAY" --no-deps -r requirements-ppu.txt
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
ASR_PY=/path/to/asr-env/bin/python
```

Pass `--asr-python "$ASR_PY"` to Task 2. Use `--stage acoustic` for replay, alignment and acoustic scoring without a semantic judge. Use the default `all` for semantic scoring with a configured Judge.
