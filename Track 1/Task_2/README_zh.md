# HEARTBEAT Track I Task 2 · 全双工语音交互

[English](README.md) · [简体中文](README_zh.md)

## 安装

安装 Python 3.12 和 CUDA 版 PyTorch。

```bash
cd "Track 1/Task_2"
python -m pip install -r requirements.txt
```

## 模型准备

```bash
bash download_model.sh models
MODEL="$PWD/models/Realtime-Venus-Audio"
```

```bash
ASR_ROOT="$PWD/models"
hf download Qwen/Qwen3-ASR-1.7B --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 --local-dir "$ASR_ROOT/asr"
hf download Qwen/Qwen3-ForcedAligner-0.6B --revision c7cbfc2048c462b0d63a45797104fc9db3ad62b7 --local-dir "$ASR_ROOT/aligner"
```

回放需要与模型配套的 MiniCPM-o-Demo。设置 `DEMO` 为含 `core/processors/unified.py` 的目录。需要 ASR 时，在对应环境安装 `requirements-asr.txt`。

## 数据准备

按 [Track 1 数据准备](../README_zh.md#数据准备)下载数据后，在本任务目录下设置 Parquet 根目录。

```bash
DATA="$PWD/../../downloads/track1/task2/export/parquet/track1"
```

## 训练

```bash
python -m train --model "$MODEL" --parquet-dir "$DATA/task2/train" \
  --prepared-dir prepared/train --output outputs/task2
```

默认配置在 [configs/task2.yaml](configs/task2.yaml)。短跑加 `--limit 16 --max-steps 2`。续训使用 `--resume-from-checkpoint PATH`。

## 评测

```bash
for subset in static dynamic; do
  python -m eval --subset "$subset" --model "$MODEL" --adapter outputs/task2 \
    --parquet-dir "$DATA/task2/dev/$subset" --output eval_outputs/dev \
    --duplex-repo "$DEMO" --asr-model "$ASR_ROOT/asr" --aligner "$ASR_ROOT/aligner"
done
```

如需评测另行准备的有标签 Test 集，将 `TEST` 和 `TEST_LABELS` 设为本地清单和标签的根目录。

```bash
for subset in static dynamic; do
  python -m eval --subset "$subset" --model "$MODEL" --adapter outputs/task2 \
    --test-manifest "$TEST/task2/test/$subset/manifest.jsonl" --test-labels "$TEST_LABELS/$subset" \
    --output eval_outputs/test --duplex-repo "$DEMO" \
    --asr-model "$ASR_ROOT/asr" --aligner "$ASR_ROOT/aligner"
done
```

完整版使用 `HEARTBEAT_JUDGE_BASE_URL`、`HEARTBEAT_JUDGE_API_KEY` 和 `HEARTBEAT_JUDGE_MODEL`。只跑回放加 `--stage infer`。跑回放、ASR 和行为评分加 `--stage acoustic`。开始回应和打断后停止都须在 2 秒内，行为与内容分别汇总。

## PPU 环境

沿用镜像自带的 PPU PyTorch、TorchVision 和 TorchAudio。训练和推理用独立 Python 依赖目录，ASR 用已安装的环境。

```bash
OVERLAY=/tmp/heartbeat-track1-task2-ppu
python -m pip install --target "$OVERLAY" --no-deps -r requirements-ppu.txt
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
ASR_PY=/path/to/asr-env/bin/python
```

Task 2 传入 `--asr-python "$ASR_PY"`。使用 `--stage acoustic` 完成回放、ASR 对齐和声学评分。默认 `all` 会额外调用已配置的语义 Judge。
