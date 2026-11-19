# HEARTBEAT Track I Task 1 · 离线音频选择题

[English](README.md) · [简体中文](README_zh.md)

## 安装

安装 Python 3.12 和 CUDA 版 PyTorch。

```bash
cd "Track 1/Task_1"
python -m pip install -r requirements.txt
```

## 模型准备

```bash
bash download_model.sh models
MODEL="$PWD/models/Realtime-Venus-Audio"
```

## 数据准备

按 [Track 1 数据准备](../README_zh.md#数据准备)下载数据后，在本任务目录下设置 Parquet 根目录。

```bash
DATA="$PWD/../../downloads/track1/task1/export/parquet/track1"
```

## 训练

```bash
python -m train --model "$MODEL" --parquet-dir "$DATA/task1/train" \
  --prepared-dir prepared/train --output outputs/task1
```

默认配置在 [configs/task1.yaml](configs/task1.yaml)。短跑加 `--limit 16 --max-steps 2`。续训使用 `--resume-from-checkpoint PATH`。

## 评测

```bash
python -m eval --model "$MODEL" --adapter outputs/task1 \
  --parquet-dir "$DATA/task1/dev" --output eval_outputs/dev
```

如需评测另行准备的有标签 Test 集，将 `TEST_MANIFEST`、`TEST_QUESTIONS` 和 `TEST_LABELS` 设为本地文件路径。

```bash
python -m eval --model "$MODEL" --adapter outputs/task1 \
  --test-manifest "$TEST_MANIFEST" \
  --questions "$TEST_QUESTIONS" \
  --test-labels "$TEST_LABELS" --output eval_outputs/test
```

输出逐题预测和 Top-1 准确率。内部 Test 的答案通过题目和音频哈希核对。

## PPU 环境

沿用镜像自带的 PPU PyTorch、TorchVision 和 TorchAudio。训练和推理用独立 Python 依赖目录。

```bash
OVERLAY=/tmp/heartbeat-track1-task1-ppu
python -m pip install --target "$OVERLAY" --no-deps -r requirements-ppu.txt
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
```
