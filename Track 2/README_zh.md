# HEARTBEAT Track 2 Baseline

[English](README.md) | 简体中文

本目录提供 HEARTBEAT Challenge Track 2 的训练与开发集评测代码，基于 [Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus) 对音频和视频任务进行 LoRA 微调。

| 任务 | 模型 | 训练配置 |
|---|---|---|
| Audio | Venus Audio | [audio.yaml](configs/audio.yaml) |
| Video | Venus Omni | [video.yaml](configs/video.yaml) |

[训练代码](train/)负责准备 Parquet 数据和启动微调，[评测代码](eval/)负责流式推理和 Dev 评分。

## 安装

从仓库根目录进入基线目录，安装训练和评测共用的依赖。

```bash
cd "Track 2"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

安装 [FFmpeg](https://ffmpeg.org/download.html)，确保终端能运行 `ffmpeg` 和 `ffprobe`。

## 模型准备

使用 Venus 下载脚本获取模型：

```bash
VENUS_REVISION=a1d95d3ac61416c8db34736b152241396850a6ed
hf download inclusionAI/Realtime-Venus download_models.py \
  --revision "$VENUS_REVISION" --local-dir models
python models/download_models.py --model audio --local-dir models \
  --revision "$VENUS_REVISION"
```

`--model audio` 下载 Venus Audio，`--model omni` 下载 Venus Omni，`--model all` 下载两者。下载器会将模型及共享的 `config.yaml` 保存到 `models/`。

## 数据准备

在 `Track 2/` 下运行所选任务的命令，将训练集下载并导出为 Parquet。下方训练命令使用相应的导出目录。

**Audio**

```bash
python ../data.py --track 2 --task task2_audio --split train --format parquet --output-dir downloads/audio
```

**Video**

```bash
python ../data.py --track 2 --task task1_video --split train --format parquet --output-dir downloads/video
```

各任务的 Parquet 字段见[数据集字段说明](https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README_zh.md)。

## 训练

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

默认使用 8 张 GPU 训练 1 个 epoch，LoRA rank 为 128。训练结果保存在 `--output` 指定目录，包含 LoRA adapter、checkpoint 和训练指标。

使用 `--config` 指定自定义配置；小规模试跑可添加 `--limit 16 --max-steps 2`，并使用新的准备目录和输出目录。断点续训时，使用 `--resume-from-checkpoint` 指定同一输出目录下的 checkpoint。

## 评测

从 `Track 2/` 运行对应任务的一条命令。程序下载公开 Dev 数据，加载训练好的 LoRA adapter 进行流式推理，按任务自动评分。

内容裁判使用 Gemini 3.8 Flash（`gemini-3.8-flash`）。评测前设置你自己的 [Gemini API key](https://aistudio.google.com/apikey)。只有时间匹配且需要判断回答内容的样本才会调用 API。

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

已有 Dev Parquet 可加 `--parquet-dir PATH`。用 `--limit 5` 小规模试跑，用 `--output DIR` 选择输出目录。中断后用相同参数加 `--resume` 继续。

Audio Dev 共 1,200 条，覆盖 Sound、Speech、Music 和 Mixed，其中 Music 有 91 条。L1 评触发时间，L2/L3 根据指令和参考答案自动选择时间匹配、规则匹配或内容评分。Video Dev 共 1,000 条，提醒类评时间，计数和位置任务使用代码随附的原始标注。状态变化的 Dev 标注是文字描述，与叙述、步骤指导一起按参考文本评内容。默认时间窗口为 ±3 秒。

结果写入 `eval_outputs/<task>/metrics.json`，逐样本结果在 `metrics.cases.json`。Audio 总分为三个等级的平均 F1，Video 总分为九类任务的平均 F1。

可用仓库内的[示例参考答案](examples/references.jsonl)和[示例预测](examples/predictions.jsonl)单独试跑评分器。

```bash
python -m eval.score --references examples/references.jsonl --predictions examples/predictions.jsonl --output eval_outputs/example/metrics.json
```

## 许可证

代码采用 [Apache-2.0](../LICENSE)。模型与数据集遵循各自的许可证，第三方声明见 [NOTICE](../NOTICE)。
