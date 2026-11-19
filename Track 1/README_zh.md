# HEARTBEAT Track I · 双向音频轮次交互

[English](README.md) · [简体中文](README_zh.md)

选择任务，查看训练和开发集评测命令。

- [Task 1 · 离线音频选择题](Task_1/README_zh.md)
- [Task 2 · 全双工语音交互](Task_2/README_zh.md)

## 数据准备

在仓库根目录，使用[数据工具](../README_zh.md#快速开始)下载所选任务的训练集和开发集 Parquet 文件。

```bash
python data.py --track 1 --task 1 --format parquet --output-dir downloads/track1/task1
python data.py --track 1 --task 2 --format parquet --output-dir downloads/track1/task2
```

两个任务分别配置环境。进入对应任务目录后，再执行该任务指南中的命令。
