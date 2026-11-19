# HEARTBEAT Track I · Reciprocal Audio Turn-Taking

[English](README.md) · [简体中文](README_zh.md)

Choose a task and follow its guide for training and Dev evaluation.

- [Task 1 · Offline audio multiple choice](Task_1/README.md)
- [Task 2 · Full-duplex spoken interaction](Task_2/README.md)

## Data preparation

From the repository root, use the [data tools](../README.md#get-started) to download the train and Dev Parquet files for your task.

```bash
python data.py --track 1 --task 1 --format parquet --output-dir downloads/track1/task1
python data.py --track 1 --task 2 --format parquet --output-dir downloads/track1/task2
```

Each task uses its own environment. Run the commands in the task guide from its task directory.
