<div align="center">

# ICASSP 2027 HEARTBEAT Challenge

**数据获取与使用指南**

<a href="https://heartbeat-challenge.github.io/"><img src="https://img.shields.io/badge/HEARTBEAT-Challenge_Website-2563EB?style=for-the-badge" alt="HEARTBEAT Challenge Website"></a>
<a href="https://huggingface.co/inclusionAI/Realtime-Venus/tree/main/data/HEARTBEAT_Challenge"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Dataset"></a>
<a href="https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README_zh.md"><img src="https://img.shields.io/badge/Data-Field_Reference-0F766E?style=for-the-badge" alt="Data Field Reference"></a>

[English](README.md) · [简体中文](README_zh.md)

</div>

HEARTBEAT 关注实时交互中的核心问题：**系统应该何时回应，以及回应什么？** 本次 Challenge 面向连续音频与音视频输入，评测系统持续理解上下文、识别交互时机并作出适当回应的能力。

本仓库为参赛者提供数据获取与准备工具。选择赛道、任务和数据划分后，通过一条命令即可下载对应数据，并导出为 **Parquet** 或 **JSON＋原始音视频文件**。也可以先读取少量样本，了解字段与媒体内容，再下载完整分区用于实验。

## 比赛赛道

### Track I · Reciprocal Audio Turn-Taking

**双向音频轮次交互。** 本赛道关注自然对话中的参与决策：何时开始回应、继续聆听、停止说话或继续说话。场景涵盖澄清、纠正、打断、信息更新，以及任务的暂停与恢复。

- **Offline 评测：** 根据音频前缀、助手指令和四个候选动作选择一个动作，以 Top-1 准确率评测。
- **Online 评测：** 增量处理音频，在持续聆听的同时生成语音。输出音频和交互日志记录回应内容、语音时序及未回应区间。

Offline 与 Online 分别评测、分别排名。

### Track II · Streaming Proactive Understanding

**流式主动理解。** 本赛道关注系统接收用户指令后，针对连续输入流中的相关事件主动回应的能力。任务覆盖 **L1 感知（Perception）**、**L2 理解（Comprehension）**和 **L3 推理（Reasoning）**。

- **音视频理解：** 综合连续输入的画面与声音，识别相关事件并作出回应。
- **纯音频理解：** 根据连续音频输入识别相关事件并作出回应。

两个设置均采用 Online 评测，分别排名。系统按顺序处理输入块，为每个块提交 JSON 输出，包括没有回应的块；回应时间取产生该回应的输入块的结束时间。

## 快速开始

克隆仓库并安装数据依赖。支持 Python 3.10–3.12；本地验证使用 Python 3.12。

```bash
git clone https://github.com/heartbeat-challenge/HEARTBEAT_challenge_baseline.git
cd HEARTBEAT_challenge_baseline

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

建议先从 Track I / Task 2 的 static 开发子集取 **两条样本**：

```bash
bash data.sh --track 1 --task 2 --split dev --subset static \
  --format parquet --sample-rows 2 --output-dir downloads/quickstart
```

这条命令按需读取远端数据块，生成一个小型 Parquet 文件，不下载完整分区。输出位置为：

```text
downloads/quickstart/sample/parquet/track1/task2/dev/static/00000/data.parquet
```

使用 PyArrow 读取：

```python
import pyarrow.parquet as pq

path = "downloads/quickstart/sample/parquet/track1/task2/dev/static/00000/data.parquet"
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
print(row.keys())
```

Bash 入口适用于 macOS、Linux，以及 Windows 的 WSL/Git Bash 环境。没有 Bash 时，可使用 `python data.py`，参数相同。如需指定 Python 解释器，可设置 `PYTHON=/path/to/python`。

## 选择所需数据

当前数据发布包含 **132,571 条记录、9 个分区**，打包数据总大小约 **599.83 GB**。参赛者可按实验需要选择分区。

| 赛道 | 任务标识 | 数据划分 | 子集 | 记录数 | 打包大小 |
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

大小按十进制 GB 四舍五入。命令中的 `--task 1` 在 Track I 对应 `task1`，在 Track II 对应 `task1_video`；`--task 2` 分别对应 `task2` 和 `task2_audio`。

下载前可以预览选择范围和体积：

```bash
bash data.sh --track 2 --split train --dry-run
```

| 参数 | 用途 |
|---|---|
| `--track` | 选择 `1`、`2`，也可传入多个值，如 `1 2` |
| `--task` | 选择 `1`、`2` 或具体任务标识，如 `task1_video` |
| `--split` | 选择 `train`、`dev` 或两者 |
| `--subset` | 选择 `dynamic` 或 `static`，仅用于 Track I / Task 2 / dev |
| `--format` | 导出为 `parquet`（默认）、`json` 或保留 `packed` 数据 |
| `--output-dir` | 下载与导出数据的保存目录 |
| `--sample-rows` | 每个选中分区仅导出前 N 条记录 |
| `--dry-run` | 仅显示分区、文件和完整下载大小 |
| `--list` | 列出全部分区 |
| `--all` | 选择完整数据发布，不与其他筛选条件同时使用 |

未指定的筛选项包含所有匹配值。例如，`--task 2` 会选择两个赛道的 Task 2，并包含 train 和 dev；选择 Track I / Task 2 / dev 时，若不指定子集，则同时包含 dynamic 和 static。不传入选择参数不会开始下载。

## 下载与格式转换

### Parquet：用于训练与数据处理

下载一个完整开发分区并导出记录：

```bash
bash data.sh --track 1 --task 1 --split dev \
  --format parquet --output-dir downloads/track1-task1-dev
```

每个源分片单独导出。Parquet 保留原始 Arrow 字段类型、嵌套结构、空值和二进制媒体字段；音视频仍以原始编码 bytes 存放在对应字段中。

逐条读取，避免一次将整个分片加载到内存：

```python
import pyarrow.parquet as pq

path = "downloads/track1-task1-dev/export/parquet/track1/task1/dev/00000/data.parquet"
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
print(row["sample_id"])
audio_bytes = row["audio"]["bytes"]
```

以上字段示例属于 Track I / Task 1。其他分区的字段请参阅[数据字段说明](https://huggingface.co/inclusionAI/Realtime-Venus/blob/main/data/HEARTBEAT_Challenge/fields/README_zh.md)。

### JSON＋媒体文件：用于基于文件路径的处理流程

将记录导出为 JSON Lines，并将二进制内容保存为独立文件：

```bash
bash data.sh --track 1 --task 2 --split dev --subset static \
  --format json --output-dir downloads/track1-task2-static
```

`records.jsonl` 每行对应一条记录，其中二进制字段替换为相对文件引用：

```json
{
  "file": "media/<sha256>.wav",
  "bytes": 123456,
  "sha256": "<sha256>"
}
```

音视频**不经过转码**，保持原始字节。文件以 SHA256 命名，避免原路径重名；同一分片内相同内容复用同一文件。扩展名根据内容签名识别，无法识别时使用 `.bin`。其他字段保留嵌套结构；Decimal 和特殊标量使用结合 schema 可无损还原的表示。

请将 `records.jsonl`、`schema.arrow` 和 `media/` 一起保存。需要将某个 JSON 分片还原为 Parquet 时，运行：

```bash
python json_to_parquet.py \
  --input-dir downloads/track1-task2-static/export/json/track1/task2/dev/static/00000 \
  --output downloads/static-restored.parquet
```

### Packed 数据：用于离线 Zarr 读取

使用 `--format packed` 下载所选数据包并生成本地 ReferenceFS 索引，不额外导出其他格式：

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

读取器同时提供 `as_pandas` 和 `as_datasets`。在 Python 中选择 Track II 时，任务名使用 `task1_video` 或 `task2_audio`；选择 Track I / Task 2 / dev 时，传入 `subset="dynamic"` 或 `subset="static"`。

## 输出目录

```text
<output-dir>/
├── packed/                         # 完整下载的数据包与离线索引
├── export/                         # 完整分区导出
│   ├── parquet/<分区>/<源分片>/
│   │   ├── data.parquet
│   │   ├── schema.arrow
│   │   └── export.json
│   └── json/<分区>/<源分片>/
│       ├── records.jsonl
│       ├── schema.arrow
│       ├── media/
│       └── export.json
├── sample/                         # 样本导出使用此目录，代替 export/
└── export-manifest.json
```

每次只生成指定格式。完整导出会保留下载的数据包，因此请为下载和转换结果都预留空间。Parquet 导出保留逻辑内容和 schema，文件压缩方式、行组布局及物理文件 SHA256 可能与原始源文件不同。
