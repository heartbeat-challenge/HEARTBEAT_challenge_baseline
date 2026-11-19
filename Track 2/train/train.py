# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Launch a single-node Venus Audio or Omni LoRA training job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def validate_prepared(directory, task, config):
    name = f"heartbeat_{task}_train"
    info = json.loads((directory / "dataset_info.json").read_text())
    data = directory / info[name]["file_name"]
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256()
    with data.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["jsonl_sha256"] or manifest["split"] != "train" or manifest["subtrack"] != task:
        raise ValueError("Prepared dataset changed or has the wrong split/task; rerun the converter")
    count, checked = 0, set()
    with data.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            messages = row["raw_prompt"] + row["raw_response"]
            units = math.ceil(row["duration"])
            if (units > config["minicpmo_max_video_seconds"] or
                    units > config["minicpmo_tail_listen_units"]):
                raise ValueError(f"{row['id']} exceeds configured duration/tail limits; increase both limits explicitly")
            valid = (row["raw_system"] == "Streaming Omni Conversation." and len(messages) == 2 * units
                     and row["use_vision"] == (task == "video") and row["use_audio"] is True
                     and row["training_mode"] == "duplex_streaming" and row["instruction_mode"] == "text")
            for i, message in enumerate(messages):
                valid = valid and message["role"] == ("assistant" if i % 2 else "user")
                if i % 2:
                    target = message["content"]
                    valid = valid and (target == "<|listen|>" or (
                        target.startswith("<|speak|>") and target.endswith("<|turn_eos|><|chunk_eos|>")))
            if not valid:
                raise ValueError(f"Invalid prepared stream: {row['id']}")
            paths = row["raw_videos"] if task == "video" else row["raw_audios"]
            if len(paths) != 1:
                raise ValueError("Each task must have one full media file")
            media = Path(paths[0])
            if media not in checked:
                if not media.is_file() or not media.stat().st_size:
                    raise ValueError(f"Missing media: {media}")
                checked.add(media)
            count += 1
    if not count or count != manifest["tasks"]:
        raise ValueError("Empty or incomplete dataset")
    return count, digest.hexdigest()


def launch(args):
    config = yaml.safe_load((args.config or ROOT / "configs" / (args.task + ".yaml")).read_text())
    model, dataset, output = args.model.resolve(), args.dataset_dir.resolve(), args.output.resolve()
    if not (model / "config.json").is_file() or (model / "adapter_config.json").exists():
        raise ValueError("--model must be a full released Venus model directory, not a LoRA adapter")
    if args.nproc < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid GPU count or port")
    config.update(model_name_or_path=str(model), dataset_dir=str(dataset),
                  dataset=f"heartbeat_{args.task}_train", output_dir=str(output), overwrite_output_dir=False)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive; omit it for one full epoch")
        config.update(max_steps=args.max_steps, save_steps=min(200, args.max_steps))
    count, digest = validate_prepared(dataset, args.task, config)
    resume = args.resume_from_checkpoint.resolve() if args.resume_from_checkpoint else None
    if resume:
        if resume.parent != output:
            raise ValueError("Resume from a checkpoint-* directory inside this output directory")
        for name in ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt"):
            if not (resume / name).is_file():
                raise ValueError(f"Incomplete training checkpoint: missing {name}")
        previous = json.loads((output / "run_manifest.json").read_text())
        if previous["dataset_sha256"] != digest or previous["task"] != args.task or previous["nproc"] != args.nproc:
            raise ValueError("Resume requires the same task, data, and GPU count")
        old = yaml.safe_load((output / "train_config.yaml").read_text())
        if old != config:
            raise ValueError("Resume requires the original training configuration")
        config["resume_from_checkpoint"] = str(resume)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new directory or explicitly resume a checkpoint")
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    env.update(HEARTBEAT_STRICT_DATA="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", USE_FLAX="0", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
               HEARTBEAT_MODULES_ROOT=str(output / ".cache" / "dynamic_modules"))
    env.setdefault("OMP_NUM_THREADS", "4")
    # Honor CUDA_VISIBLE_DEVICES chosen by the participant.
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    config_path = output / ("resume_config.yaml" if resume else "train_config.yaml")
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               f"--nproc_per_node={args.nproc}", "--master_addr=127.0.0.1", f"--master_port={args.port}",
               str(Path(__file__).with_name("distributed_entry.py")), str(config_path)]
    manifest = dict(task=args.task, tasks=count, dataset_sha256=digest, nproc=args.nproc,
                    effective_batch_size=args.nproc * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"],
                    command=command, cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", "all"))
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, config=config), indent=2))
        return
    for executable in ("ffmpeg", "ffprobe"):
        import shutil
        if not shutil.which(executable):
            raise ValueError(f"Install {executable} before training")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc:
        raise ValueError("Not enough visible CUDA GPUs")
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if not resume:
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["audio", "video"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional custom recipe; paths are set by the CLI")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--port", type=int, default=29621)
    parser.add_argument("--max-steps", type=int, help="Short smoke run; omit for the configured epochs")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print config; no GPUs or output writes")
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
