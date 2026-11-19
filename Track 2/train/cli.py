# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Validate inputs, prepare published data, then launch the Venus trainer.

The adapter layer owns paths, caches and process locks. The retained training
engine, one-second supervision and task recipes are not changed here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

BASELINE = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
PATH_OPTIONS = (
    "parquet_dir", "prepared_dir", "model", "output", "config", "resume_from_checkpoint",
)


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Copy CLI options and resolve all paths relative to the caller's cwd."""
    normalized = argparse.Namespace(**vars(args))
    for name in PATH_OPTIONS:
        value = getattr(normalized, name)
        if value is not None:
            setattr(normalized, name, Path(value).expanduser().resolve())
    return normalized


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def validate_options(args: argparse.Namespace) -> None:
    """Reject invalid options and writes overlapping data or base-model paths."""
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.prepare_only and (args.dry_run or args.resume_from_checkpoint):
        raise ValueError("--prepare-only cannot be combined with --dry-run or --resume-from-checkpoint")
    if args.nproc < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid GPU count or port")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if not args.prepare_only and (args.model is None or args.output is None):
        raise ValueError("Training / --dry-run requires --model and --output")
    if args.model is not None:
        if not (args.model / "config.json").is_file() or (args.model / "adapter_config.json").exists():
            raise ValueError("--model must be a full released Venus model directory, not an adapter")
    if args.config is not None and not args.config.is_file():
        raise ValueError("--config must name an existing training recipe")

    writable = {"--prepared-dir": args.prepared_dir, "--output": args.output}
    protected = {"--parquet-dir": args.parquet_dir, "--model": args.model}
    for name, directory in writable.items():
        if directory is None:
            continue
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"{name} must be a directory: {directory}")
        for protected_name, source in protected.items():
            if source is not None and paths_overlap(directory, source):
                raise ValueError(f"{name} must not overlap {protected_name}; use separate directories")
    if args.output is not None:
        if paths_overlap(args.prepared_dir, args.output):
            raise ValueError("--output and --prepared-dir must be separate directories")
        if args.output.exists() and any(args.output.iterdir()) and not args.resume_from_checkpoint:
            raise ValueError("Output is not empty; choose a new directory or explicitly resume a checkpoint")
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
        if checkpoint.parent != args.output:
            raise ValueError("Resume from a checkpoint directory inside the original --output")
        required = ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt")
        for name in required:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete training checkpoint: missing {name}")
        for name in ("run_manifest.json", "train_config.yaml"):
            if not (args.output / name).is_file():
                raise ValueError(f"Resume requires the original output's {name}")


def load_recipe(args: argparse.Namespace) -> dict[str, Any]:
    """Check the recipe before potentially extracting hundreds of GB of media."""
    import yaml

    path = args.config or BASELINE / "configs" / (args.task + ".yaml")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML recipe: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Training recipe must be a YAML mapping: {path}")
    for name in ("minicpmo_max_video_seconds", "minicpmo_tail_listen_units"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Recipe requires a finite positive '{name}': {path}")
    for name in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Recipe requires a positive integer '{name}': {path}")
    return config


def validate_model_weights(model: Path) -> None:
    """Check that a real training request has the local checkpoint files."""
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = model / name
        if not index.is_file():
            continue
        metadata = json.loads(index.read_text(encoding="utf-8"))
        weights = metadata.get("weight_map") if isinstance(metadata, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Invalid model weight index: {index}")
        filenames = list(weights.values())
        if not all(isinstance(filename, str) for filename in filenames):
            raise ValueError(f"Invalid shard name in {index}")
        for filename in sorted(set(filenames)):
            shard = (model / filename).resolve()
            if model not in shard.parents or not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f"Missing or invalid model weight shard: {filename}")
        return
    if not any((model / name).is_file() and (model / name).stat().st_size > 0
               for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("No model weights found; download the complete Venus checkpoint before training")


def validate_training_runtime(args: argparse.Namespace) -> None:
    """Dry runs need media tools, but never import Torch or load model weights."""
    import pyarrow.parquet  # noqa: F401 — fail before creating a preparation cache

    executables = ("ffprobe", "ffmpeg") if args.decode_check or not args.prepare_only else ("ffprobe",)
    for executable in executables:
        if not shutil.which(executable):
            raise ValueError(f"Install system {executable} and add it to PATH before preparing media")
    if args.prepare_only or args.dry_run:
        return
    validate_model_weights(args.model)
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc:
        raise ValueError(f"Training requires {args.nproc} visible CUDA GPUs; "
                         "check CUDA_VISIBLE_DEVICES, or use --prepare-only / --dry-run")


def source_inventory(source: Path, task: str, limit: int, decode_check: bool) -> dict[str, Any]:
    from .prepare import parquet_files

    # Cache identity is based on source path/size/mtime. Media bytes and the
    # resulting supervision are independently hashed during preparation.
    return {
        "version": 1,
        "task": task,
        "split": "train",
        "limit": limit,
        "decode_check": decode_check,
        "converter_sha256": hashlib.sha256(Path(__file__).with_name("prepare.py").read_bytes()).hexdigest(),
        "shards": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in parquet_files(source)
        ],
    }


@contextmanager
def exclusive_directory_lock(directory: Path, purpose: str) -> Iterator[None]:
    """Hold a nonblocking Unix lock; a crashed process releases it automatically."""
    import fcntl

    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.{purpose}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process holds the {purpose} lock for {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_cached_data(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    from .prepare import prepare
    from .train import validate_prepared

    directory = args.prepared_dir
    inventory = source_inventory(args.parquet_dir, args.task, args.limit, args.decode_check)
    cache_file = directory / "source_inventory.json"
    with exclusive_directory_lock(directory, "prepare"):
        if directory.exists() and any(directory.iterdir()):
            if not cache_file.is_file() or json.loads(cache_file.read_text()) != inventory:
                raise ValueError("Prepared cache is incomplete or its source/options changed. "
                                 "Choose a new --prepared-dir; existing files will not be overwritten.")
            try:
                count, _ = validate_prepared(directory, args.task, config)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Invalid prepared manifest in {directory}; use a new --prepared-dir") from exc
            LOGGER.info("Reusing prepared %s train partition: %d tasks", args.task, count)
        else:
            LOGGER.info("Preparing %s train partition from %s", args.task, args.parquet_dir)
            prepare(args.parquet_dir, directory, args.task, "train", args.limit, args.decode_check)
            if inventory != source_inventory(args.parquet_dir, args.task, args.limit, args.decode_check):
                raise ValueError("Source Parquet files changed during preparation; use a new --prepared-dir")
            temporary = cache_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
            temporary.replace(cache_file)
    return directory


def run(args: argparse.Namespace) -> None:
    args = normalize_paths(args)
    validate_options(args)
    config = load_recipe(args)
    validate_training_runtime(args)
    # Keep the training output exclusively owned throughout preparation and the
    # blocking torchrun call. Dry runs never create a checkpoint-directory lock.
    lock = (exclusive_directory_lock(args.output, "train")
            if not args.prepare_only and not args.dry_run else nullcontext())
    with lock:
        args.dataset_dir = prepare_cached_data(args, config)
        if not args.prepare_only:
            from .train import launch

            launch(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train",
        description="Prepare Track II Parquet data and train Venus Audio or Omni.",
    )
    parser.add_argument("--task", choices=["audio", "video"], required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True,
                        help="One train partition: recursive shards or a single Parquet file")
    parser.add_argument("--prepared-dir", type=Path, required=True,
                        help="Media/supervision cache; keep separate from inputs and training outputs")
    parser.add_argument("--model", type=Path, help="Complete local Venus Audio/Omni checkpoint")
    parser.add_argument("--output", type=Path, help="Training checkpoints and logs")
    parser.add_argument("--config", type=Path, help="Optional YAML training recipe")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--port", type=int, default=29621)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only: 0 prepares all rows")
    parser.add_argument("--max-steps", type=int, help="Smoke training limit; default uses configured epochs")
    parser.add_argument("--decode-check", action="store_true", help="Fully decode media during preparation")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data without a model or GPUs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare/reuse data and print config; no GPU training or checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def process_error_message(exc: subprocess.SubprocessError) -> str:
    """Keep FFmpeg failures actionable without printing a Python traceback."""
    command = exc.cmd
    executable = command[0] if isinstance(command, (list, tuple)) else command
    detail = exc.stderr or ""
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", errors="replace")
    detail = detail.strip()[-1500:]
    if isinstance(exc, subprocess.TimeoutExpired):
        message = f"{executable} timed out after {exc.timeout} seconds"
    else:
        message = f"{executable} failed (exit {exc.returncode})"
    return message + (f": {detail}" if detail else "; see the process output above")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(args)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"error: {process_error_message(exc)}\n")
    except ImportError as exc:
        parser.exit(2, f"error: Missing or incompatible dependency ({exc}). From the Track 2 directory, "
                    "run python -m pip install -r requirements.txt.\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Existing data and checkpoints have been preserved.\n")


if __name__ == "__main__":
    main()
