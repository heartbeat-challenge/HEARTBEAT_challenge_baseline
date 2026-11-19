"""Prepare a Track I partition and launch single-GPU or torchrun LoRA SFT."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from data_io import positive_int

from .prepare import DATASET_NAME, add_arguments, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument(
        "--gpus",
        type=positive_int,
        help="Local ranks, defaults to visible CUDA devices",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()

    import yaml

    root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    output = args.output.resolve()
    if not (model / "config.json").is_file():
        parser.error(f"Missing model config in {model}")
    resume = args.resume_from_checkpoint
    if resume and not (resume / "trainer_state.json").is_file():
        parser.error(f"Missing trainer_state.json in {resume}")
    if output.exists() and any(output.iterdir()) and not resume:
        parser.error(
            f"Output already has files: {output}. Choose a new output or resume a checkpoint."
        )
    config_path = args.config or root / "configs" / "task2.yaml"
    config = yaml.safe_load(config_path.read_text())
    if args.gpus is None:
        import torch

        ranks = torch.cuda.device_count()
    else:
        ranks = args.gpus
    if ranks < 1:
        parser.error("No CUDA devices detected")

    prepare(args.parquet_dir, args.prepared_dir, args.limit)
    output.mkdir(parents=True, exist_ok=True)
    config.update(
        dataset=DATASET_NAME,
        dataset_dir=str(args.prepared_dir.resolve()),
        model_name_or_path=str(model),
        output_dir=str(output),
        logging_dir=str(output / "logs"),
    )
    config.pop("resume_from_checkpoint", None)
    if resume:
        config["resume_from_checkpoint"] = str(resume.resolve())
    if args.max_steps:
        config["max_steps"] = args.max_steps
    run_config = output / "train_config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=True))
    package_path = os.pathsep.join(
        [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    with tempfile.TemporaryDirectory(prefix="modules-", dir=output) as modules:
        env = dict(
            os.environ,
            PYTHONPATH=package_path,
            HEARTBEAT_MODULES_ROOT=modules,
            HEARTBEAT_STRICT_DATA="1",
            TOKENIZERS_PARALLELISM="false",
            USE_TF="0",
            TRANSFORMERS_NO_TF="1",
        )
        command = [sys.executable]
        if ranks > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(ranks),
            ]
        command += [str(root / "train" / "distributed_entry.py"), str(run_config)]
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
