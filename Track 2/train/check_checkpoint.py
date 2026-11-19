# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Check a saved standard-LoRA adapter and trainer state on CPU."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    state = json.loads((args.checkpoint / "trainer_state.json").read_text())
    losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
    if state["global_step"] < 1 or not losses or not all(math.isfinite(x) for x in losses):
        raise ValueError("Missing training progress or non-finite logged losses")
    weights = load_file(str(args.checkpoint / "adapter_model.safetensors"), device="cpu")
    if not weights or not all(torch.isfinite(t).all().item() for t in weights.values()):
        raise ValueError("Empty or non-finite adapter")
    if not all("lora_" in name and (".q_proj." in name or ".v_proj." in name) for name in weights):
        raise ValueError("Unexpected parameters for the standard Q/V LoRA recipe")
    b = {name: tensor for name, tensor in weights.items() if ".lora_B." in name}
    if not b or not all(t.count_nonzero().item() for t in b.values()):
        raise ValueError("Some standard-LoRA B tensors remain zero")
    print(json.dumps(dict(global_step=state["global_step"], adapter_tensors=len(weights),
                          nonzero_lora_B_tensors=len(b), last_logged_loss=losses[-1], all_finite=True), indent=2))


if __name__ == "__main__":
    main()
