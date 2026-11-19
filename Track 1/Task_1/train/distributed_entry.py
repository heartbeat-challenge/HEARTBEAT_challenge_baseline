# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Isolate custom-model imports per rank, then enter the supplied Venus trainer."""

import os
import runpy
import sys
from pathlib import Path


def main():
    # Set this before importing Transformers; concurrent copies into a shared
    # dynamic-module cache caused partially written imports in eight-rank runs.
    cache = Path(os.environ["HEARTBEAT_MODULES_ROOT"]) / os.environ.get(
        "LOCAL_RANK", "0"
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache)
    # Local model directories can contain transitive relative imports. Seed all
    # Python modules, using Transformers' own sanitized directory-name mapping.
    import shutil

    import yaml
    from transformers.dynamic_module_utils import get_cached_module_file

    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    model = Path(config["model_name_or_path"])
    auto_map = (
        __import__("json")
        .loads((model / "config.json").read_text())
        .get("auto_map", {})
    )
    entry = auto_map.get("AutoModel") or auto_map.get("AutoModelForCausalLM")
    if isinstance(entry, str) and "--" not in entry:
        module = entry.rsplit(".", 1)[0].replace(".", "/") + ".py"
        cached = get_cached_module_file(str(model), module, local_files_only=True)
        package = cache / Path(cached).parent
        for source in model.glob("*.py"):
            shutil.copyfile(source, package / source.name)
    if not config.get("quantization_bit"):
        from runtime_compat import configure_lora

        configure_lora()
    if not config.get("deepspeed"):
        from accelerate.utils import other

        # Accelerate unwraps ordinary models by checking all optional engines.
        other.is_deepspeed_available = lambda: False
    launcher = (
        Path(__file__).resolve().parents[1] / "src" / "llamafactory" / "launcher.py"
    )
    sys.argv[0] = str(launcher)
    runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
