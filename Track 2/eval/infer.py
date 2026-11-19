# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Native Venus stream replay. Reads only inference inputs, never GT or captions."""
import argparse
import hashlib
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

from .common import atomic_json, case_file, digest_file, digest_json, finite, load_rows, write_rows
from .media import Media
from .audio import format_instruction

SYSTEM = "Streaming Omni Conversation."


def validate_input(row):
    allowed = {"id", "subtrack", "level", "audio_type", "task_type", "duration", "question", "media_path", "media_sha256", "answer_format"}
    if set(row) - allowed:
        raise ValueError("Inference inputs must not contain references, captions or additional fields")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("A nonempty ID is required")
    if row["subtrack"] not in ("audio", "video") or finite(row["duration"]) <= 0 or len(row["question"]) != 1:
        raise ValueError("Invalid task/duration/question")
    q = row["question"][0]
    if not 0 <= finite(q["time"]) <= row["duration"] or not isinstance(q["content"], str) or not q["content"].strip():
        raise ValueError("Invalid question")
    if "answer_format" in row:
        if row["subtrack"] != "audio":
            raise ValueError("answer_format is currently defined for Audio rules only")
        format_instruction(row["answer_format"])


def replay(duplex, row, media, settings, synchronize=lambda: None):
    chunks = []
    try:
        # Native prepare resets the decoder and mel extractor but retains Whisper history.
        duplex.model.audio_past_key_values = None
        duplex.prepare(prefix_system_prompt=SYSTEM)
        instruction = row["question"][0]["content"]
        if "answer_format" in row:
            instruction += "\n"+format_instruction(row["answer_format"])
        origin = time.monotonic()
        question_unit = max(0, math.ceil(row["question"][0]["time"]) - 1)
        for i in range(math.ceil(row["duration"])):
            end = min(i + 1, row["duration"])
            audio, frames, frame_time = media.chunk(i)
            if settings["realtime"]:
                time.sleep(max(0, origin + end - time.monotonic()))
            tick = time.monotonic()
            kwargs = {"audio_waveform": audio, "text_list": [instruction] if i == question_unit else None}
            if row["subtrack"] == "video":
                kwargs.update(frame_list=frames, max_slice_nums=1)
            result = duplex.streaming_prefill(**kwargs)
            if not result.get("success"):
                raise RuntimeError("Native streaming_prefill failed")
            result = duplex.streaming_generate(max_new_speak_tokens_per_chunk=settings["max_tokens"],
                decode_mode="sampling", temperature=.7, top_k=100, top_p=.8, listen_prob_scale=1)
            synchronize()
            if type(result.get("is_listen")) is not bool or not isinstance(result.get("text", ""), str):
                raise ValueError("Unexpected native generation result")
            listen = result["is_listen"]
            chunks.append(dict(index=i, input_start=float(i), input_end=end,
                               wall_time=time.monotonic() - origin, compute_seconds=time.monotonic() - tick,
                               text="" if listen else result.get("text", "").strip(), is_listen=listen,
                               end_of_turn=bool(result.get("end_of_turn")), frame_time=frame_time))
        return "complete", chunks, None
    except Exception as exc:
        # Do not silently discard already emitted false/partial responses.
        return "error", chunks, type(exc).__name__
    finally:
        duplex.set_session_stop()


def model_identity(model):
    files = sorted(p for p in model.iterdir() if p.is_file())
    return {p.name: (digest_file(p) if p.suffix in (".json", ".py") else
                      {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}) for p in files}


def load_model(model_path, adapter, device, cache):
    # Must precede Transformers imports. Each worker owns a separate output/cache.
    os.environ["HF_MODULES_CACHE"] = str(cache)
    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import torch
    from transformers import AutoModel
    from .venus_compat import adapt_venus_duplex
    # Populate all relative modules before loading either custom architecture.
    from transformers.dynamic_module_utils import get_cached_module_file
    import shutil
    config = json.loads((model_path / "config.json").read_text())
    mapping = config.get("auto_map", {})
    entry = mapping.get("AutoModel") or mapping.get("AutoModelForCausalLM")
    if isinstance(entry, str) and "--" not in entry:
        cached = get_cached_module_file(str(model_path), entry.rsplit(".", 1)[0].replace(".", "/") + ".py",
                                        local_files_only=True)
        target = cache / Path(cached).parent
        for file in model_path.glob("*.py"):
            shutil.copyfile(file, target / file.name)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True,
                                     attn_implementation="sdpa", dtype=torch.bfloat16).eval().to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).merge_and_unload(safe_merge=True).eval()
    # The released factory initializes an unused output vocoder even in text-only mode.
    original = model.init_tts
    model.init_tts = lambda *args, **kwargs: None
    try:
        duplex = model.as_duplex(generate_audio=False, force_listen_count=0)
    finally:
        model.init_tts = original
    adapt_venus_duplex(duplex)
    return duplex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realtime", action="store_true", help="Pace chunks in wall time; also record output delivery time")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.max_tokens < 1:
        raise ValueError("Invalid shard or token limit")
    if args.retry_errors and not args.resume:
        raise ValueError("--retry-errors requires --resume")
    rows = load_rows(args.inputs)
    if not rows:
        raise ValueError("No inference inputs")
    for row in rows:
        validate_input(row)
    selected = rows[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError("Empty shard; reduce the number of workers")
    model = args.model.resolve()
    adapter = args.adapter.resolve() if args.adapter else None
    if not (model / "config.json").is_file() or (model / "adapter_config.json").exists():
        raise ValueError("Supply a full local Venus model as --model; use --adapter for the trained LoRA")
    protocol = dict(schema_version="heartbeat_eval_v1", inputs_sha256=digest_file(args.inputs),
                    runtime={pkg: version(pkg) for pkg in ("torch", "transformers", "tokenizers", "peft")},
                    model_identity=model_identity(model), adapter_identity=model_identity(adapter) if adapter else None,
                    adapter_sha256=digest_file(adapter / "adapter_model.safetensors") if adapter else None,
                    chunk_seconds=1, max_tokens=args.max_tokens, seed=args.seed, system=SYSTEM,
                    realtime=args.realtime, temperature=.7, top_k=100, top_p=.8,
                    frame_policy="last PTS at/before chunk start; initial frame only after available",
                    episode_policy="end_of_turn or listen", generate_audio=False)
    signature = digest_json(protocol)
    settings_signature = digest_json({k: v for k, v in protocol.items()
                                      if k not in ("model_identity", "adapter_identity", "adapter_sha256")})
    output = args.output.resolve()
    if args.resume:
        saved = json.loads((output / "protocol.json").read_text())
        if saved["protocol_sha256"] != signature or saved["shard_index"] != args.shard_index or saved["num_shards"] != args.num_shards:
            raise ValueError("Resume requires the same inputs/model/settings/sharding")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    # An exclusive lock prevents two workers overwriting this shard.
    import fcntl
    lock = (output / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(output / "protocol.json", dict(protocol, protocol_sha256=signature,
                shard_index=args.shard_index, num_shards=args.num_shards))
    cached, todo = {}, []
    for row in selected:
        path = case_file(output, row["id"])
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("protocol_sha256") != signature or old.get("id") != row["id"]:
                raise ValueError("Incompatible cached prediction")
            if old["status"] == "complete" or not args.retry_errors:
                cached[row["id"]] = old
                continue
        todo.append(row)
    duplex = None
    if todo:
        duplex = load_model(model, adapter, args.device, output / ".cache" / "modules")
        import torch
        from transformers import set_seed
        device = torch.device(args.device)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        checked = set()
        for row in todo:
            started = time.monotonic()
            media = None
            seed = (args.seed + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)) % 2**32
            try:
                path = Path(row["media_path"])
                key = (str(path), row["media_sha256"])
                if key not in checked:
                    if digest_file(path) != row["media_sha256"]:
                        raise ValueError("Media integrity check failed")
                    checked.add(key)
                media = Media(path, row["subtrack"], row["duration"])
                set_seed(seed)
                with torch.inference_mode():
                    status, chunks, error = replay(duplex, row, media, protocol, synchronize)
            except Exception as exc:
                status, chunks, error = "error", [], type(exc).__name__
            finally:
                if media is not None:
                    media.close()
            result = dict(id=row["id"], duration=row["duration"], status=status, chunks=chunks, error=error,
                          seed=seed, realtime=args.realtime, protocol_sha256=signature,
                          settings_sha256=settings_signature,
                          elapsed_seconds=time.monotonic() - started)
            atomic_json(case_file(output, row["id"]), result)
            cached[row["id"]] = result
            print(row["id"], status, len(chunks), "chunks", flush=True)
    write_rows(output / "predictions.jsonl", [cached[row["id"]] for row in selected])
    failed = [r["id"] for r in selected if cached[r["id"]]["status"] != "complete"]
    atomic_json(output / "status.json", dict(status="incomplete" if failed else "complete", tasks=len(selected), failed=failed))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
