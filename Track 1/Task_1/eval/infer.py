"""Run offline multiple-choice inference and score labeled rows."""

import io
import json
import math
import os
import re
from itertools import islice

from data_io import mcq_prompt, parquet_rows, write_json


def metrics(rows):
    labeled = [row for row in rows if row["answer"] is not None]
    return {
        "n": len(rows),
        "labeled": len(labeled),
        "accuracy": sum(row["correct"] for row in labeled) / len(labeled)
        if labeled
        else None,
        "invalid": sum(row["prediction"] is None for row in rows),
    }


def test_rows(source, questions, limit, labels=None):
    manifest = source / "easy_audio_manifest.json" if source.is_dir() else source
    audio_rows = json.loads(manifest.read_text(encoding="utf-8"))
    question_rows = json.loads(questions.read_text(encoding="utf-8"))
    by_id = {row["scene_id"]: row for row in question_rows}
    if len(by_id) != len(question_rows):
        raise ValueError(f"Duplicate question IDs in {questions}")
    records = {}
    if labels:
        with labels.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                sample_id = record["scene_id"]
                if sample_id in records:
                    raise ValueError(f"Duplicate Test label ID: {sample_id}")
                records[sample_id] = record
    seen = set()
    for audio in islice(audio_rows, limit):
        sample_id = audio["scene_id"]
        if sample_id in seen or sample_id not in by_id:
            raise ValueError(f"Missing question or duplicate audio ID: {sample_id}")
        seen.add(sample_id)
        question = by_id[sample_id]
        options = {choice["id"]: choice["text"] for choice in question["choices"]}
        if set(options) != {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid choice IDs for {sample_id}")
        prompt = "\n".join(
            item for item in [question.get("instruction"), question["question"]] if item
        )
        answer = question.get("answer")
        if labels:
            record = records.get(sample_id)
            if (
                not record
                or record["question"] != question
                or record["audio_sha256"] != audio["sha256"]
            ):
                raise ValueError(f"Test label audio/question mismatch for {sample_id}")
            answer = record["script"]["gold_letter"]
            if answer not in options:
                raise ValueError(f"Invalid Test label for {sample_id}")
        path = manifest.parent / audio["audio"]
        yield dict(
            sample_id=sample_id,
            prompt=prompt,
            options=options,
            audio={"bytes": path.read_bytes()},
            answer=answer,
        )
    if not seen:
        raise ValueError(f"No Test audio in {manifest}")


def evaluate(args):
    os.environ.update(USE_TF="0", TRANSFORMERS_NO_TF="1")
    import numpy as np
    import soundfile as sf
    import torch
    from peft import PeftModel
    from scipy.signal import resample_poly
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
            self.get_seq_length(layer_idx)
        )
    base = str(args.model.resolve())
    torch.manual_seed(42)
    model = AutoModel.from_pretrained(
        base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        init_tts=False,
        init_vision=False,
    )
    if args.adapter:
        from runtime_compat import configure_lora

        configure_lora()
        model = PeftModel.from_pretrained(
            model, str(args.adapter.resolve())
        ).merge_and_unload()
    model = model.eval().cuda()
    for layer in model.apm.modules():
        if hasattr(layer, "self_attn"):

            def attention(*args, original=layer.self_attn.forward, **kwargs):
                result = original(*args, **kwargs)
                return (
                    (*result, None)
                    if isinstance(result, tuple) and len(result) == 2
                    else result
                )

            layer.self_attn.forward = attention
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)

    def decode(media):
        audio, rate = sf.read(io.BytesIO(media["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        return np.asarray(audio, dtype=np.float32)

    rows = []
    samples = (
        test_rows(args.test_manifest, args.questions, args.limit, args.test_labels)
        if args.test_manifest
        else parquet_rows(args.parquet_dir, args.limit)
    )
    output = args.output / "predictions.jsonl"
    with output.with_suffix(".tmp").open("w", encoding="utf-8") as stream:
        for row in samples:
            content = []
            if row.get("assistant_reference_audio"):
                content.extend(
                    [
                        "以下是助手声音参考：",
                        decode(row["assistant_reference_audio"]),
                        "以下是待判断对话：",
                    ]
                )
            content.extend([decode(row["audio"]), mcq_prompt(row)])
            with torch.inference_mode():
                text = model.chat(
                    msgs=[{"role": "user", "content": content}],
                    tokenizer=tokenizer,
                    processor=processor,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=16,
                    generate_audio=False,
                    enable_thinking=False,
                    merge_audio_from_same_content=False,
                    max_inp_length=16000,
                )
            match = re.fullmatch(r"\s*([ABCD])[.。]?\s*", str(text))
            prediction = match.group(1) if match else None
            answer = row.get("answer")
            item = dict(
                sample_id=row["sample_id"],
                prediction=prediction,
                raw_output=str(text),
                answer=answer,
                correct=prediction == answer if answer is not None else None,
                has_reference=bool(row.get("assistant_reference_audio")),
            )
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows.append(item)
    output.with_suffix(".tmp").replace(output)
    write_json(args.output / "metrics.json", metrics(rows))
