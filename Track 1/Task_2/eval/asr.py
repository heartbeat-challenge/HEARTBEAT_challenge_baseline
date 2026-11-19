"""ASR of actual output audio, mapped to both media and software playback clocks."""

import argparse
import hashlib
import json
import math
import os

from .common import MODEL_NAME, ROOT, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        os.environ["HEARTBEAT_ASR_MODEL"],
        dtype=torch.bfloat16 if a.device.startswith("cuda") else torch.float32,
        device_map=a.device,
        max_inference_batch_size=1,
        max_new_tokens=1536,
        forced_aligner=os.environ["HEARTBEAT_ALIGNER"],
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16 if a.device.startswith("cuda") else torch.float32,
            device_map=a.device,
        ),
    )

    def align(path, lo, hi, lang):
        w, sr = sf.read(
            path,
            start=max(0, round(lo * 16000)),
            stop=round(hi * 16000),
            dtype="float32",
        )
        assert sr == 16000
        words = []
        texts = []
        valid = True
        if len(w) < 1600:
            return dict(text="", words=[], valid=True, silent_or_short=True)
        for k in range(0, len(w), 24 * sr):
            x = w[k : k + 24 * sr]
            if len(x) < 1600:
                continue
            z = model.transcribe(
                audio=(np.pad(x, (4800, 4800)), sr),
                language="Chinese" if lang == "zh" else "English",
                return_time_stamps=True,
            )[0]
            texts.append(z.text)
            local = [
                dict(
                    text=q.text,
                    start_s=float(q.start_time) - 0.3,
                    end_s=float(q.end_time) - 0.3,
                )
                for q in z.time_stamps or []
                if str(q.text).strip()
            ]
            ok = (
                bool(local)
                and all(
                    -0.1 <= q["start_s"] <= q["end_s"] <= len(x) / sr + 0.1
                    for q in local
                )
                and local[0]["start_s"] < local[0]["end_s"]
                and local[-1]["start_s"] < local[-1]["end_s"]
            )
            valid = valid and ok
            words.extend(
                dict(
                    text=q["text"],
                    start_s=lo + k / sr + max(0, q["start_s"]),
                    end_s=lo + k / sr + min(len(x) / sr, q["end_s"]),
                )
                for q in local
            )
        return dict(text=" ".join(texts), words=words, valid=bool(words) and valid)

    manifest = [json.loads(s) for s in open(ROOT / "data" / f"{a.subset}.jsonl")][
        a.shard :: a.shards
    ]
    if a.limit:
        manifest = manifest[: a.limit]
    for row in manifest:
        folder = ROOT / "runs" / MODEL_NAME / row["sample_id"]
        out = folder / "acoustic.json"
        if out.exists() and json.loads(out.read_text()).get("status") == "complete":
            continue
        try:
            r = json.loads((folder / "run.json").read_text())
            assert not r["error"], r["error"]
            request = align(
                folder / "input_media.wav",
                r["request_start_s"],
                r["request_end_s"],
                r["language"],
            )
            event = (
                align(
                    folder / "input_media.wav",
                    r["trigger"]["media_start_s"],
                    r["trigger"]["media_start_s"] + r["event_duration_s"],
                    r["language"],
                )
                if r["trigger"]
                else None
            )
            if row["subset"] == "static" and row["ground_truth"].get("event_span_s"):
                lo, hi = row["ground_truth"]["event_span_s"]
                event = align(folder / "input_media.wav", lo, hi, r["language"])
            by = {}
            for s in r["play_segments"]:
                by.setdefault(s["turn_id"], []).append(s)
            turns = []
            for tid, ss in by.items():
                lo = min(s["media_start_s"] for s in ss)
                hi = max(s["media_end_s"] for s in ss)
                z = align(folder / "output_media.wav", lo, hi, r["language"])
                for w in z["words"]:
                    for edge in ["start", "end"]:
                        t = w[edge + "_s"]
                        idx = min(
                            len(r["frames"]) - 1,
                            max(0, math.floor(t - (1e-9 if edge == "end" else 0))),
                        )
                        fr = r["frames"][idx]
                        w["wall_" + edge + "_s"] = (
                            fr["wall_start_s"] + t - fr["media_start_s"]
                        )
                turns.append(dict(turn_id=tid, alignment=z))
            save(
                out,
                dict(
                    status="complete",
                    request=request,
                    event=event,
                    turns=turns,
                    audio_sha256=hashlib.sha256(
                        (folder / "output_media.wav").read_bytes()
                    ).hexdigest(),
                    method="Qwen3-ASR + ForcedAligner; actual waveform, no generated-text substitution",
                ),
            )
        except Exception as e:
            save(
                out, dict(status="error", error=type(e).__name__, message=str(e)[:500])
            )
            raise
        print("ASR", row["sample_id"], flush=True)


if __name__ == "__main__":
    main()
