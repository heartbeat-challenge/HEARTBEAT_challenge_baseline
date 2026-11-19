"""FDB serial send/wait/pop/pacing; dual media/wall software sinks. No VAD.
Backend is the unmodified frozen colleague UnifiedProcessor, called in-process.
Transport network latency is not included. Playback is a software sink, not hardware.
"""

import argparse
import base64
import collections
import json
import math
import os
import time
import traceback

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import ROOT, TYPES, rows, save

SR = 16000
N = SR


def audio(path, lo=0, hi=None):
    info = sf.info(path)
    w, sr = sf.read(
        path,
        start=round(lo * info.samplerate),
        stop=None if hi is None else round(hi * info.samplerate),
        dtype="float32",
    )
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        from math import gcd

        g = gcd(sr, SR)
        w = resample_poly(w, SR // g, sr // g)
    return np.asarray(w, dtype="float32")


def run(row, b, model):
    out = ROOT / "runs" / model / row["sample_id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        previous = json.loads((out / "run.json").read_text())
        if not previous.get("error"):
            return
        out.rename(out.with_name(out.name + ".failed." + str(time.time_ns())))
        out.mkdir(parents=True)
    sp = row["dynamic_spec"]
    typ = row["events"][0]["event_type"]
    dynamic = row["subset"] == "dynamic" and (
        typ in TYPES[2:] or (typ == "unlabeled" and bool(sp.get("clip_source")))
    )
    lo = sp["context_start_s"]
    hi = sp["prefix_end_s"]
    history = row.get("history", [])
    request = audio(row["input_audio"], lo, hi)
    source = request
    duration = len(source) / SR
    clip = (
        audio(sp["clip_source"], sp["clip_start_s"], sp["clip_end_s"])
        if dynamic
        else np.zeros(0, np.float32)
    )
    target_delay = float(sp.get("minimum_speech_history_s", 2.0))
    b.prepare(row["sample_id"], history)
    need_gate = False
    save(
        out / "model_context.json", dict(system_prompt=b.system_prompt, history=history)
    )
    tail_s = sp.get("post_event_s", 45) if dynamic else 30
    start = time.monotonic()
    pending = collections.deque()
    media = []
    input_frames = []
    frames = []
    steps = []
    play_segments = []
    active = None
    serial = 0
    texts = {}
    trigger = None
    event_pos = 0
    error = None
    finish = "hard_limit"
    first_audio = None
    delegate_locked = False
    play_end = 0.0
    events = []
    last_send = None

    def now():
        return time.monotonic() - start

    def pop_frame():
        y = np.zeros(N, np.float32)
        parts = []
        used = 0
        while pending and used < N:
            w, pos, tid = pending[0]
            n = min(N - used, len(w) - pos)
            y[used : used + n] = w[pos : pos + n]
            parts.append((used, n, tid))
            used += n
            pending[0][1] += n
            if pending[0][1] >= len(w):
                pending.popleft()
        return y, parts

    def emit(y, parts, index, received):
        nonlocal play_end
        begin = max(received, play_end)
        media.append(y)
        frames.append(
            dict(
                index=index,
                media_start_s=index,
                wall_start_s=begin,
                wall_end_s=begin + 1,
                received_wall_s=received,
            )
        )
        for off, n, tid in parts:
            seg = dict(
                turn_id=tid,
                media_start_s=index + off / SR,
                media_end_s=index + (off + n) / SR,
                wall_start_s=begin + off / SR,
                wall_end_s=begin + (off + n) / SR,
            )
            play_segments.append(seg)
        play_end = begin + 1

    try:
        for i in range(math.ceil(duration) + 150):
            if last_send is not None:
                delay = last_send + 1 - now()
                if delay > 0:
                    time.sleep(delay)
            wall = now()
            playing = next(
                (
                    s
                    for s in reversed(play_segments)
                    if s["wall_start_s"] <= wall < s["wall_end_s"]
                ),
                None,
            )
            if dynamic and trigger is None and i >= math.ceil(duration):
                tid = playing["turn_id"] if playing else None
                played = (
                    sum(
                        max(0, min(wall, s["wall_end_s"]) - s["wall_start_s"])
                        for s in play_segments
                        if s["turn_id"] == tid
                    )
                    if tid
                    else 0
                )
                # Wait only inside a currently scheduled playback segment. The model's
                # end_of_turn/listen flags do not cancel queued playback.
                if tid and played < target_delay:
                    wait = target_delay - played
                    if wall + wait < playing["wall_end_s"]:
                        time.sleep(wait)
                        wall = now()
                        playing = next(
                            (
                                s
                                for s in reversed(play_segments)
                                if s["wall_start_s"] <= wall < s["wall_end_s"]
                            ),
                            None,
                        )
                        tid = playing["turn_id"] if playing else None
                        played = (
                            sum(
                                max(0, min(wall, s["wall_end_s"]) - s["wall_start_s"])
                                for s in play_segments
                                if s["turn_id"] == tid
                            )
                            if tid
                            else 0
                        )
                if (
                    tid
                    and played >= target_delay
                    and playing["wall_end_s"] - wall
                    >= float(sp.get("minimum_queued_audio_s", 0.01))
                ):
                    trigger = dict(
                        input_index=i,
                        media_start_s=float(i),
                        delivery_wall_s=wall,
                        turn_id=tid,
                        played_s=played,
                        target_delay_s=target_delay,
                        gate_required=need_gate,
                        actual_playback_at_delivery=True,
                    )
                    events.append(dict(type="dynamic_trigger", **trigger))
                elif i >= math.ceil(duration) + sp.get("arm_timeout_s", 30):
                    finish = "trigger_timeout"
                    break
            x = np.zeros(N, np.float32)
            if i * N < len(source):
                w = source[i * N : (i + 1) * N]
                x[: len(w)] = w
                kind = "context"
            elif trigger is not None and event_pos < len(clip):
                w = clip[event_pos : event_pos + N]
                x[: len(w)] = w
                event_pos += len(w)
                kind = "event"
            else:
                kind = "silence"
            if not dynamic and i >= math.ceil(duration) + tail_s:
                finish = "static_tail_complete"
                break
            if (
                trigger is not None
                and i >= trigger["input_index"] + math.ceil(len(clip) / SR) + tail_s
            ):
                finish = "event_tail_complete"
                break
            input_frames.append(x)
            sent = now()
            last_send = sent
            r = b.step(x, force_listen=delegate_locked)
            received = now()
            if r.is_listen:
                if active is not None:
                    active = None
            else:
                if active is None:
                    serial += 1
                    active = f"turn_{serial}"
                texts[active] = texts.get(active, "") + (r.text or "")
            raw = np.zeros(0, np.float32)
            if not r.is_listen and r.audio_data:
                raw = np.frombuffer(
                    base64.b64decode(r.audio_data), dtype=np.float32
                ).copy()
                raw = resample_poly(raw, 2, 3).astype(np.float32)
                if len(raw):
                    pending.append([raw, 0, active])
                    if first_audio is None:
                        first_audio = dict(
                            input_index=i,
                            block_endpoint_s=i + 1,
                            received_wall_s=received,
                        )
            if r.end_of_turn and active is not None:
                active = None
            y, parts = pop_frame()
            emit(y, parts, i, received)
            step = dict(
                index=i,
                input_start_s=i,
                input_end_s=i + 1,
                sent_wall_s=sent,
                received_wall_s=received,
                compute_s=received - sent,
                is_listen=r.is_listen,
                end_of_turn=r.end_of_turn,
                text=r.text,
                input_kind=kind,
                generated_samples=len(raw),
                is_delegate=r.is_delegate,
                delegate_text=r.delegate_text,
            )
            steps.append(step)
            b.finalize()
            # Match frozen client: keep sending input, force listen after a delegate.
            if r.is_delegate:
                delegate_locked = True
                if active is not None:
                    active = None
        while pending:
            y, parts = pop_frame()
            emit(y, parts, len(media), now())
        # Let the real software playback deadline elapse; do not claim future audio has played.
        if play_end > now():
            time.sleep(play_end - now())
    except Exception:
        error = traceback.format_exc()
        finish = "error"

    iw = np.concatenate(input_frames) if input_frames else np.zeros(N, np.float32)
    ow = np.concatenate(media) if media else np.zeros(N, np.float32)
    sf.write(out / "input_media.wav", iw, SR, subtype="PCM_16")
    sf.write(out / "output_media.wav", ow, SR, subtype="PCM_16")
    actual = np.zeros(max(1, math.ceil(play_end * SR)), np.float32)
    for f, y in zip(frames, media):
        a = round(f["wall_start_s"] * SR)
        n = min(len(y), len(actual) - a)
        actual[a : a + n] = y[:n]
    sf.write(out / "output_playback.wav", actual, SR, subtype="PCM_16")
    # Full reference input has media timestamps only. No invented real microphone timeline.
    save(
        out / "run.json",
        dict(
            end_protocol=os.environ.get("HEARTBEAT_END_PROTOCOL", "legacy"),
            end_protocol_closed_units=(
                b.end_protocol["closed_units"] if b.end_protocol else 0
            ),
            runtime_host=__import__("socket").gethostname(),
            runtime_torch=__import__("torch").__version__,
            runtime_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
            sample_id=row["sample_id"],
            model=model,
            type=typ,
            language=row["language"],
            row=row,
            protocol="heartbeat_dev_release_v2_force0_plainhistory_dataset_spec",
            target_delay_s=target_delay,
            steps=steps,
            frames=frames,
            play_segments=play_segments,
            events=events,
            trigger=trigger,
            first_audio=first_audio,
            texts=texts,
            gate_receipts=[],
            termination=finish,
            error=error,
            wall_seconds=now(),
            input_duration_s=duration,
            request_start_s=0.0,
            request_end_s=row["request_alignment_end_s"],
            event_duration_s=len(clip) / SR,
            config=b.config.model_dump(),
            delegate_disabled_token_ids=b.delegate_disabled_ids,
            system_prompt=b.system_prompt,
            playback="software FIFO of fixed one-second frames, listen does not cancel",
            transport="in-process frozen colleague backend; websocket/network overhead excluded",
            input_clock="prerecorded media paced by serial processing; wall reference uses actual block delivery",
        ),
    )
    print(
        json.dumps(
            dict(
                sample=row["sample_id"],
                model=model,
                type=typ,
                trigger=trigger is not None,
                seconds=round(now(), 2),
                error=bool(error),
            )
        ),
        flush=True,
    )
    if error:
        raise RuntimeError(error)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    from .backend import Backend

    b = Backend(a.model)
    print("BACKEND_READY", a.model, flush=True)
    for i, r in enumerate(rows(a.manifest)):
        if i % a.shards == a.shard:
            run(r, b, a.model)


if __name__ == "__main__":
    main()
