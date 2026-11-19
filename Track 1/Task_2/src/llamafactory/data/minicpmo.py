# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..extras.constants import IGNORE_INDEX
from ..extras import logging

logger = logging.get_logger(__name__)

MEDIA_PROBE_TIMEOUT = 30
MEDIA_DECODE_TIMEOUT = 120


def _probe_media(path: str) -> dict[str, Any]:
    """Probe streams and container duration for a single media path."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path],
            capture_output=True,
            text=True,
            timeout=MEDIA_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffprobe timed out after {MEDIA_PROBE_TIMEOUT}s for {path}") from exc
    if probe.returncode != 0:
        raise ValueError(f"ffprobe failed for {path}: {probe.stderr.strip()}")
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams", [])
    if not streams or not any(x.get("codec_type") in ("video", "audio") for x in streams):
        raise ValueError(f"Neither video nor audio stream found: {path}")
    return {"streams": streams, "duration": float(info.get("format", {}).get("duration") or 0.0)}


def _decode_video_frames(path: str, max_seconds: int) -> list[Image.Image]:
    """Decode 1fps causal RGB frames into a temp dir."""
    with tempfile.TemporaryDirectory(prefix="minicpmo_frames_") as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
            "-t", str(max_seconds), "-vf", "fps=1", "-q:v", "2", "-y", frame_pattern,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=MEDIA_DECODE_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Video decode timed out after {MEDIA_DECODE_TIMEOUT}s for {path}") from exc
        if result.returncode != 0:
            raise ValueError(f"Video decode failed for {path}: {result.stderr.decode(errors='ignore').strip()}")
        frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
        frames = [Image.open(fp).convert("RGB").copy() for fp in frame_paths]
    if not frames:
        raise ValueError(f"Cannot decode video frames: {path}")
    return frames


def _decode_pcm(path: str, max_seconds: int) -> np.ndarray:
    """Decode mono 16k float32 PCM. Raises if no samples could be decoded."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
        "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=MEDIA_DECODE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Audio decode timed out after {MEDIA_DECODE_TIMEOUT}s for {path}") from exc
    if result.returncode != 0:
        raise ValueError(f"Audio decode failed for {path}: {result.stderr.decode(errors='ignore').strip()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"Declared audio stream cannot be decoded: {path}")
    return audio


def _black_frames(num: int) -> list[Image.Image]:
    """Synthesize a dummy 1fps black-frame track for pure-audio samples."""
    return [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(max(1, int(num)))]


def _load_media(
    paths: list[str], max_seconds: int, use_vision: bool
) -> tuple[list[Image.Image], np.ndarray | None, float]:
    """Load media and return ``(frames, waveform|None, duration)``.

    * Video protocol (the first listed file has a video stream): decode 1fps frames plus
      the mono 16k audio track from that container, like the original ``_decode_media``.
    * Pure-audio protocol (the first listed file has no video stream, e.g.
      ``speech_chat_streaming``/``audiomcq``): only the **first** supplied track is
      consumed; any extra entries are ignored (matches official-LlamaFactory dual-track
      handling). A synthetic black video track is produced only to satisfy the per-unit
      frame cadence; with ``use_vision=False`` it is never fed to the vision encoder.
    """
    if not paths:
        raise ValueError("MiniCPM-o runtime sample requires a media path.")
    if not use_vision and Path(paths[0]).suffix.lower() in {".wav", ".flac"}:
        from math import gcd

        import soundfile as sf
        from scipy.signal import resample_poly

        with sf.SoundFile(paths[0]) as audio:
            rate = audio.samplerate
            waveform = audio.read(frames=max_seconds * rate, dtype="float32", always_2d=True).mean(axis=1)
        if not waveform.size:
            raise ValueError(f"Audio contains no samples: {paths[0]}")
        duration = len(waveform) / rate
        if rate != 16000:
            divisor = gcd(rate, 16000)
            waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
        return _black_frames(1), np.asarray(waveform, dtype=np.float32), duration

    first = _probe_media(paths[0])
    if any(s.get("codec_type") == "video" for s in first["streams"]):
        duration = min(first["duration"], float(max_seconds))
        frames = _decode_video_frames(paths[0], max_seconds)
        audio = (
            _decode_pcm(paths[0], max_seconds)
            if any(s.get("codec_type") == "audio" for s in first["streams"])
            else None
        )
        return frames, audio, duration

    # Pure-audio protocol: only the first supplied track is consumed.
    duration = min(first["duration"], float(max_seconds))
    waveform = _decode_pcm(paths[0], max_seconds)
    frame_count = int(math.ceil(duration)) if duration > 0 else int(max_seconds)
    frame_count = max(1, min(int(max_seconds), frame_count))
    frames = _black_frames(frame_count) if use_vision else _black_frames(1)
    return frames, waveform, duration


def _decode_media(path: str, max_seconds: int) -> tuple[list[Image.Image], np.ndarray | None, float]:
    """Backward-compatible single-media loader; see ``_load_media`` for pure-audio support."""
    return _load_media([path], max_seconds, use_vision=True)


def _offline_timestamps(duration: float, max_frames: int, long_video_fps: float) -> list[float]:
    """Match the released MiniCPM-o offline timestamp sampler."""
    if duration <= 0:
        raise ValueError(f"Invalid offline video duration: {duration}.")
    if duration <= max_frames:
        return [float(index) for index in range(max(1, math.ceil(duration)))]

    candidate_count = max(1, int(duration * long_video_fps))
    selected = np.linspace(0, candidate_count - 1, max_frames, dtype=np.int64)
    return [float(index) / long_video_fps for index in selected]


def _decode_offline_media(
    path: str,
    max_frames: int,
    long_video_fps: float,
) -> tuple[list[Image.Image], list[np.ndarray], list[float], float, bool]:
    """Decode only selected full-timeline frames and partition complete audio at their timestamps."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", path],
            capture_output=True,
            text=True,
            timeout=MEDIA_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"ffprobe timed out after {MEDIA_PROBE_TIMEOUT}s for {path}") from exc
    if probe.returncode != 0:
        raise ValueError(f"ffprobe failed for {path}: {probe.stderr.strip()}")
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams", [])
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise ValueError(f"No video stream: {path}")
    duration = float(info.get("format", {}).get("duration") or 0.0)
    is_long_video = duration > max_frames
    candidate_fps = long_video_fps if is_long_video else 1.0

    with tempfile.TemporaryDirectory(prefix="minicpmo_offline_frames_") as tmpdir:
        frame_pattern = str(Path(tmpdir) / "frame_%06d.jpg")

        def extract_selected(candidate_count: int) -> tuple[list[Image.Image], list[float]]:
            # NOTE: an earlier implementation built `select='eq(n,0)+eq(n,1)+...'` listing every
            # target frame index. FFmpeg's select expression parser rejects ~120+ terms
            # ("Error while parsing expression"), which silently dropped most offline videos.
            # Decode the whole timeline at candidate_fps and pick the target frames in Python
            # instead: timestamps stay on the original candidate_fps grid (so the offline
            # audio-embedding layout is still fully consumed), just without the giant expression.
            for old_frame in Path(tmpdir).glob("frame_*.jpg"):
                old_frame.unlink()
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                "-vf", f"fps={candidate_fps:g}", "-fps_mode", "vfr", "-q:v", "2", "-y", frame_pattern,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=max(MEDIA_DECODE_TIMEOUT, int(duration * 2)))
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f"Offline video decode timed out for {path}") from exc
            if result.returncode != 0:
                raise ValueError(
                    f"Offline video decode failed for {path}: {result.stderr.decode(errors='ignore').strip()}"
                )
            frame_paths = sorted(Path(tmpdir).glob("frame_*.jpg"))
            if is_long_video:
                n = len(frame_paths)
                if n >= max_frames:
                    pick = np.linspace(0, n - 1, max_frames, dtype=np.int64).tolist()
                    timestamps = [float(i) / candidate_fps for i in pick]
                    frames = [Image.open(frame_paths[i]).convert("RGB").copy() for i in pick]
                else:
                    timestamps = [float(i) / candidate_fps for i in range(n)]
                    frames = [Image.open(p).convert("RGB").copy() for p in frame_paths]
            else:
                timestamps = [float(index) for index in range(len(frame_paths))]
                frames = [Image.open(p).convert("RGB").copy() for p in frame_paths]
            return frames, timestamps

        initial_count = max(1, int(duration * candidate_fps) if is_long_video else math.ceil(duration))
        frames, timestamps = extract_selected(initial_count)
        if not is_long_video:
            # Official FFmpeg path derives short-video timestamps from the actual 1 FPS output.
            # Container duration can overestimate the filter output by one frame at the tail.
            timestamps = [float(index) for index in range(len(frames))]

    if not frames:
        raise ValueError(f"Cannot decode offline video frames: {path}")
    if len(frames) != len(timestamps):
        raise ValueError(
            f"Offline frame/timestamp mismatch after tail correction for {path}: "
            f"frames={len(frames)}, timestamps={len(timestamps)}."
        )

    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    if not has_audio:
        return frames, [], timestamps, duration, False
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=max(MEDIA_DECODE_TIMEOUT, int(duration * 2)))
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Offline audio decode timed out for {path}") from exc
    if result.returncode != 0:
        raise ValueError(f"Offline audio decode failed for {path}: {result.stderr.decode(errors='ignore').strip()}")
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if waveform.size == 0:
        raise ValueError(f"Declared audio stream cannot be decoded: {path}")

    audio_segments: list[np.ndarray] = []
    for index, timestamp in enumerate(timestamps):
        next_timestamp = timestamps[index + 1] if index + 1 < len(timestamps) else len(waveform) / 16000.0
        start_sample = min(int(timestamp * 16000), len(waveform))
        end_sample = min(int(next_timestamp * 16000), len(waveform))
        segment = waveform[start_sample:end_sample]
        if index + 1 == len(timestamps) and len(segment) < 1600:
            segment = np.pad(segment, (0, 1600 - len(segment)))
        audio_segments.append(segment)
    return frames, audio_segments, timestamps, duration, True


def _pad_2d(items: list[torch.Tensor], value: int | float) -> torch.Tensor:
    max_len = max(x.shape[0] for x in items)
    return torch.stack([F.pad(x, (0, max_len - x.shape[0]), value=value) for x in items])


@dataclass
class MiniCPMORuntimeDataCollator:
    tokenizer: Any
    processor: Any
    data_args: Any
    finetuning_args: Any
    template: Any
    model: Any

    def __post_init__(self):
        self.unit_id = self.tokenizer.convert_tokens_to_ids("<unit>")
        self.unit_end_id = self.tokenizer.convert_tokens_to_ids("</unit>")
        self.listen_id = self.tokenizer.convert_tokens_to_ids("<|listen|>")
        self.speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        self.chunk_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.turn_eos_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        self.unk_id = self.tokenizer.unk_token_id
        self.image_start_id = self.tokenizer.im_start_id
        self.image_end_id = self.tokenizer.im_end_id
        self.max_slice_nums = 1
        self._ref_audio_cache: tuple[Any, int, int] | None = None

    def _get_ref_audio(self) -> tuple[Any, int, int]:
        """Load and cache the reference audio (mel, mel_len, token_count) for the duplex
        system-prompt voice block, matching official duplex inference with ref audio."""
        if self._ref_audio_cache is None:
            path = self.data_args.minicpmo_ref_audio_path
            if not path:
                base = getattr(getattr(self.model, "config", None), "_name_or_path", "") or ""
                path = os.path.join(base, "assets", "HT_ref_audio.wav")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
                "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=MEDIA_DECODE_TIMEOUT)
            if result.returncode != 0 or not result.stdout:
                raise ValueError(f"Cannot decode reference audio: {path}")
            wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
            audio_inputs = self.processor.audio_processor(
                [wav], sampling_rate=16000, return_attention_mask=True,
                padding="longest", truncation=False, return_tensors="pt",
            )
            mel = audio_inputs["input_features"][0]
            mel_len = int(audio_inputs["attention_mask"][0].sum())
            ref_tokens = int(((mel_len - 1) // 2 + 1 - 5) // 5 + 1)
            self._ref_audio_cache = (mel, mel_len, ref_tokens)
        return self._ref_audio_cache

    def _target_weights(self, ids: list[int], mode: str) -> list[float]:
        if mode == "offline_chat":
            return [self.finetuning_args.minicpmo_offline_text_weight] * len(ids)
        out, speaking = [], False
        for tid in ids:
            if tid == self.listen_id:
                out.append(self.finetuning_args.minicpmo_listen_weight)
            elif tid == self.speak_id:
                speaking = True
                out.append(self.finetuning_args.minicpmo_speak_weight)
            else:
                out.append(self.finetuning_args.minicpmo_spoken_text_weight if speaking else 1.0)
        return out

    def _media_paths(self, feature: dict[str, Any]) -> list[str]:
        paths = feature.get("raw_videos") or feature.get("raw_audios") or []
        if not paths:
            raise ValueError("MiniCPM-o runtime sample requires a media path (raw_videos/raw_audios).")
        return paths

    def _video_path(self, feature: dict[str, Any]) -> str:
        return self._media_paths(feature)[0]

    def _build_duplex(self, feature: dict[str, Any]) -> dict[str, Any]:
        messages = (feature["raw_prompt"] or []) + (feature["raw_response"] or [])
        if len(messages) % 2:
            raise ValueError("Duplex messages must be user/assistant pairs.")
        units = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
        # Trim only redundant listen tail after the last completed response.
        last_turn = -1
        for i, (_, assistant) in enumerate(units):
            if "<|turn_eos|>" in assistant["content"]:
                last_turn = i
        if last_turn >= 0:
            keep = last_turn + 1 + max(0, self.data_args.minicpmo_tail_listen_units)
            units = units[:keep]
        if len(units) > self.data_args.minicpmo_max_video_seconds + 1:
            raise ValueError("Duplex sample exceeds configured video duration.")

        use_vision = bool(feature.get("use_vision", True))
        frames, waveform, _ = _load_media(self._media_paths(feature), len(units), use_vision=use_vision)
        if len(frames) < len(units):
            frames.extend([frames[-1].copy() for _ in range(len(units) - len(frames))])
        use_audio = bool(feature.get("use_audio", True)) and waveform is not None
        if feature.get("instruction_mode") == "audio" and not use_audio:
            raise ValueError("Speech-In sample has no audio track.")

        image_inputs = self.processor.process_image(frames[: len(units)], max_slice_nums=1) if use_vision else None
        pixel_values = image_inputs["pixel_values"][0] if image_inputs is not None else []
        tgt_sizes = image_inputs["tgt_sizes"][0] if image_inputs is not None else torch.empty((0, 2), dtype=torch.int32)

        streaming_audio = []
        audio_token_counts: list[int] = []
        if use_audio and len(units) > 0:
            # Match official MiniCPM-o streaming units: unit t consumes frame t and audio chunk t
            # before predicting that unit's <|listen|>/<|speak|> target. Encode all unit-aligned
            # audio with the native 1-second subsequent-chunk mask, preserving history while
            # preventing access to future chunks.
            required = len(units) * 16000
            train_audio = waveform[:required]
            if len(train_audio) < required:
                train_audio = np.pad(train_audio, (0, required - len(train_audio)))
            # Whisper positional embeddings cover 30 seconds. Split longer streams into
            # independent 30-second encoder blocks; the LLM still retains all previous blocks.
            for start in range(0, len(train_audio), 30 * 16000):
                wav_block = train_audio[start : start + 30 * 16000]
                audio_inputs = self.processor.audio_processor(
                    [wav_block], sampling_rate=16000, return_attention_mask=True,
                    padding="longest", truncation=False, return_tensors="pt",
                )
                mel = audio_inputs["input_features"][0]
                mel_len = int(audio_inputs["attention_mask"][0].sum())
                streaming_audio.append((mel, mel_len, False))
                block_tokens = int(((mel_len - 1) // 2 + 1 - 5) // 5 + 1)
                remaining_seconds = min(30, math.ceil(len(wav_block) / 16000))
                remaining_tokens = block_tokens
                for _ in range(remaining_seconds):
                    count = min(10, remaining_tokens)
                    audio_token_counts.append(max(0, count))
                    remaining_tokens -= count
                if remaining_tokens > 0 and audio_token_counts:
                    audio_token_counts[-1] += remaining_tokens


        system = feature.get("raw_system") or "Streaming Omni Conversation."
        use_ref_audio = (
            self.data_args.minicpmo_ref_audio_prob > 0
            and random.random() < self.data_args.minicpmo_ref_audio_prob
        )
        if use_ref_audio:
            # Match official duplex inference with a voice reference: the system block carries
            # `<|audio_start|>` + ref-audio embeddings + `<|audio_end|>` before <|im_end|>.
            # The ref mel is prepended to runtime_audio and its bound to audio_bounds so the
            # model-side scatter consumes it first, before any per-unit audio.
            ref_mel, ref_mel_len, ref_tokens = self._get_ref_audio()
            prefix_ids = self.tokenizer.encode(
                f"<|im_start|>system\n{system}\n<|audio_start|>", add_special_tokens=False
            )
            suffix_ids = self.tokenizer.encode("<|audio_end|><|im_end|>", add_special_tokens=False)
            ids = prefix_ids + [self.unk_id] * ref_tokens + suffix_ids
            labels = [IGNORE_INDEX] * len(ids)
            weights = [0.0] * len(ids)
            audio_bounds = [[len(prefix_ids), len(prefix_ids) + ref_tokens]]
            streaming_audio.insert(0, (ref_mel, ref_mel_len, False))
        else:
            ids = self.tokenizer.encode(f"<|im_start|>system\n{system}<|im_end|>", add_special_tokens=False)
            labels = [IGNORE_INDEX] * len(ids)
            weights = [0.0] * len(ids)
            audio_bounds = []
        image_bounds = []
        image_idx = 0
        for t, (user, assistant) in enumerate(units):
            ids.append(self.unit_id); labels.append(IGNORE_INDEX); weights.append(0.0)
            if use_vision:
                start = len(ids) + 1
                ids.extend([self.image_start_id] + [self.unk_id] * 64 + [self.image_end_id])
                labels.extend([IGNORE_INDEX] * 66); weights.extend([0.0] * 66)
                image_bounds.append([start, start + 64]); image_idx += 1
            if t < len(audio_token_counts) and audio_token_counts[t] > 0:
                audio_tokens = audio_token_counts[t]
                start = len(ids)
                ids.extend([self.unk_id] * audio_tokens)
                labels.extend([IGNORE_INDEX] * audio_tokens); weights.extend([0.0] * audio_tokens)
                audio_bounds.append([start, start + audio_tokens])
            content = user["content"]
            marker = "<image>"
            text = content.split(marker, 1)[-1] if marker in content else content
            text = text.replace("</unit>", "").replace("<unit>", "").replace("<image>", "").replace("<audio>", "").strip()
            if feature.get("instruction_mode") == "audio" and text:
                # Speech-In invariant: questions live only in the TTS-mixed audio track, so
                # unit text must be empty. Leftover text means a data-conversion bug (e.g.
                # the historical zh rows trained as unintended text+audio double input).
                # Fail loudly into a dummy sample instead of silently training on it.
                raise ValueError(f"Speech-In sample carries unexpected unit text: {text[:80]!r}")
            if text:
                text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids.extend(text_ids); labels.extend([IGNORE_INDEX] * len(text_ids)); weights.extend([0.0] * len(text_ids))
            target_ids = self.tokenizer.encode(assistant["content"], add_special_tokens=False)
            # Standard causal labels: label at each target position; Trainer shifts internally.
            ids.extend(target_ids); labels.extend(target_ids); weights.extend(self._target_weights(target_ids, "duplex_streaming"))
            ids.append(self.unit_end_id); labels.append(IGNORE_INDEX); weights.append(0.0)

        if len(ids) > self.data_args.minicpmo_max_expanded_length:
            raise ValueError(f"Expanded duplex length {len(ids)} exceeds limit.")
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": torch.tensor(image_bounds, dtype=torch.long).reshape(-1, 2),
            "audio_bounds": torch.tensor(audio_bounds, dtype=torch.long).reshape(-1, 2),
            "runtime_audio": streaming_audio,
            "audio_mode": "chunk_causal",
            "training_mode_id": 1,
        }

    def _build_offline(self, feature: dict[str, Any]) -> dict[str, Any]:
        prompt = feature["raw_prompt"] or []
        response = feature["raw_response"] or []
        if not response:
            raise ValueError("Offline sample has no assistant response.")

        system = feature.get("raw_system") or "You are a helpful assistant."
        use_vision = bool(feature.get("use_vision", True))
        use_audio = bool(feature.get("use_audio", True))
        audio_segments: list[np.ndarray] = []
        media_parts: list[str] = []
        sampled_frames: list[Image.Image] = []
        if use_vision:
            # Audio-video offline: official MiniCPM-o half-duplex chat (`adjust_audio_length=False`).
            # Each sampled frame timestamp owns the audio interval up to the next sampled frame
            # timestamp; the final frame owns the remainder. The LLM layout stays IAIA... after
            # sparse full-timeline sampling; no audio gaps are discarded.
            max_frames = int(getattr(self.data_args, "minicpmo_offline_max_frames", 128))
            long_video_fps = float(getattr(self.data_args, "minicpmo_offline_long_video_fps", 10.0))
            sampled_frames, decoded_audio_segments, sampled_timestamps, _duration, has_audio = _decode_offline_media(
                self._video_path(feature), max_frames, long_video_fps
            )
            use_audio = use_audio and has_audio
            for frame_idx, _timestamp in enumerate(sampled_timestamps):
                media_parts.append("(<image>./</image>)")
                if use_audio:
                    audio_segments.append(decoded_audio_segments[frame_idx])
                    media_parts.append("(<audio>./</audio>)")
        else:
            # Pure-audio offline (e.g. audiomcq / UltraChat): the whole waveform is one audio
            # segment behind a single placeholder, matching official chat() for audio-only input
            # (the processor chunks it into 30s Whisper blocks under config.audio_chunk_length).
            # Preserve independently supplied assistant reference and question tracks.
            audio_segments = []
            if use_audio:
                for media_path in self._media_paths(feature):
                    _, waveform, _ = _load_media(
                        [media_path], int(self.data_args.minicpmo_max_video_seconds), use_vision=False
                    )
                    if waveform is not None:
                        audio_segments.append(waveform)
                        media_parts.append("(<audio>./</audio>)")
            use_audio = bool(audio_segments)
        media_prefix = "".join(media_parts)

        # Preserve the original multi-turn dialogue. Media placeholders are inserted once in the
        # first user turn and shared by all following turns. All assistant turns, including history,
        # are supervised; user/system turns remain context-only.
        messages = list(prompt) + list(response)
        rendered = [f"<|im_start|>system\n{system}<|im_end|>\n"]
        media_inserted = False
        assistant_turns = 0
        for message_idx, message in enumerate(messages):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported offline chat role: {role!r}.")
            content = message.get("content") or ""
            content = content.replace("<video>", "").replace("<image>", "").replace("<audio>", "").strip()
            if role == "user" and not media_inserted:
                content = media_prefix + content
                media_inserted = True
            if role == "assistant":
                assistant_turns += 1
                if (
                    message_idx == len(messages) - 1
                    and getattr(self.finetuning_args, "minicpmo_offline_instruct_mode", True)
                    and not content.lstrip().startswith("<think>")
                ):
                    # Official MiniCPM-o instruct chat template places an empty thinking block
                    # before the final assistant answer when no reasoning_content is provided.
                    # OmniEvalKit additionally prefills <|tts_bos|> (use_tts_template=True);
                    # include it so the training context matches the eval prompt. It is
                    # excluded from supervision below (prefill token, never generated).
                    content = f"<think>\n\n</think>\n\n<|tts_bos|>{content}"
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if media_prefix and not media_inserted:
            raise ValueError("Offline multimodal sample has no user turn for media placeholders.")
        if assistant_turns == 0:
            raise ValueError("Offline sample has no assistant turn to supervise.")

        text = "".join(rendered)
        audios = [audio_segments] if use_audio else [[]]
        images = [sampled_frames] if use_vision else [[]]
        # All timestamp-aligned audio segments belong to the first user message. Official chat()
        # uses audio_parts to merge them for efficient Whisper processing while preserving one
        # audio placeholder/bound after each sampled frame in the LLM token sequence.
        audio_parts = ([list(range(len(audio_segments)))] if not use_vision else [[0] * len(audio_segments)]) if use_audio else None
        out = self.processor(
            [text], images, audios, audio_parts=audio_parts, max_slice_nums=1, use_image_id=False,
            chunk_input=True, return_tensors="pt", sampling_rate=16000,
        )
        ids = out["input_ids"][0].long()
        assistant_marker = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_marker = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not assistant_marker or not im_end_marker:
            raise ValueError("Cannot tokenize offline assistant/im_end markers.")
        # The eval chat template prefills an empty think block (enable_thinking=False) and,
        # under OmniEvalKit's use_tts_template=True, a leading <|tts_bos|> ahead of the
        # answer. Those are prompt context, never generation targets: keep them unsupervised.
        prefill_prefix = self.tokenizer.encode("<think>\n\n</think>\n\n<|tts_bos|>", add_special_tokens=False)

        labels = torch.full_like(ids, IGNORE_INDEX)
        weights = torch.zeros(len(ids), dtype=torch.float)
        found_turns = 0
        pos = 0
        while pos <= len(ids) - len(assistant_marker):
            if ids[pos : pos + len(assistant_marker)].tolist() != assistant_marker:
                pos += 1
                continue
            target_start = pos + len(assistant_marker)
            if prefill_prefix and ids[target_start : target_start + len(prefill_prefix)].tolist() == prefill_prefix:
                # labels/weights stay IGNORE/0 for the prefilled think-block + tts_bos prefix.
                target_start += len(prefill_prefix)
            target_end = -1
            for end_pos in range(target_start, len(ids) - len(im_end_marker) + 1):
                if ids[end_pos : end_pos + len(im_end_marker)].tolist() == im_end_marker:
                    target_end = end_pos + len(im_end_marker)
                    break
            if target_end < 0:
                raise ValueError("Cannot locate <|im_end|> for an offline assistant turn.")
            labels[target_start:target_end] = ids[target_start:target_end]
            weights[target_start:target_end] = self.finetuning_args.minicpmo_offline_text_weight
            found_turns += 1
            pos = target_end

        if found_turns != assistant_turns:
            raise ValueError(
                f"Offline assistant turn mismatch after multimodal expansion: rendered={assistant_turns}, found={found_turns}."
            )
        if len(ids) > self.data_args.minicpmo_max_expanded_length:
            raise ValueError(f"Expanded offline length {len(ids)} exceeds limit.")
        return {
            "input_ids": ids,
            "labels": labels,
            "loss_weights": weights,
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "pixel_values": out.get("pixel_values", [[]])[0],
            "tgt_sizes": out.get("tgt_sizes", [torch.empty((0, 2), dtype=torch.int32)])[0],
            "image_bound": out.get("image_bound", [torch.empty((0, 2), dtype=torch.long)])[0],
            "audio_bounds": out.get("audio_bounds", [torch.empty((0, 2), dtype=torch.long)])[0],
            "runtime_audio": [
                (out["audio_features"][idx], int(length), False)
                for idx, length in enumerate(out["audio_feature_lens"][0])
            ] if use_audio else [],
            # Randomly tag a fraction of offline samples for full-bidirectional audio
            # attention (chunk_length=-1, OmniEvalKit protocol); the model patch maps
            # the mode to the chunk_length. Covers both eval protocols in one run.
            "audio_mode": (
                "offline_full"
                if use_audio
                and self.data_args.minicpmo_offline_full_attention_prob > 0
                and random.random() < self.data_args.minicpmo_offline_full_attention_prob
                else "offline"
            ),
            "training_mode_id": 0,
        }

    def _build(self, feature: dict[str, Any]) -> dict[str, Any]:
        return self._build_duplex(feature) if feature["training_mode"] == "duplex_streaming" else self._build_offline(feature)

    def _dummy_sample(self) -> dict[str, Any]:
        # Keep at least two tokens: causal LM loss indexes the position before a target.
        # A one-token all-ignore placeholder made `supervised_positions + 1` out of bounds.
        token_id = self.tokenizer.bos_token_id or self.tokenizer.pad_token_id
        ids = torch.tensor([token_id, token_id], dtype=torch.long)
        return {
            "input_ids": ids, "labels": torch.tensor([IGNORE_INDEX, IGNORE_INDEX]),
            "loss_weights": torch.zeros(2),
            "attention_mask": torch.ones(2, dtype=torch.bool), "pixel_values": [],
            "tgt_sizes": torch.empty((0, 2), dtype=torch.int32),
            "image_bound": torch.empty((0, 2), dtype=torch.long),
            "audio_bounds": torch.empty((0, 2), dtype=torch.long), "runtime_audio": [],
            "audio_mode": "none", "training_mode_id": 0,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        built = []
        errors = []
        for feature in features:
            try:
                built.append(self._build(feature))
            except Exception as exc:
                if os.environ.get("HEARTBEAT_STRICT_DATA", "0") == "1":
                    raise RuntimeError(f"Invalid HEARTBEAT sample {feature.get('dataset_name')}: {exc}") from exc
                errors.append(f"{feature.get('dataset_name')}: {exc}")
                logger.warning_rank0(f"Replaced corrupt/overlength MiniCPM-o sample with zero-loss placeholder: {errors[-1]}")
                # Preserve local batch cardinality across DDP ranks.
                built.append(self._dummy_sample())
        if os.environ.get("MINICPMO_DEBUG_BATCH", "0") == "1":
            rank = os.environ.get("RANK", "0")
            for feature, item in zip(features, built):
                video = (feature.get("raw_videos") or ["<none>"])[0]
                logger.warning(
                    "MiniCPM-o batch rank=%s mode=%s duration=%s seq_len=%s images=%s audio_bounds=%s "
                    "audio_blocks=%s supervised=%s video=%s",
                    rank,
                    feature.get("training_mode"),
                    feature.get("duration"),
                    len(item["input_ids"]),
                    len(item["image_bound"]),
                    len(item["audio_bounds"]),
                    len(item["runtime_audio"]),
                    int((item["labels"] != IGNORE_INDEX).sum()),
                    video,
                )

        input_ids = _pad_2d([x["input_ids"] for x in built], self.tokenizer.pad_token_id)
        labels = _pad_2d([x["labels"] for x in built], IGNORE_INDEX)
        weights = _pad_2d([x["loss_weights"] for x in built], 0.0)
        mask = _pad_2d([x["attention_mask"] for x in built], 0).bool()
        data = {
            "input_ids": input_ids, "labels": labels, "loss_weights": weights, "attention_mask": mask,
            "position_ids": torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1),
            "pixel_values": [x["pixel_values"] for x in built], "tgt_sizes": [x["tgt_sizes"] for x in built],
            "image_bound": [x["image_bound"] for x in built], "audio_bounds": [x["audio_bounds"] for x in built],
            "runtime_audio": [x["runtime_audio"] for x in built], "audio_mode": [x["audio_mode"] for x in built],
            "training_mode_ids": torch.tensor([x["training_mode_id"] for x in built], dtype=torch.long),
        }
        return {"data": data, "input_ids": input_ids, "labels": labels, "loss_weights": weights,
                "training_mode_ids": data["training_mode_ids"]}
