# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""FFmpeg decoding with explicit frame presentation timestamps; no PyAV."""
import bisect
import json
import math
import subprocess
import tempfile
from pathlib import Path


def choose_frames(timestamps, duration):
    if not timestamps or any(not math.isfinite(t) for t in timestamps) or timestamps != sorted(timestamps):
        raise ValueError("Missing, non-finite or unordered video timestamps")
    chosen = []
    for i in range(math.ceil(duration)):
        # Prefer a frame at/before chunk start (the training cadence). If the
        # first frame starts later, it is usable only once that chunk has seen it.
        index = bisect.bisect_right(timestamps, float(i) + 1e-9) - 1
        if index < 0:
            index = 0 if timestamps[0] <= min(i + 1, duration) else None
        chosen.append(index)
    return chosen


class Media:
    def __init__(self, path, task, duration):
        import numpy as np
        self.duration, self.task = duration, task
        self.temp = tempfile.TemporaryDirectory(prefix="heartbeat_eval_frames_")
        self.frames = {}
        self.chosen = []
        self.timestamps = []
        self.audio = None
        try:
            info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                             "-of", "json", str(path)], capture_output=True, text=True,
                                            check=True, timeout=120).stdout)
            if abs(float(info["format"]["duration"]) - duration) > .25:
                raise ValueError("Declared media duration differs from the file")
            has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
            if task == "audio" and not has_audio:
                raise ValueError("Audio task has no audio stream")
            if has_audio:
                raw = subprocess.run(["ffmpeg", "-v", "error", "-copyts", "-start_at_zero", "-i", str(path),
                                      "-map", "0:a:0", "-vn", "-af", "aresample=16000:async=1:first_pts=0",
                                      "-ac", "1", "-ar", "16000", "-t", str(duration), "-f", "f32le", "pipe:1"],
                                     capture_output=True, check=True, timeout=1200).stdout
                audio = np.frombuffer(raw, dtype="<f4").copy()
                if not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("Empty or non-finite decoded audio")
                if task == "audio" and abs(len(audio) / 16000 - duration) > .25:
                    raise ValueError("Audio duration mismatch")
                length = round(duration * 16000)
                self.audio = np.pad(audio[:length], (0, max(0, length - len(audio))))
            if task == "video":
                frames = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)],
                    capture_output=True, text=True, check=True, timeout=1200).stdout)["frames"]
                offset = float(info["format"].get("start_time", 0))
                self.timestamps = [float(f["best_effort_timestamp_time"]) - offset for f in frames]
                self.chosen = choose_frames(self.timestamps, duration)
                indices = sorted({i for i in self.chosen if i is not None})
                if not indices:
                    raise ValueError("No visible frames in the clip interval")
                select = "+".join(f"eq(n\\,{i})" for i in indices)
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
                                "-vf", "select=" + select, "-vsync", "0", "-q:v", "2",
                                str(Path(self.temp.name) / "%06d.jpg")], capture_output=True, check=True, timeout=1200)
                files = sorted(Path(self.temp.name).glob("*.jpg"))
                if len(files) != len(indices):
                    raise ValueError("Decoded frame count differs from selected PTS indices")
                self.frames = dict(zip(indices, files))
        except BaseException:
            self.close()
            raise

    def chunk(self, i):
        import numpy as np
        from PIL import Image
        audio, frames, timestamp = None, None, None
        if self.audio is not None:
            audio = self.audio[i * 16000:(i + 1) * 16000]
            audio = np.pad(audio, (0, 16000 - len(audio)))
        if self.task == "video" and self.chosen[i] is not None:
            index = self.chosen[i]
            timestamp = self.timestamps[index]
            if timestamp > min(i + 1, self.duration) + 1e-6:
                raise ValueError("A future video frame would be exposed")
            with Image.open(self.frames[index]) as image:
                frames = [image.convert("RGB").copy()]
        return audio, frames, timestamp

    def close(self):
        self.temp.cleanup()
