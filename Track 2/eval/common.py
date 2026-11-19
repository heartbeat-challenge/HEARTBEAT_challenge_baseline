# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Small I/O helpers shared by evaluation tools (no model dependencies)."""
import hashlib
import json
import math
import os
import re
from pathlib import Path


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
             "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
             "seventeenth", "eighteenth", "nineteenth", "twentieth")
_ORDINAL_PREFIX = re.compile(
    r"^(?:(?:that(?:'s| is| marks)|this is|a|the)\s+)?(?:the\s+)?(" + "|".join(_ORDINALS) + r")\b"
)
_EXPLICIT_COUNT = re.compile(
    r"\b(?:i (?:can )?count\s+|(?:count|total)\s*(?::|is|to|of|=)\s*|there (?:are|is)\s+)"
    r"(\d+)\b(?![.,]\d|[:/-]\d)"
)


def reference_count(answer, *, running=False):
    """Read an explicit count or ordinal from a counting-task reference."""
    text = answer.strip().casefold()
    if re.fullmatch(r"[0-9]+[.!]?", text):
        return int(text.rstrip(".!"))
    values = {int(n) for n in _EXPLICIT_COUNT.findall(text)}
    if running:
        ordinal = _ORDINAL_PREFIX.match(text)
        if ordinal:
            values.add(_ORDINALS.index(ordinal[1]) + 1)
    # Distinct quantities in one answer require semantic interpretation.
    if len(values) == 1:
        value = next(iter(values))
        if all(int(n) == value for n in re.findall(r"\b[0-9]+\b", text)):
            return value
    return None


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def finite(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric timestamp")
    return float(value)


def case_file(root, sample_id):
    return Path(root) / "cases" / (hashlib.sha256(sample_id.encode()).hexdigest() + ".json")


def episodes_from_chunks(chunks, clock="stream"):
    """A response is one speak episode, preserving the time of each text prefix."""
    episodes, active = [], None
    for chunk in chunks:
        timestamp = chunk["input_end"] if clock == "stream" else chunk["wall_time"]
        text = chunk["text"].strip() if not chunk["is_listen"] else ""
        if text:
            if active is None:
                active = {"time": timestamp, "prefixes": []}
                episodes.append(active)
            previous = active["prefixes"][-1]["content"] if active["prefixes"] else ""
            active["prefixes"].append({"time": timestamp, "content": (previous + " " + text).strip()})
        if chunk["is_listen"] or chunk["end_of_turn"]:
            active = None
    return episodes


def validate_prediction(row, reference, clock="stream"):
    """Reject malformed logs; an explicitly failed case may contain a partial prefix."""
    if row["id"] != reference["id"] or row.get("status") not in ("complete", "error"):
        raise ValueError("Invalid prediction ID/status")
    duration = finite(reference["duration"])
    if abs(finite(row["duration"]) - duration) > 1e-6:
        raise ValueError("Prediction/reference duration mismatch")
    chunks = row["chunks"]
    if clock == "wall" and row.get("realtime") is not True:
        raise ValueError("Wall-clock scores require a paced --realtime run")
    previous_wall, previous_end = 0., 0.
    for i, chunk in enumerate(chunks):
        if chunk["index"] != i or type(chunk["index"]) is not int:
            raise ValueError("Chunks must be contiguous and ordered")
        start, end = finite(chunk["input_start"]), finite(chunk["input_end"])
        if abs(start - previous_end) > 1e-6 or not start < end <= duration + 1e-6:
            raise ValueError("Invalid input chunk intervals")
        previous_end = end
        wall = finite(chunk["wall_time"])
        if wall < previous_wall or (row.get("realtime") and wall + 1e-6 < chunk["input_end"]):
            raise ValueError("Invalid wall-clock output time")
        previous_wall = wall
        if not isinstance(chunk["text"], str) or type(chunk["is_listen"]) is not bool or type(chunk["end_of_turn"]) is not bool:
            raise ValueError("Invalid chunk output types")
        if chunk["is_listen"] and chunk["text"].strip():
            raise ValueError("A listening chunk cannot contain response text")
    if row["status"] == "complete" and (not chunks or abs(previous_end - duration) > 1e-6):
        raise ValueError("Chunk coverage mismatch: listening chunks must also be recorded")
    return episodes_from_chunks(chunks, clock)
