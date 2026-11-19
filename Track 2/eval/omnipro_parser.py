# Adapted from RuixiangZhao/OmniPro (MIT). See the repository root NOTICE.
"""Unified parser for streaming-mode model outputs.

Any non-empty model output counts as an emit. The parser extracts
task-specific structured fields from the full text:

  - IEA, SCA, EN, SSI (text): description = full text
  - ETG: description + position (9-region)
  - SOC, CC, DC: integer count
  - RSM: state name

Empty string / None / pure whitespace -> STANDBY (not an emit).

Online prompts no longer require any keyword prefix (TRIGGER:/UPDATE:/...),
so the parser treats the entire non-empty output as the payload.
"""

import re
from typing import Any, Dict, Optional

_REGIONS = {
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
}


def is_standby(raw: Optional[str]) -> bool:
    """Return True if the model chose not to emit this tick."""
    if raw is None:
        return True
    return len(raw.strip()) == 0


def _extract_integer(text: str) -> Optional[int]:
    """Pull first integer from a string. Strips timestamps first."""
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)
    m = re.search(r"-?\d+", cleaned)
    return int(m.group(0)) if m else None


def _extract_region(text: str) -> Optional[str]:
    """Pull a 9-region label out of text.

    1. Prefer explicit `Position: <region>` anchor.
    2. Fallback: longest-match scan.
    """
    t = text.lower()
    m = re.search(r"position\s*:\s*([\w\-]+)", t)
    if m:
        cand = m.group(1).strip()
        if cand in _REGIONS:
            return cand
    for r in sorted(_REGIONS, key=len, reverse=True):
        if r in t:
            return r
    return None


def parse_streaming_output(raw: str, task: str) -> Dict[str, Any]:
    """Parse one raw emit into a structured dict.

    Any non-empty output is a valid emit. The full text is treated as the
    payload; task-specific structured fields (position / count / state)
    are extracted from it.
    """
    out: Dict[str, Any] = {
        "raw": raw,
        "payload": None,
        "valid": False,
    }
    if is_standby(raw):
        return out

    payload = raw.strip()
    out["payload"] = payload
    out["valid"] = True

    # Task-specific structured fields
    if task == "explicit_target_grounding":
        out["position"] = _extract_region(raw)
        desc = re.sub(r"position\s*:\s*[\w\-]+", "", payload,
                       flags=re.IGNORECASE).rstrip(". ").strip()
        out["description"] = desc if desc else payload

    elif task in ("snapshot_counting", "cumulative_counting",
                  "dedup_counting"):
        out["count"] = _extract_integer(payload)
        if out["count"] is None:
            out["valid"] = False  # can't score without a number

    elif task == "realtime_state_monitor":
        out["state"] = payload.strip().strip("'\"").lower()

    else:
        # text tasks: IEA, SCA, EN, SSI
        out["description"] = payload

    return out
