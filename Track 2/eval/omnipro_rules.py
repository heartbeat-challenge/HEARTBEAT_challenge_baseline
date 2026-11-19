# Adapted from RuixiangZhao/OmniPro (MIT). See the repository root NOTICE.
"""Official task rules and greedy matcher; episode construction is handled separately."""
from typing import Dict, List, Optional, Tuple
from .omnipro_parser import parse_streaming_output

TASK_CONTENT_KIND = {
    # Alert tasks — content is not evaluated; any time-matched emit is correct.
    "instant_event_alert":         "time_only",
    "semantic_condition_alert":    "time_only",
    # Structured tasks — rule-based exact matching.
    "explicit_target_grounding":   "position",
    "snapshot_counting":      "count",
    "cumulative_counting":         "count",
    "dedup_counting":              "count",
    "realtime_state_monitor":      "state",
    # Free-text narration tasks — need GPT judge (fallback: non-empty check).
    "event_narration":             "gpt_judge",
    "sequential_step_instruction": "gpt_judge",
}

def _mmss_to_sec(s):
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return 0.0
    parts = str(s).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(s)

def _gt_trigger_sec(gt: Dict) -> float:
    if "trigger_time_sec" in gt:
        return float(gt["trigger_time_sec"])
    return _mmss_to_sec(gt.get("trigger_time", "00:00"))

def match_emits_to_gt(
    emits: List[Dict],
    gts: List[Dict],
    tolerance: float = 3.0,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy 1-to-1 matching: repeatedly pick the globally closest (emit, gt)
    pair with |Δt| <= tolerance, remove both, continue.

    Returns:
        matches: list of (emit_idx, gt_idx, abs_dt)
        unmatched_emits: list of emit_idx (false positives)
        unmatched_gts:   list of gt_idx   (false negatives)
    """
    emit_times = [float(e["t_sec"]) for e in emits]
    gt_times = [_gt_trigger_sec(g) for g in gts]

    pairs = []
    for i, et in enumerate(emit_times):
        for j, gt in enumerate(gt_times):
            dt = abs(et - gt)
            if dt <= tolerance:
                pairs.append((dt, i, j))
    pairs.sort()

    used_e, used_g = set(), set()
    matches = []
    for dt, i, j in pairs:
        if i in used_e or j in used_g:
            continue
        matches.append((i, j, dt))
        used_e.add(i)
        used_g.add(j)

    unmatched_emits = [i for i in range(len(emits)) if i not in used_e]
    unmatched_gts = [j for j in range(len(gts)) if j not in used_g]
    return matches, unmatched_emits, unmatched_gts

def _get_parsed(emit: Dict, task: str) -> Dict:
    """Get parsed fields from an emit, parsing on-the-fly if needed."""
    if "parsed" in emit and emit["parsed"]:
        return emit["parsed"]
    # New format: only {t_sec, raw} — parse now
    return parse_streaming_output(emit.get("raw", ""), task)

def _score_content(task: str, emit: Dict, gt: Dict,
                   gts_all: List[Dict]) -> Optional[bool]:
    """Return True/False for TP if scorable, None if unscorable (e.g. GPT
    judge needed but not provided).

    Strategy per task kind:
      - time_only  : content is ignored → always True when time matches
      - position   : 9-region exact match
      - count      : integer exact match (against paired GT)
      - state      : state name exact match (to state_to)
      - gpt_judge  : requires external judge; returns None here (caller
                     can re-score with a GPT judge later)
    """
    parsed = _get_parsed(emit, task)
    kind = TASK_CONTENT_KIND.get(task, "gpt_judge")

    # time_only: content is not part of the metric at all.
    if kind == "time_only":
        return True

    # gpt_judge tasks: cannot auto-judge; leave for caller to fill in.
    if kind == "gpt_judge":
        return None

    if not parsed.get("valid", False):
        return False

    if kind == "position":
        pp = (parsed.get("position") or "").lower().strip()
        gp = (gt.get("position") or "").lower().strip()
        if not pp or not gp:
            return False
        return pp == gp

    if kind == "count":
        pc = parsed.get("count")
        gc = gt.get("count")
        if pc is None or gc is None:
            return False
        return int(pc) == int(gc)

    if kind == "state":
        ps = (parsed.get("state") or "").lower().strip()
        gs_to = (gt.get("state_to") or gt.get("state") or "").lower().strip()
        if not ps or not gs_to:
            return False
        return ps == gs_to

    return False
