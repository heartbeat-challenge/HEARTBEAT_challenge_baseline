import math

RESPONSE_START_DEADLINE_S = 2.0
CUT_STOP_DEADLINE_S = 2.0


def point(r, a):
    z = {
        "sample_id": r["sample_id"],
        "model": r["model"],
        "type": r["type"],
        "language": r["language"],
        "triggered": r["trigger"] is not None,
        "error": r["error"],
        "behavior": None,
        "eligible": None,
    }
    if a.get("status") != "complete":
        return dict(z, uncertain="alignment_not_complete")
    words = sorted(
        [
            dict(w, turn_id=t["turn_id"])
            for t in a["turns"]
            if t["alignment"]["valid"]
            for w in t["alignment"]["words"]
        ],
        key=lambda w: w["start_s"],
    )
    req = a["request"]
    ev = a.get("event")
    typ = r["type"]
    g = r["row"]["events"][0]
    lo = r["request_start_s"]
    if not req["valid"]:
        return dict(z, uncertain="request_alignment_invalid")
    end = req["words"][-1]["end_s"]
    idx = min(len(r["steps"]) - 1, max(0, math.ceil(end) - 1))
    ref = r["steps"][idx]
    if typ == "response_start":
        z["eligible"] = True
        # First output after the target request; cross-boundary speech is an early-start failure.
        early = any(w["start_s"] < end < w["end_s"] for w in words)
        found = next((w for w in words if w["start_s"] >= end - 0.08), None)
        z["behavior"] = bool(
            found and not early and found["start_s"] <= end + RESPONSE_START_DEADLINE_S
        )
        if found:
            z["media_asr_latency_s"] = found["start_s"] - end
            z["playback_after_last_input_packet_s"] = (
                found["wall_start_s"] - ref["sent_wall_s"]
            )
            z["playback_lexical_estimate_s"] = z[
                "playback_after_last_input_packet_s"
            ] + (ref["input_end_s"] - end)
            audio_step = next(
                (s for s in r["steps"] if s["generated_samples"] > 0), None
            )
            if audio_step:
                z["block_latency_s"] = audio_step["input_end_s"] - end
                z["block_after_last_input_packet_s"] = (
                    audio_step["input_end_s"] - ref["input_end_s"]
                )
                z["playback_minus_block_s"] = (
                    z["playback_lexical_estimate_s"] - z["block_latency_s"]
                )
        return z
    if typ == "pause":
        span = g["pause_span"]
        ps, pe = [x + lo for x in span]
        z.update(
            eligible=True,
            behavior=not any(w["start_s"] < pe and w["end_s"] > ps for w in words),
        )
        return z
    if not r["trigger"]:
        return dict(z, eligible=False, uncertain="not_triggered")
    if not ev or not ev["valid"]:
        return dict(z, eligible=False, uncertain="event_alignment_invalid")
    tr = r["trigger"]
    old = [w for w in words if w["turn_id"] == tr["turn_id"]]
    # Verify old actual output still spans the actual delivery of the first audible packet.
    ei = min(len(r["steps"]) - 1, math.floor(ev["words"][0]["start_s"]))
    delivery = r["steps"][ei]["sent_wall_s"]
    eligible = bool(
        old
        and min(w["wall_start_s"] for w in old) < delivery
        and max(w["wall_end_s"] for w in old) > delivery
    )
    z.update(eligible=eligible)
    if not eligible:
        return dict(z, uncertain="no_lexical_overlap_at_audible_packet_delivery")
    if typ == "cut":
        stop = max(w["wall_end_s"] for w in old)
        latency = stop - delivery
        z["stop_after_event_packet_s"] = latency
        z["behavior"] = 0 <= latency <= CUT_STOP_DEADLINE_S
        st = next((s for s in r["steps"][ei:] if s["is_listen"]), None)
        if st:
            z["block_stop_latency_s"] = (
                st["input_end_s"] - r["steps"][ei]["input_end_s"]
            )
    else:
        z["uncertain"] = "requires_context_completion_judge"
    return z
