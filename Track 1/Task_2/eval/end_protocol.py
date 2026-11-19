"""Close the current unit immediately after turn_eos, without event labels.

The native decoder feeds EOS into KV before requesting the closing token; its
normal terminator path feeds chunk_eos + </unit> and finalizes the TTS turn.
"""

import torch


def install_end_protocol(duplex):
    if getattr(duplex, "_heartbeat_end_protocol", None):
        return duplex._heartbeat_end_protocol
    original = duplex.decoder.decode
    pending = [False]
    receipt = {"version": "turn_eos_then_chunk_eos_v1", "closed_units": 0}

    def decode(*args, **kwargs):
        if pending[0]:
            pending[0] = False
            receipt["closed_units"] += 1
            return torch.tensor(
                [duplex.chunk_eos_token_id],
                dtype=torch.long,
                device=kwargs["logits"].device,
            )
        token = original(*args, **kwargs)
        pending[0] = int(token.item()) == duplex.turn_eos_token_id
        return token

    duplex.decoder.decode = decode
    # Avoid carrying a pending terminator into the next unit at token budget.
    receipt["reset"] = lambda: pending.__setitem__(0, False)
    duplex._heartbeat_end_protocol = receipt
    return receipt
