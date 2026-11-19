# Copyright 2026 HEARTBEAT contributors.
# Licensed under the Apache License, Version 2.0.
"""Bridge the released Venus text-output duplex models to Transformers 4.57."""
from functools import partial

from transformers.cache_utils import DynamicCache


class _AudioCache(DynamicCache):
    def get_usable_length(self, new_seq_length, layer_idx=0):
        # Venus uses an unbounded Whisper cache and resets it at the position limit.
        return self.get_seq_length(layer_idx)


def _audio_encoder_forward(forward, *args, **kwargs):
    if kwargs.get("use_cache") and kwargs.get("past_key_values") is None:
        kwargs["past_key_values"] = _AudioCache()
    return forward(*args, **kwargs)


def _audio_attention_forward(forward, *args, past_key_value=None, **kwargs):
    hidden, weights = forward(*args, past_key_values=past_key_value, **kwargs)
    # Whisper updates the supplied cache in place. Venus expects the self-attention
    # portion as the third result and indexes it to size its next chunk's mask.
    cache = past_key_value.self_attention_cache if past_key_value is not None else None
    return hidden, weights, cache


def _decoder_cache_length(decoder):
    return 0 if decoder.cache is None else decoder.cache.get_seq_length()


def adapt_venus_duplex(duplex):
    """Adapt only this duplex instance, including its encoder and Qwen stream decoder."""
    encoder = duplex.model.apm
    encoder.forward = partial(_audio_encoder_forward, encoder.forward)
    for layer in encoder.layers:
        attention = layer.self_attn
        attention.forward = partial(_audio_attention_forward, attention.forward)
    duplex.decoder.get_cache_length = partial(_decoder_cache_length, duplex.decoder)
