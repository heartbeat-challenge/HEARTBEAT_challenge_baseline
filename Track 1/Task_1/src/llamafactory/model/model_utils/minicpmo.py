# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from types import MethodType
from typing import Any

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...extras import logging

logger = logging.get_logger(__name__)


def patch_minicpmo_stage1_forward(model: torch.nn.Module) -> None:
    """Add mixed offline/duplex audio embedding support without entering the TTS path."""
    if getattr(model.config, "model_type", None) not in ("minicpmo", "realtime_venus_omni") or getattr(model, "_lf_stage1_patched", False):
        return

    # MiniCPM-o 4.5 remote Whisper code still calls the pre-4.57 Cache API.
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length

    # Transformers 4.57 WhisperAttention returns (hidden, attn) without cache when
    # use_cache=False, while MiniCPM-o's custom encoder layer always unpacks three values.
    for layer in getattr(getattr(model, "apm", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_lf_return_compat", False):
            continue
        original_forward = attn.forward
        def compatible_forward(*args, _original=original_forward, **kwargs):
            result = _original(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                return result + (kwargs.get("past_key_value"),)
            return result
        attn.forward = compatible_forward
        attn._lf_return_compat = True

    def _get_runtime_omni_embedding(self, data: dict[str, Any], input_embeddings: torch.Tensor) -> torch.Tensor:
        runtime_audio = data.get("runtime_audio")
        audio_modes = data.get("audio_mode")
        if runtime_audio is None or audio_modes is None:
            return self.get_omni_embedding(data, input_embeddings, chunk_length=self.config.audio_chunk_length)

        output = input_embeddings.clone()
        for sample_idx, (items, mode) in enumerate(zip(runtime_audio, audio_modes)):
            if not items:
                continue
            bounds = data["audio_bounds"][sample_idx]
            sample_embeds = []
            if mode == "chunk_causal":
                encoded_blocks = []
                for mel, mel_len, _ in items:
                    batch = {
                        "audio_features": mel.unsqueeze(0),
                        "audio_feature_lens": [torch.tensor([mel_len], device=mel.device)],
                    }
                    encoded_blocks.append(self.get_audio_embedding(batch, chunk_length=1, dummy=False)[0][0])
                embeds = torch.cat(encoded_blocks, dim=0)
                offset = 0
                for bound in bounds:
                    length = int(bound[1] - bound[0])
                    sample_embeds.append(embeds[offset : offset + length])
                    offset += length
                if offset != embeds.shape[0]:
                    raise ValueError(f"Unused chunk-causal audio embeddings: used {offset}, total {embeds.shape[0]}.")
            elif mode in ("offline", "offline_full"):
                # "offline" uses the checkpoint default (1s causal chunks, = official chat()).
                # "offline_full" uses full bidirectional attention within each 30s block
                # (chunk_length=-1, the OmniEvalKit eval protocol).
                chunk_length = -1 if mode == "offline_full" else self.config.audio_chunk_length
                encoded_blocks = []
                for mel, mel_len, _ in items:
                    batch = {
                        "audio_features": mel.unsqueeze(0),
                        "audio_feature_lens": [torch.tensor([mel_len], device=mel.device)],
                    }
                    encoded_blocks.extend(self.get_audio_embedding(
                        batch, chunk_length=chunk_length, dummy=False
                    )[0])
                flat_embeds = torch.cat(encoded_blocks, dim=0)
                offset = 0
                for bound in bounds:
                    length = int(bound[1] - bound[0])
                    sample_embeds.append(flat_embeds[offset : offset + length])
                    offset += length
                if offset != flat_embeds.shape[0]:
                    raise ValueError(f"Unused offline audio embeddings: used {offset}, total {flat_embeds.shape[0]}.")
            else:
                continue

            if len(sample_embeds) != len(bounds):
                raise ValueError(
                    f"MiniCPM-o audio embedding/bound mismatch: {len(sample_embeds)} embeddings vs {len(bounds)} bounds."
                )
            for emb, bound in zip(sample_embeds, bounds):
                start, end = int(bound[0]), int(bound[1])
                if emb.shape[0] != end - start:
                    raise ValueError(
                        f"MiniCPM-o audio shape mismatch: embedding {emb.shape[0]} vs placeholder {end - start}."
                    )
                output[sample_idx, start:end] = emb.to(device=output.device, dtype=output.dtype)
        return output

    def _forward(self, data, **kwargs):
        # PEFT's causal-LM wrapper injects standard inputs in addition to our `data` argument.
        # MiniCPM-o stage-1 uses `data` as the single source of truth.
        for key in ("input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"):
            kwargs.pop(key, None)
        # Trainer-facing helper columns are consumed by the custom loss and must not reach the LLM.
        data = dict(data)
        data.pop("loss_weights", None)
        data.pop("training_mode_ids", None)
        vllm_embedding, _ = self.get_vllm_embedding(data)
        vllm_embedding = self._get_runtime_omni_embedding(data, vllm_embedding)
        if self.training:
            # Mixed-modality batches can bypass a trainable projector (or replace a corrupt
            # sample with a zero-loss placeholder) on only some distributed ranks. DeepSpeed
            # ZeRO-2 requires the same gradient-reduction graph everywhere. Attach a zero-valued
            # dependency for every trainable parameter so skipped modules receive an explicit
            # zero gradient without changing embeddings, logits, or loss.
            trainable_anchor = None
            # Only modality-dependent projectors can be skipped by a sample. LLM/LoRA parameters
            # are always used and must not be anchored: activation checkpoint recomputation plus
            # a second out-of-checkpoint dependency makes ZeRO-2 reduce them twice.
            optional_trainable_modules = (
                getattr(self, "resampler", None),
                getattr(self, "audio_avg_pooler", None),
                getattr(self, "audio_projection_layer", None),
            )
            for module in optional_trainable_modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if parameter.requires_grad and parameter.numel() > 0:
                        term = parameter.reshape(-1)[0] * 0.0
                        trainable_anchor = term if trainable_anchor is None else trainable_anchor + term
            if trainable_anchor is not None:
                vllm_embedding = vllm_embedding + trainable_anchor.to(vllm_embedding.dtype)
        position_ids = data["position_ids"].long()
        if kwargs.pop("return_sparse_hidden_states", False):
            sparse_positions = kwargs.pop("logits_to_keep")
            outputs = self.llm.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=data.get("attention_mask"),
                inputs_embeds=vllm_embedding,
                **kwargs,
            )
            sparse_hidden_states = outputs.last_hidden_state[:, sparse_positions, :]
            # The custom trainer interprets logits as pre-lm_head hidden states in this mode.
            # CausalLMOutputWithPast keeps Transformers evaluation/prediction plumbing compatible.
            return CausalLMOutputWithPast(
                logits=sparse_hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=data.get("attention_mask"),
            inputs_embeds=vllm_embedding,
            **kwargs,
        )

    def _safe_get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_embedding(data)
        if hasattr(self.llm.config, "scale_emb"):
            embeddings = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            embeddings = self.llm.model.embed_tokens(data["input_ids"])
        embeddings = embeddings.clone()
        vision_hidden_states = [
            item.to(dtype=embeddings.dtype) if isinstance(item, torch.Tensor) else item
            for item in vision_hidden_states
        ]
        for sample_idx in range(len(data["input_ids"])):
            current = vision_hidden_states[sample_idx]
            bounds = data["image_bound"][sample_idx]
            if len(current) > 0 and len(bounds) > 0:
                indices = torch.cat(
                    [torch.arange(int(r[0]), int(r[1]), device=embeddings.device) for r in bounds], dim=0
                )
                values = current.reshape(-1, current.shape[-1]).to(device=embeddings.device, dtype=embeddings.dtype)
                if values.shape[0] != indices.shape[0]:
                    raise ValueError(
                        f"MiniCPM-o vision shape mismatch: {values.shape[0]} embeddings vs {indices.shape[0]} positions."
                    )
                embeddings[sample_idx] = embeddings[sample_idx].scatter(
                    0, indices[:, None].expand(-1, embeddings.shape[-1]), values
                )
            elif len(current) > 0 and self.training:
                embeddings = embeddings + current[0].mean() * 0
        return embeddings, vision_hidden_states

    model.get_vllm_embedding = MethodType(_safe_get_vllm_embedding, model)
    model._get_runtime_omni_embedding = MethodType(_get_runtime_omni_embedding, model)
    model.forward = MethodType(_forward, model)
    model._lf_stage1_patched = True
    logger.info_rank0("Patched MiniCPM-o stage-1 mixed offline/duplex forward.")
