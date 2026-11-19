# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


def _apply_stream_focal_multiplier(
    token_loss: torch.Tensor,
    labels: torch.Tensor,
    state_token_ids: tuple[int, ...],
    gamma: float,
    state_alphas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Streamo focal reweighting to state-token CE while leaving all other tokens unchanged."""
    focal_multiplier = torch.ones_like(token_loss)
    for state_idx, token_id in enumerate(state_token_ids):
        state_mask = labels.eq(token_id)
        if state_mask.any():
            # exp(-CE) is the full-vocabulary probability assigned to the target token.
            state_factor = (1.0 - torch.exp(-token_loss[state_mask])).pow(gamma)
            if state_alphas is not None:
                state_factor = state_factor * state_alphas[state_idx]
            focal_multiplier[state_mask] = state_factor

    return token_loss * focal_multiplier


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        optimizer = super().create_optimizer(*args, **kwargs)
        # Transformers creates separate decay/no-decay groups even when one is empty.  Pure
        # LLM-LoRA tuning can have no bias/normalization parameters, leaving an empty no-decay
        # group. DeepSpeed ZeRO removes that group after the scheduler has captured two base LRs,
        # so the first scheduler.step() fails with "zip() argument 2 is longer than argument 1".
        # Remove empty groups before both scheduler construction and DeepSpeed initialization.
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            nonempty_groups = [group for group in optimizer.param_groups if len(group.get("params", [])) > 0]
            if len(nonempty_groups) != len(optimizer.param_groups):
                logger.info_rank0(
                    "Removed %d empty optimizer parameter group(s) before scheduler/DeepSpeed initialization.",
                    len(optimizer.param_groups) - len(nonempty_groups),
                )
                optimizer.param_groups[:] = nonempty_groups
        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if "loss_weights" in inputs:
            labels = inputs["labels"]
            loss_weights = inputs["loss_weights"].to(dtype=torch.float32)
            mode_ids = inputs.get("training_mode_ids")
            # MiniCPM-o long-video sequences are mostly multimodal context. Computing the full
            # [batch, sequence, vocab] tensor wastes several GB although only assistant positions
            # are supervised. Qwen3 natively accepts tensor-valued `logits_to_keep`; use the union
            # of causal positions whose next token is supervised. Hidden states and causal attention
            # are still computed for the complete sequence, so this is exactly loss-equivalent.
            if labels.size(1) < 2:
                raise ValueError("MiniCPM-o causal training requires sequence length >= 2.")
            causal_valid = labels[:, 1:].ne(IGNORE_INDEX)
            supervised_positions = causal_valid.any(dim=0).nonzero(as_tuple=False).flatten()
            if supervised_positions.numel() == 0:
                # Keep causal position 0 so every rank executes the same trainable graph.
                # The selected next-token label remains IGNORE_INDEX and its weight is zero.
                supervised_positions = torch.zeros(1, dtype=torch.long, device=labels.device)
            if int(supervised_positions.max()) + 1 >= labels.size(1):
                raise ValueError(
                    f"MiniCPM-o supervised position exceeds sequence length: "
                    f"max={int(supervised_positions.max())}, seq_len={labels.size(1)}."
                )
            outputs = model(
                data=inputs["data"],
                use_cache=False,
                return_dict=True,
                logits_to_keep=supervised_positions,
                return_sparse_hidden_states=True,
            )
            sparse_hidden_states = outputs.logits.contiguous()
            shift_labels = labels[:, supervised_positions + 1].contiguous()
            shift_weights = loss_weights[:, supervised_positions + 1].to(dtype=torch.float32).contiguous()
            valid = shift_labels.ne(IGNORE_INDEX)
            shift_weights = shift_weights * valid

            stream_focal_token_ids: tuple[int, ...] = ()
            stream_focal_alphas = None
            if self.finetuning_args.minicpmo_use_stream_focal_loss:
                tokenizer = self.processing_class
                stream_focal_token_ids = tuple(
                    tokenizer.convert_tokens_to_ids(token) for token in ("<|listen|>", "<|speak|>")
                )
                invalid_ids = {None, getattr(tokenizer, "unk_token_id", None)}
                if any(token_id in invalid_ids for token_id in stream_focal_token_ids):
                    raise ValueError(
                        "Cannot enable MiniCPM-o Streamo focal loss because <|listen|>/<|speak|> "
                        "are not valid tokenizer tokens."
                    )
                if self.finetuning_args.minicpmo_stream_focal_auto_alpha:
                    # Match Streamo: alpha_c = total_state_count / (num_present_states * count_c).
                    # Counts are computed over the complete local batch before lm_head chunking.
                    state_counts = torch.stack(
                        [shift_labels.eq(token_id).sum() for token_id in stream_focal_token_ids]
                    ).to(dtype=torch.float32)
                    present = state_counts.gt(0)
                    if present.any():
                        total_count = state_counts.sum()
                        num_present = present.sum().to(dtype=torch.float32)
                        stream_focal_alphas = torch.where(
                            present,
                            total_count / (state_counts.clamp_min(1.0) * num_present),
                            torch.ones_like(state_counts),
                        )

            base_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
            lm_head = base_model.get_output_embeddings()
            chunk_size = max(1, self.finetuning_args.minicpmo_lm_head_chunk_size)
            weighted_loss_sum = torch.zeros(shift_labels.size(0), device=shift_labels.device, dtype=torch.float32)

            def chunked_weighted_ce(hidden_chunk, label_chunk, weight_chunk):
                # Accelerate can return FP32 hidden states outside the forward autocast
                # context, while a frozen output head remains BF16.
                logits_chunk = lm_head(hidden_chunk.to(dtype=lm_head.weight.dtype))
                loss_chunk = F.cross_entropy(
                    logits_chunk.reshape(-1, logits_chunk.size(-1)).float(),
                    label_chunk.reshape(-1),
                    reduction="none",
                    ignore_index=IGNORE_INDEX,
                ).view_as(label_chunk).float()
                if stream_focal_token_ids:
                    # Streamo applies focal reweighting only to streaming state tokens; all text
                    # and other special tokens retain their original weighted cross entropy.
                    loss_chunk = _apply_stream_focal_multiplier(
                        loss_chunk,
                        label_chunk,
                        stream_focal_token_ids,
                        self.finetuning_args.minicpmo_stream_focal_gamma,
                        stream_focal_alphas,
                    )
                return (loss_chunk * weight_chunk).sum(dim=1)

            for start in range(0, sparse_hidden_states.size(1), chunk_size):
                end = min(start + chunk_size, sparse_hidden_states.size(1))
                # Non-reentrant checkpoint drops each chunk's huge vocab logits after forward and
                # recomputes only that lm_head/CE chunk during backward. This is mathematically
                # equivalent to full CE while bounding peak memory for long offline answers.
                weighted_loss_sum = weighted_loss_sum + checkpoint(
                    chunked_weighted_ce,
                    sparse_hidden_states[:, start:end, :],
                    shift_labels[:, start:end],
                    shift_weights[:, start:end],
                    use_reentrant=False,
                )

            if self.finetuning_args.minicpmo_loss_normalization == "global_token":
                loss = weighted_loss_sum.sum() / shift_weights.sum().clamp_min(1e-8)
            else:
                sample_loss = weighted_loss_sum / shift_weights.sum(dim=1).clamp_min(1e-8)
                if mode_ids is not None:
                    mode_weights = torch.where(
                        mode_ids.to(sample_loss.device).eq(1),
                        torch.as_tensor(self.finetuning_args.minicpmo_duplex_sample_weight, device=sample_loss.device),
                        torch.as_tensor(self.finetuning_args.minicpmo_offline_sample_weight, device=sample_loss.device),
                    )
                    sample_loss = sample_loss * mode_weights
                loss = sample_loss.mean()
            return (loss, outputs) if kwargs.get("return_outputs", False) else loss

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
