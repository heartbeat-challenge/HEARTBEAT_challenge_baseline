# Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .processor_utils import DatasetProcessor


@dataclass
class MiniCPMORuntimeDatasetProcessor(DatasetProcessor):
    """Keep aligned examples raw and build MiniCPM-o multimodal tensors lazily in the collator."""

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        outputs = defaultdict(list)
        size = len(examples["_prompt"])
        for i in range(size):
            outputs["raw_prompt"].append(examples["_prompt"][i])
            outputs["raw_response"].append(examples["_response"][i])
            outputs["raw_system"].append(examples["_system"][i])
            outputs["raw_images"].append(examples["_images"][i])
            outputs["raw_videos"].append(examples["_videos"][i])
            outputs["raw_audios"].append(examples["_audios"][i])
            outputs["training_mode"].append(examples["_training_mode"][i])
            outputs["instruction_mode"].append(examples["_instruction_mode"][i])
            outputs["use_vision"].append(examples["_use_vision"][i])
            outputs["use_audio"].append(examples["_use_audio"][i])
            outputs["dataset_name"].append(examples["_dataset_name"][i])
            outputs["duration"].append(examples["_duration"][i])
        return outputs

    def print_data_example(self, example: dict[str, Any]) -> None:
        print("training_mode:", example["training_mode"])
        print("instruction_mode:", example["instruction_mode"])
        print("system:", example["raw_system"])
        print("prompt turns:", len(example["raw_prompt"]))
        print("response turns:", len(example["raw_response"]))
