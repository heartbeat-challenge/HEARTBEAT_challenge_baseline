"""Compatibility for the unquantized Venus LoRA recipe."""


def configure_lora():
    from peft.tuners.lora import torchao as lora_torchao

    if lora_torchao.is_torchao_available():
        try:
            from torchao.quantization import (
                LinearActivationQuantizedTensor,  # noqa: F401
            )
        except ImportError:
            # PEFT checks this optional dispatcher even for ordinary linear layers.
            lora_torchao.is_torchao_available = lambda: False
