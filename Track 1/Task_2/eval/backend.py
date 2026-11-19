import hashlib
import os
import sys

from .common import FROZEN, MODELS, REF

os.environ.update(
    USE_TF="0",
    TRANSFORMERS_NO_TF="1",
    OMP_NUM_THREADS="4",
    TOKENIZERS_PARALLELISM="false",
)
sys.path.insert(0, FROZEN)
import numpy as np
import soundfile as sf
import torch
import torchaudio
import transformers.utils.import_utils as hf

hf._detectron2_available = False
hf._kernels_available = False
hf.is_kernels_available = lambda: False
import transformers.integrations.deepspeed as ds

ds.is_deepspeed_available = lambda: False


def sf_load(path, frame_offset=0, num_frames=-1, channels_first=True, **kw):
    w, sr = sf.read(
        path, start=frame_offset, frames=num_frames, dtype="float32", always_2d=True
    )
    return torch.from_numpy(np.ascontiguousarray(w.T if channels_first else w)), sr


torchaudio.load = sf_load
import diffusers.utils.import_utils as diff_imports

diff_imports._xformers_available = False
from core.processors.unified import UnifiedProcessor
from core.schemas.duplex import DuplexConfig


class Backend:
    def __init__(self, name):
        from runtime_compat import configure_lora

        if os.environ.get("HEARTBEAT_ADAPTER"):
            configure_lora()
        self.name = name
        self.delegate_disabled_ids = []
        torch.set_num_threads(4)
        torch.manual_seed(20261022)
        self.config = DuplexConfig(
            force_listen_count=0,
            length_penalty=1.0,
            listen_prob_scale=1.0,
            chunk_ms=1000,
            sample_rate=16000,
        )
        self.processor = UnifiedProcessor(
            model_path=MODELS[name],
            ref_audio_path=REF,
            lora_path=os.environ.get("HEARTBEAT_ADAPTER") or None,
            duplex_config=self.config,
            preload_both_tts=False,
            attn_implementation="sdpa",
            compile=False,
        )
        self.view = self.processor.set_duplex_mode()
        self.end_protocol = None
        if os.environ.get("HEARTBEAT_END_PROTOCOL", "legacy") == "close_eos":
            from .end_protocol import install_end_protocol

            self.end_protocol = install_end_protocol(self.processor.model.duplex)

    def prepare(self, sample_id, history):
        if self.end_protocol:
            self.end_protocol["reset"]()
            self.end_protocol["closed_units"] = 0
        seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        self.system_prompt = (
            "Streaming Omni Conversation.\nPrevious conversation:\n"
            + "\n".join(str(h["role"]) + ": " + h["text"] for h in history)
        )
        with torch.inference_mode():
            self.view.prepare(system_prompt_text=self.system_prompt)
        if self.name == "venus":
            d = self.processor.model.duplex
            ids = [d.delegate_start_id, d.delegate_end_id]
            ids = [i for i in ids if isinstance(i, int) and i >= 0]
            for i in ids:
                if i not in d.decoder.forbidden_token_ids:
                    d.decoder.forbidden_token_ids.append(i)
            self.delegate_disabled_ids = ids
            print("DELEGATE_HARD_MASK", ids, flush=True)

    def step(self, pcm, force_listen=False):
        with torch.inference_mode():
            if self.end_protocol:
                self.end_protocol["reset"]()
            status = self.view.prefill(audio_waveform=pcm)
            if status.get("success") is False:
                raise RuntimeError(str(status))
            if self.name == "venus":
                assert set(self.delegate_disabled_ids) <= set(
                    self.processor.model.duplex.decoder.forbidden_token_ids
                )
            result = self.view.generate(force_listen=force_listen)
            if self.name == "venus" and result.is_delegate:
                raise RuntimeError("Delegate hard mask violated")
            return result

    def finalize(self):
        with torch.inference_mode():
            self.view.finalize()
