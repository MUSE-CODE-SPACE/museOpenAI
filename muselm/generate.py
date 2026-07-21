"""Load a trained checkpoint and generate text from a prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from muselm.config import MuseLMConfig, resolve_device
from muselm.model import MuseLM
from muselm.tokenizer import ByteBPETokenizer


class Generator:
    def __init__(
        self,
        model: MuseLM,
        tokenizer: ByteBPETokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "auto",
    ) -> Generator:
        device = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = MuseLMConfig.from_dict(ckpt["model_config"])
        model = MuseLM(cfg)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
        return cls(model, tokenizer, device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)
        ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt, allow_special=False)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token=self.tokenizer.eos_id,
        )
        generated = out[0, len(ids):].tolist()
        if self.tokenizer.eos_id in generated:
            generated = generated[: generated.index(self.tokenizer.eos_id)]
        return self.tokenizer.decode(generated)
