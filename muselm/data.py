"""Dataset utilities: tokenize a corpus into a memory-mapped token array
and sample contiguous training batches from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from muselm.tokenizer import ByteBPETokenizer


def prepare_dataset(
    text: str,
    tokenizer: ByteBPETokenizer,
    out_dir: str | Path,
    val_fraction: float = 0.1,
) -> dict[str, int]:
    """Tokenize ``text`` and write ``train.bin`` / ``val.bin`` (uint16/uint32).

    Returns a small stats dict with token counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tokenizer.encode(text, allow_special=False)
    dtype = np.uint16 if tokenizer.vocab_size <= 2**16 else np.uint32

    n_val = int(len(ids) * val_fraction)
    split = len(ids) - n_val
    train_ids = np.array(ids[:split], dtype=dtype)
    val_ids = np.array(ids[split:], dtype=dtype)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta = {"dtype": np.dtype(dtype).name, "vocab_size": tokenizer.vocab_size}
    (out_dir / "meta.json").write_text(__import__("json").dumps(meta))
    return {"train_tokens": len(train_ids), "val_tokens": len(val_ids)}


class TokenDataset:
    """Memory-mapped token stream with random contiguous-window sampling."""

    def __init__(self, bin_path: str | Path, dtype: str = "uint16") -> None:
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self, batch_size: int, seq_len: int, device: str, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - seq_len - 1
        if max_start <= 0:
            raise ValueError("dataset too small for the requested sequence length")
        ix = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix]
        )
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        return x.to(device), y.to(device)
