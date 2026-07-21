"""Benchmark MuseLM inference throughput and report the active-vs-total
parameter ratio — the number that makes Mixture-of-Experts worthwhile.

Usage:
    python scripts/benchmark_inference.py --config configs/moe-small.json
    python scripts/benchmark_inference.py --config configs/tiny.json --tokens 128
"""

from __future__ import annotations

import argparse
import time

import torch

from muselm.config import MuseLMConfig, TrainConfig, resolve_device
from muselm.model import MuseLM


def benchmark(cfg: MuseLMConfig, n_tokens: int, warmup: int, device: str) -> dict:
    model = MuseLM(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8), device=device)

    with torch.no_grad():
        model.generate(idx, max_new_tokens=warmup, temperature=0.8, top_k=20)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        model.generate(idx, max_new_tokens=n_tokens, temperature=0.8, top_k=20)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

    total = model.num_parameters()
    active = model.num_parameters(exclude_embeddings=True)
    return {
        "device": device,
        "is_moe": cfg.is_moe,
        "total_params_M": round(total / 1e6, 2),
        "non_embedding_params_M": round(active / 1e6, 2),
        "tokens": n_tokens,
        "seconds": round(elapsed, 3),
        "tokens_per_sec": round(n_tokens / elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cfg = TrainConfig.from_json(args.config).model
    result = benchmark(cfg, args.tokens, args.warmup, device)

    print(f"{'metric':<26} value")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k:<26} {v}")


if __name__ == "__main__":
    main()
