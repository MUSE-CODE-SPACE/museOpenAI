"""Training loop with cosine LR schedule, gradient accumulation, gradient
clipping, periodic evaluation, and rotating checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from muselm.config import TrainConfig, resolve_device
from muselm.data import TokenDataset
from muselm.model import MuseLM


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to ``min_learning_rate``."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return cfg.min_learning_rate
    ratio = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


@torch.no_grad()
def evaluate(model: MuseLM, dataset: TokenDataset, cfg: TrainConfig, device: str,
             generator: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = dataset.get_batch(cfg.batch_size, cfg.model.max_seq_len, device, generator)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: MuseLM, optimizer, step: int, val_loss: float,
                    cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": cfg.to_dict(),
            "model_config": cfg.model.to_dict(),
        },
        path,
    )


def _rotate_checkpoints(ckpt_dir: Path, keep: int) -> None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep]:
        old.unlink()


def train(cfg: TrainConfig) -> dict:
    """Run a full training loop. Returns a summary dict with final metrics."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    dtype = meta["dtype"]
    train_ds = TokenDataset(data_dir / "train.bin", dtype)
    val_ds = TokenDataset(data_dir / "val.bin", dtype)

    cfg.model.vocab_size = meta["vocab_size"]
    model = MuseLM(cfg.model).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    params = model.num_parameters()
    active = model.num_parameters(exclude_embeddings=True)
    print(f"device={device}  params={params/1e6:.2f}M  vocab={cfg.model.vocab_size}")

    decay, no_decay = [], []
    for _n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    best_val = float("inf")
    history: list[dict] = []
    model.train()
    t0 = time.time()

    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_ds.get_batch(cfg.batch_size, cfg.model.max_seq_len, device, generator)
            _, loss, _ = model(x, targets=y)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = (time.time() - t0) / max(step, 1)
            print(
                f"step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} "
                f"| grad {grad_norm:.2f} | {dt*1000:.0f} ms/step", flush=True
            )

        if step > 0 and step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_ds, cfg, device, generator)
            history.append({"step": step, "val_loss": val_loss})
            print(f"  eval | step {step} | val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            save_checkpoint(model, optimizer, step, val_loss, cfg, ckpt_dir / f"step_{step}.pt")
            _rotate_checkpoints(ckpt_dir, cfg.keep_last_checkpoints)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, val_loss, cfg, ckpt_dir / "best.pt")

    final_val = evaluate(model, val_ds, cfg, device, generator)
    save_checkpoint(model, optimizer, cfg.max_steps, final_val, cfg, ckpt_dir / "final.pt")
    if final_val < best_val:
        best_val = final_val
        save_checkpoint(model, optimizer, cfg.max_steps, final_val, cfg, ckpt_dir / "best.pt")

    summary = {
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "final_ppl": math.exp(final_val),
        "params": params,
        "active_params": active,
        "steps": cfg.max_steps,
        "history": history,
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done | best_val {best_val:.4f} | ppl {math.exp(best_val):.2f}")
    return summary
