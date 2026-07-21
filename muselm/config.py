"""Configuration dataclasses for model architecture and training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MuseLMConfig:
    """Architecture hyperparameters for a MuseLM model.

    Setting ``n_experts=0`` yields a dense SwiGLU feed-forward network;
    any positive value enables Mixture-of-Experts routing with
    ``n_active_experts`` experts selected per token plus
    ``n_shared_experts`` always-on shared experts (DeepSeek/Kimi style).
    """

    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    max_seq_len: int = 512
    ffn_hidden_dim: int = 704
    # Mixture-of-Experts
    n_experts: int = 0
    n_active_experts: int = 2
    n_shared_experts: int = 1
    expert_hidden_dim: int = 176
    moe_aux_loss_coef: float = 0.01
    # Misc
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    qk_norm: bool = True

    def __post_init__(self) -> None:
        if self.n_heads % max(self.n_kv_heads, 1) != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_experts > 0 and self.n_active_experts > self.n_experts:
            raise ValueError(
                f"n_active_experts ({self.n_active_experts}) cannot exceed n_experts ({self.n_experts})"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MuseLMConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainConfig:
    """Hyperparameters and bookkeeping for a training run."""

    # Data
    data_dir: str = "data/shakespeare"
    tokenizer_path: str = "data/shakespeare/tokenizer.json"
    # Optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 5000
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Evaluation / checkpointing
    eval_interval: int = 250
    eval_steps: int = 40
    log_interval: int = 20
    checkpoint_dir: str = "checkpoints/run"
    keep_last_checkpoints: int = 3
    # Runtime
    device: str = "auto"
    seed: int = 1337
    compile: bool = False
    model: MuseLMConfig = field(default_factory=MuseLMConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        d = dict(d)
        model_d = d.pop("model", {})
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        cfg.model = MuseLMConfig.from_dict(model_d)
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def resolve_device(device: str = "auto") -> str:
    """Pick the best available torch device."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
