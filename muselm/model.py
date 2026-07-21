"""MuseLM: a modern decoder-only transformer with optional Mixture-of-Experts.

Architecture highlights (the same family of choices used by frontier
open-weight models such as DeepSeek-V3 and Kimi K2):

* RMSNorm pre-normalization
* Rotary positional embeddings (RoPE)
* Grouped-query attention (GQA) with optional QK normalization
* SwiGLU feed-forward networks
* Optional fine-grained MoE with top-k routing, shared experts, and a
  Switch-style load-balancing auxiliary loss
* KV-cache incremental decoding for fast generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from muselm.config import MuseLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.

    x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim // 2).
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


@dataclass
class KVCache:
    """Per-layer key/value tensors accumulated during incremental decoding."""

    k: torch.Tensor  # (B, n_kv_heads, T_cached, head_dim)
    v: torch.Tensor


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class Attention(nn.Module):
    def __init__(self, cfg: MuseLMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_cache: Optional[KVCache] = None
        if cache is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)
        if not self.training:
            new_cache = KVCache(k=k, v=v)

        k_r = repeat_kv(k, self.n_rep)
        v_r = repeat_kv(v, self.n_rep)

        # With a cache, queries attend to all cached positions plus a causal
        # mask over the new block; is_causal only applies to square attention.
        is_causal = T > 1 and (cache is None or cache.k.size(2) == 0)
        attn_mask = None
        if T > 1 and cache is not None and cache.k.size(2) > 0:
            total = k.size(2)
            attn_mask = torch.ones(T, total, dtype=torch.bool, device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=total - T)
        out = F.scaled_dot_product_attention(
            q,
            k_r,
            v_r,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    """Fine-grained Mixture-of-Experts with shared experts.

    Routing: a linear router produces per-expert logits; the top-k experts
    are selected per token and their outputs combined with softmax-renormalized
    gates. Shared experts run for every token. A Switch-Transformer style
    load-balancing loss is stored in ``self.aux_loss`` on every forward pass.
    """

    def __init__(self, cfg: MuseLMConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_active_experts
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.dim, cfg.expert_hidden_dim) for _ in range(cfg.n_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLU(cfg.dim, cfg.expert_hidden_dim) for _ in range(cfg.n_shared_experts)
        )
        self.aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)  # (N, E)
        probs = logits.softmax(dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        gates = top_p / top_p.sum(dim=-1, keepdim=True)  # (N, k)

        # Load-balancing auxiliary loss: E * sum_i(fraction_i * prob_i)
        with torch.no_grad():
            counts = torch.zeros_like(probs).scatter_(1, top_i, 1.0)
            fraction = counts.mean(dim=0)
        self.aux_loss = self.n_experts * (fraction * probs.mean(dim=0)).sum()

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            token_idx, slot_idx = (top_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](flat[token_idx])
            out.index_add_(0, token_idx, expert_out * gates[token_idx, slot_idx, None])

        for shared in self.shared_experts:
            out = out + shared(flat)
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, cfg: MuseLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn: nn.Module = MoE(cfg) if cfg.is_moe else SwiGLU(cfg.dim, cfg.ffn_hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_cache


class MuseLM(nn.Module):
    def __init__(self, cfg: MuseLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: Optional[list[KVCache]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[list[KVCache]]]:
        """Run the model.

        Returns (logits, loss, new_caches). ``loss`` includes the MoE
        auxiliary loss when targets are given and the model is MoE.
        """
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence of length {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.drop(self.tok_emb(idx))
        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, new_cache = block(x, cos, sin, cache)
            if new_cache is not None:
                new_caches.append(new_cache)
        x = self.final_norm(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            if self.cfg.is_moe:
                aux = torch.stack(
                    [b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, MoE)]
                ).mean()
                loss = loss + self.cfg.moe_aux_loss_coef * aux

        return logits, loss, (new_caches if new_caches else None)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with KV-cache incremental decoding.

        When the KV cache fills up, generation re-prefills from the most
        recent ``max_seq_len // 2`` tokens (sliding window); the returned
        tensor always contains the full prompt plus all sampled tokens.
        """
        self.eval()
        caches: Optional[list[KVCache]] = None
        pos = 0
        tokens = idx

        for _ in range(max_new_tokens):
            if caches is None:
                step_input = tokens[:, -(self.cfg.max_seq_len - 1) :]
                pos = 0
            elif pos >= self.cfg.max_seq_len:
                # Cache full: restart from a truncated window of recent tokens.
                caches = None
                step_input = tokens[:, -self.cfg.max_seq_len // 2 :]
                pos = 0
            else:
                step_input = tokens[:, -1:]
            logits, _, caches = self(step_input, caches=caches, start_pos=pos)
            pos += step_input.size(1)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(tokens.size(0)):
                    seen = tokens[b].unique()
                    penalized = logits[b, seen]
                    logits[b, seen] = torch.where(
                        penalized > 0, penalized / repetition_penalty, penalized * repetition_penalty
                    )

            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    remove = cum > top_p
                    # Keep the first token that crosses the threshold.
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    remove = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
                    logits = logits.masked_fill(remove, float("-inf"))
                probs = logits.softmax(dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_tok], dim=1)
            if stop_token is not None and (next_tok == stop_token).all():
                break

        return tokens
