# MuseLM Architecture

This document explains every design choice in the model, top to bottom. The
goal is that after reading it you could re-derive `muselm/model.py` from
memory.

## 1. Overview

MuseLM is a decoder-only transformer — the same family as GPT, Llama, and
DeepSeek. Input token ids are embedded, passed through `n_layers` identical
blocks, normalized, and projected back to vocabulary logits. Training is plain
next-token prediction with cross-entropy loss.

```
tokens ─► embedding ─► [ block × n_layers ] ─► RMSNorm ─► lm_head ─► logits
                          │
                          ├─ RMSNorm ─► attention ─► + residual
                          └─ RMSNorm ─► FFN/MoE   ─► + residual
```

Each block is **pre-norm**: normalization happens *before* each sub-layer and
the sub-layer output is added back to the residual stream. Pre-norm is what
makes deep transformers trainable without careful warmup gymnastics.

## 2. Normalization — RMSNorm

We use RMSNorm instead of LayerNorm. It drops the mean-subtraction and bias,
normalizing only by the root-mean-square of the activations:

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) · g
```

Fewer operations, no centering, and empirically as good as LayerNorm for
transformers (used by Llama, Mistral, DeepSeek). We compute it in float32 for
numerical stability regardless of the model's dtype, then cast back.

## 3. Positional information — RoPE

There are no learned position embeddings. Instead we use **Rotary Positional
Embeddings**: each query/key vector is rotated by an angle proportional to its
absolute position. Because a dot product between two rotated vectors depends
only on their *relative* offset, attention becomes naturally relative-position
aware, and the model generalizes to positions it can index within
`max_seq_len`.

The cos/sin tables are precomputed once (`precompute_rope`) and sliced by
`start_pos` during generation so cached tokens keep their original rotation.

## 4. Attention — Grouped-Query Attention (GQA)

Standard multi-head attention keeps one key and value head per query head. GQA
keeps *fewer* KV heads (`n_kv_heads < n_heads`) and shares each across a group
of query heads. This shrinks the KV cache — the memory that dominates
long-context inference — by `n_heads / n_kv_heads×` with negligible quality
loss.

```
q: (B, n_heads,    T, head_dim)
k: (B, n_kv_heads, T, head_dim)  ─ repeat_kv ─►  (B, n_heads, T, head_dim)
v: (B, n_kv_heads, T, head_dim)  ─ repeat_kv ─►  (B, n_heads, T, head_dim)
```

Two refinements over the textbook version:

- **QK-norm**: queries and keys are RMSNorm'd (per head) before the dot
  product. This bounds attention-logit magnitude and stabilizes training at
  higher learning rates — a trick from more recent frontier models.
- **SDPA**: the actual softmax-attention is delegated to PyTorch's
  `scaled_dot_product_attention`, which dispatches to a fused/flash kernel when
  available.

### Causal masking with a KV cache

During training, attention is a square causal matrix (`is_causal=True`). During
incremental decoding the query block is short (often length 1) but attends to a
long cache, so we build an explicit rectangular lower-triangular mask. The unit
test `test_kv_cache_matches_full_forward` asserts that cached decoding produces
logits identical (to 1e-4) to a full forward pass — the single most important
correctness guarantee in the codebase.

## 5. Feed-forward — SwiGLU

The dense FFN is SwiGLU:

```
FFN(x) = W_down( silu(W_gate·x) ⊙ (W_up·x) )
```

The gated activation consistently beats a plain ReLU/GELU MLP at equal
parameter count, which is why every modern LLM uses some GLU variant.

## 6. Mixture-of-Experts

Setting `n_experts > 0` replaces the dense FFN with a fine-grained MoE layer,
the mechanism behind the "huge total, small active" parameter counts of
DeepSeek-V3 and Kimi K2.

**Routing.** A linear router scores all experts per token; the top-`k` are
selected and their outputs combined with softmax-renormalized gates:

```
logits = router(x)                      # (N_tokens, n_experts)
gates, idx = topk(softmax(logits), k)   # keep k experts per token
y = Σ_j gates_j · expert_{idx_j}(x)
```

**Shared experts.** `n_shared_experts` experts run for *every* token
(DeepSeek-style). They capture common patterns so the routed experts can
specialize, which stabilizes training.

**Load balancing.** Left alone, routers collapse — a few experts win every
token and the rest die. We add a Switch-Transformer auxiliary loss that
penalizes imbalance between the fraction of tokens routed to each expert and
the router's mean probability mass on it:

```
aux = n_experts · Σ_e (fraction_e · mean_prob_e)
```

It is scaled by `moe_aux_loss_coef` and added to the training loss. Each MoE
block stores its `aux_loss` every forward pass; the model averages them.

**Why it matters.** A MoE layer with 8 experts and top-2 routing has ~4× the
FFN parameters of the dense model but activates only 2 of them per token — more
capacity at roughly constant compute. `muselm info` reports total vs. active
parameters for any config.

## 7. Generation

`MuseLM.generate` does autoregressive sampling with a **KV cache**: after the
prompt is processed once, each new step feeds only the single newest token and
reuses cached keys/values, making per-token cost independent of context length.

Supported decoding controls:

- `temperature` (0 = greedy/deterministic),
- `top_k` — restrict to the k most likely tokens,
- `top_p` (nucleus) — restrict to the smallest set covering probability p,
- `repetition_penalty` — down-weight already-generated tokens,
- `stop_token` — halt on EOS.

When the sequence would exceed `max_seq_len`, generation re-prefills from a
sliding window of recent tokens, so text can be produced indefinitely.

## 8. Weight initialization

Linear and embedding weights use `N(0, 0.02)`. Residual projections (`wo`,
`w_down`) are additionally scaled by `1/sqrt(2·n_layers)` — the GPT-2 trick that
keeps the residual stream's variance from growing with depth.

## 9. Tokenizer

See `muselm/tokenizer.py`. It is a byte-level BPE tokenizer: text is UTF-8
encoded (so nothing is ever out-of-vocabulary), pre-split with a GPT-style
regex, and greedily merged using pairs learned during training. Byte ids
`0–255` are always present; merges and special tokens (`<|bos|>`, `<|eos|>`,
`<|pad|>`) occupy the ids above. Training operates on a frequency-weighted
multiset of unique chunks, so it scales to large corpora.
