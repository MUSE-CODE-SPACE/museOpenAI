# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-21

### Added

- **Byte-level BPE tokenizer** (`muselm.tokenizer`) — train from scratch,
  encode/decode with full Unicode round-trip, special tokens, JSON
  serialization.
- **Transformer model** (`muselm.model`) — RMSNorm, RoPE, grouped-query
  attention with optional QK-norm, SwiGLU feed-forward, GPT-2 scaled init.
- **Mixture-of-Experts** — fine-grained top-k routing, shared experts, and a
  Switch-style load-balancing auxiliary loss; enabled by `n_experts > 0`.
- **KV-cache generation** — incremental decoding with temperature, top-k,
  top-p, and repetition-penalty sampling; sliding-window past `max_seq_len`.
- **Training loop** (`muselm.train`) — cosine LR schedule with warmup, gradient
  accumulation and clipping, periodic evaluation/perplexity, rotating
  checkpoints, best-checkpoint tracking, run summary.
- **Data pipeline** (`muselm.data`) — corpus → memory-mapped token bins with a
  random contiguous-window batch sampler.
- **CLI** (`muselm`) — `tokenizer-train`, `prepare`, `train`, `generate`,
  `info`.
- **Configs** — ready-to-run dense (`tiny`) and MoE (`moe-small`) presets.
- **Released checkpoint** — ~10M-param Tiny-Shakespeare model with tokenizer.
- **Tests** — 19 unit + integration tests, including cache/no-cache numerical
  equivalence and a full train→generate pipeline.
- **CI** — lint + test matrix on Python 3.9 / 3.11 / 3.12, plus an end-to-end
  smoke-train job.
- **Docs** — README, architecture deep-dive, roadmap, model card, contributing
  guide, code of conduct.

[0.1.0]: https://github.com/MUSE-CODE-SPACE/museOpenAI/releases/tag/v0.1.0
