# Roadmap

MuseLM's goal is to be the most *readable* complete LLM stack while steadily
growing toward production capability. Milestones below are grouped by theme;
each is a natural PR (or series of them). Contributions are welcome — pick one
and open an issue to claim it.

## v0.1 — Foundation ✅ (shipped)

- [x] Byte-level BPE tokenizer (train / encode / decode / serialize)
- [x] Transformer with RMSNorm, RoPE, GQA, QK-norm, SwiGLU
- [x] Fine-grained MoE with shared experts + load-balancing loss
- [x] Training loop: cosine schedule, warmup, grad accum/clip, checkpoints
- [x] KV-cached generation with temperature / top-k / top-p / repetition penalty
- [x] CLI + library API
- [x] 19 unit + integration tests, 3-version CI matrix
- [x] Released Tiny-Shakespeare checkpoint

## v0.2 — Training at scale

- [ ] **Streaming / sharded datasets** — train on corpora larger than RAM (webtext, code).
- [ ] **Mixed precision (bf16/fp16) + gradient scaler** for CUDA.
- [ ] **Distributed data parallel (DDP)** multi-GPU training.
- [ ] **Resume-from-checkpoint** (optimizer + step state) and Weights & Biases logging.
- [ ] **Expert parallelism** for MoE so experts shard across devices.

## v0.3 — Inference & serving

- [ ] **FlashAttention-2** path and a fused RoPE kernel.
- [ ] **Batched / continuous-batching generation** server (OpenAI-compatible `/v1/completions`).
- [ ] **Weight quantization** (int8 / int4) for CPU + edge inference.
- [ ] **KV-cache paging** for long contexts.
- [ ] **Speculative decoding** with a small draft model.

## v0.4 — Alignment & post-training

- [ ] **Supervised fine-tuning (SFT)** with chat templates and loss masking.
- [ ] **Preference optimization** — DPO first, then a minimal PPO/GRPO loop.
- [ ] **Instruction dataset tooling** and eval harness (MMLU-lite, HellaSwag-lite).
- [ ] **LoRA / QLoRA** parameter-efficient fine-tuning adapters.

## v0.5 — Ecosystem

- [ ] **Hugging Face Hub** export/import (`safetensors` + config round-trip).
- [ ] **GGUF export** for llama.cpp interop.
- [ ] **Model card automation** and reproducible training manifests.
- [ ] **Multi-node** training recipes and cost/throughput benchmarks.

## Design principles (won't change)

1. **Readable over clever.** If a feature can't be understood from the source
   in one sitting, it needs a doc or it doesn't merge.
2. **No hidden magic.** Plain PyTorch. Optional accelerated paths must have a
   pure-PyTorch fallback.
3. **Tested behavior.** New capabilities ship with tests. Correctness
   invariants (like cache/no-cache equivalence) are non-negotiable.
4. **Runs on a laptop.** Every feature must be demonstrable at tiny scale
   without a GPU cluster.
