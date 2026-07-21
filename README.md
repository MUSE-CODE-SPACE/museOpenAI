<div align="center">

# 🎼 MuseLM

**A fully open, from-scratch language-model stack you can read in an afternoon and train on a laptop.**

Byte-level BPE tokenizer · Mixture-of-Experts transformer · training loop · KV-cached inference — all in plain PyTorch, no hidden dependencies.

[![CI](https://github.com/MUSE-CODE-SPACE/museOpenAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MUSE-CODE-SPACE/museOpenAI/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

**[→ museOpenAI landing page](https://muse-code-space.github.io/museOpenAI/)**

</div>

---

## Why MuseLM?

Most "open" LLMs ship weights but keep the interesting parts — the tokenizer trainer, the MoE routing, the data pipeline — behind a wall of framework code. MuseLM is the opposite: **every component that turns raw text into a talking model is here, small enough to read, and correct enough to trust.** The same architectural choices that power frontier open-weight models (DeepSeek-V3, Kimi K2, Llama 3) are implemented in a few hundred lines each:

- **Byte-level BPE tokenizer** trained from scratch — round-trips *any* Unicode, no `<unk>`.
- **Modern transformer**: RMSNorm, RoPE, grouped-query attention, QK-norm, SwiGLU.
- **Fine-grained Mixture-of-Experts** with shared experts + top-k routing + load-balancing loss — flip one config field to go from dense to sparse.
- **Real training loop**: cosine schedule, warmup, grad accumulation, grad clipping, rotating checkpoints, eval/perplexity.
- **Fast inference**: KV-cache incremental decoding, temperature / top-k / top-p / repetition-penalty sampling.
- **Tested**: 19 unit + integration tests, including a proof that KV-cache decoding is numerically identical to a full forward pass. CI on Python 3.9 / 3.11 / 3.12.

It trains a coherent model on [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) on a MacBook (Apple Silicon / MPS) in minutes, and the exact same code scales to a real corpus and a real GPU.

### See it work right now

A ~5M-parameter model trained with this repo ships in [`checkpoints/release/`](checkpoints/release/) — no training required:

```bash
muselm generate \
  --checkpoint checkpoints/release/best.pt \
  --tokenizer checkpoints/release/tokenizer.json \
  --prompt "KING HENRY:" --max-new-tokens 80
```

```
KING HENRY:
Fear not, my lord, I am the king's crown.

WARWICK:
O, if you were not this greater man
Upon your hands and honourable state!

QUEEN ELIZABETH:
You, good Gloucester, and Cobray Srook Buckingham,
Which often thou canst perfect the king ...
```

That checkpoint reaches **validation perplexity 47.8** in 1,500 steps (~8 min on an M-series Mac). Full run details in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Quickstart

```bash
git clone https://github.com/MUSE-CODE-SPACE/museOpenAI.git
cd museOpenAI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Train the whole thing end-to-end (tokenizer → data → model → text):

```bash
# 0. get a corpus
curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o corpus.txt

# 1. train a byte-BPE tokenizer
muselm tokenizer-train --input corpus.txt --output data/tinyshakespeare/tokenizer.json --vocab-size 2048

# 2. tokenize the corpus into train/val bins
muselm prepare --input corpus.txt --tokenizer data/tinyshakespeare/tokenizer.json --output data/tinyshakespeare

# 3. train (auto-selects CUDA / MPS / CPU)
muselm train --config configs/tiny.json

# 4. generate
muselm generate \
  --checkpoint checkpoints/tiny/best.pt \
  --tokenizer data/tinyshakespeare/tokenizer.json \
  --prompt "To be, or not to be" --max-new-tokens 200
```

Prefer sparse? Swap in the MoE config — same corpus, same commands:

```bash
muselm train --config configs/moe-small.json
```

## Use it as a library

```python
from muselm import MuseLM, MuseLMConfig, ByteBPETokenizer

tok = ByteBPETokenizer.train(open("corpus.txt").read(), vocab_size=2048)
model = MuseLM(MuseLMConfig(vocab_size=tok.vocab_size, n_experts=8, n_active_experts=2))
print(f"{model.num_parameters()/1e6:.1f}M params, MoE={model.cfg.is_moe}")
```

## Configuration

Everything is a dataclass with sane defaults (`muselm/config.py`). Turn dense into MoE by setting `n_experts > 0`:

| Field | Dense (`tiny`) | MoE (`moe-small`) | Meaning |
|---|---|---|---|
| `dim` | 256 | 256 | model width |
| `n_layers` | 6 | 6 | transformer blocks |
| `n_heads` / `n_kv_heads` | 8 / 4 | 8 / 4 | GQA query vs. KV heads |
| `n_experts` | 0 | 8 | 0 = dense SwiGLU FFN |
| `n_active_experts` | – | 2 | experts routed per token |
| `n_shared_experts` | – | 1 | always-on experts |

Inspect any config's parameter count without training:

```bash
muselm info --config configs/moe-small.json
```

## How it works

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full walkthrough — the attention math, RoPE, the MoE router and its auxiliary loss, and how the KV cache makes generation `O(1)` per token. See **[docs/ROADMAP.md](docs/ROADMAP.md)** for where this is going (FlashAttention path, distributed training, RLHF/DPO, quantized inference).

## Project layout

```
muselm/
  config.py      # architecture + training dataclasses
  tokenizer.py   # byte-level BPE: train / encode / decode / save / load
  model.py       # RMSNorm, RoPE, GQA attention, SwiGLU, MoE, KV-cache, generate()
  data.py        # corpus -> memmapped token bins, batch sampler
  train.py       # cosine LR, warmup, grad accum/clip, eval, checkpoints
  generate.py    # load a checkpoint and sample text
  cli.py         # `muselm` command-line entrypoint
tests/           # 19 unit + integration tests
configs/         # ready-to-run dense + MoE configs
docs/            # architecture, roadmap, model card
```

## Contributing

PRs welcome — this is meant to be hacked on. Read **[CONTRIBUTING.md](CONTRIBUTING.md)** and check the [good first issues](https://github.com/MUSE-CODE-SPACE/museOpenAI/labels/good%20first%20issue). Every PR runs lint + the full test matrix.

## License

[Apache 2.0](LICENSE) — free for commercial and research use.

## Acknowledgements

Standing on the shoulders of the open-source LLM community: the byte-BPE approach follows GPT-2 / tiktoken, the architecture follows Llama and DeepSeek-V3, and the training-loop ergonomics are inspired by nanoGPT. MuseLM re-implements these ideas independently so you can learn from and build on every line.
