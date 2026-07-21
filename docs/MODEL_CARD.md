# Model Card — MuseLM Tiny (Shakespeare)

A small demonstration checkpoint released with the repository so anyone can
generate text immediately, without training. It exists to prove the stack works
end-to-end and to give newcomers a working artifact to poke at — **not** to be a
capable general-purpose assistant.

## Model details

| | |
|---|---|
| Architecture | Decoder-only transformer (RMSNorm, RoPE, GQA, QK-norm, SwiGLU) |
| Parameters | ~5M (dense; 4.4M non-embedding) |
| Layers / dim / heads | 6 / 256 / 8 (4 KV heads) |
| Context length | 256 tokens |
| Vocabulary | 2048 (byte-level BPE) |
| Regularization | dropout 0.2, weight decay 0.1 |
| Best validation | loss 3.87 / **perplexity 47.8** (step 750 of 1500) |
| Config | [`configs/tiny.json`](../configs/tiny.json) |
| License | Apache 2.0 |

## Training data

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
— ~1.1 MB of the collected works of Shakespeare, a public-domain corpus. Split
90/10 train/validation. No other data was used.

## Training procedure

Trained from scratch with the loop in `muselm/train.py`: AdamW, cosine learning-
rate schedule with warmup, gradient clipping. See `configs/tiny.json` for exact
hyperparameters and `checkpoints/release/summary.json` for the realized
loss/perplexity curve. Fully reproducible via
[`scripts/reproduce_tinyshakespeare.sh`](../scripts/reproduce_tinyshakespeare.sh).

## Intended use

- **Educational**: read the code, watch a real training run converge, inspect
  generations.
- **A smoke test**: verify an install works before scaling to a real corpus.

## Limitations & biases

- Trained only on archaic English theatrical text; output is Shakespeare-flavored
  and frequently nonsensical as modern prose.
- ~10M parameters and a 256-token context — far below any coherent-reasoning
  threshold. It models surface style, not facts.
- No safety tuning, alignment, or content filtering. Do not deploy it as an
  assistant. It reflects the biases of its 16th–17th-century source text.

## How to use

```bash
muselm generate \
  --checkpoint checkpoints/release/best.pt \
  --tokenizer checkpoints/release/tokenizer.json \
  --prompt "To be, or not to be" --max-new-tokens 200
```

## Scaling up

The identical code trains a real model on a real corpus. Point the config at a
larger dataset, increase `dim` / `n_layers` / `max_seq_len`, enable MoE
(`n_experts > 0`), and run on a GPU. See [ROADMAP.md](ROADMAP.md) for the path to
scale.
