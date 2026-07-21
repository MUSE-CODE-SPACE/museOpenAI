# Training Guide

How to train a MuseLM model on your own data, from a text file to samples.

## 1. Prepare a corpus

Any UTF-8 text file works — the tokenizer is byte-level, so code, prose, and
mixed languages are all fine. Bigger is better; a few MB is a reasonable minimum
for the tiny config.

```bash
cat book1.txt book2.txt > corpus.txt
```

## 2. Train a tokenizer

```bash
muselm tokenizer-train --input corpus.txt --output data/mine/tokenizer.json --vocab-size 8192
```

Larger `vocab-size` = shorter sequences (faster) but a bigger embedding table.
2k–32k is the usual range. Training stops early if the corpus runs out of
repeated byte pairs, so tiny corpora may yield a smaller vocab than requested.

## 3. Tokenize into train/val bins

```bash
muselm prepare --input corpus.txt --tokenizer data/mine/tokenizer.json --output data/mine
```

This writes `train.bin`, `val.bin`, and `meta.json` (memory-mapped uint16/uint32
token streams). `--val-fraction` controls the split (default 0.1).

## 4. Write a config

Copy `configs/tiny.json` and edit. Key knobs:

| Knob | Effect |
|---|---|
| `dim`, `n_layers` | Model capacity (and cost). |
| `max_seq_len` | Context window. Must fit the batch in memory. |
| `batch_size`, `grad_accum_steps` | Effective batch = `batch_size × grad_accum_steps`. |
| `max_steps`, `warmup_steps` | Length of training and LR ramp. |
| `learning_rate`, `min_learning_rate` | Cosine schedule endpoints. |
| `n_experts` (>0) | Switch to Mixture-of-Experts. |

Check the parameter count before committing to a run:

```bash
muselm info --config configs/mine.json
```

## 5. Train

```bash
muselm train --config configs/mine.json          # auto CPU/CUDA/MPS
muselm train --config configs/mine.json --device cuda --max-steps 50000
```

The loop prints `loss`, `lr`, `grad`, and `ms/step`, evaluates every
`eval_interval` steps (reporting validation loss and perplexity), and writes
rotating checkpoints plus `best.pt`, `final.pt`, and `summary.json`.

## 6. Generate

```bash
muselm generate \
  --checkpoint checkpoints/mine/best.pt \
  --tokenizer data/mine/tokenizer.json \
  --prompt "Once upon a time" \
  --max-new-tokens 300 --temperature 0.8 --top-k 40 --top-p 0.95
```

## Tips

- **Loss not dropping?** Lower the LR, or increase warmup. Watch `grad` — if it
  spikes, your LR is too high.
- **Out of memory?** Reduce `batch_size` and raise `grad_accum_steps` to keep the
  effective batch, or shrink `max_seq_len`.
- **MoE tips.** Start with `n_experts=8`, `n_active_experts=2`,
  `n_shared_experts=1`. If experts collapse (one dominates), raise
  `moe_aux_loss_coef`.
- **Overfitting** (val loss rising while train falls) means it's time for more
  data or fewer steps. `best.pt` always holds the lowest-val checkpoint.
