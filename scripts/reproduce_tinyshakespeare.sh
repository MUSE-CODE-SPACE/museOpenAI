#!/usr/bin/env bash
# Reproduce the released Tiny-Shakespeare checkpoint end-to-end.
# Usage: bash scripts/reproduce_tinyshakespeare.sh
set -euo pipefail

CORPUS="data/tinyshakespeare_input.txt"
DATA_DIR="data/tinyshakespeare"
CONFIG="configs/tiny.json"

echo "==> [1/4] Downloading corpus"
mkdir -p "$DATA_DIR"
if [ ! -f "$CORPUS" ]; then
  curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o "$CORPUS"
fi
wc -c "$CORPUS"

echo "==> [2/4] Training tokenizer (vocab 2048)"
muselm tokenizer-train --input "$CORPUS" --output "$DATA_DIR/tokenizer.json" --vocab-size 2048

echo "==> [3/4] Preparing dataset"
muselm prepare --input "$CORPUS" --tokenizer "$DATA_DIR/tokenizer.json" --output "$DATA_DIR"

echo "==> [4/4] Training model ($CONFIG)"
muselm train --config "$CONFIG"

echo "==> Sample:"
muselm generate \
  --checkpoint checkpoints/tiny/best.pt \
  --tokenizer "$DATA_DIR/tokenizer.json" \
  --prompt "To be, or not to be" \
  --max-new-tokens 200 --seed 0
