"""Copy the best trained checkpoint + tokenizer + summary into
checkpoints/release/ so they ship with the repo.

Usage: python scripts/package_release.py --run checkpoints/tiny \\
           --tokenizer data/tinyshakespeare/tokenizer.json
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/tiny")
    ap.add_argument("--tokenizer", default="data/tinyshakespeare/tokenizer.json")
    ap.add_argument("--out", default="checkpoints/release")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in ["best.pt", "summary.json"]:
        src = run / name
        if src.exists():
            shutil.copy2(src, out / name)
            print(f"copied {src} -> {out / name}")
    shutil.copy2(args.tokenizer, out / "tokenizer.json")
    print(f"copied {args.tokenizer} -> {out / 'tokenizer.json'}")
    print("release packaged.")


if __name__ == "__main__":
    main()
