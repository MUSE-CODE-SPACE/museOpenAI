"""Command-line interface for MuseLM.

Subcommands:
    muselm tokenizer-train   Train a byte-BPE tokenizer on a text file
    muselm prepare           Tokenize a corpus into train/val bins
    muselm train             Train a model from a JSON config
    muselm generate          Sample text from a checkpoint
    muselm info              Print parameter counts for a config
    muselm benchmark         Measure inference throughput for a config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from muselm.tokenizer import ByteBPETokenizer

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"training tokenizer: {len(text)} chars -> vocab {args.vocab_size}")
    tok = ByteBPETokenizer.train(text, args.vocab_size, verbose=args.verbose)
    tok.save(args.output)
    print(f"saved tokenizer ({tok.vocab_size} tokens) -> {args.output}")
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    from muselm.data import prepare_dataset
    from muselm.tokenizer import ByteBPETokenizer

    text = Path(args.input).read_text(encoding="utf-8")
    tok = ByteBPETokenizer.load(args.tokenizer)
    stats = prepare_dataset(text, tok, args.output, val_fraction=args.val_fraction)
    print(f"prepared dataset -> {args.output}: {stats}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from muselm.config import TrainConfig
    from muselm.train import train

    cfg = TrainConfig.from_json(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.device is not None:
        cfg.device = args.device
    train(cfg)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from muselm.generate import Generator

    gen = Generator.from_checkpoint(args.checkpoint, args.tokenizer, device=args.device)
    text = gen.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    print(args.prompt + text)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from muselm.config import TrainConfig
    from muselm.model import MuseLM

    cfg = TrainConfig.from_json(args.config)
    model = MuseLM(cfg.model)
    total = model.num_parameters()
    active = model.num_parameters(exclude_embeddings=True)
    print(json.dumps({
        "total_params": total,
        "total_params_M": round(total / 1e6, 2),
        "non_embedding_params_M": round(active / 1e6, 2),
        "is_moe": cfg.model.is_moe,
        "config": cfg.model.to_dict(),
    }, indent=2))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from benchmark_inference import benchmark  # noqa: E402

    from muselm.config import TrainConfig, resolve_device

    device = resolve_device(args.device)
    cfg = TrainConfig.from_json(args.config).model
    result = benchmark(cfg, args.tokens, args.warmup, device)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="muselm", description="MuseLM: an open LLM stack")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokenizer-train", help="train a byte-BPE tokenizer")
    t.add_argument("--input", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--vocab-size", type=int, default=4096, dest="vocab_size")
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=_cmd_tokenizer_train)

    pr = sub.add_parser("prepare", help="tokenize a corpus into train/val bins")
    pr.add_argument("--input", required=True)
    pr.add_argument("--tokenizer", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--val-fraction", type=float, default=0.1, dest="val_fraction")
    pr.set_defaults(func=_cmd_prepare)

    tr = sub.add_parser("train", help="train a model")
    tr.add_argument("--config", required=True)
    tr.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    tr.add_argument("--device", default=None)
    tr.set_defaults(func=_cmd_train)

    g = sub.add_parser("generate", help="generate text from a checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--prompt", default="")
    g.add_argument("--max-new-tokens", type=int, default=200, dest="max_new_tokens")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40, dest="top_k")
    g.add_argument("--top-p", type=float, default=None, dest="top_p")
    g.add_argument("--repetition-penalty", type=float, default=1.1, dest="repetition_penalty")
    g.add_argument("--device", default="auto")
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=_cmd_generate)

    i = sub.add_parser("info", help="print model parameter counts")
    i.add_argument("--config", required=True)
    i.set_defaults(func=_cmd_info)

    b = sub.add_parser("benchmark", help="measure inference throughput")
    b.add_argument("--config", required=True)
    b.add_argument("--tokens", type=int, default=256)
    b.add_argument("--warmup", type=int, default=16)
    b.add_argument("--device", default="auto")
    b.set_defaults(func=_cmd_benchmark)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
