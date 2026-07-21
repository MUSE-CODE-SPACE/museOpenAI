"""Minimal end-to-end example using MuseLM as a library (no CLI).

Run: python examples/library_usage.py
"""

from muselm import ByteBPETokenizer, MuseLM, MuseLMConfig
from muselm.config import TrainConfig
from muselm.data import prepare_dataset
from muselm.generate import Generator
from muselm.train import train

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "muselm is a tiny but complete language model. "
) * 500


def main() -> None:
    # 1. Train a tokenizer directly from text.
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=512)
    tok.save("data/example/tokenizer.json")
    print(f"tokenizer: {tok.vocab_size} tokens")

    # 2. Inspect a model without training.
    cfg = MuseLMConfig(vocab_size=tok.vocab_size, dim=128, n_layers=4,
                       n_heads=4, n_kv_heads=2, max_seq_len=64, n_experts=4)
    model = MuseLM(cfg)
    print(f"model: {model.num_parameters()/1e6:.2f}M params, MoE={cfg.is_moe}")

    # 3. Tokenize data and train briefly.
    prepare_dataset(CORPUS, tok, "data/example")
    train_cfg = TrainConfig(
        data_dir="data/example",
        max_steps=100, eval_interval=50, eval_steps=5, warmup_steps=10,
        batch_size=16, checkpoint_dir="checkpoints/example", device="cpu",
        model=cfg,
    )
    summary = train(train_cfg)
    print(f"trained: val_loss={summary['best_val_loss']:.3f}")

    # 4. Generate.
    gen = Generator.from_checkpoint(
        "checkpoints/example/best.pt", "data/example/tokenizer.json", device="cpu"
    )
    print("sample:", repr(gen.generate("the quick", max_new_tokens=40, seed=0)))


if __name__ == "__main__":
    main()
