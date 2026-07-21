"""End-to-end: train a tokenizer, prepare data, train a tiny model, generate."""

import torch

from muselm.config import MuseLMConfig, TrainConfig
from muselm.data import TokenDataset, prepare_dataset
from muselm.generate import Generator
from muselm.tokenizer import ByteBPETokenizer
from muselm.train import get_lr, train

CORPUS = "hello world. the quick brown fox jumps over the lazy dog. " * 200


def test_lr_schedule_shape():
    cfg = TrainConfig(warmup_steps=10, max_steps=100, learning_rate=1e-3, min_learning_rate=1e-4)
    assert get_lr(0, cfg) < cfg.learning_rate
    assert abs(get_lr(9, cfg) - cfg.learning_rate) < 1e-9
    assert get_lr(100, cfg) == cfg.min_learning_rate
    assert cfg.min_learning_rate <= get_lr(55, cfg) <= cfg.learning_rate


def test_full_pipeline(tmp_path):
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=320)
    tok_path = tmp_path / "tok.json"
    tok.save(tok_path)

    data_dir = tmp_path / "data"
    stats = prepare_dataset(CORPUS, tok, data_dir, val_fraction=0.1)
    assert stats["train_tokens"] > 0 and stats["val_tokens"] > 0

    ds = TokenDataset(data_dir / "train.bin", "uint16")
    x, y = ds.get_batch(4, 16, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])

    cfg = TrainConfig(
        data_dir=str(data_dir),
        tokenizer_path=str(tok_path),
        batch_size=8,
        max_steps=30,
        eval_interval=15,
        eval_steps=3,
        log_interval=10,
        warmup_steps=5,
        checkpoint_dir=str(tmp_path / "ckpt"),
        device="cpu",
        model=MuseLMConfig(
            vocab_size=tok.vocab_size,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            ffn_hidden_dim=128,
        ),
    )
    summary = train(cfg)
    assert summary["best_val_loss"] < 100
    assert (tmp_path / "ckpt" / "best.pt").exists()

    gen = Generator.from_checkpoint(tmp_path / "ckpt" / "best.pt", tok_path, device="cpu")
    text = gen.generate("hello", max_new_tokens=20, temperature=0.7, seed=0)
    assert isinstance(text, str)


def test_moe_pipeline(tmp_path):
    tok = ByteBPETokenizer.train(CORPUS, vocab_size=320)
    data_dir = tmp_path / "data"
    prepare_dataset(CORPUS, tok, data_dir)
    cfg = TrainConfig(
        data_dir=str(data_dir),
        batch_size=8,
        max_steps=20,
        eval_interval=10,
        eval_steps=2,
        warmup_steps=3,
        checkpoint_dir=str(tmp_path / "ckpt"),
        device="cpu",
        model=MuseLMConfig(
            vocab_size=tok.vocab_size,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            n_experts=4,
            n_active_experts=2,
            n_shared_experts=1,
            expert_hidden_dim=32,
        ),
    )
    summary = train(cfg)
    assert summary["active_params"] < summary["params"] or not cfg.model.tie_embeddings
    assert (tmp_path / "ckpt" / "final.pt").exists()
