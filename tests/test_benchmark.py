import sys
from pathlib import Path

from muselm.config import MuseLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_inference import benchmark  # noqa: E402


def test_benchmark_returns_metrics():
    cfg = MuseLMConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4,
                       n_kv_heads=2, max_seq_len=64, ffn_hidden_dim=128)
    result = benchmark(cfg, n_tokens=16, warmup=2, device="cpu")
    assert result["tokens"] == 16
    assert result["tokens_per_sec"] > 0
    assert result["non_embedding_params_M"] <= result["total_params_M"]
    assert result["is_moe"] is False


def test_benchmark_moe_active_less_than_total():
    cfg = MuseLMConfig(vocab_size=128, dim=64, n_layers=2, n_heads=4,
                       n_kv_heads=2, max_seq_len=64, n_experts=4,
                       n_active_experts=2, n_shared_experts=1, expert_hidden_dim=32)
    result = benchmark(cfg, n_tokens=16, warmup=2, device="cpu")
    assert result["is_moe"] is True
    assert result["tokens_per_sec"] > 0
