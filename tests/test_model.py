import torch

from muselm.config import MuseLMConfig
from muselm.model import MuseLM


def tiny_config(**kw) -> MuseLMConfig:
    base = dict(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        ffn_hidden_dim=128,
    )
    base.update(kw)
    return MuseLMConfig(**base)


def test_forward_shapes():
    cfg = tiny_config()
    model = MuseLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(x, targets=x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_loss_decreases_on_overfit():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = MuseLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    first = None
    for _ in range(50):
        _, loss, _ = model(x, targets=y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_moe_forward_and_aux_loss():
    cfg = tiny_config(n_experts=4, n_active_experts=2, n_shared_experts=1, expert_hidden_dim=32)
    assert cfg.is_moe
    model = MuseLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = model(x, targets=x)
    assert torch.isfinite(loss)
    # aux loss should be populated on every MoE block
    from muselm.model import MoE

    moe_blocks = [b.ffn for b in model.blocks if isinstance(b.ffn, MoE)]
    assert moe_blocks and all(torch.isfinite(b.aux_loss) for b in moe_blocks)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = MuseLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 10))

    # Full forward
    full_logits, _, _ = model(x)

    # Incremental with cache, token by token
    caches = None
    pos = 0
    step_logits = []
    with torch.no_grad():
        for t in range(x.size(1)):
            inp = x[:, t : t + 1]
            lg, _, caches = model(inp, caches=caches, start_pos=pos)
            pos += 1
            step_logits.append(lg[:, -1, :])
    incremental = torch.stack(step_logits, dim=1)
    assert torch.allclose(full_logits, incremental, atol=1e-4)


def test_generate_runs_and_stays_in_vocab():
    cfg = tiny_config()
    model = MuseLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=10)
    assert out.shape == (1, 24)
    assert out.max().item() < cfg.vocab_size


def test_generate_past_context_window():
    cfg = tiny_config(max_seq_len=16)
    model = MuseLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=40, temperature=0.7, top_k=10)
    assert out.shape[1] == 44


def test_greedy_is_deterministic():
    cfg = tiny_config()
    model = MuseLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    a = model.generate(idx, max_new_tokens=10, temperature=0.0)
    b = model.generate(idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)


def test_param_count_monotonic():
    small = MuseLM(tiny_config(n_layers=2)).num_parameters()
    big = MuseLM(tiny_config(n_layers=4)).num_parameters()
    assert big > small
