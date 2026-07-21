# MuseLM 홍보 게시글 초안

각 플랫폼 톤에 맞춘 복붙용 초안입니다. 링크: https://github.com/MUSE-CODE-SPACE/museOpenAI

---

## X / Twitter (스레드)

**1/**
대부분의 "오픈" LLM은 가중치만 공개하고 정작 재미있는 부분 — 토크나이저 학습기, MoE 라우팅, 데이터 파이프라인 — 은 프레임워크 코드 속에 숨겨둡니다.

MuseLM은 정반대입니다. 텍스트를 말하는 모델로 바꾸는 모든 부품이, 한 번에 읽을 수 있을 만큼 작게, 순수 PyTorch로 🧵

**2/**
프론티어 오픈모델(DeepSeek-V3, Kimi K2, Llama 3)이 쓰는 바로 그 선택들을 각각 수백 줄로 구현했습니다:

• Byte-level BPE 토크나이저 (모든 유니코드 round-trip)
• RMSNorm · RoPE · GQA · QK-norm · SwiGLU
• Fine-grained Mixture-of-Experts

**3/**
`n_experts > 0` 한 줄로 dense → sparse 전환.
top-k 라우팅 + shared experts + load-balancing loss까지 전부 들어있습니다.

MoE의 "총 파라미터는 크게, 활성 연산은 작게"를 실제로 측정하는 `muselm benchmark`도 있습니다.

**4/**
데모가 아니라 릴리즈입니다:
✅ 테스트 21개 (KV-cache = full-forward 수치 동일성 증명 포함)
✅ CI 매트릭스 (Python 3.9/3.11/3.12)
✅ 학습된 체크포인트 동봉 — 학습 없이 바로 생성

**5/**
맥북(Apple Silicon)에서 ~8분 만에 셰익스피어를 학습해 validation perplexity 47.8.

Apache-2.0. 상업·연구 자유.
읽고, 배우고, 위에 쌓아 올리세요.

⭐ https://github.com/MUSE-CODE-SPACE/museOpenAI

---

## Hacker News (Show HN)

**제목:**
Show HN: MuseLM – A from-scratch LLM stack (tokenizer, MoE, training) in plain PyTorch

**본문:**
I built MuseLM because most "open" LLMs ship weights but keep the interesting machinery — the tokenizer trainer, MoE routing, the data pipeline — behind framework abstractions. MuseLM is the opposite: every component that turns raw text into a working model is here, small enough to read in an afternoon, in plain PyTorch with no hidden dependencies.

It implements the same architectural choices as frontier open-weight models (DeepSeek-V3, Kimi K2, Llama 3), each in a few hundred lines:

- Byte-level BPE tokenizer trained from scratch (round-trips any Unicode, no <unk>)
- RMSNorm, RoPE, grouped-query attention with QK-norm, SwiGLU
- Fine-grained Mixture-of-Experts: top-k routing, shared experts, load-balancing aux loss — enable with one config field
- Training loop with cosine schedule, grad accumulation/clipping, eval/perplexity
- KV-cached generation with temperature/top-k/top-p/repetition penalty

It's a release, not a demo: 21 tests (including a proof that KV-cache decoding is numerically identical to a full forward pass), a CI matrix on Python 3.9/3.11/3.12, and a trained ~5M-param Tiny-Shakespeare checkpoint that ships in the repo so you can generate text immediately. It reaches validation perplexity 47.8 in ~8 minutes on an M-series Mac, and the exact same code scales to a real corpus on a GPU.

Apache-2.0. Feedback and PRs very welcome — there's a roadmap toward distributed training, quantized inference, and SFT/DPO.

https://github.com/MUSE-CODE-SPACE/museOpenAI

---

## Reddit — r/MachineLearning

**제목:**
[P] MuseLM: a fully open, from-scratch LLM stack (byte-BPE tokenizer + MoE transformer + training + inference) in plain PyTorch

**본문:**
I wanted a codebase where *every* part of building an LLM is visible and readable — not just the weights. So I wrote MuseLM: a complete stack in plain PyTorch (torch + numpy only), where each component mirrors what frontier open-weight models actually do.

What's in it:

- **Tokenizer**: byte-level BPE trained from scratch, full Unicode round-trip, no `<unk>`.
- **Model**: RMSNorm, RoPE, grouped-query attention with QK-norm, SwiGLU.
- **Mixture-of-Experts**: fine-grained top-k routing + shared experts + Switch-style load-balancing loss. Flip `n_experts > 0` to go dense → sparse.
- **Training**: cosine LR + warmup, grad accumulation/clipping, eval/perplexity, rotating checkpoints.
- **Inference**: KV-cache incremental decoding with temperature/top-k/top-p/repetition penalty.

Why it might be useful to you:

- It's tested (21 unit + integration tests), including one that asserts KV-cache decoding is numerically identical to a full forward pass — the kind of invariant that's easy to get subtly wrong.
- A trained Tiny-Shakespeare checkpoint ships in the repo (val perplexity 47.8), so you can `muselm generate` in 30 seconds.
- CI runs lint + tests on Python 3.9/3.11/3.12.

The architecture doc walks through the attention math, RoPE, the MoE router and its auxiliary loss, and how the KV cache makes generation O(1) per token. Apache-2.0.

I'd love feedback on the MoE implementation and the training ergonomics. Repo: https://github.com/MUSE-CODE-SPACE/museOpenAI

---

## LinkedIn

프론티어 오픈모델들이 쓰는 아키텍처를 처음부터, 읽을 수 있게.

지난 며칠간 **MuseLM**을 만들었습니다 — byte-level BPE 토크나이저부터 Mixture-of-Experts 트랜스포머, 학습 루프, KV-cache 추론까지 전부 순수 PyTorch로 구현한 완전 오픈소스 LLM 스택입니다.

대부분의 "오픈" LLM은 가중치만 공개합니다. MuseLM은 텍스트를 모델로 바꾸는 모든 과정을 투명하게 보여주는 것을 목표로 했습니다:

🎼 DeepSeek-V3 · Kimi K2 · Llama 3가 쓰는 바로 그 설계 (RMSNorm, RoPE, GQA, QK-norm, SwiGLU, MoE)
🎼 `n_experts > 0` 한 줄로 dense ↔ sparse 전환
🎼 테스트 21개 + CI, 학습된 체크포인트 동봉 (맥북 ~8분, perplexity 47.8)
🎼 Apache-2.0, 상업·연구 자유

LLM 내부를 진짜로 이해하고 싶은 분, 위에 무언가 쌓아 올리고 싶은 분께 유용할 겁니다.

⭐ https://github.com/MUSE-CODE-SPACE/museOpenAI

#MachineLearning #LLM #DeepLearning #PyTorch #OpenSource #AI
