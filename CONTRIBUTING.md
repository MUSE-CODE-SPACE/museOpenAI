# Contributing to MuseLM

Thanks for your interest — MuseLM is built to be hacked on. This guide gets you
from clone to merged PR.

## Development setup

```bash
git clone https://github.com/MUSE-CODE-SPACE/museOpenAI.git
cd museOpenAI
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify your environment:

```bash
ruff check muselm/ tests/     # lint
pytest -q                     # full suite (should be all green in < 1 min on CPU)
```

## Workflow

1. **Open (or claim) an issue** describing the change. For anything non-trivial,
   agree on the approach before writing code.
2. **Branch** from `main`: `git checkout -b feat/streaming-dataset`.
3. **Write code + tests.** Every behavior change needs a test. Bug fixes should
   add a regression test that fails before your fix.
4. **Run `ruff check` and `pytest`** locally — CI runs both on Python 3.9, 3.11,
   and 3.12 and must pass.
5. **Open a PR** against `main`. Fill in the template: what, why, how tested.

## Coding standards

- **Style**: `ruff` enforces formatting and lint. Line length 100. Run
  `ruff check --fix` before pushing.
- **Types**: annotate public functions. `from __future__ import annotations` is
  already imported everywhere, so use modern syntax freely.
- **Docstrings**: every module and public class/function gets one. Say *why*,
  not just *what*.
- **No new hard dependencies** without discussion. The core stays PyTorch +
  NumPy. Optional accelerated paths must degrade gracefully.
- **Readability first.** See the design principles in
  [docs/ROADMAP.md](docs/ROADMAP.md). If a reviewer can't follow it, it needs
  simplifying or documenting.

## What makes a good PR

- Focused: one logical change. Split unrelated work.
- Tested: green CI, plus new tests for new behavior.
- Documented: update the README / architecture doc if you change the interface
  or add a concept.
- Benchmarked: if you touch the hot path (attention, MoE routing, the training
  step), include before/after numbers.

## Good first issues

Look for the [`good first issue`](https://github.com/MUSE-CODE-SPACE/museOpenAI/labels/good%20first%20issue)
label. The v0.2 roadmap items are also a great place to start — resume-from-
checkpoint and W&B logging are self-contained and high-value.

## Reporting bugs

Open an issue with: the config/command you ran, the full traceback, your Python
and torch versions (`pip show torch`), and your device (CPU/CUDA/MPS). A minimal
reproduction is worth a thousand words.

## Code of Conduct

Be kind, be constructive, assume good faith. Harassment or discrimination of any
kind is not tolerated. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
