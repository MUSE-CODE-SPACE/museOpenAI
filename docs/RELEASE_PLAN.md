# Release & PR Plan

This document records how MuseLM v0.1.0 was released and the branching model for
ongoing contributions.

## Initial public release (v0.1.0)

1. `main` — the stable branch. Protected: PRs must pass CI (lint + test matrix).
2. The initial commit lands the full stack: core library, tests, CI, docs, and a
   released Tiny-Shakespeare checkpoint under `checkpoints/release/`.
3. Tag `v0.1.0` and cut a GitHub Release with the checkpoint attached.

## Branching model

- `main` — always green, always releasable.
- `feat/*`, `fix/*`, `docs/*` — short-lived branches, one logical change each.
- Every change reaches `main` via a pull request that passes CI and review.

## Demonstration PR: MoE inference benchmark

To exercise the contribution workflow end-to-end, the first PR after the initial
release adds an **MoE-vs-dense inference benchmark** utility — a self-contained,
useful feature that:

- adds `scripts/benchmark_inference.py` measuring tokens/sec and active-vs-total
  parameter ratios for a given config,
- wires a `muselm benchmark` CLI subcommand,
- ships a test,
- updates the README with a benchmark section.

This proves the loop: branch → implement + test → green CI → PR against `main`.

## Checklist for each release

- [ ] `ruff check` + `pytest` green locally and in CI
- [ ] `CHANGELOG.md` updated
- [ ] Version bumped in `pyproject.toml` and `muselm/__init__.py`
- [ ] Docs reflect any interface changes
- [ ] Tag + GitHub Release
