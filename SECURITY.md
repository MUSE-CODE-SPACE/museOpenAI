# Security Policy

## Supported versions

MuseLM is pre-1.0; security fixes land on `main` and in the latest release.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's [private vulnerability reporting](https://github.com/MUSE-CODE-SPACE/museOpenAI/security/advisories/new)
or email the maintainer. Include a description, reproduction steps, and impact.
We aim to acknowledge within 72 hours.

## Scope & model-safety note

MuseLM is a from-scratch research/education stack. Checkpoints trained with it —
including the released Tiny-Shakespeare model — have **no safety tuning or
content filtering** and should not be deployed as user-facing assistants without
your own alignment and moderation layer. Loading a checkpoint executes
`torch.load`; only load checkpoints you trust.
