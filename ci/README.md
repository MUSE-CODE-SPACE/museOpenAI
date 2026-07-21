# Continuous Integration

The GitHub Actions workflow for MuseLM lives here as [`ci.yml`](ci.yml). It runs
`ruff` lint and the full `pytest` suite on Python 3.9 / 3.11 / 3.12, plus an
end-to-end smoke-train job.

## Enabling it

GitHub blocks pushing files under `.github/workflows/` unless your credential
has the `workflow` OAuth scope, so the workflow ships here instead. To activate
it once:

```bash
# 1. Grant the workflow scope to your GitHub CLI token (one time)
gh auth refresh -h github.com -s workflow

# 2. Move the workflow into place and push
mkdir -p .github/workflows
git mv ci/ci.yml .github/workflows/ci.yml
git commit -m "ci: enable GitHub Actions workflow"
git push
```

After that, every push and pull request runs the matrix automatically.
