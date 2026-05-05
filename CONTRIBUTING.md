# Contributing to comfyui-cloud-storage

Thanks for your interest in improving this package. The goal is a small, focused
set of S3-compatible cloud storage nodes for ComfyUI that "just work" across
every major provider — please keep that goal in mind when proposing changes.

## Getting set up

```bash
git clone https://github.com/backblaze-labs/comfyui-cloud-storage
cd comfyui-cloud-storage
pip install -e ".[dev]"
```

Tests do not require a running ComfyUI install — `tests/conftest.py` mocks the
ComfyUI surface. The exception is `test_user_agent.py`, which spins up a local
HTTP server to verify the `b2ai-comfyui` user-agent on real `boto3` calls.

```bash
python -m pytest tests/ -v
```

## What we're looking for

- Bug fixes (with a reproducing test).
- Provider preset additions for S3-compatible services not in `providers.py`.
- Improvements to credential resolution, caching, or error messages.
- Symmetric save/load support for content types we don't yet cover.

## What we're cautious about

- Net-new dependencies. `boto3` is intentionally the only runtime dep.
- Changes that break workflows already in the wild (saved JSON references node
  IDs, input names, and output types — keep these stable).
- Provider-specific quirks leaking into shared code paths. Add an entry in
  `PROVIDERS` and let the abstraction handle it.

## PR checklist

- [ ] Tests added or updated; `pytest tests/ -v` passes locally.
- [ ] New nodes registered in `__init__.py`.
- [ ] Provider presets validated against the real endpoint.
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`.
- [ ] Public-facing changes (new nodes, env vars, behavior) documented in `README.md`.

## Reporting security issues

Please do **not** open a public issue for security-sensitive bugs. Email the
maintainers at the address listed on the GitHub organization page, or use
GitHub's private security advisory feature on this repository.

## Publishing to the ComfyUI Manager registry

The package is set up to be listed at https://registry.comfy.org/ but needs
two things filled in before submission:

- `[tool.comfy] PublisherId` in `pyproject.toml` — must match a registered
  publisher account on the registry.
- `[tool.comfy] Icon` (optional) — a public URL to a 256x256 PNG.

Once those are populated, follow the registry's submission flow: create a
publisher account, point it at this repository, and tag a release.
