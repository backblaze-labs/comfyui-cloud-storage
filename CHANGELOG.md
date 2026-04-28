# Changelog

All notable changes to this project will be documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-28

### Added
- `TestCloudConnection` node — issues `head_bucket` and a 3-key list against
  the configured profile and returns a one-line OK report with wall-clock
  latency. Use it to surface credential/bucket misconfigurations before
  queueing a real workflow.
- Save nodes (`SaveImageToCloud`, `SaveVideoToCloud`, `SaveAudioToCloud`) now
  expose a `key` STRING graph output. For batches, the first key is returned;
  this is the only safe shape to feed into downstream single-key nodes like
  `Generate Sharing URL`.
- `SaveImageToCloud` gained `presign_url` (bool) + `expires_hours` (int)
  inputs and a `url` STRING output. Toggle on to emit a presigned URL inline
  with the upload — no separate node required.
- New filename-template tokens for save nodes: `%date%` (YYYY-MM-DD),
  `%time%` (HHMMSS), `%uuid%` (8-char). `%batch_num%` continues to work.
- `read_only: true` flag in `profiles.json` blocks save nodes from uploading
  through that profile, enforced centrally via `validate_config(mode="write")`.
- `default_tags: {...}` in `profiles.json` applies S3 object tags to every
  uploaded object. Skipped automatically on Backblaze B2, which does not
  implement the S3 PutObjectTagging API.
- `LoadModelFromCloud` returns a UI text status line — `cached: foo.safetensors`
  on cache hit, or `downloaded: foo.safetensors (4.2 GB in 3m12s)` on miss.
- Module-level boto3 client cache (LRU, max 16 entries) keyed on credential
  fields. Identical config across nodes reuses the same boto3 client. Note:
  rotated credentials require a ComfyUI restart to take effect.
- Platform-specific shell hints in credential-not-configured error messages
  (`setx` on Windows, `export ... ~/.zshrc` on macOS, `~/.bashrc` on Linux).
- Example workflows in `examples/` plus an `examples/README.md` walkthrough.
- `[tool.comfy] DisplayName` populated. `PublisherId` and `Icon` left as
  documented TODOs in `CONTRIBUTING.md`.

### Changed
- `validate_config` accepts a `mode` parameter (`"read"` default, `"write"`
  for save paths). Save nodes pass `mode="write"`.

### Removed
- `MinIO` provider preset is no longer shipped. Users running MinIO can still
  point at it via the `Custom` provider plus `endpoint_url`.

### Known limitations
- `BrowseCloudFiles` (a Combo dropdown populated from bucket contents) was
  evaluated and intentionally not shipped: calling boto3 inside `define_schema`
  would block ComfyUI startup for offline users. A future enhancement using
  ComfyUI's frontend-extension API would be the right path.
- Concurrent batch uploads in `SaveImageToCloud` were evaluated and deferred:
  boto3 clients are not thread-safe under shared use, and the existing
  user-agent test infrastructure is single-threaded. Needs a per-thread client
  design and test-infra rework.

## [0.2.0] - 2026-04-28

### Added
- `LoadAudioFromCloud` node, mirroring `LoadImageFromCloud` for audio assets.
- Upload progress bar on `SaveVideoToCloud` for large video payloads.
- `apply_prefix` helper used consistently by all save/load/browse nodes for
  predictable `path_prefix` semantics.
- Strict provider validation in `validate_config`: unknown providers, R2
  without `account_id`/`endpoint_url`, and Custom without `endpoint_url`
  now fail fast with a clear message.
- `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, GitHub Actions test workflow.
- `typing_extensions` declared as an explicit runtime dependency.
- Optional `[dev]` extras in `pyproject.toml` for `pytest` and coverage.

### Changed
- `SaveAudioToCloud` now iterates the batch dimension and uploads N files per
  batch, mirroring `SaveImageToCloud` (previously squeezed batch dim and
  uploaded one file).
- `SaveVideoToCloud` and `SaveAudioToCloud` honor the `%batch_num%` filename
  template via the shared `_build_key` helper.
- `fingerprint_inputs` returns a deterministic input-derived sentinel instead
  of an empty string when the remote head request fails. Fixes a subtle
  cache-hit bug on transient network errors.
- `.s3etag` cache file is now written via tmp + `os.replace` for atomicity.
- Dependency on `comfy.cli_args.args` is lazy and tolerant of forks that omit
  `disable_metadata`.

### Fixed
- `profile.py` no longer swallows arbitrary exceptions when locating the
  ComfyUI system user directory; only `ImportError` is caught.
- `profiles.json` is now read with `encoding="utf-8"` for cross-platform
  consistency.

### Removed
- Undocumented leading-slash escape that bypassed `path_prefix` on
  `LoadImageFromCloud`. Leading slashes are now stripped uniformly.

## [0.1.0] - 2026-02-19

Initial release.
