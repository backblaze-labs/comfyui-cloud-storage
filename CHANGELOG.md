# Changelog

All notable changes to this project will be documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
