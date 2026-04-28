"""
Cloud storage profile resolution.

Credentials are resolved in layers:
1. Environment variables (COMFY_S3_*)
2. Named profiles from JSON file in ComfyUI's system user directory
3. Per-node widget overrides (bucket, prefix only - never keys)
"""

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)


def _shell_hint(env_var: str, value_placeholder: str) -> str:
    """Platform-appropriate command for setting an environment variable.

    Returned text is appended to error messages so users see a copy-pastable
    command for their OS instead of a generic "set the env var" instruction.
    """
    if sys.platform == "win32":
        return f"setx {env_var} \"{value_placeholder}\"  (then restart ComfyUI)"
    if sys.platform == "darwin":
        return f"export {env_var}=\"{value_placeholder}\"  (add to ~/.zshrc to persist)"
    return f"export {env_var}=\"{value_placeholder}\"  (add to ~/.bashrc to persist)"

ENV_PREFIX = "COMFY_S3_"

# Map of env var suffix -> profile dict key
ENV_KEYS = {
    "PROVIDER": "provider",
    "ACCESS_KEY": "access_key",
    "SECRET_KEY": "secret_key",
    "REGION": "region",
    "BUCKET": "bucket",
    "ENDPOINT_URL": "endpoint_url",
    "ACCOUNT_ID": "account_id",
    "PATH_PREFIX": "path_prefix",
}


def _get_profiles_path() -> str:
    """Get path to the profiles JSON file in ComfyUI's system user directory.

    Uses ComfyUI's HTTP-inaccessible system directory when available; falls back
    to a user-home path only when running outside ComfyUI (e.g. for tests).
    """
    try:
        import folder_paths
    except ImportError:
        return os.path.join(os.path.expanduser("~"), ".comfyui-cloud-storage", "profiles.json")
    sys_dir = folder_paths.get_system_user_directory("cloud_storage")
    return os.path.join(sys_dir, "profiles.json")


def _load_profiles() -> dict:
    """Load named profiles from the JSON file."""
    path = _get_profiles_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("profiles", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load cloud storage profiles from %s: %s", path, e)
        return {}


def load_profile_names() -> list[str]:
    """Return list of available named profile names."""
    return list(_load_profiles().keys())


def _profile_from_env() -> dict:
    """Build a profile dict from environment variables."""
    profile = {}
    for env_suffix, key in ENV_KEYS.items():
        val = os.environ.get(f"{ENV_PREFIX}{env_suffix}", "")
        if val:
            profile[key] = val
    return profile


def resolve_profile(
    profile_name: str = "(env vars)",
    provider_override: str = "",
    bucket_override: str = "",
    path_prefix_override: str = "",
) -> dict:
    """Resolve a complete profile config dict.

    Layers: env vars -> named profile -> widget overrides.

    Returns dict with keys:
        provider, access_key, secret_key, region, bucket,
        endpoint_url, account_id, path_prefix
    """
    # Start with env vars. `read_only` and `default_tags` are profile-only
    # (not env-driven) — they're orthogonal to credential resolution.
    config = {
        "provider": "AWS S3",
        "access_key": "",
        "secret_key": "",
        "region": "",
        "bucket": "",
        "endpoint_url": "",
        "account_id": "",
        "path_prefix": "",
        "read_only": False,
        "default_tags": {},
    }
    config.update({k: v for k, v in _profile_from_env().items() if v})

    # Overlay named profile
    if profile_name and profile_name != "(env vars)":
        profiles = _load_profiles()
        named = profiles.get(profile_name, {})
        if not named:
            logger.warning("Cloud storage profile '%s' not found", profile_name)
        else:
            config.update({k: v for k, v in named.items() if v})

    # Overlay per-node widget overrides (never override credentials this way)
    if provider_override and provider_override != "(from profile)":
        config["provider"] = provider_override
    if bucket_override:
        config["bucket"] = bucket_override
    if path_prefix_override:
        config["path_prefix"] = path_prefix_override

    return config


def resolve_default_profile() -> dict:
    """Resolve the default profile (env vars only, no named profile)."""
    return resolve_profile("(env vars)")


def validate_config(config: dict, mode: str = "read") -> None:
    """Raise ValueError with a clear message if the config is missing or invalid.

    `mode` is "read" by default, or "write" for save/upload paths. When the
    profile carries `read_only: true`, write-mode access is rejected — this
    lets shared inference machines hold read-only credentials without nodes
    accidentally uploading.
    """
    # Lazy import to avoid a circular dependency at module load time.
    from .providers import PROVIDERS

    if not config.get("access_key"):
        raise ValueError(
            "Cloud storage access key not configured. "
            f"{_shell_hint('COMFY_S3_ACCESS_KEY', '<your-key-id>')}, "
            f"or create a profile in {_get_profiles_path()}, "
            "or connect a CloudStorageProfile node."
        )
    if not config.get("secret_key"):
        raise ValueError(
            "Cloud storage secret key not configured. "
            f"{_shell_hint('COMFY_S3_SECRET_KEY', '<your-secret>')}, "
            "or configure a profile."
        )
    if not config.get("bucket"):
        raise ValueError(
            "Cloud storage bucket not configured. "
            f"{_shell_hint('COMFY_S3_BUCKET', '<bucket-name>')}, "
            "or configure a profile."
        )

    provider = config.get("provider", "")
    if provider and provider not in PROVIDERS:
        raise ValueError(
            f"Unknown cloud storage provider: {provider!r}. "
            f"Valid options: {', '.join(PROVIDERS)}."
        )

    # Cloudflare R2's preset endpoint requires an account_id; users can also
    # bypass the preset by supplying endpoint_url directly.
    if provider == "Cloudflare R2" and not config.get("endpoint_url") and not config.get("account_id"):
        raise ValueError(
            "Cloudflare R2 requires either COMFY_S3_ACCOUNT_ID (account_id) "
            "or an explicit COMFY_S3_ENDPOINT_URL."
        )

    # Custom provider has no preset, so endpoint_url is the only way to reach it.
    if provider == "Custom" and not config.get("endpoint_url"):
        raise ValueError(
            "Custom provider requires COMFY_S3_ENDPOINT_URL (endpoint_url)."
        )

    if mode == "write" and config.get("read_only"):
        raise ValueError(
            "This cloud storage profile is marked read_only. "
            "Remove `read_only: true` from the profile to allow uploads."
        )


def apply_prefix(config: dict, key: str) -> str:
    """Combine `path_prefix` from the config with a user-supplied key.

    Strips any leading slash from `key` to keep S3 object keys clean — S3 keys
    do not begin with "/". This is the single source of truth for prefix
    composition across save, load, and browse nodes.
    """
    return f"{config.get('path_prefix', '')}{key.lstrip('/')}"
