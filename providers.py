"""
S3-compatible provider presets and client factory.

Supports: AWS S3, Backblaze B2, Cloudflare R2, Wasabi,
DigitalOcean Spaces, GCS (S3 interop), and any custom S3 endpoint.
"""

from collections import OrderedDict
from dataclasses import dataclass

# Bounded LRU cache of boto3 clients. Each ComfyUI workflow tends to reuse the
# same one or two profiles across many node executions; rebuilding the client
# every time pays the boto3 init cost (~tens of ms) needlessly. The cap of 16
# is well above the realistic profile count and small enough not to leak
# memory if a user cycles through many short-lived profiles. Note: cached
# clients hold credentials in memory until eviction or process restart, so
# rotated keys require a ComfyUI restart to take effect.
_CLIENT_CACHE_MAX = 16
_client_cache: "OrderedDict[tuple, object]" = OrderedDict()


def _client_cache_key(provider, access_key, secret_key, region, endpoint_url, account_id):
    return (provider, access_key, secret_key, region, endpoint_url, account_id)


def clear_client_cache() -> None:
    """Drop all cached boto3 clients. Mostly useful for tests."""
    _client_cache.clear()


@dataclass(frozen=True)
class ProviderPreset:
    endpoint_template: str  # "" means boto3 default (AWS), else "{region}" placeholders
    default_region: str = "us-east-1"
    force_path_style: bool = False


PROVIDERS: dict[str, ProviderPreset] = {
    "AWS S3": ProviderPreset(
        endpoint_template="",
        default_region="us-east-1",
    ),
    "Backblaze B2": ProviderPreset(
        endpoint_template="https://s3.{region}.backblazeb2.com",
        default_region="us-west-004",
    ),
    "Cloudflare R2": ProviderPreset(
        endpoint_template="https://{account_id}.r2.cloudflarestorage.com",
        default_region="auto",
    ),
    "Wasabi": ProviderPreset(
        endpoint_template="https://s3.{region}.wasabisys.com",
        default_region="us-east-1",
    ),
    "DigitalOcean Spaces": ProviderPreset(
        endpoint_template="https://{region}.digitaloceanspaces.com",
        default_region="nyc3",
    ),
    "GCS (S3 interop)": ProviderPreset(
        endpoint_template="https://storage.googleapis.com",
        default_region="auto",
    ),
    "Custom": ProviderPreset(
        endpoint_template="",
        default_region="",
    ),
}

PROVIDER_NAMES = list(PROVIDERS.keys())


def create_s3_client(
    provider: str = "AWS S3",
    access_key: str = "",
    secret_key: str = "",
    region: str = "",
    endpoint_url: str = "",
    account_id: str = "",
    **_ignored,
):
    """Create or reuse a boto3 S3 client configured for the given provider.

    Clients are memoized by their credential-relevant fields, so callers can
    invoke this once per node execution without repeatedly paying boto3's
    init cost. Lazy-imports boto3 so the dependency only loads when actually
    used. Accepts and ignores extra keyword arguments (e.g. `bucket`,
    `path_prefix`, `default_tags`) so callers can splat the full profile dict.
    """
    cache_key = _client_cache_key(provider, access_key, secret_key, region, endpoint_url, account_id)
    cached = _client_cache.get(cache_key)
    if cached is not None:
        _client_cache.move_to_end(cache_key)
        return cached

    import boto3
    from botocore.config import Config

    preset = PROVIDERS.get(provider, PROVIDERS["Custom"])
    effective_region = region or preset.default_region

    # Resolve endpoint: explicit override > preset template
    if endpoint_url:
        effective_endpoint = endpoint_url
    elif preset.endpoint_template:
        effective_endpoint = preset.endpoint_template.format(
            region=effective_region,
            account_id=account_id,
        )
    else:
        effective_endpoint = ""

    kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": effective_region,
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if preset.force_path_style else "auto"},
            retries={"max_attempts": 3, "mode": "adaptive"},
            user_agent_extra="b2ai-comfyui",
        ),
    }
    if effective_endpoint:
        kwargs["endpoint_url"] = effective_endpoint

    client = boto3.client("s3", **kwargs)

    _client_cache[cache_key] = client
    if len(_client_cache) > _CLIENT_CACHE_MAX:
        _client_cache.popitem(last=False)
    return client


def encode_tags(tags) -> str:
    """Encode a tag dict as a URL-encoded string for the S3 Tagging header.

    AWS' Tagging format is `k1=v1&k2=v2` with values URL-encoded. boto3 accepts
    a pre-encoded string for `put_object(Tagging=...)` and
    `upload_fileobj(ExtraArgs={"Tagging": ...})`. Returns "" when tags is
    falsy so callers can skip the parameter cleanly.
    """
    if not tags:
        return ""
    from urllib.parse import urlencode
    return urlencode({str(k): str(v) for k, v in tags.items()})


def provider_supports_tagging(provider: str) -> bool:
    """Whether a provider implements S3 PutObjectTagging.

    Backblaze B2's S3-compatible API does not support object tagging at the
    time of writing; including a Tagging header on B2 will cause uploads to
    fail. Skipping silently with a debug log keeps cross-provider workflows
    portable. AWS, R2, Wasabi, DO Spaces, and GCS support it.
    """
    return provider != "Backblaze B2"
