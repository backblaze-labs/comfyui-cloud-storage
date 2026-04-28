"""Tests for providers.py - provider presets and S3 client factory."""

import sys
import pytest
from unittest.mock import patch, MagicMock


class TestProviderPresets:
    def test_all_providers_have_required_fields(self):
        from comfyui_cloud_storage.providers import PROVIDERS
        for name, preset in PROVIDERS.items():
            assert hasattr(preset, "endpoint_template"), f"{name} missing endpoint_template"
            assert hasattr(preset, "default_region"), f"{name} missing default_region"
            assert hasattr(preset, "force_path_style"), f"{name} missing force_path_style"

    def test_provider_names_matches_keys(self):
        from comfyui_cloud_storage.providers import PROVIDERS, PROVIDER_NAMES
        assert PROVIDER_NAMES == list(PROVIDERS.keys())

    def test_b2_endpoint_template(self):
        from comfyui_cloud_storage.providers import PROVIDERS
        b2 = PROVIDERS["Backblaze B2"]
        endpoint = b2.endpoint_template.format(region="us-west-004", account_id="")
        assert endpoint == "https://s3.us-west-004.backblazeb2.com"

    def test_r2_endpoint_template(self):
        from comfyui_cloud_storage.providers import PROVIDERS
        r2 = PROVIDERS["Cloudflare R2"]
        endpoint = r2.endpoint_template.format(region="auto", account_id="abc123")
        assert endpoint == "https://abc123.r2.cloudflarestorage.com"

    def test_aws_has_empty_endpoint(self):
        from comfyui_cloud_storage.providers import PROVIDERS
        assert PROVIDERS["AWS S3"].endpoint_template == ""

    def test_all_providers_use_auto_style(self):
        # All shipped providers use virtual-host-style addressing. The
        # `force_path_style` field on ProviderPreset stays available so
        # custom-endpoint users can still opt in via a fork or PR.
        from comfyui_cloud_storage.providers import PROVIDERS
        for name, preset in PROVIDERS.items():
            assert preset.force_path_style is False, f"{name} should not use path style"


class TestCreateS3Client:
    def _call_with_mock_boto3(self, **kwargs):
        """Call create_s3_client with a mocked boto3 and return the mock + call args."""
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            # Need to re-import to pick up the mock
            from comfyui_cloud_storage.providers import create_s3_client
            create_s3_client(**kwargs)
        return mock_boto3

    def test_aws_no_endpoint_url(self):
        mock_boto3 = self._call_with_mock_boto3(
            provider="AWS S3", access_key="AKID", secret_key="SECRET",
        )
        call_kwargs = mock_boto3.client.call_args
        assert "endpoint_url" not in call_kwargs.kwargs

    def test_b2_sets_endpoint(self):
        mock_boto3 = self._call_with_mock_boto3(
            provider="Backblaze B2", access_key="AKID", secret_key="SECRET",
            region="eu-central-003",
        )
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "https://s3.eu-central-003.backblazeb2.com"

    def test_custom_endpoint_overrides_preset(self):
        mock_boto3 = self._call_with_mock_boto3(
            provider="Backblaze B2", access_key="AKID", secret_key="SECRET",
            endpoint_url="https://custom.example.com",
        )
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "https://custom.example.com"

    def test_credentials_passed_through(self):
        mock_boto3 = self._call_with_mock_boto3(
            provider="AWS S3", access_key="mykey", secret_key="mysecret",
            region="us-west-2",
        )
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["aws_access_key_id"] == "mykey"
        assert call_kwargs.kwargs["aws_secret_access_key"] == "mysecret"
        assert call_kwargs.kwargs["region_name"] == "us-west-2"

    def test_unknown_provider_uses_custom(self):
        mock_boto3 = self._call_with_mock_boto3(
            provider="SomeUnknown", access_key="AKID", secret_key="SECRET",
            endpoint_url="https://unknown.example.com",
        )
        call_kwargs = mock_boto3.client.call_args
        assert call_kwargs.kwargs["endpoint_url"] == "https://unknown.example.com"

    def test_user_agent_set(self):
        from botocore.config import Config
        mock_boto3 = self._call_with_mock_boto3(
            provider="Backblaze B2", access_key="AKID", secret_key="SECRET",
        )
        config = mock_boto3.client.call_args.kwargs["config"]
        assert isinstance(config, Config)
        # RFC 7231 product token: `b2ai-comfyui/<version>` when the package
        # is installed, falling back to the bare name from source loads.
        ua = config.user_agent_extra
        assert ua == "b2ai-comfyui" or ua.startswith("b2ai-comfyui/")

    def test_extra_kwargs_ignored(self):
        # Splatting a full profile dict (with bucket/path_prefix/default_tags)
        # must not raise — the factory only consumes credential-relevant fields.
        mock_boto3 = self._call_with_mock_boto3(
            provider="AWS S3", access_key="AKID", secret_key="SECRET",
            bucket="ignored", path_prefix="also-ignored", default_tags={"k": "v"},
            read_only=True,
        )
        assert mock_boto3.client.called

class TestClientCache:
    def test_identical_calls_return_same_client(self):
        from unittest.mock import patch
        import sys
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from comfyui_cloud_storage.providers import (
                create_s3_client, clear_client_cache,
            )
            clear_client_cache()
            a = create_s3_client(provider="AWS S3", access_key="K", secret_key="S")
            b = create_s3_client(provider="AWS S3", access_key="K", secret_key="S")
        assert a is b
        assert mock_boto3.client.call_count == 1

    def test_different_creds_get_different_clients(self):
        from unittest.mock import patch
        import sys
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from comfyui_cloud_storage.providers import (
                create_s3_client, clear_client_cache,
            )
            clear_client_cache()
            create_s3_client(provider="AWS S3", access_key="K1", secret_key="S")
            create_s3_client(provider="AWS S3", access_key="K2", secret_key="S")
        assert mock_boto3.client.call_count == 2

    def test_cache_evicts_oldest_when_full(self):
        # Add 17 distinct clients; cache cap is 16, so the oldest should be gone.
        from unittest.mock import patch
        import sys
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            from comfyui_cloud_storage.providers import (
                create_s3_client, clear_client_cache, _client_cache,
            )
            clear_client_cache()
            for i in range(17):
                create_s3_client(
                    provider="AWS S3", access_key=f"K{i}", secret_key="S",
                )
            assert len(_client_cache) == 16


class TestUserAgentSuffix:
    """The custom suffix should be `b2ai-comfyui/<version>` when the package
    is installed and resolvable, falling back to the bare name otherwise."""

    def test_versioned_when_installed(self):
        from unittest.mock import patch
        from comfyui_cloud_storage import providers
        with patch.object(providers, "version", return_value="9.9.9"):
            assert providers._user_agent_suffix() == "b2ai-comfyui/9.9.9"

    def test_falls_back_when_not_installed(self):
        from unittest.mock import patch
        from importlib.metadata import PackageNotFoundError
        from comfyui_cloud_storage import providers
        with patch.object(providers, "version", side_effect=PackageNotFoundError):
            assert providers._user_agent_suffix() == "b2ai-comfyui"


class TestEncodeTags:
    def test_empty_returns_empty_string(self):
        from comfyui_cloud_storage.providers import encode_tags
        assert encode_tags({}) == ""
        assert encode_tags(None) == ""

    def test_single_tag(self):
        from comfyui_cloud_storage.providers import encode_tags
        assert encode_tags({"env": "prod"}) == "env=prod"

    def test_special_chars_url_encoded(self):
        from comfyui_cloud_storage.providers import encode_tags
        # `=` and `&` in values must be percent-encoded so they don't break
        # the Tagging header parser.
        out = encode_tags({"k": "a=b&c"})
        assert "%3D" in out or "=" not in out.split("=", 1)[1]
        assert "%26" in out or "&" not in out.split("=", 1)[1]


class TestProviderTaggingSupport:
    def test_b2_excluded(self):
        from comfyui_cloud_storage.providers import provider_supports_tagging
        assert provider_supports_tagging("Backblaze B2") is False

    def test_other_providers_supported(self):
        from comfyui_cloud_storage.providers import provider_supports_tagging
        for p in ["AWS S3", "Cloudflare R2", "Wasabi", "DigitalOcean Spaces", "Custom"]:
            assert provider_supports_tagging(p) is True
