"""Tests for node logic - save, load, browse."""

import io
import os
import sys
import tempfile
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image


def _make_image_tensor(width=64, height=64, batch=1):
    """Create a dummy image tensor matching ComfyUI format: (B, H, W, 3) float32 [0,1]."""
    return torch.rand(batch, height, width, 3, dtype=torch.float32)


class TestTensorToImageBytes:
    def test_png_output(self):
        from comfyui_cloud_storage.nodes_save import _tensor_to_image_bytes
        tensor = _make_image_tensor()[0]  # single image
        with patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            data = _tensor_to_image_bytes(tensor, fmt="png")
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_jpg_output(self):
        from comfyui_cloud_storage.nodes_save import _tensor_to_image_bytes
        tensor = _make_image_tensor()[0]
        with patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            data = _tensor_to_image_bytes(tensor, fmt="jpg", quality=80)
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_webp_output(self):
        from comfyui_cloud_storage.nodes_save import _tensor_to_image_bytes
        tensor = _make_image_tensor()[0]
        with patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            data = _tensor_to_image_bytes(tensor, fmt="webp")
        assert data[:4] == b"RIFF"  # WebP magic bytes


class TestBuildKey:
    def test_basic_key(self):
        from comfyui_cloud_storage.nodes_save import _build_key
        config = {"path_prefix": ""}
        key = _build_key(config, "images/", "test_%batch_num%", 0, "png")
        assert key == "images/test_0.png"

    def test_with_path_prefix(self):
        from comfyui_cloud_storage.nodes_save import _build_key
        config = {"path_prefix": "myproject/"}
        key = _build_key(config, "images/", "test_%batch_num%", 2, "jpg")
        assert key == "myproject/images/test_2.jpg"

    def test_batch_num_substitution(self):
        from comfyui_cloud_storage.nodes_save import _build_key
        config = {"path_prefix": ""}
        key = _build_key(config, "", "img_%batch_num%_%batch_num%", 5, "png")
        assert key == "img_5_5.png"

    def test_strips_leading_slash_from_prefix(self):
        # apply_prefix is the source of truth; verify _build_key inherits it.
        from comfyui_cloud_storage.nodes_save import _build_key
        config = {"path_prefix": "outputs/"}
        key = _build_key(config, "/images/", "test", 0, "png")
        assert key == "outputs/images/test.png"

    def test_video_audio_get_batch_num_substitution(self):
        # Regression: prior versions left %batch_num% literal in video/audio keys.
        from comfyui_cloud_storage.nodes_save import _build_key
        config = {"path_prefix": ""}
        key = _build_key(config, "videos/", "clip_%batch_num%", 3, "mp4")
        assert key == "videos/clip_3.mp4"


class TestS3ErrorMessage:
    def test_no_such_bucket(self):
        from comfyui_cloud_storage.nodes_save import _s3_error_message
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "mybucket"}},
            "PutObject",
        )
        msg = _s3_error_message(err)
        assert "Bucket not found" in msg

    def test_access_denied(self):
        from comfyui_cloud_storage.nodes_save import _s3_error_message
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "PutObject",
        )
        msg = _s3_error_message(err)
        assert "Access denied" in msg

    def test_generic_error(self):
        from comfyui_cloud_storage.nodes_save import _s3_error_message
        from botocore.exceptions import ClientError
        err = ClientError(
            {"Error": {"Code": "InternalError", "Message": "oops"}},
            "PutObject",
        )
        msg = _s3_error_message(err)
        assert "InternalError" in msg
        assert "oops" in msg

    def test_non_client_error(self):
        from comfyui_cloud_storage.nodes_save import _s3_error_message
        msg = _s3_error_message(RuntimeError("something"))
        assert "something" in msg


class TestFingerprintFallback:
    """fingerprint_inputs must return a stable, input-derived sentinel on failure
    rather than an empty string (silent cache hit) or a timestamp (forced miss)."""

    def test_returns_sentinel_when_head_fails(self):
        from comfyui_cloud_storage.nodes_load import _fingerprint_remote_object

        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   side_effect=RuntimeError("network down")):
            fp = _fingerprint_remote_object("foo.png", {"bucket": "b"})
        assert fp.startswith("noetag:")
        assert "b" in fp
        assert "foo.png" in fp

    def test_returns_etag_on_success(self):
        from comfyui_cloud_storage.nodes_load import _fingerprint_remote_object

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ETag": '"abcdef"'}
        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   return_value=mock_client):
            fp = _fingerprint_remote_object(
                "foo.png",
                {"bucket": "b", "access_key": "x", "secret_key": "y"},
            )
        assert fp == '"abcdef"'

    def test_sentinel_is_stable_across_calls(self):
        # Same inputs -> same fingerprint, otherwise ComfyUI's cache would invalidate
        # on every queue when the network is flaky.
        from comfyui_cloud_storage.nodes_load import _fingerprint_remote_object

        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   side_effect=RuntimeError("x")):
            a = _fingerprint_remote_object("k", {"bucket": "b"})
            b = _fingerprint_remote_object("k", {"bucket": "b"})
        assert a == b


class TestAtomicEtagWrite:
    """LoadModelFromCloud writes .s3etag via tmp + os.replace so a crash mid-write
    can never leave a corrupt sentinel beside a complete model file."""

    def test_etag_written_atomically(self, tmp_path):
        from comfyui_cloud_storage.nodes_load import LoadModelFromCloud

        local_dir = tmp_path / "checkpoints"
        local_dir.mkdir()

        mock_client = MagicMock()
        mock_client.head_object.return_value = {
            "ContentLength": 4, "ETag": '"abc123"',
        }

        def fake_download(bucket, key, path, Callback=None):
            with open(path, "wb") as f:
                f.write(b"data")

        mock_client.download_file.side_effect = fake_download

        mock_fp = sys.modules["folder_paths"]
        mock_fp.get_folder_paths.return_value = [str(local_dir)]

        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_load.validate_config"), \
             patch("comfyui_cloud_storage.nodes_load.os.replace",
                   wraps=os.replace) as mock_replace:
            LoadModelFromCloud.execute(
                model_type="checkpoints", key="model.safetensors",
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        # Both renames happen via os.replace: the .download -> final, then .s3etag.tmp -> .s3etag
        assert mock_replace.call_count >= 2
        etag_path = local_dir / "model.safetensors.s3etag"
        assert etag_path.exists()
        assert etag_path.read_text(encoding="utf-8") == '"abc123"'


class TestExpandFilenameTokens:
    def test_batch_num(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        assert _expand_filename_tokens("img_%batch_num%", 3) == "img_3"

    def test_date_format(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        out = _expand_filename_tokens("%date%", 0)
        # YYYY-MM-DD
        assert len(out) == 10 and out[4] == "-" and out[7] == "-"

    def test_time_format(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        out = _expand_filename_tokens("%time%", 0)
        # HHMMSS, exactly 6 digits
        assert len(out) == 6 and out.isdigit()

    def test_uuid_8_chars(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        out = _expand_filename_tokens("%uuid%", 0)
        assert len(out) == 8
        # hex chars only
        int(out, 16)

    def test_uuid_stable_within_one_call(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        out = _expand_filename_tokens("%uuid%_%uuid%", 0)
        a, b = out.split("_")
        assert a == b

    def test_combined_template(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        out = _expand_filename_tokens("%date%/img_%batch_num%_%uuid%", 7)
        # Must contain literal "img_7_" and a uuid suffix
        assert "img_7_" in out

    def test_unknown_token_left_alone(self):
        from comfyui_cloud_storage.nodes_save import _expand_filename_tokens
        # %workflow_id% intentionally not implemented; must pass through.
        out = _expand_filename_tokens("x_%workflow_id%", 0)
        assert out == "x_%workflow_id%"


class TestPutObjectKwargsTagging:
    def test_no_tags_no_tagging_kwarg(self):
        from comfyui_cloud_storage.nodes_save import _put_object_kwargs
        kw = _put_object_kwargs(
            {"provider": "AWS S3", "default_tags": {}}, b"data", "image/png",
        )
        assert "Tagging" not in kw
        assert kw["Body"] == b"data"

    def test_aws_with_tags_includes_tagging(self):
        from comfyui_cloud_storage.nodes_save import _put_object_kwargs
        kw = _put_object_kwargs(
            {"provider": "AWS S3", "default_tags": {"env": "prod"}},
            b"data", "image/png",
        )
        assert kw["Tagging"] == "env=prod"

    def test_b2_strips_tagging(self):
        # B2 does not implement S3 tagging — the helper must skip silently.
        from comfyui_cloud_storage.nodes_save import _put_object_kwargs
        kw = _put_object_kwargs(
            {"provider": "Backblaze B2", "default_tags": {"env": "prod"}},
            b"data", "image/png",
        )
        assert "Tagging" not in kw


class TestSaveImagePresign:
    """SaveImageToCloud should generate a presigned URL only when toggled on."""

    def _common_mocks(self):
        mock_client = MagicMock()
        mock_client.put_object.return_value = {}
        mock_client.generate_presigned_url.return_value = "https://signed.example/img.png"
        return mock_client

    def test_presign_disabled_returns_empty_url(self):
        from comfyui_cloud_storage.nodes_save import SaveImageToCloud
        mock_client = self._common_mocks()

        with patch("comfyui_cloud_storage.nodes_save.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_save.validate_config"), \
             patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            SaveImageToCloud.hidden = MagicMock(prompt=None, extra_pnginfo=None)
            result = SaveImageToCloud.execute(
                images=torch.rand(1, 32, 32, 3),
                key_prefix="x/", filename="f", format="png", quality=95,
                presign_url=False,
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        # NodeOutput conftest mock: result["result"] = (key, url)
        assert result["result"][1] == ""
        mock_client.generate_presigned_url.assert_not_called()

    def test_presign_enabled_returns_url(self):
        from comfyui_cloud_storage.nodes_save import SaveImageToCloud
        mock_client = self._common_mocks()

        with patch("comfyui_cloud_storage.nodes_save.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_save.validate_config"), \
             patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            SaveImageToCloud.hidden = MagicMock(prompt=None, extra_pnginfo=None)
            result = SaveImageToCloud.execute(
                images=torch.rand(1, 32, 32, 3),
                key_prefix="x/", filename="f", format="png", quality=95,
                presign_url=True, expires_hours=12,
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        assert result["result"][1] == "https://signed.example/img.png"
        # ExpiresIn = expires_hours * 3600
        mock_client.generate_presigned_url.assert_called_once()
        assert mock_client.generate_presigned_url.call_args.kwargs["ExpiresIn"] == 12 * 3600

    def test_returns_first_key_for_batch(self):
        # For batch>1, the STRING output is the first uploaded key only —
        # newline-joined would break downstream nodes that expect a single key.
        from comfyui_cloud_storage.nodes_save import SaveImageToCloud
        mock_client = self._common_mocks()

        with patch("comfyui_cloud_storage.nodes_save.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_save.validate_config"), \
             patch("comfyui_cloud_storage.nodes_save._disable_metadata", return_value=True):
            SaveImageToCloud.hidden = MagicMock(prompt=None, extra_pnginfo=None)
            result = SaveImageToCloud.execute(
                images=torch.rand(3, 32, 32, 3),
                key_prefix="x/", filename="img_%batch_num%", format="png", quality=95,
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        first_key = result["result"][0]
        assert first_key.endswith("img_0.png")
        assert "\n" not in first_key


class TestModelLoadUIText:
    """LoadModelFromCloud emits a UI text status describing cached vs downloaded."""

    def test_cached_branch_reports_cached(self, tmp_path):
        from comfyui_cloud_storage.nodes_load import LoadModelFromCloud

        local_dir = tmp_path / "checkpoints"
        local_dir.mkdir()
        model = local_dir / "model.safetensors"
        model.write_bytes(b"existing")
        etag_file = local_dir / "model.safetensors.s3etag"
        etag_file.write_text('"abc"', encoding="utf-8")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ETag": '"abc"', "ContentLength": 8}

        mock_fp = sys.modules["folder_paths"]
        mock_fp.get_folder_paths.return_value = [str(local_dir)]

        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_load.validate_config"):
            result = LoadModelFromCloud.execute(
                model_type="checkpoints", key="model.safetensors",
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        assert result["ui"] is not None
        assert "cached" in result["ui"]["text"][0]

    def test_downloaded_branch_reports_size_and_time(self, tmp_path):
        from comfyui_cloud_storage.nodes_load import LoadModelFromCloud

        local_dir = tmp_path / "checkpoints"
        local_dir.mkdir()

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ETag": '"new"', "ContentLength": 100}

        def fake_download(bucket, key, path, Callback=None):
            with open(path, "wb") as f:
                f.write(b"x" * 100)

        mock_client.download_file.side_effect = fake_download

        mock_fp = sys.modules["folder_paths"]
        mock_fp.get_folder_paths.return_value = [str(local_dir)]

        with patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_load.validate_config"):
            result = LoadModelFromCloud.execute(
                model_type="checkpoints", key="new.safetensors",
                profile={"access_key": "a", "secret_key": "b", "bucket": "x", "path_prefix": ""},
            )

        text = result["ui"]["text"][0]
        assert "downloaded" in text
        assert "100" in text  # the byte count


class TestTestCloudConnection:
    def test_returns_ok_with_keys(self):
        from comfyui_cloud_storage.nodes_browse import TestCloudConnection

        mock_client = MagicMock()
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "a.png"}, {"Key": "b.png"}]
        }

        with patch("comfyui_cloud_storage.nodes_browse.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_browse.validate_config"):
            result = TestCloudConnection.execute(
                profile={"access_key": "a", "secret_key": "b", "bucket": "x"},
            )

        report = result["result"][0]
        assert report.startswith("OK")
        assert "a.png" in report
        assert "b.png" in report

    def test_head_bucket_failure_raises(self):
        from comfyui_cloud_storage.nodes_browse import TestCloudConnection
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no perms"}},
            "HeadBucket",
        )

        with patch("comfyui_cloud_storage.nodes_browse.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_browse.validate_config"):
            with pytest.raises(ValueError, match="AccessDenied"):
                TestCloudConnection.execute(
                    profile={"access_key": "a", "secret_key": "b", "bucket": "x"},
                )

    def test_list_failure_keeps_report_alive(self):
        # If head succeeds but list fails (e.g. limited IAM), still return OK.
        from comfyui_cloud_storage.nodes_browse import TestCloudConnection
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_client.head_bucket.return_value = {}
        mock_client.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no list"}},
            "ListObjectsV2",
        )

        with patch("comfyui_cloud_storage.nodes_browse.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_browse.validate_config"):
            result = TestCloudConnection.execute(
                profile={"access_key": "a", "secret_key": "b", "bucket": "x"},
            )

        report = result["result"][0]
        assert report.startswith("OK")
        assert "list permission" in report or "empty" in report


class TestLoadAudioFromCloud:
    def test_returns_audio_dict(self):
        from comfyui_cloud_storage.nodes_load import LoadAudioFromCloud

        # Fake torchaudio.load returning a 2-channel waveform at 44.1 kHz.
        fake_torchaudio = MagicMock()
        fake_torchaudio.load.return_value = (torch.zeros(2, 16000), 44100)

        mock_client = MagicMock()
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = b"\x00" * 1024
        mock_client.get_object.return_value = mock_response

        with patch.dict(sys.modules, {"torchaudio": fake_torchaudio}), \
             patch("comfyui_cloud_storage.nodes_load.create_s3_client",
                   return_value=mock_client), \
             patch("comfyui_cloud_storage.nodes_load.validate_config"):
            result = LoadAudioFromCloud.execute(
                key="clip.flac",
                profile={"access_key": "a", "secret_key": "b", "bucket": "x"},
            )

        # NodeOutput conftest mock: result["result"] is a tuple of positional args
        audio = result["result"][0]
        assert "waveform" in audio
        assert "sample_rate" in audio
        assert audio["sample_rate"] == 44100
        # Batch dim added: shape (1, 2, 16000)
        assert audio["waveform"].shape == (1, 2, 16000)
