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
