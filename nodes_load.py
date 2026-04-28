"""Load nodes - download images, audio, and models from S3-compatible storage."""

import io as io_stdlib
import os
import logging
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from comfy_api.latest import io
import comfy.utils

from .nodes_profile import S3_PROFILE_TYPE
from .profile import apply_prefix, resolve_default_profile, validate_config
from .providers import create_s3_client


def _format_bytes(n: int) -> str:
    """Compact, human-readable byte size for status text."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def _format_duration(seconds: float) -> str:
    """Compact, human-readable duration for status text."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"

logger = logging.getLogger(__name__)


def _client_error_to_value_error(e, bucket: str, key: str) -> ValueError:
    """Translate a botocore ClientError into a user-friendly ValueError."""
    code = e.response["Error"]["Code"]
    if code in ("NoSuchKey", "404"):
        return ValueError(f"Object not found: s3://{bucket}/{key}")
    return ValueError(f"S3 error [{code}]: {e.response['Error']['Message']}")


def _fingerprint_remote_object(key, profile):
    """Shared fingerprint logic for cloud-load nodes.

    Returns the S3 ETag when reachable, else a deterministic sentinel keyed
    on the inputs. The sentinel keeps ComfyUI's cache stable per (bucket, key)
    on transient failures without forcing a redundant re-fetch on every run.
    """
    try:
        config = profile or resolve_default_profile()
        client = create_s3_client(**config)
        full_key = apply_prefix(config, key)
        resp = client.head_object(Bucket=config["bucket"], Key=full_key)
        etag = resp.get("ETag", "")
        if etag:
            return etag
        return f"noetag:{config.get('bucket', '')}:{full_key}"
    except Exception as e:
        logger.debug("fingerprint_inputs failed: %s", e)
        bucket = (profile or {}).get("bucket", "")
        return f"noetag:{bucket}:{key}"


class LoadImageFromCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadImageFromCloud",
            display_name="Load Image from Cloud",
            category="cloud_storage/load",
            description="Download an image from S3-compatible cloud storage into the pipeline.",
            search_aliases=["s3 image", "download image", "cloud image", "b2 image"],
            inputs=[
                io.String.Input(
                    "key",
                    default="",
                    tooltip="S3 object key, e.g. 'comfyui/images/photo.png'",
                ),
                io.Custom(S3_PROFILE_TYPE).Input(
                    "profile",
                    optional=True,
                    tooltip="Cloud storage profile. Uses env vars if not connected.",
                ),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def execute(cls, key, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        validate_config(config)
        client = create_s3_client(**config)
        bucket = config["bucket"]
        full_key = apply_prefix(config, key)

        try:
            response = client.get_object(Bucket=bucket, Key=full_key)
        except ClientError as e:
            raise _client_error_to_value_error(e, bucket, full_key) from e

        image_data = response["Body"].read()
        img = Image.open(io_stdlib.BytesIO(image_data))
        img = ImageOps.exif_transpose(img)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))

        image_rgb = img.convert("RGB")
        image_np = np.array(image_rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros(
                (image_np.shape[0], image_np.shape[1]),
                dtype=torch.float32,
            )

        return io.NodeOutput(image_tensor, mask.unsqueeze(0))

    @classmethod
    def fingerprint_inputs(cls, key, profile=None):
        return _fingerprint_remote_object(key, profile)


class LoadModelFromCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadModelFromCloud",
            display_name="Download Model from Cloud",
            category="cloud_storage/models",
            description=(
                "Download and cache a model file from S3-compatible cloud storage. "
                "Returns the local filename for use with standard loader nodes."
            ),
            search_aliases=["s3 model", "cloud model", "download checkpoint", "b2 model"],
            inputs=[
                io.Combo.Input(
                    "model_type",
                    options=[
                        "checkpoints", "loras", "vae", "text_encoders",
                        "controlnet", "diffusion_models", "upscale_models",
                        "embeddings", "clip_vision",
                    ],
                    default="checkpoints",
                    tooltip="Which model category to save to (determines local directory).",
                ),
                io.String.Input(
                    "key",
                    default="",
                    tooltip="S3 object key, e.g. 'models/sd_xl_base_1.0.safetensors'",
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    tooltip="Re-download even if cached locally.",
                ),
                io.Custom(S3_PROFILE_TYPE).Input("profile", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="model_filename"),
            ],
        )

    @classmethod
    def execute(cls, model_type, key, force_redownload=False, profile=None) -> io.NodeOutput:
        import folder_paths
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        validate_config(config)
        client = create_s3_client(**config)
        bucket = config["bucket"]
        full_key = apply_prefix(config, key)

        model_paths = folder_paths.get_folder_paths(model_type)
        if not model_paths:
            raise ValueError(f"No directory configured for model type: {model_type}")
        local_dir = model_paths[0]
        filename = os.path.basename(full_key)
        local_path = os.path.join(local_dir, filename)
        etag_path = local_path + ".s3etag"

        # Cache hit path: compare cached ETag against remote.
        if os.path.exists(local_path) and not force_redownload:
            try:
                remote_head = client.head_object(Bucket=bucket, Key=full_key)
                remote_etag = remote_head.get("ETag", "")
                if os.path.exists(etag_path):
                    with open(etag_path, "r", encoding="utf-8") as f:
                        cached_etag = f.read().strip()
                    if cached_etag == remote_etag:
                        logger.info("Model cached: %s", local_path)
                        return io.NodeOutput(
                            filename,
                            ui={"text": [f"cached: {filename}"]},
                        )
            except ClientError as e:
                # Network/auth blip but local copy exists — use it and warn.
                logger.warning(
                    "Could not verify cached model against remote (%s); using local copy.", e,
                )
                return io.NodeOutput(
                    filename,
                    ui={"text": [f"cached (unverified): {filename}"]},
                )

        try:
            head = client.head_object(Bucket=bucket, Key=full_key)
        except ClientError as e:
            raise _client_error_to_value_error(e, bucket, full_key) from e

        file_size = head["ContentLength"]
        remote_etag = head.get("ETag", "")

        logger.info(
            "Downloading %s (%.2f GB) from s3://%s/%s",
            filename, file_size / (1024**3), bucket, full_key,
        )

        os.makedirs(local_dir, exist_ok=True)
        temp_path = local_path + ".download"

        pbar = comfy.utils.ProgressBar(file_size)
        downloaded = 0
        start = time.monotonic()

        def progress_callback(bytes_amount):
            nonlocal downloaded
            downloaded += bytes_amount
            pbar.update_absolute(downloaded, file_size)

        try:
            client.download_file(
                bucket, full_key, temp_path,
                Callback=progress_callback,
            )
            os.replace(temp_path, local_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        elapsed = time.monotonic() - start

        # Atomic ETag write: write to a tmp file then rename, so a crash
        # mid-write can't leave a half-written sentinel beside a complete model.
        if remote_etag:
            etag_tmp = etag_path + ".tmp"
            with open(etag_tmp, "w", encoding="utf-8") as f:
                f.write(remote_etag)
            os.replace(etag_tmp, etag_path)

        status = (
            f"downloaded: {filename} ({_format_bytes(file_size)} "
            f"in {_format_duration(elapsed)})"
        )
        logger.info("Model downloaded to: %s", local_path)
        return io.NodeOutput(filename, ui={"text": [status]})


class LoadAudioFromCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadAudioFromCloud",
            display_name="Load Audio from Cloud",
            category="cloud_storage/load",
            description="Download an audio file from S3-compatible cloud storage into the pipeline.",
            search_aliases=["s3 audio", "download audio", "cloud audio", "b2 audio"],
            inputs=[
                io.String.Input(
                    "key",
                    default="",
                    tooltip="S3 object key, e.g. 'comfyui/audio/clip.flac'",
                ),
                io.Custom(S3_PROFILE_TYPE).Input(
                    "profile",
                    optional=True,
                    tooltip="Cloud storage profile. Uses env vars if not connected.",
                ),
            ],
            outputs=[
                io.Audio.Output(),
            ],
        )

    @classmethod
    def execute(cls, key, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError
        import torchaudio

        config = profile or resolve_default_profile()
        validate_config(config)
        client = create_s3_client(**config)
        bucket = config["bucket"]
        full_key = apply_prefix(config, key)

        try:
            response = client.get_object(Bucket=bucket, Key=full_key)
        except ClientError as e:
            raise _client_error_to_value_error(e, bucket, full_key) from e

        # torchaudio.load accepts a file-like object; BytesIO keeps the
        # download fully in memory which is fine for clips. Add a (B=1) axis
        # so the output matches ComfyUI's audio tensor convention.
        buf = io_stdlib.BytesIO(response["Body"].read())
        waveform, sample_rate = torchaudio.load(buf)
        return io.NodeOutput({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})

    @classmethod
    def fingerprint_inputs(cls, key, profile=None):
        return _fingerprint_remote_object(key, profile)
