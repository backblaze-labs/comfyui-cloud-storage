"""Save nodes - upload generated images, video, and audio to S3-compatible storage."""

import io as io_stdlib
import json
import logging
import uuid
from datetime import datetime

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_api.latest import io
import comfy.utils

from .nodes_profile import S3_PROFILE_TYPE
from .profile import apply_prefix, resolve_default_profile, validate_config
from .providers import create_s3_client, encode_tags, provider_supports_tagging

logger = logging.getLogger(__name__)

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}

AUDIO_MIME_TYPES = {
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
}


def _disable_metadata() -> bool:
    """Whether ComfyUI is configured to strip workflow metadata from saves.

    Reads `comfy.cli_args.args` lazily; degrades gracefully on forks that
    don't expose the flag.
    """
    try:
        from comfy.cli_args import args
    except ImportError:
        return False
    return getattr(args, "disable_metadata", False)


def _tensor_to_image_bytes(
    image_tensor,
    fmt="png",
    quality=95,
    prompt=None,
    extra_pnginfo=None,
) -> bytes:
    """Convert a single image tensor to bytes in the specified format."""
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    buf = io_stdlib.BytesIO()
    save_kwargs = {}

    if fmt == "png":
        metadata = None
        if not _disable_metadata():
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k in extra_pnginfo:
                    metadata.add_text(k, json.dumps(extra_pnginfo[k]))
        save_kwargs["pnginfo"] = metadata
        save_kwargs["compress_level"] = 4
        img.save(buf, format="PNG", **save_kwargs)
    elif fmt == "jpg":
        img.save(buf, format="JPEG", quality=quality)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=quality)

    return buf.getvalue()


def _expand_filename_tokens(filename: str, batch_idx: int) -> str:
    """Substitute %batch_num%, %date%, %time%, %uuid% in the user filename.

    %uuid% is generated once per call so multiple substitutions on the same
    line yield the same token. %date% and %time% use the local clock; if the
    user wants a single moment across a batch they should pre-compute it.
    """
    if "%uuid%" in filename:
        filename = filename.replace("%uuid%", uuid.uuid4().hex[:8])
    if "%date%" in filename or "%time%" in filename:
        now = datetime.now()
        filename = filename.replace("%date%", now.strftime("%Y-%m-%d"))
        filename = filename.replace("%time%", now.strftime("%H%M%S"))
    return filename.replace("%batch_num%", str(batch_idx))


def _build_key(config: dict, prefix: str, filename: str, batch_idx: int, ext: str) -> str:
    """Build the full S3 object key.

    Filename templates (%batch_num%, %date%, %time%, %uuid%) are expanded
    here. The `path_prefix` from the profile is applied via `apply_prefix`.
    """
    name = _expand_filename_tokens(filename, batch_idx)
    return apply_prefix(config, f"{prefix}{name}.{ext}")


def _put_object_kwargs(config: dict, body: bytes, content_type: str) -> dict:
    """Assemble put_object kwargs, optionally including object Tagging.

    Tagging is opt-in via `default_tags` in the profile and is skipped on
    providers that don't implement S3 PutObjectTagging (currently B2).
    """
    kwargs = {"Body": body, "ContentType": content_type}
    tags = config.get("default_tags") or {}
    if tags and provider_supports_tagging(config.get("provider", "")):
        kwargs["Tagging"] = encode_tags(tags)
    elif tags:
        logger.debug(
            "Skipping object tags on %s — provider does not support tagging.",
            config.get("provider"),
        )
    return kwargs


def _s3_error_message(e) -> str:
    """Extract a user-friendly message from a botocore ClientError."""
    from botocore.exceptions import ClientError
    if isinstance(e, ClientError):
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        if code == "NoSuchBucket":
            return f"Bucket not found: {msg}"
        if code in ("AccessDenied", "403"):
            return f"Access denied. Check credentials and bucket policy. ({msg})"
        if code == "InvalidAccessKeyId":
            return f"Invalid access key. ({msg})"
        return f"S3 error [{code}]: {msg}"
    return str(e)


class SaveImageToCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveImageToCloud",
            display_name="Save Image to Cloud",
            category="cloud_storage/save",
            description="Upload images to S3-compatible cloud storage (B2, S3, R2, etc.).",
            search_aliases=["upload image", "s3 save", "cloud save", "b2 save"],
            inputs=[
                io.Image.Input("images", tooltip="The images to upload."),
                io.String.Input(
                    "key_prefix",
                    default="comfyui/images/",
                    tooltip="S3 key prefix (folder path in bucket).",
                ),
                io.String.Input(
                    "filename",
                    default="ComfyUI_%batch_num%",
                    tooltip=(
                        "Filename template. Tokens: %batch_num%, %date% (YYYY-MM-DD), "
                        "%time% (HHMMSS), %uuid% (8-char)."
                    ),
                ),
                io.Combo.Input("format", options=["png", "jpg", "webp"], default="png"),
                io.Int.Input(
                    "quality",
                    default=95,
                    min=1,
                    max=100,
                    tooltip="JPEG/WebP quality (ignored for PNG).",
                ),
                io.Boolean.Input(
                    "presign_url",
                    default=False,
                    tooltip="If true, generate a presigned sharing URL for the first uploaded image.",
                    optional=True,
                ),
                io.Int.Input(
                    "expires_hours",
                    default=24,
                    min=1,
                    max=168,
                    tooltip="Expiration for the presigned URL in hours (max 7 days).",
                    optional=True,
                ),
                io.Custom(S3_PROFILE_TYPE).Input(
                    "profile",
                    optional=True,
                    tooltip="Cloud storage profile. Uses env vars if not connected.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="key"),
                io.String.Output(display_name="url"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        key_prefix="comfyui/images/",
        filename="ComfyUI_%batch_num%",
        format="png",
        quality=95,
        presign_url=False,
        expires_hours=24,
        profile=None,
    ) -> io.NodeOutput:
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        validate_config(config, mode="write")
        client = create_s3_client(**config)
        bucket = config["bucket"]

        uploaded = []
        keys = []
        for batch_idx, image_tensor in enumerate(images):
            img_bytes = _tensor_to_image_bytes(
                image_tensor,
                fmt=format,
                quality=quality,
                prompt=cls.hidden.prompt,
                extra_pnginfo=cls.hidden.extra_pnginfo,
            )
            key = _build_key(config, key_prefix, filename, batch_idx, format)
            content_type = MIME_TYPES.get(format, "application/octet-stream")

            try:
                client.put_object(
                    Bucket=bucket, Key=key,
                    **_put_object_kwargs(config, img_bytes, content_type),
                )
            except ClientError as e:
                raise ValueError(_s3_error_message(e)) from e

            uploaded.append(f"s3://{bucket}/{key}")
            keys.append(key)
            logger.info("Uploaded %s (%d bytes)", key, len(img_bytes))

        first_key = keys[0] if keys else ""
        url = ""
        if presign_url and first_key:
            try:
                url = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": first_key},
                    ExpiresIn=expires_hours * 3600,
                )
                uploaded.append(f"url: {url}")
            except ClientError as e:
                # Don't fail the upload over a presign issue — surface the
                # error in the UI text and return an empty url string so the
                # downstream graph behaves predictably.
                logger.warning("Presign failed: %s", e)
                uploaded.append(f"presign failed: {_s3_error_message(e)}")

        return io.NodeOutput(first_key, url, ui={"text": uploaded})


class SaveVideoToCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        from comfy_api.latest import Types
        return io.Schema(
            node_id="SaveVideoToCloud",
            display_name="Save Video to Cloud",
            category="cloud_storage/save",
            description="Upload video to S3-compatible cloud storage.",
            search_aliases=["upload video", "s3 video", "cloud video"],
            inputs=[
                io.Video.Input("video", tooltip="The video to upload."),
                io.String.Input("key_prefix", default="comfyui/videos/"),
                io.String.Input(
                    "filename", default="ComfyUI_video",
                    tooltip="Tokens: %batch_num%, %date%, %time%, %uuid%.",
                ),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto"),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto"),
                io.Custom(S3_PROFILE_TYPE).Input("profile", optional=True),
            ],
            outputs=[io.String.Output(display_name="key")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video, key_prefix, filename, format, codec, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError
        from comfy_api.latest import Types

        config = profile or resolve_default_profile()
        validate_config(config, mode="write")
        client = create_s3_client(**config)
        bucket = config["bucket"]

        buf = io_stdlib.BytesIO()
        saved_metadata = None
        if not _disable_metadata():
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if metadata:
                saved_metadata = metadata

        video.save_to(
            buf,
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata,
        )
        size = buf.tell()
        buf.seek(0)

        ext = Types.VideoContainer.get_extension(format)
        # batch_idx=0 since video is a single object; %batch_num% still substitutes for symmetry.
        key = _build_key(config, key_prefix, filename, 0, ext)

        # Progress bar for large video uploads (model uploads tend to be small;
        # video can be hundreds of MB).
        pbar = comfy.utils.ProgressBar(size) if size else None
        uploaded = 0

        def progress_callback(bytes_amount):
            nonlocal uploaded
            uploaded += bytes_amount
            if pbar is not None:
                pbar.update_absolute(uploaded, size)

        # upload_fileobj uses ExtraArgs for tagging/content-type instead of
        # the put_object kwargs format.
        extra_args = {}
        tags = config.get("default_tags") or {}
        if tags and provider_supports_tagging(config.get("provider", "")):
            extra_args["Tagging"] = encode_tags(tags)

        try:
            client.upload_fileobj(
                buf, bucket, key,
                ExtraArgs=extra_args or None,
                Callback=progress_callback,
            )
        except ClientError as e:
            raise ValueError(_s3_error_message(e)) from e

        logger.info("Uploaded video %s (%d bytes)", key, size)
        return io.NodeOutput(key, ui={"text": [f"s3://{bucket}/{key}"]})


class SaveAudioToCloud(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveAudioToCloud",
            display_name="Save Audio to Cloud",
            category="cloud_storage/save",
            description="Upload audio to S3-compatible cloud storage.",
            search_aliases=["upload audio", "s3 audio", "cloud audio"],
            inputs=[
                io.Audio.Input("audio", tooltip="The audio to upload."),
                io.String.Input("key_prefix", default="comfyui/audio/"),
                io.String.Input(
                    "filename", default="ComfyUI_audio",
                    tooltip="Tokens: %batch_num%, %date%, %time%, %uuid%.",
                ),
                io.Combo.Input("format", options=["flac", "mp3", "wav"], default="flac"),
                io.Custom(S3_PROFILE_TYPE).Input("profile", optional=True),
            ],
            outputs=[io.String.Output(display_name="key")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio, key_prefix, filename, format, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError
        import torchaudio

        config = profile or resolve_default_profile()
        validate_config(config, mode="write")
        client = create_s3_client(**config)
        bucket = config["bucket"]

        # audio is a dict with "waveform" (B, C, N) and "sample_rate" keys.
        # Iterate the batch dim so multi-clip audio batches save as N files,
        # mirroring SaveImageToCloud.
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        content_type = AUDIO_MIME_TYPES.get(format, "application/octet-stream")

        uploaded = []
        keys = []
        for batch_idx in range(waveform.shape[0]):
            buf = io_stdlib.BytesIO()
            torchaudio.save(buf, waveform[batch_idx], sample_rate, format=format)
            key = _build_key(config, key_prefix, filename, batch_idx, format)

            try:
                client.put_object(
                    Bucket=bucket, Key=key,
                    **_put_object_kwargs(config, buf.getvalue(), content_type),
                )
            except ClientError as e:
                raise ValueError(_s3_error_message(e)) from e

            uploaded.append(f"s3://{bucket}/{key}")
            keys.append(key)
            logger.info("Uploaded audio %s", key)

        return io.NodeOutput(keys[0] if keys else "", ui={"text": uploaded})
