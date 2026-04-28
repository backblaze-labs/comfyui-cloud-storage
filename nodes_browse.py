"""Browse nodes - list bucket contents, generate presigned URLs, test connection."""

import logging
import time

from comfy_api.latest import io

from .nodes_profile import S3_PROFILE_TYPE
from .profile import apply_prefix, resolve_default_profile, validate_config
from .providers import create_s3_client

logger = logging.getLogger(__name__)


class TestCloudConnection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TestCloudConnection",
            display_name="Test Cloud Connection",
            category="cloud_storage",
            description=(
                "Diagnose your cloud storage profile: HEADs the bucket and lists "
                "up to 3 keys. Use this before queuing real workflows to surface "
                "credential or bucket misconfigurations early."
            ),
            search_aliases=["s3 test", "cloud test", "diagnose", "ping bucket"],
            inputs=[
                io.Custom(S3_PROFILE_TYPE).Input(
                    "profile",
                    optional=True,
                    tooltip="Cloud storage profile. Uses env vars if not connected.",
                ),
            ],
            outputs=[io.String.Output(display_name="report")],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        # mode="read" — diagnostic, never writes.
        validate_config(config, mode="read")
        client = create_s3_client(**config)
        bucket = config["bucket"]

        start = time.monotonic()
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ValueError(
                f"Cannot reach bucket s3://{bucket} on {config.get('provider', '?')} "
                f"[{code}]: {e.response['Error']['Message']}"
            ) from e

        # List a few keys to confirm the access key has list permission.
        listed = []
        try:
            resp = client.list_objects_v2(Bucket=bucket, MaxKeys=3)
            for obj in resp.get("Contents", []):
                listed.append(obj["Key"])
        except ClientError as e:
            # head_bucket succeeded but list failed — keep going and report it.
            logger.warning("list_objects_v2 failed during connection test: %s", e)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        report_lines = [
            f"OK · s3://{bucket} · {config.get('provider', '?')} · {elapsed_ms} ms",
        ]
        if listed:
            report_lines.append("First keys:")
            report_lines.extend(f"  {k}" for k in listed)
        else:
            report_lines.append("(bucket is empty or list permission missing)")
        return io.NodeOutput("\n".join(report_lines))


class ListBucket(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ListBucketContents",
            display_name="List Bucket Contents",
            category="cloud_storage/browse",
            description="List files in an S3-compatible bucket.",
            search_aliases=["s3 list", "bucket browse", "cloud files"],
            inputs=[
                io.String.Input(
                    "prefix",
                    default="",
                    tooltip="Filter results to keys starting with this prefix.",
                ),
                io.Int.Input(
                    "max_results",
                    default=100,
                    min=1,
                    max=1000,
                ),
                io.Custom(S3_PROFILE_TYPE).Input("profile", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="file_list"),
            ],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, prefix="", max_results=100, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        validate_config(config)
        client = create_s3_client(**config)
        bucket = config["bucket"]

        full_prefix = apply_prefix(config, prefix)

        try:
            paginator = client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(
                Bucket=bucket,
                Prefix=full_prefix,
                PaginationConfig={"MaxItems": max_results},
            ):
                for obj in page.get("Contents", []):
                    size_mb = obj["Size"] / (1024 * 1024)
                    keys.append(f"{obj['Key']}  ({size_mb:.1f} MB)")
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ValueError(f"S3 error [{code}]: {e.response['Error']['Message']}") from e

        return io.NodeOutput("\n".join(keys))


class GeneratePresignedURL(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GeneratePresignedURL",
            display_name="Generate Sharing URL",
            category="cloud_storage/browse",
            description="Generate a time-limited presigned URL for sharing a cloud storage file.",
            search_aliases=["s3 url", "share link", "presigned"],
            inputs=[
                io.String.Input(
                    "key",
                    default="",
                    tooltip="S3 object key to generate a URL for.",
                ),
                io.Int.Input(
                    "expires_hours",
                    default=24,
                    min=1,
                    max=168,
                    tooltip="URL expiration time in hours (max 7 days).",
                ),
                io.Custom(S3_PROFILE_TYPE).Input("profile", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="url"),
            ],
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, key, expires_hours=24, profile=None) -> io.NodeOutput:
        from botocore.exceptions import ClientError

        config = profile or resolve_default_profile()
        validate_config(config)
        client = create_s3_client(**config)
        bucket = config["bucket"]

        full_key = apply_prefix(config, key)

        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": full_key},
                ExpiresIn=expires_hours * 3600,
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            raise ValueError(f"S3 error [{code}]: {e.response['Error']['Message']}") from e

        return io.NodeOutput(url)
