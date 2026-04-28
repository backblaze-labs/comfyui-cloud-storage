# Example workflows

Drag any of these `.json` files into the ComfyUI canvas to load the workflow.
Each example is a minimal, self-contained graph that demonstrates one common
pattern. Update the bucket name, key, and (when present) the profile widget to
match your setup before queueing.

| File | What it shows |
|------|---------------|
| `01_b2_save_image.json` | Save a generated image to a Backblaze B2 bucket via env-var credentials. |
| `02_test_connection.json` | Verify your profile is configured correctly before doing anything else. |
| `03_save_and_share.json` | Save an image and emit a presigned URL in one node. |
| `04_load_model_from_cloud.json` | Download a checkpoint from cloud storage with local ETag-based caching. |

## Before running

1. Install the package and `boto3` in ComfyUI's Python environment.
2. Set credentials — environment variables are easiest for first-time use:
   ```bash
   export COMFY_S3_PROVIDER="Backblaze B2"
   export COMFY_S3_ACCESS_KEY="<your-key-id>"
   export COMFY_S3_SECRET_KEY="<your-application-key>"
   export COMFY_S3_BUCKET="<your-bucket>"
   export COMFY_S3_REGION="us-west-004"
   ```
3. Restart ComfyUI so it picks up the env vars.
4. Run `02_test_connection.json` first — if that doesn't return `OK`, fix the
   reported error before trying the others.

## Notes on the examples

- These are **graph fragments**, not full image-generation pipelines. They
  assume you wire them into your usual checkpoint/sampler/decode chain.
- The `key` and `key_prefix` defaults use placeholder paths. Edit them to
  match your bucket layout.
- All examples leave the `profile` input unconnected, which means they fall
  back to environment variables. Add a `Cloud Storage Profile` node and wire
  it in if you use `profiles.json` instead.
