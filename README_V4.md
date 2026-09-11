# Rosely Wan 2.2 V4 — matched Enhanced FP8 pair

This patch replaces the old v3 hybrid model stack with the matched four-file FP8 bundle:

- `models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8H.safetensors`
  - SHA256 `96a4603ac80992b33713ae279ee833325a83b83181b97cd2946beafb73175374`
- `models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8L.safetensors`
  - SHA256 `fa74873fad4f92d6125bf592369996b26f3792935f856a18f837a5e0dea8eab9`
- `models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - SHA256 `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68`
- `models/vae/wan_2.1_vae.safetensors`
  - SHA256 `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b`

## What changed from v3

Removed:

- `Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf`
- official `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
- `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`
- `ComfyUI-GGUF`
- `UnetLoaderGGUF`

Both HIGH and LOW now load through ComfyUI's built-in `UNETLoader` and route directly to their respective `ModelSamplingSD3` nodes.

## Build the model bundle on the temporary EC2

```bash
chmod +x make_model_bundle.sh
./make_model_bundle.sh \
  /mnt/h3-build/wan22-enhanced-bundle \
  /mnt/h3-build/wan22-enhanced-fp8-v4.tar.zst
```

The archive intentionally contains `models/...` at its root. Do not wrap it in an additional `wan22-enhanced-bundle/` directory.

## Upload the bundle

Example:

```bash
aws s3 cp \
  /mnt/h3-build/wan22-enhanced-fp8-v4.tar.zst \
  s3://YOUR_BUCKET/serverless/wan22-v4/wan22-enhanced-fp8-v4.tar.zst
```

## Vast environment

```text
SERVERLESS=true
PYWORKER_REPO=https://github.com/YOUR_GITHUB_USER/rosely-wan22-v4-serverless.git
PYWORKER_REF=main
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/YOUR_GITHUB_USER/rosely-wan22-v4-serverless/main/provision.sh

MODEL_S3_BUCKET=YOUR_BUCKET
MODEL_S3_BUNDLE_KEY=serverless/wan22-v4/wan22-enhanced-fp8-v4.tar.zst
MODEL_S3_REGION=us-east-1
MODEL_S3_ENDPOINT_URL=

COMFYUI_ARGS=--lowvram --disable-smart-memory
GENERATION_TIMEOUT_SECONDS=2400
KEEP_LOCAL_OUTPUTS=false
```

Provide `ROSELY_WAN22_S3_ACCESS_KEY_ID` / `ROSELY_WAN22_S3_SECRET_ACCESS_KEY` only when the worker cannot use the default AWS credential chain.

## Provisioning behavior

`provision.sh`:

1. installs only the packages needed for ComfyUI and zstd;
2. installs the pinned ComfyUI commit;
3. downloads one model bundle from S3;
4. validates archive paths;
5. extracts directly to `/workspace/ComfyUI/models`;
6. SHA-verifies all four files;
7. validates the workflow has no GGUF or LightX2V node;
8. starts ComfyUI and the FastAPI model server.

During provisioning the compressed bundle and extracted weights coexist temporarily, so use a worker disk with enough headroom (100 GB is a sensible minimum).

## Workflow

The workflow is `workflows/wan22_i2v_enhanced_fp8_v4.json`.

Default sampling remains the previous 4-step split for an apples-to-apples first test:

- HIGH: steps 0–2
- LOW: steps 2–4
- Euler / simple
- CFG 1

Benchmark this first before changing steps, scheduler, or resolution. That isolates the model-pair change from inference-setting changes.
