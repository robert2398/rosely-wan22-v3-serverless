# Rosely Wan 2.2 V3 on Vast Serverless

This repository deploys **only** the currently working Wan I2V v3 workflow:

`workflows/wan22_i2v_custom_high_lora_v3.json`

It does not install ZiT, ReActor, ComfyUI Manager, VideoHelperSuite, or the old custom HIGH LoRAs.

## Included runtime stack

- ComfyUI pinned to `8deaa4d911497f93bbd434a3821efab396f6981f`
- ComfyUI-GGUF pinned to `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
- ComfyUI on `127.0.0.1:18189`
- Local FastAPI model server on `127.0.0.1:18288`
- Vast PyWorker route: `POST /generate/sync`
- One request at a time per GPU worker

## Models installed — no extras

1. `Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf`
2. `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
3. `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
4. `wan_2.1_vae.safetensors`
5. `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors`

Total model storage is roughly 37 GB. Use at least a **100 GB disk** for the worker because ComfyUI, Python packages, temporary frames, and output videos also need space.

## 1. Push this folder to a public GitHub repository

Example repository name:

`rosely-wan22-v3-serverless`

Do not commit API keys or model files.

## 2. Add Vast account secrets

In Vast account settings, add:

- `CIVITAI_API_TOKEN`
- `HF_TOKEN` only if needed
- Optional S3/R2 variables from `.env.example`

For production output URLs, configure the S3/R2 variables. Without S3, request with `return_base64: true` for testing.

## 3. Create a Vast template

Use this base image:

`vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312`

Template environment variables:

```text
SERVERLESS=true
PYWORKER_REPO=https://github.com/YOUR_GITHUB_USER/rosely-wan22-v3-serverless.git
PYWORKER_REF=main
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/YOUR_GITHUB_USER/rosely-wan22-v3-serverless/main/provision.sh
WAN_HIGH_MODEL_URL=https://civitai.com/api/download/models/2540892
WAN_HIGH_MODEL_FILENAME=Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf
COMFYUI_ARGS=--lowvram --disable-smart-memory
GENERATION_TIMEOUT_SECONDS=2400
KEEP_LOCAL_OUTPUTS=false
```

Use 100 GB disk. No public ComfyUI, wrapper, or Cloudflare tunnel port is needed for Serverless.

## 4. Create the Serverless endpoint

Recommended first configuration:

```text
Endpoint name: rosely-wan22-v3
Minimum Workers: 1
Maximum Workers: 1
Minimum Load: 0
Target Utilization: 0.90
Minimum Cold Load: 1
Inactivity Timeout: 300 seconds
Max Queue Time: 1200 seconds
Target Queue Time: 120 seconds
```

Keeping one inactive/cold worker avoids downloading approximately 37 GB again for every request while still allowing the GPU to stop when idle.

For the first stable production workergroup, select a verified GPU with **48 GB VRAM** and at least **64 GB system RAM**. The workflow can run on a 24 GB RTX 4090 using aggressive offloading, but it is much easier to hit OOM at higher resolutions.

## 5. API request

```json
{
  "input": {
    "request_id": "video_123",
    "input_image_url": "https://cdn.example.com/input.png",
    "prompt": "Subtle natural movement, stable camera, consistent identity and lighting.",
    "width": 384,
    "height": 512,
    "length": 33,
    "fps": 16,
    "seed": 12345,
    "return_base64": false
  }
}
```

Rules enforced by the model server:

- Width and height must be divisible by 16.
- Frame length must be `4n+1`, such as `17`, `33`, or `49`.
- Batch size is always fixed to 1.
- The v3 HIGH branch always stays `Enhanced HIGH GGUF -> ModelSamplingSD3`.

## 6. Test from Python

```bash
pip install vastai
export VAST_API_KEY='...'
export VAST_ENDPOINT_NAME='rosely-wan22-v3'
export INPUT_IMAGE_URL='https://cdn.example.com/input.png'
python test_vast_endpoint.py
```

## Response

With S3/R2 configured:

```json
{
  "request_id": "video_123",
  "status": "completed",
  "output_url": "https://cdn.example.com/generated/wan22/video_123.mp4",
  "size_bytes": 1234567
}
```

Without S3 and with `return_base64: true`, the response includes `video_base64` and `video_data_url`.
