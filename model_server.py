from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("wan22-model-server")

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:18189")
WORKFLOW_PATH = Path(
    os.getenv(
        "WAN_WORKFLOW_PATH",
        "/workspace/workflows/wan22_i2v_enhanced_fp8_v4.json",
    )
)
INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/workspace/ComfyUI/input"))
OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output"))
GENERATION_TIMEOUT = int(os.getenv("GENERATION_TIMEOUT_SECONDS", "2400"))
KEEP_LOCAL_OUTPUTS = os.getenv("KEEP_LOCAL_OUTPUTS", "false").lower() == "true"

# Output S3 is deliberately isolated from generic S3_* environment variables.
# This prevents Vast/base-image/template variables from silently redirecting
# generated media to another bucket.
OUTPUT_S3_BUCKET = os.getenv("ROSELY_WAN22_OUTPUT_S3_BUCKET") or None
OUTPUT_S3_REGION = os.getenv("ROSELY_WAN22_OUTPUT_S3_REGION", "us-east-1")
OUTPUT_S3_ENDPOINT_URL = os.getenv("ROSELY_WAN22_OUTPUT_S3_ENDPOINT_URL") or None
OUTPUT_S3_PREFIX = os.getenv(
    "ROSELY_WAN22_OUTPUT_S3_PREFIX",
    "generated/wan22-v4",
).strip("/")
OUTPUT_S3_PRESIGNED_URL_EXPIRES_SECONDS = int(
    os.getenv("ROSELY_WAN22_OUTPUT_S3_PRESIGNED_URL_EXPIRES_SECONDS", "3600")
)

EXPECTED_MODELS = {
    "high": Path(
        "/workspace/ComfyUI/models/diffusion_models/"
        "wan22EnhancedNSFWSVICamera_nsfwV2FP8H.safetensors"
    ),
    "low": Path(
        "/workspace/ComfyUI/models/diffusion_models/"
        "wan22EnhancedNSFWSVICamera_nsfwV2FP8L.safetensors"
    ),
    "clip": Path(
        "/workspace/ComfyUI/models/text_encoders/"
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    ),
    "vae": Path("/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors"),
}

app = FastAPI(title="Rosely Wan 2.2 V4 Matched FP8 Serverless Model Server")
generation_lock = asyncio.Lock()


class GenerateEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: dict[str, Any] = Field(default_factory=dict)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise HTTPException(status_code=422, detail=f"Invalid boolean value: {value!r}")


def _load_base_workflow() -> dict[str, Any]:
    if not WORKFLOW_PATH.exists():
        raise RuntimeError(f"Workflow not found: {WORKFLOW_PATH}")

    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    required = [
        "123",
        "116:84",
        "116:90",
        "116:95",
        "116:96",
        "116:103",
        "116:104",
        "116:86",
        "116:85",
        "116:122",
        "116:98",
        "116:93",
        "116:89",
        "116:87",
        "116:121",
    ]

    missing = [node_id for node_id in required if node_id not in workflow]
    if missing:
        raise RuntimeError(f"Workflow missing nodes: {missing}")

    for node_id in ("116:95", "116:96"):
        if workflow[node_id].get("class_type") != "UNETLoader":
            raise RuntimeError(f"Node {node_id} must use UNETLoader")

    if workflow["116:104"]["inputs"].get("model") != ["116:95", 0]:
        raise RuntimeError("HIGH branch must route 116:95 -> 116:104")

    if workflow["116:103"]["inputs"].get("model") != ["116:96", 0]:
        raise RuntimeError("LOW branch must route 116:96 -> 116:103")

    if "116:102" in workflow:
        raise RuntimeError(
            "v4 workflow must not contain the old LOW LightX2V LoRA node 116:102"
        )

    return workflow


BASE_WORKFLOW = _load_base_workflow()


def _payload(envelope: GenerateEnvelope) -> dict[str, Any]:
    data = dict(envelope.input)
    if not data:
        raise HTTPException(status_code=422, detail="input object is required")
    return data


def _validate_dimensions(width: int, height: int, length: int, fps: int) -> None:
    if width < 256 or height < 256:
        raise HTTPException(
            status_code=422,
            detail="width and height must be at least 256",
        )

    if width % 16 or height % 16:
        raise HTTPException(
            status_code=422,
            detail="width and height must be divisible by 16",
        )

    if width > 1024 or height > 1024:
        raise HTTPException(
            status_code=422,
            detail="width and height must not exceed 1024",
        )

    if length < 5 or (length - 1) % 4:
        raise HTTPException(
            status_code=422,
            detail="length must follow 4n+1, for example 17, 33 or 49",
        )

    if length > 161:
        raise HTTPException(status_code=422, detail="length must not exceed 161")

    if fps < 1 or fps > 60:
        raise HTTPException(status_code=422, detail="fps must be between 1 and 60")


async def _download_input_image(data: dict[str, Any], request_id: str) -> str:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw: bytes

    encoded = data.get("input_image_base64")
    url = data.get("input_image_url")

    if encoded:
        if isinstance(encoded, str) and encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[1]

        try:
            raw = base64.b64decode(encoded, validate=False)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input_image_base64: {exc}",
            ) from exc

    elif url:
        headers = data.get("input_image_headers") or {}
        timeout = httpx.Timeout(60.0, connect=20.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(str(url), headers=headers)
            response.raise_for_status()
            raw = response.content

    else:
        raise HTTPException(
            status_code=422,
            detail="Provide input_image_url or input_image_base64",
        )

    target = INPUT_DIR / f"wan22_{request_id}.png"

    try:
        from io import BytesIO

        with Image.open(BytesIO(raw)) as image:
            image.convert("RGB").save(target, format="PNG", optimize=True)

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input image: {exc}",
        ) from exc

    return target.name


def _patch_workflow(
    data: dict[str, Any],
    image_name: str,
    request_id: str,
) -> dict[str, Any]:
    workflow = copy.deepcopy(BASE_WORKFLOW)

    width = int(data.get("width", 384))
    height = int(data.get("height", 512))
    length = int(data.get("length", 33))
    fps = int(data.get("fps", 16))
    seed = int(data.get("seed", int.from_bytes(os.urandom(6), "big")))

    prompt = str(
        data.get(
            "prompt",
            "Smooth natural movement, stable camera, consistent identity and lighting.",
        )
    )

    negative_prompt = data.get("negative_prompt")

    _validate_dimensions(width, height, length, fps)

    workflow["116:122"]["inputs"]["image"] = image_name
    workflow["116:98"]["inputs"].update(
        {
            "width": width,
            "height": height,
            "length": length,
            "batch_size": 1,
        }
    )

    workflow["116:93"]["inputs"]["text"] = prompt

    if negative_prompt is not None:
        workflow["116:89"]["inputs"]["text"] = str(negative_prompt)

    workflow["116:86"]["inputs"]["noise_seed"] = seed
    workflow["116:121"]["inputs"]["fps"] = fps
    workflow["123"]["inputs"]["filename_prefix"] = f"video/{request_id}"

    # Enforce the matched v4 pair on every request.
    workflow["116:104"]["inputs"]["model"] = ["116:95", 0]
    workflow["116:103"]["inputs"]["model"] = ["116:96", 0]

    return workflow


async def _submit_and_wait(
    workflow: dict[str, Any],
    request_id: str,
) -> tuple[str, dict[str, Any]]:
    client_id = str(uuid.uuid4())
    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"ComfyUI rejected prompt: {response.text}",
            )

        body = response.json()

        if body.get("error") or body.get("node_errors"):
            raise HTTPException(status_code=502, detail=body)

        prompt_id = body["prompt_id"]
        deadline = time.monotonic() + GENERATION_TIMEOUT

        while time.monotonic() < deadline:
            history_response = await client.get(
                f"{COMFY_URL}/history/{prompt_id}"
            )
            history_response.raise_for_status()

            history = history_response.json()

            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})

                if status.get("status_str") == "error":
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "request_id": request_id,
                            "status": status,
                        },
                    )

                return prompt_id, item

            await asyncio.sleep(5)

        try:
            await client.post(f"{COMFY_URL}/interrupt")
        except Exception:
            logger.exception(
                "Failed to interrupt timed-out prompt %s",
                prompt_id,
            )

        raise HTTPException(
            status_code=504,
            detail="Wan generation timed out",
        )


def _find_output_metadata(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        filename = value.get("filename")

        if filename and Path(str(filename)).suffix.lower() in {
            ".mp4",
            ".webm",
            ".mov",
            ".mkv",
        }:
            return value

        for nested in value.values():
            found = _find_output_metadata(nested)
            if found:
                return found

    elif isinstance(value, list):
        for nested in value:
            found = _find_output_metadata(nested)
            if found:
                return found

    return None


def _resolve_output(
    item: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    metadata = _find_output_metadata(item.get("outputs", {}))

    if metadata:
        subfolder = str(metadata.get("subfolder") or "")
        output_path = OUTPUT_DIR / subfolder / str(metadata["filename"])

        if output_path.exists():
            return output_path, metadata

    candidates = sorted(
        (
            p
            for p in OUTPUT_DIR.rglob("*")
            if p.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise HTTPException(
            status_code=500,
            detail="ComfyUI completed but no video file was found",
        )

    return candidates[0], {
        "filename": candidates[0].name,
        "subfolder": str(candidates[0].parent.relative_to(OUTPUT_DIR)),
        "type": "output",
    }


def _output_s3_client():
    if not OUTPUT_S3_BUCKET:
        return None

    # Prefer output-specific credentials. If they are not supplied, reuse the
    # dedicated Wan model credentials. Never use generic AWS_* or S3_* env vars.
    access_key = (
        os.getenv("ROSELY_WAN22_OUTPUT_S3_ACCESS_KEY_ID")
        or os.getenv("ROSELY_WAN22_S3_ACCESS_KEY_ID")
    )
    secret_key = (
        os.getenv("ROSELY_WAN22_OUTPUT_S3_SECRET_ACCESS_KEY")
        or os.getenv("ROSELY_WAN22_S3_SECRET_ACCESS_KEY")
    )
    session_token = (
        os.getenv("ROSELY_WAN22_OUTPUT_S3_SESSION_TOKEN")
        or os.getenv("ROSELY_WAN22_S3_SESSION_TOKEN")
        or None
    )

    if not access_key or not secret_key:
        raise RuntimeError(
            "Output S3 is configured but dedicated credentials are missing. "
            "Set ROSELY_WAN22_OUTPUT_S3_ACCESS_KEY_ID / "
            "ROSELY_WAN22_OUTPUT_S3_SECRET_ACCESS_KEY, or the existing "
            "ROSELY_WAN22_S3_* credentials."
        )

    kwargs: dict[str, Any] = {
        "service_name": "s3",
        "region_name": OUTPUT_S3_REGION,
        "endpoint_url": OUTPUT_S3_ENDPOINT_URL,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }

    if session_token:
        kwargs["aws_session_token"] = session_token

    return boto3.client(**kwargs)


def _upload_output(
    path: Path,
    request_id: str,
) -> str | None:
    if not OUTPUT_S3_BUCKET:
        return None

    client = _output_s3_client()
    if client is None:
        return None

    key = (
        f"{OUTPUT_S3_PREFIX}/{request_id}{path.suffix.lower()}"
        if OUTPUT_S3_PREFIX
        else f"{request_id}{path.suffix.lower()}"
    )

    content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"

    logger.info(
        "Uploading generated video to s3://%s/%s",
        OUTPUT_S3_BUCKET,
        key,
    )

    client.upload_file(
        str(path),
        OUTPUT_S3_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )

    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": OUTPUT_S3_BUCKET,
            "Key": key,
        },
        ExpiresIn=OUTPUT_S3_PRESIGNED_URL_EXPIRES_SECONDS,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    missing = [
        name
        for name, path in EXPECTED_MODELS.items()
        if not path.exists()
    ]

    comfy_ok = False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{COMFY_URL}/system_stats")
            comfy_ok = response.status_code == 200
    except Exception:
        comfy_ok = False

    if missing or not comfy_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "comfyui": comfy_ok,
                "missing_models": missing,
            },
        )

    return {
        "status": "ok",
        "comfyui": True,
        "workflow": WORKFLOW_PATH.name,
        "model_pair": "wan22EnhancedNSFWSVICamera_nsfwV2FP8H/L",
        "output_s3_configured": bool(OUTPUT_S3_BUCKET),
        "output_s3_bucket": OUTPUT_S3_BUCKET,
        "output_s3_prefix": OUTPUT_S3_PREFIX if OUTPUT_S3_BUCKET else None,
    }


@app.post("/generate/sync")
async def generate_sync(
    envelope: GenerateEnvelope,
) -> dict[str, Any]:
    data = _payload(envelope)

    request_id = str(
        data.get("request_id")
        or uuid.uuid4().hex
    )

    return_base64 = _as_bool(
        data.get("return_base64"),
        default=True,
    )

    upload_to_s3 = _as_bool(
        data.get("upload_to_s3"),
        default=True,
    )

    image_name: str | None = None
    output_path: Path | None = None

    async with generation_lock:
        try:
            image_name = await _download_input_image(
                data,
                request_id,
            )

            workflow = _patch_workflow(
                data,
                image_name,
                request_id,
            )

            prompt_id, history_item = await _submit_and_wait(
                workflow,
                request_id,
            )

            output_path, metadata = _resolve_output(
                history_item
            )

            output_url: str | None = None

            if upload_to_s3:
                output_url = await asyncio.to_thread(
                    _upload_output,
                    output_path,
                    request_id,
                )

            response: dict[str, Any] = {
                "request_id": request_id,
                "prompt_id": prompt_id,
                "status": "completed",
                "filename": output_path.name,
                "size_bytes": output_path.stat().st_size,
                "upload_to_s3": upload_to_s3,
                "output_url": output_url,
                "comfyui_output": metadata,
            }

            if return_base64 and not output_url:
                encoded = await asyncio.to_thread(
                    base64.b64encode,
                    output_path.read_bytes(),
                )

                response["video_base64"] = encoded.decode("ascii")
                response["video_data_url"] = (
                    "video/mp4;base64,"
                    f"{response['video_base64']}"
                )

            return response

        finally:
            if image_name:
                (INPUT_DIR / image_name).unlink(missing_ok=True)

            if output_path and not KEEP_LOCAL_OUTPUTS:
                output_path.unlink(missing_ok=True)
