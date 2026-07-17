#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

APP_DIR=${APP_DIR:-/workspace/vast-pyworker}
COMFY_DIR=${COMFY_DIR:-/workspace/ComfyUI}
COMFY_COMMIT=${COMFY_COMMIT:-8deaa4d911497f93bbd434a3821efab396f6981f}
GGUF_COMMIT=${GGUF_COMMIT:-6ea2651e7df66d7585f6ffee804b20e92fb38b8a}

PYWORKER_REPO=${PYWORKER_REPO:?PYWORKER_REPO must point to the Wan serverless repository}
PYWORKER_REF=${PYWORKER_REF:-main}

# S3 location containing the five verified Wan model objects.
S3_BUCKET=${S3_BUCKET:?S3_BUCKET is required}
S3_MODEL_PREFIX=${S3_MODEL_PREFIX:-serverless/wan22-v3/models}
S3_REGION=${S3_REGION:-us-east-1}
S3_ENDPOINT_URL=${S3_ENDPOINT_URL:-}
S3_MODEL_DOWNLOAD_WORKERS=${S3_MODEL_DOWNLOAD_WORKERS:-8}
S3_MODEL_CHUNK_MIB=${S3_MODEL_CHUNK_MIB:-64}

WAN_HIGH_MODEL_FILENAME=${WAN_HIGH_MODEL_FILENAME:-Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf}
WAN_HIGH_MODEL_SHA256=${WAN_HIGH_MODEL_SHA256:-deb6dcf39fefef7850d1dd49a52b70a64c65b8d8e7b5f03bc1b6a541cf58aae9}

MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-55}

export APP_DIR COMFY_DIR
export S3_BUCKET S3_MODEL_PREFIX S3_REGION S3_ENDPOINT_URL
export S3_MODEL_DOWNLOAD_WORKERS S3_MODEL_CHUNK_MIB
export WAN_HIGH_MODEL_FILENAME WAN_HIGH_MODEL_SHA256

log() {
  printf '\n[%s] %s\n' "$(date -Iseconds)" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  log "Provisioning failed at line ${line_no} with exit code ${exit_code}"
  exit "$exit_code"
}

trap 'on_error $LINENO' ERR

ensure_system_packages() {
  local packages=()

  command -v curl >/dev/null 2>&1 || packages+=(curl)
  command -v ffmpeg >/dev/null 2>&1 || packages+=(ffmpeg)
  command -v git >/dev/null 2>&1 || packages+=(git)
  command -v jq >/dev/null 2>&1 || packages+=(jq)

  dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null \
    | grep -q 'install ok installed' || packages+=(ca-certificates)

  dpkg-query -W -f='${Status}' libgl1 2>/dev/null \
    | grep -q 'install ok installed' || packages+=(libgl1)

  dpkg-query -W -f='${Status}' libglib2.0-0t64 2>/dev/null \
    | grep -q 'install ok installed' || packages+=(libglib2.0-0)

  if (( ${#packages[@]} > 0 )); then
    log "Installing missing system packages: ${packages[*]}"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends "${packages[@]}"
    rm -rf /var/lib/apt/lists/*
  else
    log "Required system packages are already present"
  fi
}

check_disk_space() {
  local available_kb required_kb

  available_kb=$(df -Pk /workspace | awk 'NR==2 {print $4}')
  required_kb=$((MIN_FREE_DISK_GB * 1024 * 1024))

  log "Available /workspace disk: $((available_kb / 1024 / 1024)) GiB"

  if (( available_kb < required_kb )); then
    fail "At least ${MIN_FREE_DISK_GB} GiB free is required before installing Wan"
  fi
}

clone_exact_commit() {
  local repo_url=$1
  local commit=$2
  local destination=$3

  rm -rf "$destination"
  git init -q "$destination"
  git -C "$destination" remote add origin "$repo_url"
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach -q FETCH_HEAD
}

install_comfyui_preserving_models() {
  local current_commit=""
  local backup_dir=""

  if [[ -d "$COMFY_DIR/.git" ]]; then
    current_commit=$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || true)
  fi

  if [[ "$current_commit" == "$COMFY_COMMIT" ]]; then
    log "Pinned ComfyUI already installed"
    return 0
  fi

  if [[ -d "$COMFY_DIR/models" ]]; then
    backup_dir=$(mktemp -d /workspace/.wan-models-backup.XXXXXX)
    log "Temporarily preserving existing model directory"
    mv "$COMFY_DIR/models" "$backup_dir/models"
  fi

  log "Installing pinned ComfyUI"
  clone_exact_commit \
    "https://github.com/Comfy-Org/ComfyUI.git" \
    "$COMFY_COMMIT" \
    "$COMFY_DIR"

  if [[ -n "$backup_dir" && -d "$backup_dir/models" ]]; then
    rm -rf "$COMFY_DIR/models"
    mv "$backup_dir/models" "$COMFY_DIR/models"
    rmdir "$backup_dir" || true
  fi
}

ensure_system_packages
check_disk_space

log "Cloning the Wan serverless repository"
rm -rf "$APP_DIR"
git clone \
  --depth 1 \
  --single-branch \
  --branch "$PYWORKER_REF" \
  "$PYWORKER_REPO" \
  "$APP_DIR"

[[ -f /venv/main/bin/activate ]] \
  || fail "/venv/main is missing; use the Vast PyTorch CUDA 12.8 image"

log "Using PyTorch included in the Vast image"
source /venv/main/bin/activate

python - <<'PY'
try:
    import torch
    import torchvision
    import torchaudio
except Exception as exc:
    raise SystemExit(
        "PyTorch is not ready in /venv/main. Use "
        "vastai/pytorch:cuda-12.8.1-auto. "
        f"Import error: {exc}"
    )

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

print("GPU:", torch.cuda.get_device_name(0))
PY

log "Installing application dependencies"
python -m pip install \
  --prefer-binary \
  -r "$APP_DIR/requirements.txt"

install_comfyui_preserving_models

python -m pip install \
  --prefer-binary \
  -r "$COMFY_DIR/requirements.txt"

log "Installing the only required custom node: ComfyUI-GGUF"
mkdir -p "$COMFY_DIR/custom_nodes"

clone_exact_commit \
  "https://github.com/city96/ComfyUI-GGUF.git" \
  "$GGUF_COMMIT" \
  "$COMFY_DIR/custom_nodes/ComfyUI-GGUF"

python -m pip install \
  --prefer-binary \
  -r "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"

mkdir -p \
  "$COMFY_DIR/models/unet" \
  "$COMFY_DIR/models/diffusion_models" \
  "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" \
  "$COMFY_DIR/models/loras" \
  "$COMFY_DIR/input" \
  "$COMFY_DIR/output" \
  "$COMFY_DIR/temp" \
  /workspace/workflows \
  /var/log/portal

check_disk_space

log "Downloading checksum-verified Wan models from S3"

python - <<'PY'
from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config


MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

bucket = os.environ["S3_BUCKET"]
prefix = os.environ.get(
    "S3_MODEL_PREFIX",
    "serverless/wan22-v3/models",
).strip("/")
region = os.environ.get("S3_REGION", "us-east-1")
endpoint_url = os.environ.get("S3_ENDPOINT_URL") or None
workers = max(1, int(os.environ.get("S3_MODEL_DOWNLOAD_WORKERS", "8")))
chunk_size = max(8, int(os.environ.get("S3_MODEL_CHUNK_MIB", "64"))) * MIB
comfy = Path(os.environ.get("COMFY_DIR", "/workspace/ComfyUI"))
high_name = os.environ.get(
    "WAN_HIGH_MODEL_FILENAME",
    "Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf",
)
high_sha = os.environ.get(
    "WAN_HIGH_MODEL_SHA256",
    "deb6dcf39fefef7850d1dd49a52b70a64c65b8d8e7b5f03bc1b6a541cf58aae9",
).lower()

access_key = (
    os.environ.get("S3_ACCESS_KEY_ID")
    or os.environ.get("AWS_ACCESS_KEY_ID")
)
secret_key = (
    os.environ.get("S3_SECRET_ACCESS_KEY")
    or os.environ.get("AWS_SECRET_ACCESS_KEY")
)
session_token = (
    os.environ.get("S3_SESSION_TOKEN")
    or os.environ.get("AWS_SESSION_TOKEN")
)

if bool(access_key) != bool(secret_key):
    raise SystemExit(
        "Both S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be supplied together"
    )

client_kwargs = {
    "service_name": "s3",
    "region_name": region,
    "endpoint_url": endpoint_url,
    "config": Config(
        connect_timeout=60,
        read_timeout=900,
        tcp_keepalive=True,
        max_pool_connections=max(16, workers * 2),
        retries={"mode": "standard", "max_attempts": 20},
        signature_version="s3v4",
    ),
}

if access_key and secret_key:
    client_kwargs.update(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )

s3 = boto3.client(**client_kwargs)

models = [
    {
        "label": "Wan HIGH GGUF",
        "relative_key": f"unet/{high_name}",
        "destination": comfy / "models/unet" / high_name,
        "sha256": high_sha,
    },
    {
        "label": "Wan LOW FP8",
        "relative_key": (
            "diffusion_models/"
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        ),
        "destination": (
            comfy
            / "models/diffusion_models"
            / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        ),
        "sha256": (
            "5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e"
        ),
    },
    {
        "label": "UMT5 text encoder",
        "relative_key": (
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ),
        "destination": (
            comfy
            / "models/text_encoders"
            / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        ),
        "sha256": (
            "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68"
        ),
    },
    {
        "label": "Wan VAE",
        "relative_key": "vae/wan_2.1_vae.safetensors",
        "destination": comfy / "models/vae/wan_2.1_vae.safetensors",
        "sha256": (
            "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"
        ),
    },
    {
        "label": "Wan LOW LightX2V LoRA",
        "relative_key": (
            "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
        ),
        "destination": (
            comfy
            / "models/loras"
            / "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
        ),
        "sha256": (
            "024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9"
        ),
    },
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * MIB), b""):
            value.update(block)
    return value.hexdigest()


def download_model(index: int, model: dict[str, object]) -> None:
    label = str(model["label"])
    relative_key = str(model["relative_key"])
    destination = Path(model["destination"])
    expected_sha = str(model["sha256"]).lower()
    key = f"{prefix}/{relative_key}"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"[{index}/{len(models)}] Checking existing file: {label}", flush=True)
        if digest(destination) == expected_sha:
            gib = destination.stat().st_size / GIB
            print(
                f"[{index}/{len(models)}] Already valid: {label} ({gib:.2f} GiB)",
                flush=True,
            )
            return
        destination.unlink()

    head = s3.head_object(Bucket=bucket, Key=key)
    total_size = int(head["ContentLength"])
    remote_sha = str(head.get("Metadata", {}).get("sha256", "")).lower()

    if remote_sha and remote_sha != expected_sha:
        raise RuntimeError(
            f"S3 SHA metadata mismatch for {key}: "
            f"expected {expected_sha}, got {remote_sha}"
        )

    partial = destination.with_name(destination.name + ".part")
    state_dir = destination.with_name(destination.name + ".s3parts")
    total_chunks = math.ceil(total_size / chunk_size)

    if partial.exists() and partial.stat().st_size != total_size:
        partial.unlink()
        shutil.rmtree(state_dir, ignore_errors=True)

    state_dir.mkdir(parents=True, exist_ok=True)

    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, total_size)
    sync_lock = threading.Lock()

    def bounds(chunk_index: int) -> tuple[int, int, int]:
        start = chunk_index * chunk_size
        end = min(total_size - 1, start + chunk_size - 1)
        return start, end, end - start + 1

    def marker_path(chunk_index: int) -> Path:
        return state_dir / f"{chunk_index:06d}.done"

    def completed_chunk(chunk_index: int) -> bool:
        marker = marker_path(chunk_index)
        if not marker.is_file():
            return False
        _, _, expected_length = bounds(chunk_index)
        try:
            return int(marker.read_text(encoding="utf-8").strip()) == expected_length
        except (OSError, ValueError):
            return False

    def fetch_chunk(chunk_index: int) -> int:
        start, end, expected_length = bounds(chunk_index)
        marker = marker_path(chunk_index)
        last_error: Exception | None = None

        for attempt in range(1, 11):
            body = None
            try:
                response = s3.get_object(
                    Bucket=bucket,
                    Key=key,
                    Range=f"bytes={start}-{end}",
                )
                body = response["Body"]
                written = 0

                while written < expected_length:
                    block = body.read(min(8 * MIB, expected_length - written))
                    if not block:
                        break
                    os.pwrite(fd, block, start + written)
                    written += len(block)

                if written != expected_length:
                    raise IOError(
                        f"short ranged read for chunk {chunk_index}: "
                        f"expected {expected_length}, got {written}"
                    )

                with sync_lock:
                    os.fsync(fd)
                    marker.write_text(str(expected_length), encoding="utf-8")

                return expected_length
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                marker.unlink(missing_ok=True)
                if attempt == 10:
                    break
                delay = min(60, attempt * 3)
                print(
                    f"Retrying {label} chunk {chunk_index + 1}/{total_chunks} "
                    f"after error ({attempt}/10): {exc}",
                    flush=True,
                )
                time.sleep(delay)
            finally:
                if body is not None:
                    body.close()

        raise RuntimeError(
            f"Failed {label} chunk {chunk_index + 1}/{total_chunks}: {last_error}"
        )

    already_complete = [
        chunk_index
        for chunk_index in range(total_chunks)
        if completed_chunk(chunk_index)
    ]
    missing = [
        chunk_index
        for chunk_index in range(total_chunks)
        if chunk_index not in set(already_complete)
    ]
    completed_bytes = sum(bounds(i)[2] for i in already_complete)

    print(
        f"[{index}/{len(models)}] Downloading {label}: "
        f"{total_size / GIB:.2f} GiB, {total_chunks} chunks, "
        f"{workers} workers",
        flush=True,
    )

    if already_complete:
        print(
            f"[{index}/{len(models)}] Resuming at "
            f"{completed_bytes / total_size * 100:.1f}%",
            flush=True,
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_chunk, chunk_index): chunk_index
                for chunk_index in missing
            }

            for future in as_completed(futures):
                completed_bytes += future.result()
                percentage = completed_bytes / total_size * 100
                print(
                    f"\r[{index}/{len(models)}] {label}: "
                    f"{completed_bytes / GIB:.2f}/{total_size / GIB:.2f} GiB "
                    f"({percentage:.1f}%)",
                    end="",
                    flush=True,
                )
        print(flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)

    print(f"[{index}/{len(models)}] Verifying SHA-256: {label}", flush=True)
    actual_sha = digest(partial)

    if actual_sha != expected_sha:
        partial.unlink(missing_ok=True)
        shutil.rmtree(state_dir, ignore_errors=True)
        raise RuntimeError(
            f"SHA mismatch for {label}: expected {expected_sha}, got {actual_sha}"
        )

    os.replace(partial, destination)
    shutil.rmtree(state_dir, ignore_errors=True)

    print(
        f"[{index}/{len(models)}] Ready: {label} "
        f"({destination.stat().st_size / GIB:.2f} GiB)",
        flush=True,
    )


for position, item in enumerate(models, start=1):
    download_model(position, item)

print("All five S3 model files are ready", flush=True)
PY

log "Validating the exact v3 workflow and model files"

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

app = Path(os.environ.get("APP_DIR", "/workspace/vast-pyworker"))
comfy = Path(os.environ.get("COMFY_DIR", "/workspace/ComfyUI"))

high_name = os.environ.get(
    "WAN_HIGH_MODEL_FILENAME",
    "Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf",
)
high = comfy / "models/unet" / high_name
expected_high_sha = os.environ.get(
    "WAN_HIGH_MODEL_SHA256",
    "deb6dcf39fefef7850d1dd49a52b70a64c65b8d8e7b5f03bc1b6a541cf58aae9",
).strip().lower()

if not high.exists():
    raise SystemExit(f"Missing HIGH model: {high}")

if high.stat().st_size < 10_000_000_000:
    raise SystemExit(
        f"HIGH GGUF is unexpectedly small: {high.stat().st_size} bytes"
    )

with high.open("rb") as handle:
    if handle.read(4) != b"GGUF":
        raise SystemExit(f"Invalid GGUF header: {high}")

value = hashlib.sha256()
with high.open("rb") as handle:
    for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        value.update(block)

actual = value.hexdigest()
if actual != expected_high_sha:
    raise SystemExit(
        f"HIGH model SHA mismatch: expected {expected_high_sha}, got {actual}"
    )

workflow_path = app / "workflows/wan22_i2v_custom_high_lora_v3.json"
workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

expected = {
    ("116:95", "unet_name"): high_name,
    (
        "116:96",
        "unet_name",
    ): "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    (
        "116:84",
        "clip_name",
    ): "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ("116:90", "vae_name"): "wan_2.1_vae.safetensors",
    (
        "116:102",
        "lora_name",
    ): "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
}

for (node, key), expected_value in expected.items():
    actual_value = workflow[node]["inputs"][key]
    if actual_value != expected_value:
        raise SystemExit(
            f"{node}.{key}: expected {expected_value!r}, got {actual_value!r}"
        )

if workflow["116:95"]["class_type"] != "UnetLoaderGGUF":
    raise SystemExit("116:95 must use UnetLoaderGGUF")

if workflow["116:104"]["inputs"]["model"] != ["116:95", 0]:
    raise SystemExit("v3 HIGH routing must be 116:95 -> 116:104")

print("Workflow and model validation passed")
PY

cp \
  "$APP_DIR/workflows/wan22_i2v_custom_high_lora_v3.json" \
  /workspace/workflows/

log "Disabling generic ComfyUI services that conflict with the Wan stack"

for service in api-wrapper comfyui; do
  supervisorctl stop "$service" >/dev/null 2>&1 || true
done

for config in \
  /etc/supervisor/conf.d/api-wrapper.conf \
  /etc/supervisor/conf.d/comfyui.conf; do
  if [[ -f "$config" ]]; then
    mv "$config" "${config}.disabled"
  fi
done

supervisorctl reread >/dev/null 2>&1 || true
supervisorctl update >/dev/null 2>&1 || true

log "Installing Wan model services"
mkdir -p /opt/wan22-serverless

cp "$APP_DIR/scripts/start_comfyui.sh" \
  /opt/wan22-serverless/
cp "$APP_DIR/scripts/start_model_server.sh" \
  /opt/wan22-serverless/
chmod +x /opt/wan22-serverless/*.sh

cp \
  "$APP_DIR/supervisor/wan-services.conf" \
  /etc/supervisor/conf.d/wan-services.conf

supervisorctl reread
supervisorctl update
supervisorctl restart \
  wan-comfyui \
  wan-model-server \
  || true

log "Waiting for the local Wan model server health check"
health_ok=0

for _ in $(seq 1 180); do
  if curl -fsS \
    http://127.0.0.1:18288/health \
    >/dev/null 2>&1; then
    health_ok=1
    break
  fi

  sleep 2
done

(( health_ok == 1 )) || {
  supervisorctl status || true
  tail -200 /var/log/portal/comfyui.log 2>/dev/null || true
  tail -200 /var/log/portal/model-server.log 2>/dev/null || true
  fail "Wan model server did not become healthy within 360 seconds"
}

log "Provisioning complete. Service status:"
supervisorctl status \
  wan-comfyui \
  wan-model-server \
  || true

log "Relevant model files:"
find "$COMFY_DIR/models" \
  -type f \
  \( -name "*.gguf" -o -name "*.safetensors" \) \
  -printf "%s %p\n" \
  | sort -n

df -h /workspace
