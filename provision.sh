#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

APP_DIR=${APP_DIR:-/workspace/vast-pyworker}
COMFY_DIR=${COMFY_DIR:-/workspace/ComfyUI}
COMFY_COMMIT=${COMFY_COMMIT:-8deaa4d911497f93bbd434a3821efab396f6981f}

PYWORKER_REPO=${PYWORKER_REPO:?PYWORKER_REPO must point to the Wan serverless repository}
PYWORKER_REF=${PYWORKER_REF:-main}

MODEL_S3_BUCKET=${MODEL_S3_BUCKET:-${S3_BUCKET:-}}
MODEL_S3_BUCKET=${MODEL_S3_BUCKET:?MODEL_S3_BUCKET (or S3_BUCKET) is required}
MODEL_S3_BUNDLE_KEY=${MODEL_S3_BUNDLE_KEY:-serverless/wan22-v4/wan22-enhanced-fp8-v4.tar.zst}
MODEL_S3_REGION=${MODEL_S3_REGION:-${S3_REGION:-us-east-1}}
MODEL_S3_ENDPOINT_URL=${MODEL_S3_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}
MODEL_S3_DOWNLOAD_CONCURRENCY=${MODEL_S3_DOWNLOAD_CONCURRENCY:-8}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-75}

export APP_DIR COMFY_DIR
export MODEL_S3_BUCKET MODEL_S3_BUNDLE_KEY MODEL_S3_REGION MODEL_S3_ENDPOINT_URL
export MODEL_S3_DOWNLOAD_CONCURRENCY

log() { printf '\n[%s] %s\n' "$(date -Iseconds)" "$*"; }
fail() { log "ERROR: $*"; exit 1; }
trap 'rc=$?; log "Provisioning failed at line $LINENO with exit code $rc"; exit $rc' ERR

ensure_system_packages() {
  local packages=()
  command -v curl >/dev/null 2>&1 || packages+=(curl)
  command -v ffmpeg >/dev/null 2>&1 || packages+=(ffmpeg)
  command -v git >/dev/null 2>&1 || packages+=(git)
  command -v jq >/dev/null 2>&1 || packages+=(jq)
  command -v zstd >/dev/null 2>&1 || packages+=(zstd)
  dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null | grep -q 'install ok installed' || packages+=(ca-certificates)
  dpkg-query -W -f='${Status}' libgl1 2>/dev/null | grep -q 'install ok installed' || packages+=(libgl1)
  dpkg-query -W -f='${Status}' libglib2.0-0t64 2>/dev/null | grep -q 'install ok installed' || packages+=(libglib2.0-0)
  if (( ${#packages[@]} > 0 )); then
    log "Installing missing system packages: ${packages[*]}"
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends "${packages[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
}

check_disk_space() {
  local available_kb required_kb
  available_kb=$(df -Pk /workspace | awk 'NR==2 {print $4}')
  required_kb=$((MIN_FREE_DISK_GB * 1024 * 1024))
  log "Available /workspace disk: $((available_kb / 1024 / 1024)) GiB"
  (( available_kb >= required_kb )) || fail "At least ${MIN_FREE_DISK_GB} GiB free is required during provisioning"
}

clone_exact_commit() {
  local repo_url=$1 commit=$2 destination=$3
  rm -rf "$destination"
  git init -q "$destination"
  git -C "$destination" remote add origin "$repo_url"
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach -q FETCH_HEAD
}

install_comfyui_preserving_models() {
  local current_commit="" backup_dir=""
  if [[ -d "$COMFY_DIR/.git" ]]; then
    current_commit=$(git -C "$COMFY_DIR" rev-parse HEAD 2>/dev/null || true)
  fi
  if [[ "$current_commit" == "$COMFY_COMMIT" ]]; then
    log "Pinned ComfyUI already installed"
    return 0
  fi
  if [[ -d "$COMFY_DIR/models" ]]; then
    backup_dir=$(mktemp -d /workspace/.wan-models-backup.XXXXXX)
    mv "$COMFY_DIR/models" "$backup_dir/models"
  fi
  log "Installing pinned ComfyUI"
  clone_exact_commit "https://github.com/Comfy-Org/ComfyUI.git" "$COMFY_COMMIT" "$COMFY_DIR"
  if [[ -n "$backup_dir" && -d "$backup_dir/models" ]]; then
    rm -rf "$COMFY_DIR/models"
    mv "$backup_dir/models" "$COMFY_DIR/models"
    rmdir "$backup_dir" || true
  fi
}

ensure_system_packages
check_disk_space

log "Cloning Wan v4 serverless repository"
rm -rf "$APP_DIR"
git clone --depth 1 --single-branch --branch "$PYWORKER_REF" "$PYWORKER_REPO" "$APP_DIR"

[[ -f /venv/main/bin/activate ]] || fail "/venv/main is missing; use a Vast PyTorch CUDA 12.8 image"
source /venv/main/bin/activate

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
print("GPU:", torch.cuda.get_device_name(0))
PY

log "Installing application dependencies"
python -m pip install --prefer-binary -r "$APP_DIR/requirements.txt"

install_comfyui_preserving_models
python -m pip install --prefer-binary -r "$COMFY_DIR/requirements.txt"

mkdir -p \
  "$COMFY_DIR/models/diffusion_models" \
  "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" \
  "$COMFY_DIR/input" \
  "$COMFY_DIR/output" \
  "$COMFY_DIR/temp" \
  /workspace/workflows \
  /workspace/model-cache \
  /var/log/portal

check_disk_space

BUNDLE=/workspace/model-cache/wan22-enhanced-fp8-v4.tar.zst
export BUNDLE

log "Downloading matched Wan v4 model bundle from s3://${MODEL_S3_BUCKET}/${MODEL_S3_BUNDLE_KEY}"
python - <<'PY'
from __future__ import annotations
import os
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

bucket = os.environ["MODEL_S3_BUCKET"]
key = os.environ["MODEL_S3_BUNDLE_KEY"]
region = os.environ.get("MODEL_S3_REGION") or "us-east-1"
endpoint_url = os.environ.get("MODEL_S3_ENDPOINT_URL") or None
concurrency = max(1, int(os.environ.get("MODEL_S3_DOWNLOAD_CONCURRENCY", "8")))
destination = Path(os.environ["BUNDLE"])
destination.parent.mkdir(parents=True, exist_ok=True)

kwargs = {
    "service_name": "s3",
    "region_name": region,
    "endpoint_url": endpoint_url,
    "config": Config(connect_timeout=60, read_timeout=900, tcp_keepalive=True, retries={"mode": "standard", "max_attempts": 20}),
}
access = os.environ.get("ROSELY_WAN22_S3_ACCESS_KEY_ID")
secret = os.environ.get("ROSELY_WAN22_S3_SECRET_ACCESS_KEY")
token = os.environ.get("ROSELY_WAN22_S3_SESSION_TOKEN")
if access and secret:
    kwargs.update(aws_access_key_id=access, aws_secret_access_key=secret, aws_session_token=token)

s3 = boto3.client(**kwargs)
head = s3.head_object(Bucket=bucket, Key=key)
print(f"Bundle size: {int(head['ContentLength']) / (1024**3):.2f} GiB", flush=True)
config = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=concurrency,
    use_threads=True,
)
s3.download_file(bucket, key, str(destination), Config=config)
print("Bundle download complete:", destination, flush=True)
PY

log "Checking archive layout"
tar --use-compress-program=unzstd -tf "$BUNDLE" > /workspace/model-cache/bundle-files.txt
if grep -Eq '(^|/)\.\./|^/' /workspace/model-cache/bundle-files.txt; then
  fail "Unsafe path found inside model bundle"
fi
for required in \
  models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8H.safetensors \
  models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8L.safetensors \
  models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  models/vae/wan_2.1_vae.safetensors; do
  grep -Fxq "$required" /workspace/model-cache/bundle-files.txt || fail "Bundle is missing $required"
done

log "Extracting model bundle into ComfyUI"
tar --use-compress-program=unzstd -xf "$BUNDLE" -C "$COMFY_DIR"
rm -f "$BUNDLE" /workspace/model-cache/bundle-files.txt

log "Verifying all four model SHA-256 checksums"
cd "$COMFY_DIR"
cat > /workspace/model-cache/expected.sha256 <<'SHA'
96a4603ac80992b33713ae279ee833325a83b83181b97cd2946beafb73175374  models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8H.safetensors
fa74873fad4f92d6125bf592369996b26f3792935f856a18f837a5e0dea8eab9  models/diffusion_models/wan22EnhancedNSFWSVICamera_nsfwV2FP8L.safetensors
c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68  models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b  models/vae/wan_2.1_vae.safetensors
SHA
sha256sum -c /workspace/model-cache/expected.sha256

log "Validating corrected matched-pair workflow"
python - <<'PY'
import json
import os
from pathlib import Path
app = Path(os.environ.get("APP_DIR", "/workspace/vast-pyworker"))
workflow_path = app / "workflows/wan22_i2v_enhanced_fp8_v4.json"
w = json.loads(workflow_path.read_text())
expected = {
    ("116:95", "unet_name"): "wan22EnhancedNSFWSVICamera_nsfwV2FP8H.safetensors",
    ("116:96", "unet_name"): "wan22EnhancedNSFWSVICamera_nsfwV2FP8L.safetensors",
    ("116:84", "clip_name"): "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ("116:90", "vae_name"): "wan_2.1_vae.safetensors",
}
for (node, key), value in expected.items():
    actual = w[node]["inputs"][key]
    if actual != value:
        raise SystemExit(f"{node}.{key}: expected {value!r}, got {actual!r}")
if w["116:95"]["class_type"] != "UNETLoader" or w["116:96"]["class_type"] != "UNETLoader":
    raise SystemExit("Both HIGH and LOW must use built-in UNETLoader")
if w["116:104"]["inputs"]["model"] != ["116:95", 0]:
    raise SystemExit("HIGH route is incorrect")
if w["116:103"]["inputs"]["model"] != ["116:96", 0]:
    raise SystemExit("LOW route is incorrect")
if "116:102" in w:
    raise SystemExit("Old LightX2V LoRA node is still present")
print("Matched FP8 workflow validation passed")
PY

cp "$APP_DIR/workflows/wan22_i2v_enhanced_fp8_v4.json" /workspace/workflows/

log "Disabling generic ComfyUI services that conflict with the Wan stack"
for service in api-wrapper comfyui; do supervisorctl stop "$service" >/dev/null 2>&1 || true; done
for config in /etc/supervisor/conf.d/api-wrapper.conf /etc/supervisor/conf.d/comfyui.conf; do
  [[ -f "$config" ]] && mv "$config" "${config}.disabled"
done
supervisorctl reread >/dev/null 2>&1 || true
supervisorctl update >/dev/null 2>&1 || true

log "Installing Wan services"
mkdir -p /opt/wan22-serverless
cp "$APP_DIR/scripts/start_comfyui.sh" /opt/wan22-serverless/
cp "$APP_DIR/scripts/start_model_server.sh" /opt/wan22-serverless/
chmod +x /opt/wan22-serverless/*.sh
cp "$APP_DIR/supervisor/wan-services.conf" /etc/supervisor/conf.d/wan-services.conf

supervisorctl reread
supervisorctl update

log "Waiting for ComfyUI"
comfy_ok=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18189/system_stats >/dev/null 2>&1; then comfy_ok=1; break; fi
  state=$(supervisorctl status wan-comfyui 2>/dev/null | awk '{print $2}' || true)
  [[ "$state" == "FATAL" || "$state" == "EXITED" ]] && break
  sleep 2
done
(( comfy_ok == 1 )) || {
  supervisorctl status || true
  nvidia-smi || true
  tail -250 /var/log/portal/comfyui.log 2>/dev/null || true
  fail "Wan ComfyUI did not become healthy within 360 seconds"
}

state=$(supervisorctl status wan-model-server 2>/dev/null | awk '{print $2}' || true)
if [[ "$state" != "RUNNING" && "$state" != "STARTING" ]]; then supervisorctl start wan-model-server; fi

log "Waiting for model server"
health_ok=0
for _ in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:18288/health >/dev/null 2>&1; then health_ok=1; break; fi
  state=$(supervisorctl status wan-model-server 2>/dev/null | awk '{print $2}' || true)
  [[ "$state" == "FATAL" || "$state" == "EXITED" ]] && break
  sleep 2
done
(( health_ok == 1 )) || {
  supervisorctl status || true
  nvidia-smi || true
  tail -250 /var/log/portal/comfyui.log 2>/dev/null || true
  tail -250 /var/log/portal/model-server.log 2>/dev/null || true
  fail "Wan model server did not become healthy within 360 seconds"
}

log "Provisioning complete"
supervisorctl status wan-comfyui wan-model-server || true
find "$COMFY_DIR/models" -type f -name "*.safetensors" -printf "%s %p\n" | sort -n
df -h /workspace
