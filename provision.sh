#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

APP_DIR=${APP_DIR:-/workspace/wan22-serverless}
COMFY_DIR=${COMFY_DIR:-/workspace/ComfyUI}
COMFY_COMMIT=${COMFY_COMMIT:-8deaa4d911497f93bbd434a3821efab396f6981f}
GGUF_COMMIT=${GGUF_COMMIT:-6ea2651e7df66d7585f6ffee804b20e92fb38b8a}
PYWORKER_REPO=${PYWORKER_REPO:?PYWORKER_REPO must point to the Wan serverless repository}
PYWORKER_REF=${PYWORKER_REF:-main}
WAN_HIGH_MODEL_URL=${WAN_HIGH_MODEL_URL:-https://civitai.com/api/download/models/2540892}
WAN_HIGH_MODEL_FILENAME=${WAN_HIGH_MODEL_FILENAME:-Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf}
WAN_HIGH_MODEL_SHA256=${WAN_HIGH_MODEL_SHA256:-}
DOWNLOAD_CONNECTIONS=${DOWNLOAD_CONNECTIONS:-8}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-55}

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

  command -v aria2c >/dev/null 2>&1 || packages+=(aria2)
  command -v curl >/dev/null 2>&1 || packages+=(curl)
  command -v ffmpeg >/dev/null 2>&1 || packages+=(ffmpeg)
  command -v git >/dev/null 2>&1 || packages+=(git)
  command -v jq >/dev/null 2>&1 || packages+=(jq)
  dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null | grep -q 'install ok installed' || packages+=(ca-certificates)
  dpkg-query -W -f='${Status}' libgl1 2>/dev/null | grep -q 'install ok installed' || packages+=(libgl1)
  dpkg-query -W -f='${Status}' libglib2.0-0t64 2>/dev/null | grep -q 'install ok installed' || packages+=(libglib2.0-0)

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
    fail "At least ${MIN_FREE_DISK_GB} GiB free is required before downloading Wan assets"
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

sha_ok() {
  local file=$1
  local expected=$2
  [[ -f "$file" ]] && echo "$expected  $file" | sha256sum -c - >/dev/null 2>&1
}

valid_gguf() {
  local file=$1
  [[ -f "$file" ]] || return 1

  local size_bytes
  size_bytes=$(stat -c '%s' "$file")
  (( size_bytes >= 10000000000 )) || return 1

  [[ "$(head -c 4 "$file")" == "GGUF" ]]
}

download_file() {
  local url=$1
  local destination=$2
  local expected_sha=${3:-}
  local auth_token=${4:-}
  local label
  label=$(basename "$destination")

  mkdir -p "$(dirname "$destination")"

  if [[ -n "$expected_sha" ]] && sha_ok "$destination" "$expected_sha"; then
    log "Already valid: $label"
    return 0
  fi

  if [[ -z "$expected_sha" && "$destination" == *.gguf ]] && valid_gguf "$destination"; then
    log "Already valid: $label"
    return 0
  fi

  if [[ -f "$destination" && ! -f "${destination}.aria2" ]]; then
    rm -f "$destination"
  fi

  log "Downloading $label"

  local aria_args=(
    --allow-overwrite=true
    --auto-file-renaming=false
    --check-certificate=true
    --connect-timeout=30
    --console-log-level=notice
    --continue=true
    --dir="$(dirname "$destination")"
    --file-allocation=none
    --max-connection-per-server="$DOWNLOAD_CONNECTIONS"
    --max-tries=12
    --min-split-size=16M
    --out="$(basename "$destination")"
    --retry-wait=5
    --split="$DOWNLOAD_CONNECTIONS"
    --summary-interval=15
    --timeout=60
  )

  if [[ -n "$auth_token" ]]; then
    aria_args+=(--header="Authorization: Bearer ${auth_token}")
  fi

  aria2c "${aria_args[@]}" "$url"

  if [[ -n "$expected_sha" ]]; then
    echo "$expected_sha  $destination" | sha256sum -c -
  elif [[ "$destination" == *.gguf ]]; then
    valid_gguf "$destination" || fail "Invalid or incomplete GGUF file: $destination"
  fi
}

ensure_system_packages
check_disk_space

log "Cloning the Wan serverless repository"
rm -rf "$APP_DIR"
git clone --depth 1 --single-branch --branch "$PYWORKER_REF" "$PYWORKER_REPO" "$APP_DIR"

log "Using PyTorch already included in the Vast PyTorch image"
source /venv/main/bin/activate
python - <<'PY'
try:
    import torch
    import torchvision
    import torchaudio
except Exception as exc:
    raise SystemExit(
        "PyTorch is not ready in /venv/main. Use the template image "
        "vastai/pytorch:cuda-12.8.1-auto instead of vastai/base-image. "
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

log "Installing application dependencies only"
python -m pip install --prefer-binary -r "$APP_DIR/requirements.txt"

log "Installing pinned ComfyUI using a shallow commit fetch"
clone_exact_commit \
  "https://github.com/Comfy-Org/ComfyUI.git" \
  "$COMFY_COMMIT" \
  "$COMFY_DIR"
python -m pip install --prefer-binary -r "$COMFY_DIR/requirements.txt"

log "Installing the only required custom node: ComfyUI-GGUF"
mkdir -p "$COMFY_DIR/custom_nodes"
clone_exact_commit \
  "https://github.com/city96/ComfyUI-GGUF.git" \
  "$GGUF_COMMIT" \
  "$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
python -m pip install --prefer-binary -r "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"

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

HF_TOKEN_VALUE=${HF_TOKEN:-}
CIVITAI_TOKEN_VALUE=${CIVITAI_API_TOKEN:-}

log "Downloading the five model files required by the v3 workflow in parallel"

declare -a download_pids=()
declare -a download_names=()

start_download() {
  local name=$1
  shift
  download_file "$@" &
  download_pids+=("$!")
  download_names+=("$name")
}

start_download \
  "Enhanced HIGH GGUF" \
  "$WAN_HIGH_MODEL_URL" \
  "$COMFY_DIR/models/unet/$WAN_HIGH_MODEL_FILENAME" \
  "$WAN_HIGH_MODEL_SHA256" \
  "$CIVITAI_TOKEN_VALUE"

start_download \
  "Wan LOW FP8" \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors?download=true" \
  "$COMFY_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  "5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e" \
  "$HF_TOKEN_VALUE"

start_download \
  "UMT5 text encoder" \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true" \
  "$COMFY_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68" \
  "$HF_TOKEN_VALUE"

start_download \
  "Wan VAE" \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true" \
  "$COMFY_DIR/models/vae/wan_2.1_vae.safetensors" \
  "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b" \
  "$HF_TOKEN_VALUE"

start_download \
  "Wan LOW LightX2V LoRA" \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors?download=true" \
  "$COMFY_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
  "024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9" \
  "$HF_TOKEN_VALUE"

failed=0
for i in "${!download_pids[@]}"; do
  if wait "${download_pids[$i]}"; then
    log "Download completed: ${download_names[$i]}"
  else
    log "Download failed: ${download_names[$i]}"
    failed=1
  fi
done

(( failed == 0 )) || fail "One or more required model downloads failed"

log "Validating the GGUF header and exact v3 workflow model references"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

app = Path('/workspace/wan22-serverless')
comfy = Path('/workspace/ComfyUI')
high_name = os.environ.get(
    'WAN_HIGH_MODEL_FILENAME',
    'Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf',
)
high = comfy / 'models/unet' / high_name

if not high.exists():
    raise SystemExit(f'Missing HIGH model: {high}')
if high.stat().st_size < 10_000_000_000:
    raise SystemExit(f'HIGH GGUF is unexpectedly small: {high.stat().st_size} bytes')
if high.read_bytes()[:4] != b'GGUF':
    raise SystemExit(f'Invalid GGUF header: {high}')

expected_high_sha = os.environ.get('WAN_HIGH_MODEL_SHA256', '').strip().lower()
if expected_high_sha:
    digest = hashlib.sha256()
    with high.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_high_sha:
        raise SystemExit(f'HIGH model SHA mismatch: expected {expected_high_sha}, got {actual}')

workflow_path = app / 'workflows/wan22_i2v_custom_high_lora_v3.json'
workflow = json.loads(workflow_path.read_text(encoding='utf-8'))

expected = {
    ('116:95', 'unet_name'): high_name,
    ('116:96', 'unet_name'): 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors',
    ('116:84', 'clip_name'): 'umt5_xxl_fp8_e4m3fn_scaled.safetensors',
    ('116:90', 'vae_name'): 'wan_2.1_vae.safetensors',
    ('116:102', 'lora_name'): 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors',
}

for (node, key), value in expected.items():
    actual = workflow[node]['inputs'][key]
    if actual != value:
        raise SystemExit(f'{node}.{key}: expected {value!r}, got {actual!r}')

if workflow['116:95']['class_type'] != 'UnetLoaderGGUF':
    raise SystemExit('116:95 must use UnetLoaderGGUF')
if workflow['116:104']['inputs']['model'] != ['116:95', 0]:
    raise SystemExit('v3 HIGH routing must be 116:95 -> 116:104')

print('Workflow and model validation passed')
PY

cp "$APP_DIR/workflows/wan22_i2v_custom_high_lora_v3.json" /workspace/workflows/

log "Installing serverless service scripts"
mkdir -p /opt/wan22-serverless
cp "$APP_DIR/scripts/start_comfyui.sh" /opt/wan22-serverless/
cp "$APP_DIR/scripts/start_model_server.sh" /opt/wan22-serverless/
cp "$APP_DIR/scripts/start_worker.sh" /opt/wan22-serverless/
chmod +x /opt/wan22-serverless/*.sh
cp "$APP_DIR/supervisor/wan-services.conf" /etc/supervisor/conf.d/wan-services.conf

supervisorctl reread
supervisorctl update
supervisorctl restart wan-comfyui wan-model-server wan-pyworker || true

log "Waiting for the local Wan model server health check"
health_ok=0
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:18288/health >/dev/null 2>&1; then
    health_ok=1
    break
  fi
  sleep 2
done

(( health_ok == 1 )) || {
  supervisorctl status || true
  tail -200 /var/log/portal/comfyui.log 2>/dev/null || true
  fail "Wan model server did not become healthy within 240 seconds"
}

log "Provisioning complete. Service status:"
supervisorctl status wan-comfyui wan-model-server wan-pyworker || true

log "Relevant model files:"
find "$COMFY_DIR/models" -type f \( -name '*.gguf' -o -name '*.safetensors' \) -printf '%s %p\n' | sort -n

df -h /workspace
