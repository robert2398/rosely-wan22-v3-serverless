#!/usr/bin/env bash
set -Eeuo pipefail

export DEBIAN_FRONTEND=noninteractive
APP_DIR=/workspace/wan22-serverless
COMFY_DIR=/workspace/ComfyUI
COMFY_COMMIT=${COMFY_COMMIT:-8deaa4d911497f93bbd434a3821efab396f6981f}
GGUF_COMMIT=${GGUF_COMMIT:-6ea2651e7df66d7585f6ffee804b20e92fb38b8a}
PYWORKER_REPO=${PYWORKER_REPO:?PYWORKER_REPO must be set to this repository URL}
PYWORKER_REF=${PYWORKER_REF:-main}
WAN_HIGH_MODEL_URL=${WAN_HIGH_MODEL_URL:-https://civitai.com/api/download/models/2540892}
WAN_HIGH_MODEL_FILENAME=${WAN_HIGH_MODEL_FILENAME:-Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf}

log() { printf '\n[%s] %s\n' "$(date -Iseconds)" "$*"; }

apt-get update
apt-get install -y --no-install-recommends \
  aria2 ca-certificates curl ffmpeg git git-lfs libgl1 libglib2.0-0 jq
rm -rf /var/lib/apt/lists/*

git lfs install

log "Cloning the Wan serverless repository"
rm -rf "$APP_DIR"
git clone --depth 1 --branch "$PYWORKER_REF" "$PYWORKER_REPO" "$APP_DIR"

log "Installing CUDA PyTorch and Python dependencies"
source /venv/main/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  torch torchvision torchaudio
python -m pip install -r "$APP_DIR/requirements.txt"

log "Installing pinned ComfyUI"
if [[ ! -d "$COMFY_DIR/.git" ]]; then
  git clone https://github.com/Comfy-Org/ComfyUI.git "$COMFY_DIR"
fi
git -C "$COMFY_DIR" fetch --all --tags
git -C "$COMFY_DIR" checkout --force "$COMFY_COMMIT"
python -m pip install -r "$COMFY_DIR/requirements.txt"

log "Installing only the custom node required by v3: ComfyUI-GGUF"
rm -rf "$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
git clone https://github.com/city96/ComfyUI-GGUF.git "$COMFY_DIR/custom_nodes/ComfyUI-GGUF"
git -C "$COMFY_DIR/custom_nodes/ComfyUI-GGUF" checkout --force "$GGUF_COMMIT"
python -m pip install -r "$COMFY_DIR/custom_nodes/ComfyUI-GGUF/requirements.txt"

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

sha_ok() {
  local file=$1 expected=$2
  [[ -f "$file" ]] && echo "$expected  $file" | sha256sum -c - >/dev/null 2>&1
}

download_file() {
  local url=$1 dest=$2 expected_sha=${3:-} auth_header=${4:-}
  if [[ -n "$expected_sha" ]] && sha_ok "$dest" "$expected_sha"; then
    log "Already valid: $dest"
    return
  fi
  if [[ -f "$dest" && -z "$expected_sha" ]]; then
    log "Already present: $dest"
    return
  fi
  rm -f "$dest.part"
  log "Downloading $(basename "$dest")"
  local args=(-fL --retry 8 --retry-delay 5 --connect-timeout 30 --continue-at -)
  if [[ -n "$auth_header" ]]; then args+=(-H "$auth_header"); fi
  curl "${args[@]}" "$url" -o "$dest.part"
  mv "$dest.part" "$dest"
  if [[ -n "$expected_sha" ]]; then
    echo "$expected_sha  $dest" | sha256sum -c -
  fi
}

HF_AUTH=""
[[ -n "${HF_TOKEN:-}" ]] && HF_AUTH="Authorization: Bearer ${HF_TOKEN}"
CIVITAI_AUTH=""
[[ -n "${CIVITAI_API_TOKEN:-}" ]] && CIVITAI_AUTH="Authorization: Bearer ${CIVITAI_API_TOKEN}"

download_file \
  "$WAN_HIGH_MODEL_URL" \
  "$COMFY_DIR/models/unet/$WAN_HIGH_MODEL_FILENAME" \
  "" \
  "$CIVITAI_AUTH"

download_file \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors?download=true" \
  "$COMFY_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  "5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e" \
  "$HF_AUTH"

download_file \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true" \
  "$COMFY_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68" \
  "$HF_AUTH"

download_file \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true" \
  "$COMFY_DIR/models/vae/wan_2.1_vae.safetensors" \
  "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b" \
  "$HF_AUTH"

download_file \
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors?download=true" \
  "$COMFY_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
  "024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9" \
  "$HF_AUTH"

log "Validating the GGUF header and exact v3 workflow model references"
python - <<'PY'
import json
from pathlib import Path

app = Path('/workspace/wan22-serverless')
comfy = Path('/workspace/ComfyUI')
high_name = __import__('os').environ.get('WAN_HIGH_MODEL_FILENAME', 'Wan2_2_Enhanced_FastMove_HIGH_Q8.gguf')
high = comfy / 'models/unet' / high_name
if high.read_bytes()[:4] != b'GGUF':
    raise SystemExit(f'Invalid GGUF header: {high}')
workflow_path = app / 'workflows/wan22_i2v_custom_high_lora_v3.json'
workflow = json.loads(workflow_path.read_text())
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
print('Workflow validation passed')
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

log "Provisioning complete. Relevant model files:"
find "$COMFY_DIR/models" -type f \( -name '*.gguf' -o -name '*.safetensors' \) -printf '%s %p\n' | sort -n
