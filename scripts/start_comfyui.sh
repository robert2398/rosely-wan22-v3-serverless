#!/usr/bin/env bash
set -euo pipefail
mkdir -p /var/log/portal /workspace/ComfyUI/input /workspace/ComfyUI/output /workspace/ComfyUI/temp
: > /var/log/portal/comfyui.log
source /venv/main/bin/activate
cd /workspace/ComfyUI
# COMFYUI_ARGS defaults to aggressive offloading for 24 GB GPUs.
read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:---lowvram --disable-smart-memory}"
set -o pipefail
python main.py \
  --listen 127.0.0.1 \
  --port 18189 \
  --disable-auto-launch \
  --input-directory /workspace/ComfyUI/input \
  --output-directory /workspace/ComfyUI/output \
  --temp-directory /workspace/ComfyUI/temp \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a /var/log/portal/comfyui.log
