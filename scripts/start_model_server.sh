#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/wan22-serverless
exec python -m uvicorn model_server:app --host 127.0.0.1 --port 18288
