#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/wan22-serverless
exec python worker.py
