WAN 2.2 SERVERLESS DOWNLOAD FIX

Replace these files at the repository root:
- provision.sh
- requirements.txt

Then run:

chmod +x provision.sh
git add provision.sh requirements.txt
git commit -m "Fix Hugging Face model downloads"
git push origin main

Template image:
vastai/pytorch:cuda-12.8.1-auto

Keep:
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/robert2398/rosely-wan22-v3-serverless/main/provision.sh
PYWORKER_REPO=https://github.com/robert2398/rosely-wan22-v3-serverless.git
PYWORKER_REF=main

After pushing, destroy the current workergroup and create a fresh worker.
Existing workers do not pick up the new Git commit.

This version:
- removes parallel aria2 downloads for Hugging Face files
- uses huggingface_hub/hf_xet for signed Xet/CAS downloads
- validates HF_TOKEN and falls back to anonymous access if invalid
- downloads large Hugging Face files sequentially
- keeps resumable curl download for the Civitai HIGH GGUF
