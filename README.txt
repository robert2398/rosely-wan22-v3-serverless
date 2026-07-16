WAN 2.2 PYWORKER BENCHMARK FIX

1. Stop the current Serverless deployment/workergroup so it does not keep
   creating replacement workers.

2. Replace worker.py in the repository root with the included worker.py.

3. Push:

   git add worker.py
   git commit -m "Reduce Wan startup benchmark and avoid false fatal logs"
   git push origin main

4. Delete the existing workergroup and create a fresh one from the same
   template. The endpoint itself can remain.

What changed:
- benchmark reduced from 320x448x17 to 256x256x5
- do_warmup=False, so Vast runs one video generation instead of a warm-up plus
  another measured generation
- concurrency explicitly fixed at 1
- generic `Traceback` and `Value not in list` strings removed from fatal log
  matching
- queue allowance raised to 2400 seconds

Expected logs:
- Running worker.py
- request_id: vast-benchmark-wan22-v3
- one benchmark generation only
- Benchmark complete
- healthcheck confirmed
- model_loading -> idle/ready

If it still exits, capture these from Extra Debug Logs before the container is
destroyed:
- /workspace/debug.log
- /workspace/pyworker.log
- /var/log/portal/comfyui.log
Search for: Benchmark failed, No successful responses, ComfyUI rejected prompt,
Value not in list, OutOfMemoryError, Wan generation timed out.
