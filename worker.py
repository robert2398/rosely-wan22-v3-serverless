from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from vastai import (
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18288
MODEL_LOG_FILE = "/var/log/portal/comfyui.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"


def _workload(payload: dict[str, Any]) -> float:
    data = payload.get("input", payload)

    width = float(data.get("width", 384))
    height = float(data.get("height", 512))
    length = float(data.get("length", 33))

    return max(
        1.0,
        width * height * length / 1_000_000.0,
    )


BENCHMARK_IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "benchmark.png"
)

if not BENCHMARK_IMAGE_PATH.is_file():
    raise RuntimeError(
        f"Benchmark image is missing: {BENCHMARK_IMAGE_PATH}"
    )


benchmark_image = base64.b64encode(
    BENCHMARK_IMAGE_PATH.read_bytes()
).decode("ascii")


benchmark_dataset = [
    {
        "input": {
            "request_id": "vast-benchmark-wan22-v4",
            "input_image_base64": benchmark_image,
            "prompt": (
                "A person makes a very small natural head movement "
                "while the camera remains stable, facial identity "
                "remains consistent, lighting remains stable, and "
                "motion is smooth and realistic."
            ),
            "width": 256,
            "height": 256,
            "length": 5,
            "fps": 8,
            "seed": 12345,

            # The startup benchmark verifies inference only.
            # It must not depend on production object-storage permissions.
            "upload_to_s3": False,
            "return_base64": False,
        }
    }
]


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,

    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=2400.0,
            workload_calculator=_workload,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
                runs=1,
                concurrency=1,
                do_warmup=False,
            ),
        )
    ],

    log_action_config=LogActionConfig(
        on_load=[
            "To see the GUI go to: ",
        ],
        on_error=[
            "torch.OutOfMemoryError",
            "CUDA error: an illegal memory access was encountered",
            "[ERROR] Provisioning Script failed",
        ],
        on_info=[
            "Requested to load",
            "ERROR UNSUPPORTED UNET",
        ],
    ),
)


if __name__ == "__main__":
    Worker(worker_config).run()
