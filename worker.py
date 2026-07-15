from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18288
MODEL_LOG_FILE = "/var/log/portal/comfyui.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"


def _workload(payload: dict[str, Any]) -> float:
    data = payload.get("input", payload)
    width = float(data.get("width", 384))
    height = float(data.get("height", 512))
    length = float(data.get("length", 33))
    return max(1.0, width * height * length / 1_000_000.0)


benchmark_image = base64.b64encode(
    Path(__file__).with_name("assets").joinpath("benchmark.png").read_bytes()
).decode("ascii")

benchmark_dataset = [
    {
        "input": {
            "request_id": "vast-benchmark-wan22-v3",
            "input_image_base64": benchmark_image,
            "prompt": "A person makes a small natural head movement while the camera remains stable.",
            "width": 320,
            "height": 448,
            "length": 17,
            "fps": 16,
            "seed": 12345,
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
            max_queue_time=1800.0,
            workload_calculator=_workload,
            benchmark_config=BenchmarkConfig(dataset=benchmark_dataset, runs=1),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["To see the GUI go to: "],
        on_error=[
            "MetadataIncompleteBuffer",
            "Value not in list: ",
            "Traceback (most recent call last):",
            "torch.OutOfMemoryError",
            "[ERROR] Provisioning Script failed",
        ],
        on_info=["Downloading", "Requested to load"],
    ),
)

Worker(worker_config).run()
