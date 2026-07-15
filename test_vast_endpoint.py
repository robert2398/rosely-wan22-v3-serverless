from __future__ import annotations

import asyncio
import os

from vastai import Serverless

ENDPOINT_NAME = os.environ.get("VAST_ENDPOINT_NAME", "rosely-wan22-v3")
INPUT_IMAGE_URL = os.environ["INPUT_IMAGE_URL"]


async def main() -> None:
    width, height, length = 384, 512, 33
    client = Serverless()
    try:
        endpoint = await client.get_endpoint(name=ENDPOINT_NAME)
        payload = {
            "input": {
                "request_id": "wan22-v3-test",
                "input_image_url": INPUT_IMAGE_URL,
                "prompt": "Subtle natural breathing and a small head movement, stable camera, consistent lighting.",
                "width": width,
                "height": height,
                "length": length,
                "fps": 16,
                "seed": 12345,
                "return_base64": False,
            }
        }
        workload = width * height * length / 1_000_000.0
        result = await endpoint.request("/generate/sync", payload, cost=workload)
        print(result)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
