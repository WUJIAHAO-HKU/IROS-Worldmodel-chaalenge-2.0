#!/usr/bin/env python3
"""Self-contained official API contract test for the local service."""

from __future__ import annotations

import argparse
import asyncio
import uuid

import httpx
import numpy as np

from wam_pipeline.images import encode_png_base64
from wam_pipeline.profile import ACTION_DIM, API_VERSION, OFFICIAL_PROFILE_ID


def image(value: int) -> dict:
    return encode_png_base64(np.full((256, 256, 3), value, dtype=np.uint8))


def sample(index: int) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "seed": index,
        "context": {
            "frames": [image(index + frame) for frame in range(5)],
            "actions": np.zeros((4, ACTION_DIM), dtype=np.float32).tolist(),
            "states": None,
            "instruction": "adjust bottle",
        },
        "actions": np.full((8, ACTION_DIM), 0.01 * (index + 1), dtype=np.float32).tolist(),
    }


def request_payload(samples: int = 1, request_id: str | None = None, model_version: str = "ivideogpt64-track2-dev") -> dict:
    return {
        "api_version": API_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "model_version": model_version,
        "profile_id": OFFICIAL_PROFILE_ID,
        "samples": [sample(index) for index in range(samples)],
    }


async def main_async(base_url: str, token: str, model_version: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    # A real 8-sample batch is serialized by the official max-concurrency=1.
    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        health = await client.get("/v1/health")
        assert health.status_code == 200 and health.json()["status"] == "ready"
        caps = await client.get("/v1/capabilities", headers=headers)
        assert caps.status_code == 200
        assert caps.json()["profiles"][0]["profile_id"] == OFFICIAL_PROFILE_ID
        assert caps.json()["determinism"] == {"seed_supported": True, "same_seed_same_pixels": True}

        payload = request_payload(model_version=model_version)
        first = await client.post("/v1/predict", headers=headers, json=payload)
        assert first.status_code == 200, first.text
        output = first.json()
        assert len(output["predictions"]) == 1 and len(output["predictions"][0]["frames"]) == 8
        retry = await client.post("/v1/predict", headers=headers, json=payload)
        assert retry.status_code == 200 and retry.json() == output

        changed = request_payload(request_id=payload["request_id"], model_version=model_version)
        changed["samples"][0]["seed"] = 999
        conflict = await client.post("/v1/predict", headers=headers, json=changed)
        assert conflict.status_code == 409, conflict.json()

        batch = await client.post("/v1/predict", headers=headers, json=request_payload(samples=8, model_version=model_version))
        assert batch.status_code == 200 and len(batch.json()["predictions"]) == 8

        malformed = request_payload(model_version=model_version)
        malformed["samples"][0]["actions"][0].pop()
        bad = await client.post("/v1/predict", headers=headers, json=malformed)
        assert bad.status_code == 400, bad.json()

        wrong_version = request_payload(model_version=model_version)
        wrong_version["profile_id"] = "wrong-profile"
        unsupported = await client.post("/v1/predict", headers=headers, json=wrong_version)
        assert unsupported.status_code == 422, unsupported.json()

        unauthorized = await client.post("/v1/predict", json=request_payload(model_version=model_version))
        assert unauthorized.status_code == 401, unauthorized.json()
    print("contract test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="local-dev-token")
    parser.add_argument("--model-version", default="ivideogpt64-track2-dev")
    args = parser.parse_args()
    asyncio.run(main_async(args.base_url, args.token, args.model_version))


if __name__ == "__main__":
    main()
