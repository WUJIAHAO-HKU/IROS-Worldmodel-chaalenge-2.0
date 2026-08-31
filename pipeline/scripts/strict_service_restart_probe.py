#!/usr/bin/env python3
"""Record deterministic Track 2 output evidence before or after a restart."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from strict_service_acceptance import payload, pixel_hash
from wam_pipeline.profile import RECOMMENDED_TIMEOUT_MS


async def run(
    base_url: str, token: str, model_version: str, request_id: str
) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    request = payload(
        model_version,
        count=1,
        request_id=request_id,
    )
    timeout = httpx.Timeout(RECOMMENDED_TIMEOUT_MS / 1000.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        health = await client.get("/v1/health")
        health.raise_for_status()
        capabilities = await client.get("/v1/capabilities", headers=headers)
        capabilities.raise_for_status()
        started = time.perf_counter()
        response = await client.post("/v1/predict", headers=headers, json=request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.raise_for_status()

    capability_bytes = json.dumps(
        capabilities.json(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "health": health.json(),
        "capabilities_sha256": hashlib.sha256(capability_bytes).hexdigest(),
        "pixel_sha256": pixel_hash(response.json()),
        "latency_ms": latency_ms,
        "request_id": request["request_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--request-id",
        default="00000000-0000-4000-8000-000000000001",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with open(args.token_file, "r", encoding="utf-8") as token_file:
        token = token_file.read().strip()
    rendered = json.dumps(
        asyncio.run(run(args.base_url, token, args.model_version, args.request_id)),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
