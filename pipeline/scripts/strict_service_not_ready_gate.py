#!/usr/bin/env python3
"""Verify the submission service's explicit 503 startup gate in process."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
from pathlib import Path

import httpx

from strict_service_acceptance import payload
from wam_pipeline.backends import SyntheticActionBackend
from wam_pipeline.service import ServiceSettings, create_app


async def run() -> dict:
    model_version = os.environ.get(
        "WAM_MODEL_VERSION", "track2-v15.7-hybrid-action-gated-reward-safe-blend12"
    )
    app = create_app(
        ServiceSettings(model_version=model_version, bearer_token="private-test-token"),
        SyntheticActionBackend(),
    )
    app.state.ready = False
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://in-process") as client:
        health = await client.get("/v1/health")
        response = await client.post(
            "/v1/predict",
            headers={"Authorization": "Bearer private-test-token"},
            json=payload(model_version),
        )
    body = response.json()
    passed = (
        health.status_code == 200
        and health.json()["status"] == "starting"
        and response.status_code == 503
        and body["error"]["code"] == "NOT_READY"
        and body["error"]["retryable"] is True
    )
    return {
        "passed": passed,
        "health_status": health.json()["status"],
        "predict_http_status": response.status_code,
        "error_code": body.get("error", {}).get("code"),
        "retryable": body.get("error", {}).get("retryable"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(asyncio.run(run()), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
