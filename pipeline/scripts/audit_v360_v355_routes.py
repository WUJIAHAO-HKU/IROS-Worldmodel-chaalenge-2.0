#!/usr/bin/env python3
"""Audit v355 left/right routing through the local HTTP service."""

from __future__ import annotations

import argparse
import copy
import json
import uuid
from pathlib import Path

import httpx
import numpy as np

from strict_service_acceptance import payload, pixel_hash
from wam_pipeline.arm_routed_autoregressive_runtime import (
    Track2ArmRoutedAutoregressiveUNet,
)


def prediction_hash(prediction: dict) -> str:
    return pixel_hash({"predictions": [prediction]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    headers = {"Authorization": f"Bearer {token}"}

    base = payload(args.model_version)
    base_sample = base["samples"][0]
    left_sample = copy.deepcopy(base_sample)
    left_sample["sample_id"] = "explicit-left"
    left_sample["context"]["instruction"] = "use the left arm to adjust the bottle"
    right_sample = copy.deepcopy(base_sample)
    right_sample["sample_id"] = "explicit-right"
    right_sample["context"]["instruction"] = "use the right arm to adjust the bottle"

    def request(samples: list[dict]) -> dict:
        body = copy.deepcopy(base)
        body["request_id"] = str(uuid.uuid4())
        body["samples"] = samples
        return body

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        left_response = client.post("/v1/predict", headers=headers, json=request([left_sample]))
        right_response = client.post("/v1/predict", headers=headers, json=request([right_sample]))
        mixed_response = client.post(
            "/v1/predict", headers=headers, json=request([left_sample, right_sample])
        )
    for response in (left_response, right_response, mixed_response):
        response.raise_for_status()

    left_hash = prediction_hash(left_response.json()["predictions"][0])
    right_hash = prediction_hash(right_response.json()["predictions"][0])
    mixed_hashes = [prediction_hash(item) for item in mixed_response.json()["predictions"]]

    history = np.zeros((4, 14), dtype=np.float32)
    left_future = np.zeros((8, 14), dtype=np.float32)
    right_future = np.zeros((8, 14), dtype=np.float32)
    left_future[:, :7] = np.arange(8, dtype=np.float32)[:, None]
    right_future[:, 7:] = np.arange(8, dtype=np.float32)[:, None]
    route = Track2ArmRoutedAutoregressiveUNet.active_arm
    static_routes = {
        "explicit_left": route(history, right_future, "use the left arm"),
        "explicit_right": route(history, left_future, "use the right arm"),
        "action_fallback_left": route(history, left_future, "adjust the bottle"),
        "action_fallback_right": route(history, right_future, "adjust the bottle"),
    }

    passed = (
        left_hash != right_hash
        and mixed_hashes == [left_hash, right_hash]
        and static_routes
        == {
            "explicit_left": "left",
            "explicit_right": "right",
            "action_fallback_left": "left",
            "action_fallback_right": "right",
        }
    )
    report = {
        "format": "strict-track2-v360-v355-route-audit-v1",
        "passed": passed,
        "explicit_left_hash": left_hash,
        "explicit_right_hash": right_hash,
        "experts_produce_distinct_pixels": left_hash != right_hash,
        "mixed_batch_preserves_routes": mixed_hashes == [left_hash, right_hash],
        "static_routes": static_routes,
        "authorization": "local service audit only; no submission",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
