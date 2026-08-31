#!/usr/bin/env python3
"""Exercise the complete WorldArena Track 2 HTTP acceptance matrix."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from wam_pipeline.images import encode_png_base64
from wam_pipeline.profile import (
    ACTION_DIM,
    API_VERSION,
    MAX_BATCH_SIZE,
    MAX_CONCURRENCY,
    MAX_REQUEST_BYTES,
    OFFICIAL_PROFILE_ID,
    RECOMMENDED_TIMEOUT_MS,
)


def image(value: int) -> dict:
    pixels = np.full((256, 256, 3), value % 256, dtype=np.uint8)
    return encode_png_base64(pixels)


def sample(index: int) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "seed": 10_000 + index,
        "context": {
            "frames": [image(index + frame) for frame in range(5)],
            "actions": np.zeros((4, ACTION_DIM), dtype=np.float32).tolist(),
            "states": None,
            "instruction": "adjust the bottle",
        },
        "actions": np.full(
            (8, ACTION_DIM), 0.0025 * (index + 1), dtype=np.float32
        ).tolist(),
    }


def payload(model_version: str, count: int = 1, request_id: str | None = None) -> dict:
    return {
        "api_version": API_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "model_version": model_version,
        "profile_id": OFFICIAL_PROFILE_ID,
        "samples": [sample(index) for index in range(count)],
    }


def assert_error(response: httpx.Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code, response.json()


def pixel_hash(response: dict) -> str:
    digest = hashlib.sha256()
    for prediction in response["predictions"]:
        assert set(prediction) == {"sample_id", "frames"}
        for frame in prediction["frames"]:
            assert {key: frame[key] for key in frame if key != "data"} == {
                "encoding": "png_base64",
                "color_space": "RGB",
                "height": 256,
                "width": 256,
                "channels": 3,
            }
            raw = base64.b64decode(frame["data"], validate=True)
            with Image.open(io.BytesIO(raw)) as decoded:
                rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
            assert rgb.shape == (256, 256, 3)
            digest.update(rgb.tobytes())
    return digest.hexdigest()


async def timed_post(
    client: httpx.AsyncClient, headers: dict[str, str], request: dict
) -> tuple[httpx.Response, float]:
    started = time.perf_counter()
    response = await client.post("/v1/predict", headers=headers, json=request)
    return response, (time.perf_counter() - started) * 1000.0


async def run(base_url: str, token: str, model_version: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(RECOMMENDED_TIMEOUT_MS / 1000.0)
    results: dict[str, object] = {
        "base_url": base_url,
        "model_version": model_version,
        "tests": {},
        "latency_ms": {},
    }

    async with httpx.AsyncClient(
        base_url=base_url, timeout=timeout, follow_redirects=False
    ) as client:
        health = await client.get("/v1/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ready",
            "api_version": API_VERSION,
            "model_version": model_version,
        }
        results["tests"]["health"] = "passed"

        capabilities = await client.get("/v1/capabilities", headers=headers)
        assert capabilities.status_code == 200
        capability_body = capabilities.json()
        assert capability_body["profiles"][0]["profile_id"] == OFFICIAL_PROFILE_ID
        assert capability_body["limits"] == {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_concurrency": MAX_CONCURRENCY,
            "recommended_timeout_ms": RECOMMENDED_TIMEOUT_MS,
        }
        assert capability_body["determinism"] == {
            "seed_supported": True,
            "same_seed_same_pixels": True,
        }
        results["tests"]["capabilities"] = "passed"

        single_request = payload(model_version)
        first, first_ms = await timed_post(client, headers, single_request)
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert set(first_body) == {
            "api_version",
            "request_id",
            "model_version",
            "predictions",
        }
        assert first_body["request_id"] == single_request["request_id"]
        assert len(first_body["predictions"]) == 1
        assert len(first_body["predictions"][0]["frames"]) == 8
        first_hash = pixel_hash(first_body)
        results["tests"]["single_sample"] = "passed"
        results["latency_ms"]["single_uncached"] = first_ms

        retry, retry_ms = await timed_post(client, headers, single_request)
        assert retry.status_code == 200
        assert retry.json() == first_body
        assert pixel_hash(retry.json()) == first_hash
        results["tests"]["same_seed_same_pixels"] = "passed"
        results["tests"]["idempotent_retry"] = "passed"
        results["latency_ms"]["idempotent_cache_hit"] = retry_ms

        changed = json.loads(json.dumps(single_request))
        changed["samples"][0]["seed"] += 1
        conflict = await client.post("/v1/predict", headers=headers, json=changed)
        assert_error(conflict, 409, "REQUEST_ID_CONFLICT")
        results["tests"]["idempotency_conflict"] = "passed"

        batch_request = payload(model_version, MAX_BATCH_SIZE)
        batch, batch_ms = await timed_post(client, headers, batch_request)
        assert batch.status_code == 200, batch.text
        batch_body = batch.json()
        assert [item["sample_id"] for item in batch_body["predictions"]] == [
            item["sample_id"] for item in batch_request["samples"]
        ]
        assert all(len(item["frames"]) == 8 for item in batch_body["predictions"])
        pixel_hash(batch_body)
        results["tests"]["max_batch_8"] = "passed"
        results["latency_ms"]["batch_8"] = batch_ms

        too_many = await client.post(
            "/v1/predict", headers=headers, json=payload(model_version, 9)
        )
        assert_error(too_many, 400, "INVALID_ARGUMENT")
        results["tests"]["batch_9_rejected"] = "passed"

        malformed = payload(model_version)
        malformed["samples"][0]["actions"][0].pop()
        malformed_response = await client.post(
            "/v1/predict", headers=headers, json=malformed
        )
        assert_error(malformed_response, 400, "INVALID_ARGUMENT")
        results["tests"]["bad_action_dimension"] = "passed"

        bad_frame_count = payload(model_version)
        bad_frame_count["samples"][0]["context"]["frames"].pop()
        bad_frame_count_response = await client.post(
            "/v1/predict", headers=headers, json=bad_frame_count
        )
        assert_error(bad_frame_count_response, 400, "INVALID_ARGUMENT")
        results["tests"]["bad_frame_count"] = "passed"

        bad_png = payload(model_version)
        bad_png["samples"][0]["context"]["frames"][0]["data"] = "not-base64"
        bad_png_response = await client.post(
            "/v1/predict", headers=headers, json=bad_png
        )
        assert_error(bad_png_response, 400, "INVALID_ARGUMENT")
        results["tests"]["bad_png"] = "passed"

        non_finite = payload(model_version)
        non_finite["samples"][0]["actions"][0][0] = float("nan")
        non_finite_response = await client.post(
            "/v1/predict",
            headers={**headers, "Content-Type": "application/json"},
            content=json.dumps(non_finite, allow_nan=True).encode("utf-8"),
        )
        assert_error(non_finite_response, 400, "INVALID_ARGUMENT")
        results["tests"]["nan_rejected"] = "passed"

        malformed_json = await client.post(
            "/v1/predict",
            headers={**headers, "Content-Type": "application/json"},
            content=b"{",
        )
        assert_error(malformed_json, 400, "INVALID_ARGUMENT")
        results["tests"]["malformed_json"] = "passed"

        unsupported = payload(model_version)
        unsupported["profile_id"] = "unsupported-profile"
        unsupported_response = await client.post(
            "/v1/predict", headers=headers, json=unsupported
        )
        assert_error(unsupported_response, 422, "UNSUPPORTED_VERSION")
        results["tests"]["unsupported_profile"] = "passed"

        wrong_api = payload(model_version)
        wrong_api["api_version"] = "999"
        wrong_api_response = await client.post(
            "/v1/predict", headers=headers, json=wrong_api
        )
        assert_error(wrong_api_response, 422, "UNSUPPORTED_VERSION")
        results["tests"]["unsupported_api_version"] = "passed"

        wrong_model = payload(model_version)
        wrong_model["model_version"] = "wrong-model-version"
        wrong_model_response = await client.post(
            "/v1/predict", headers=headers, json=wrong_model
        )
        assert_error(wrong_model_response, 422, "UNSUPPORTED_VERSION")
        results["tests"]["unsupported_model_version"] = "passed"

        unauthorized = await client.post(
            "/v1/predict", json=payload(model_version)
        )
        assert_error(unauthorized, 401, "UNAUTHENTICATED")
        results["tests"]["authentication"] = "passed"

        oversized = await client.post(
            "/v1/predict",
            headers={**headers, "Content-Type": "application/json"},
            content=b" " * (MAX_REQUEST_BYTES + 1),
        )
        assert oversized.status_code == 413, oversized.text
        results["tests"]["payload_over_32_mib"] = "passed"

        long_task = asyncio.create_task(
            timed_post(client, headers, payload(model_version, MAX_BATCH_SIZE))
        )
        await asyncio.sleep(0.1)
        overloaded, overloaded_ms = await timed_post(
            client, headers, payload(model_version)
        )
        long_response, long_ms = await long_task
        assert long_response.status_code == 200, long_response.text
        assert_error(overloaded, 429, "OVERLOADED")
        results["tests"]["concurrency_1_overload_429"] = "passed"
        results["latency_ms"]["concurrency_primary"] = long_ms
        results["latency_ms"]["overload_rejection"] = overloaded_ms

        warm_single_ms = []
        for _ in range(5):
            warm_response, warm_ms = await timed_post(
                client, headers, payload(model_version)
            )
            assert warm_response.status_code == 200, warm_response.text
            assert pixel_hash(warm_response.json()) == first_hash
            warm_single_ms.append(warm_ms)
        results["tests"]["five_uncached_repeat_pixel_hashes"] = "passed"
        results["latency_ms"]["warm_single_samples"] = warm_single_ms
        results["latency_ms"]["warm_single_p50"] = float(
            np.percentile(warm_single_ms, 50)
        )
        results["latency_ms"]["warm_single_p95"] = float(
            np.percentile(warm_single_ms, 95)
        )
        results["latency_ms"]["warm_single_p95_under_timeout"] = bool(
            results["latency_ms"]["warm_single_p95"] < RECOMMENDED_TIMEOUT_MS
        )

    measured = [
        float(value)
        for key, value in results["latency_ms"].items()
        if key
        in {
            "single_uncached",
            "batch_8",
            "concurrency_primary",
            "warm_single_p50",
            "warm_single_p95",
        }
    ]
    results["latency_ms"]["uncached_median"] = statistics.median(measured)
    results["latency_ms"]["all_under_recommended_timeout"] = all(
        value < RECOMMENDED_TIMEOUT_MS for value in measured
    )
    results["restart_pixel_hash"] = first_hash
    results["passed"] = True
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with open(args.token_file, "r", encoding="utf-8") as token_file:
        token = token_file.read().strip()
    result = asyncio.run(run(args.base_url, token, args.model_version))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
