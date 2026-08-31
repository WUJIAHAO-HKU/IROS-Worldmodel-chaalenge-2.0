#!/usr/bin/env python3
"""Measure whether a frozen Track 2 service responds to future actions.

This is an offline parent-world-model diagnostic.  It deliberately reads only
the five RGB context frames and the 4+8 action sequence from the held-out
window split; target RGB frames, task-success labels and reward-model outputs
are never loaded.  Therefore it cannot select a model from evaluation success
or outcome labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import requests

from wam_pipeline.images import decode_png_base64_batch, encode_png_base64_batch
from wam_pipeline.profile import API_VERSION, OFFICIAL_PROFILE_ID


def array_sha256(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(value).view(np.uint8)
    return hashlib.sha256(raw).hexdigest()


def service_predict(
    *,
    service_url: str,
    token: str,
    model_version: str,
    context: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    seed: int,
    timeout: float,
) -> np.ndarray:
    payload = {
        "api_version": API_VERSION,
        "request_id": str(uuid.uuid4()),
        "model_version": model_version,
        "profile_id": OFFICIAL_PROFILE_ID,
        "samples": [
            {
                "sample_id": "sensitivity",
                "seed": int(seed),
                "context": {
                    "frames": encode_png_base64_batch(context),
                    "actions": history.astype(np.float32).tolist(),
                    "states": None,
                },
                "actions": future.astype(np.float32).tolist(),
            }
        ],
    }
    response = requests.post(
        f"{service_url.rstrip('/')}/v1/predict",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    response.raise_for_status()
    frames = response.json()["predictions"][0]["frames"]
    decoded = decode_png_base64_batch(frames)
    result = np.stack(decoded)
    if result.shape != (8, 256, 256, 3):
        raise ValueError(f"unexpected prediction shape {result.shape}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8001")
    parser.add_argument("--token", default="local-dev-token")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.max_windows < 1:
        raise SystemExit("--max-windows must be positive")

    split = json.loads(Path(args.split_manifest).read_text())
    validation_ids = set(map(int, split["validation_episodes"]))
    paths = [
        path
        for path in sorted(Path(args.windows).glob("episode*_*.npz"))
        if int(path.name.split("_")[0][7:]) in validation_ids
    ]
    indices = np.linspace(0, len(paths) - 1, min(args.max_windows, len(paths)), dtype=int)
    selected = [paths[index] for index in indices]
    if not selected:
        raise SystemExit("no held-out windows available")

    rows = []
    for index, path in enumerate(selected):
        # Do not access target_frames, capture_success or arm_right.
        with np.load(path, allow_pickle=False) as values:
            context = values["context_frames"].copy()
            history = values["history_actions"].astype(np.float32).copy()
            future = values["future_actions"].astype(np.float32).copy()
            start = int(values["start"])
        hold = np.repeat(history[-1:, :], 8, axis=0)
        reversed_future = future[::-1].copy()
        original_prediction = service_predict(
            service_url=args.service_url, token=args.token,
            model_version=args.model_version, context=context, history=history,
            future=future, seed=args.seed + index, timeout=args.timeout,
        )
        hold_prediction = service_predict(
            service_url=args.service_url, token=args.token,
            model_version=args.model_version, context=context, history=history,
            future=hold, seed=args.seed + index, timeout=args.timeout,
        )
        reversed_prediction = service_predict(
            service_url=args.service_url, token=args.token,
            model_version=args.model_version, context=context, history=history,
            future=reversed_future, seed=args.seed + index, timeout=args.timeout,
        )
        hold_action_l1 = float(np.abs(future - hold).mean())
        reverse_action_l1 = float(np.abs(future - reversed_future).mean())
        hold_pixel_mae = float(np.abs(original_prediction.astype(np.float32) - hold_prediction).mean())
        reverse_pixel_mae = float(np.abs(original_prediction.astype(np.float32) - reversed_prediction).mean())
        rows.append({
            "path": str(path.resolve()), "start": start,
            "context_sha256": array_sha256(context),
            "history_actions_sha256": array_sha256(history),
            "future_actions_sha256": array_sha256(future),
            "hold_action_l1": hold_action_l1,
            "reverse_action_l1": reverse_action_l1,
            "hold_pixel_mae": hold_pixel_mae,
            "reverse_pixel_mae": reverse_pixel_mae,
            "hold_pixel_mae_per_action_l1": hold_pixel_mae / max(hold_action_l1, 1e-12),
            "reverse_pixel_mae_per_action_l1": reverse_pixel_mae / max(reverse_action_l1, 1e-12),
        })
        print(json.dumps(rows[-1]), flush=True)

    report = {
        "format": "strict-track2-parent-action-sensitivity-v1",
        "purpose": "offline frozen parent diagnostic only",
        "data_guard": "uses held-out RGB/action windows only; no target frames, capture_success, reward or real evaluation seeds",
        "service_url": args.service_url,
        "model_version": args.model_version,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "window_count": len(rows),
        "mean_hold_pixel_mae": float(np.mean([row["hold_pixel_mae"] for row in rows])),
        "mean_reverse_pixel_mae": float(np.mean([row["reverse_pixel_mae"] for row in rows])),
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
