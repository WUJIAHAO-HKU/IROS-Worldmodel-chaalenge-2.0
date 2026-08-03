#!/usr/bin/env python3
"""Exercise one RLinf-style reset/chunk request through the Track 2 service."""

from __future__ import annotations

import argparse
import base64
import pickle

import numpy as np
import requests
import torch

from wam_pipeline.data import load_window_npz


def encode_payload(value) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def decode_payload(value: str):
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def post(url: str, path: str, value):
    response = requests.post(f"{url.rstrip('/')}{path}", json={"payload": encode_payload(value)}, timeout=600.0)
    response.raise_for_status()
    return decode_payload(response.json()["payload"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18080")
    parser.add_argument("--window", required=True)
    args = parser.parse_args()
    window = load_window_npz(args.window)
    context = torch.from_numpy(window.context_frames).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)
    current_obs = context.unsqueeze(0).unsqueeze(2)
    reset = post(
        args.bridge_url,
        "/reset",
        {
            "current_obs": current_obs,
            "condition_action": torch.from_numpy(np.vstack([np.zeros((1, 14), np.float32), window.history_actions])).unsqueeze(0),
            "task_descriptions": ["adjust bottle"],
            "elapsed_steps": 0,
        },
    )
    if reset != {"status": "reset"}:
        raise RuntimeError(f"unexpected bridge reset response: {reset}")
    first_actions = torch.from_numpy(window.future_actions).unsqueeze(0)
    result = post(args.bridge_url, "/chunk_step", {"actions": first_actions})
    current = result["current_obs"]
    if not isinstance(current, torch.Tensor) or current.shape != (1, 3, 1, 13, 256, 256):
        raise RuntimeError(f"unexpected current_obs shape: {getattr(current, 'shape', None)}")
    if result.get("elapsed_steps") != 8 or not torch.isfinite(current).all():
        raise RuntimeError("bridge returned invalid state")
    second_actions = first_actions + 0.01
    result = post(args.bridge_url, "/chunk_step", {"actions": second_actions})
    current = result["current_obs"]
    if current.shape != (1, 3, 1, 13, 256, 256) or result.get("elapsed_steps") != 16:
        raise RuntimeError("bridge did not preserve its two-round rollout state")
    print("RLinf Track 2 bridge test passed")


if __name__ == "__main__":
    main()
