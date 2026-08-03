#!/usr/bin/env python3
"""Run two published-policy/world-model/reward chunks against public reset data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def load_reset(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    frames = np.load(path, allow_pickle=True)
    if frames.shape != (5,):
        raise RuntimeError(f"{path} must contain five reset frames")
    images = np.stack([np.asarray(frame["image"], dtype=np.uint8) for frame in frames])
    actions = np.stack([np.asarray(frame["abs_action"], dtype=np.float32) for frame in frames])
    instruction = str(frames[0]["instruction"])
    if images.shape != (5, 256, 256, 3) or actions.shape != (5, 14):
        raise RuntimeError(f"{path} does not match 5x256 RGB / 5x14 action reset contract")
    return images, actions, instruction


def build_policy(checkpoint: Path, device: torch.device):
    os.environ["OPENPI_CKPT_PATH"] = str(checkpoint.resolve())
    from rlinf.models.embodiment.openpi import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint.resolve()),
            "openpi_data": None,
            "openpi": {
                "config_name": "pi05_aloha_robotwin_head_adjust_bottle",
                "num_images_in_input": 1,
                "action_chunk": 8,
                "action_env_dim": 14,
                "num_steps": 5,
                "add_value_head": False,
                "train_expert_only": True,
                "noise_method": "flow_sde",
            },
        }
    )
    return get_model(cfg).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", required=True, help="One public reset episode*.npy")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18080")
    parser.add_argument("--policy", default="artifacts/official_resources/pi05_adjust_bottle")
    parser.add_argument("--reward", default="artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt")
    parser.add_argument("--t5-model", default="artifacts/official_resources/reward_model/t5-base")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    for path in (Path(args.policy), Path(args.reward), Path(args.t5_model)):
        if not path.exists():
            raise SystemExit(f"missing official component: {path}")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    from wam_pipeline.rlinf_bridge.server import _decode_payload, _encode_payload

    import requests

    device = torch.device(args.device)
    policy = build_policy(Path(args.policy), device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(Path(args.reward)), config={"t5_model_name": str(Path(args.t5_model))}
    ).to(device).eval()
    frames, reset_actions, instruction = load_reset(Path(args.reset))
    current = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div(127.5).sub(1.0).unsqueeze(0).unsqueeze(2)
    # Match WanEnv.reset(): slot zero belongs to the immutable reference frame;
    # the four following slots condition o1..o4.  The bridge later replaces
    # those four slots with u4..u7 after each rollout chunk.
    condition = torch.zeros((1, 5, 14), dtype=torch.float32)
    condition[:, 1:, :] = torch.from_numpy(reset_actions[1:]).unsqueeze(0)
    bridge_url = args.bridge_url.rstrip("/")

    def post(path: str, payload: dict) -> dict:
        response = requests.post(
            f"{bridge_url}{path}", json={"payload": _encode_payload(payload)}, timeout=600.0
        )
        response.raise_for_status()
        return _decode_payload(response.json()["payload"])

    reset = post(
        "/reset",
        {
            "current_obs": current,
            "condition_action": condition,
            "task_descriptions": [instruction],
            "elapsed_steps": 0,
        },
    )
    if reset != {"status": "reset"}:
        raise RuntimeError(f"unexpected bridge reset: {reset}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    results = []
    for round_index in range(args.rounds):
        latest = current[:, :, 0, -1].permute(0, 2, 3, 1).to(torch.uint8)
        policy_input = {
            "main_images": latest.to(device),
            "wrist_images": None,
            "extra_view_images": None,
            "states": torch.zeros((1, 14), dtype=torch.float32, device=device),
            "task_descriptions": [instruction],
        }
        with torch.inference_mode():
            actions, _ = policy.predict_action_batch(policy_input, mode="eval", compute_values=False)
        if actions.shape != (1, 8, 14) or not torch.isfinite(actions).all():
            raise RuntimeError(f"policy returned invalid actions: {tuple(actions.shape)}")
        response = post("/chunk_step", {"actions": actions.detach().cpu()})
        current = response["current_obs"]
        if current.shape != (1, 3, 1, 13, 256, 256) or not torch.isfinite(current).all():
            raise RuntimeError("bridge returned invalid world-model state")
        generated = current[:, :, 0, -8:].permute(0, 3, 1, 2, 4).reshape(8, 3, 256, 256)
        with torch.inference_mode():
            scores = reward.compute_reward((generated.to(device) + 1.0).div(2.0), [instruction] * 8)
        if scores.shape != (8,) or not torch.isfinite(scores).all():
            raise RuntimeError(f"official reward returned invalid scores: {tuple(scores.shape)}")
        results.append(
            {
                "round": round_index + 1,
                "elapsed_steps": int(response["elapsed_steps"]),
                "action_min": float(actions.min().cpu()),
                "action_max": float(actions.max().cpu()),
                "official_reward_mean": float(scores.mean().cpu()),
                "official_reward_min": float(scores.min().cpu()),
                "official_reward_max": float(scores.max().cpu()),
            }
        )
    print(
        json.dumps(
            {
                "reset": str(Path(args.reset).resolve()),
                "instruction": instruction,
                "rounds": results,
                "note": "public-data local smoke only; not an organizer hidden held-out score",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
