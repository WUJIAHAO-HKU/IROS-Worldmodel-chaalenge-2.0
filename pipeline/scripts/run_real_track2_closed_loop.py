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


def build_policy(
    checkpoint: Path,
    device: torch.device,
    state_dict_path: Path | None = None,
):
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
    policy = get_model(cfg)
    if state_dict_path is not None:
        state_dict = torch.load(
            state_dict_path, map_location="cpu", mmap=True, weights_only=True
        )
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        # Training checkpoints include the actor value head, while this
        # inference-only policy deliberately does not construct one.
        disallowed_missing = [key for key in missing if "value" not in key]
        disallowed_unexpected = [key for key in unexpected if "value" not in key]
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "policy state dict mismatch: "
                f"missing={disallowed_missing[:8]}, unexpected={disallowed_unexpected[:8]}"
            )
    return policy.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        required=True,
        nargs="+",
        help="One or more public reset episode*.npy files",
    )
    parser.add_argument("--bridge-url", default="http://127.0.0.1:18080")
    parser.add_argument("--policy", default="artifacts/official_resources/pi05_adjust_bottle")
    parser.add_argument(
        "--policy-state-dict",
        help="Optional RLinf full_weights.pt to overlay on the base policy",
    )
    parser.add_argument("--reward", default="artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt")
    parser.add_argument("--t5-model", default="artifacts/official_resources/reward_model/t5-base")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    reset_paths = [Path(path) for path in args.reset]
    required_paths = [
        Path(args.policy),
        Path(args.reward),
        Path(args.t5_model),
        *reset_paths,
    ]
    if args.policy_state_dict:
        required_paths.append(Path(args.policy_state_dict))
    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"missing official component: {path}")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    from wam_pipeline.rlinf_bridge.server import _decode_payload, _encode_payload

    import requests

    device = torch.device(args.device)
    policy = build_policy(
        Path(args.policy),
        device,
        Path(args.policy_state_dict) if args.policy_state_dict else None,
    )
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(Path(args.reward)), config={"t5_model_name": str(Path(args.t5_model))}
    ).to(device).eval()
    bridge_url = args.bridge_url.rstrip("/")

    def post(path: str, payload: dict) -> dict:
        response = requests.post(
            f"{bridge_url}{path}", json={"payload": _encode_payload(payload)}, timeout=600.0
        )
        response.raise_for_status()
        return _decode_payload(response.json()["payload"])

    episode_results = []
    all_reward_means = []
    for episode_offset, reset_path in enumerate(reset_paths):
        frames, reset_actions, instruction = load_reset(reset_path)
        current = (
            torch.from_numpy(frames)
            .permute(3, 0, 1, 2)
            .float()
            .div(127.5)
            .sub(1.0)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        # Match WanEnv.reset(): slot zero belongs to the immutable reference
        # frame; the following four slots condition o1..o4.
        condition = torch.zeros((1, 5, 14), dtype=torch.float32)
        condition[:, 1:, :] = torch.from_numpy(reset_actions[1:]).unsqueeze(0)
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

        episode_seed = args.seed + episode_offset
        torch.manual_seed(episode_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(episode_seed)
        round_results = []
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
                actions, _ = policy.predict_action_batch(
                    policy_input, mode="eval", compute_values=False
                )
            if actions.shape != (1, 8, 14) or not torch.isfinite(actions).all():
                raise RuntimeError(f"policy returned invalid actions: {tuple(actions.shape)}")
            response = post("/chunk_step", {"actions": actions.detach().cpu()})
            current = response["current_obs"]
            if current.shape != (1, 3, 1, 13, 256, 256) or not torch.isfinite(current).all():
                raise RuntimeError("bridge returned invalid world-model state")
            generated = (
                current[:, :, 0, -8:]
                .permute(0, 3, 1, 2, 4)
                .reshape(8, 3, 256, 256)
            )
            with torch.inference_mode():
                scores = reward.compute_reward(
                    (generated.to(device) + 1.0).div(2.0), [instruction] * 8
                )
            if scores.shape != (8,) or not torch.isfinite(scores).all():
                raise RuntimeError(
                    f"official reward returned invalid scores: {tuple(scores.shape)}"
                )
            reward_mean = float(scores.mean().cpu())
            all_reward_means.append(reward_mean)
            round_results.append(
                {
                    "round": round_index + 1,
                    "elapsed_steps": int(response["elapsed_steps"]),
                    "action_min": float(actions.min().cpu()),
                    "action_max": float(actions.max().cpu()),
                    "official_reward_mean": reward_mean,
                    "official_reward_min": float(scores.min().cpu()),
                    "official_reward_max": float(scores.max().cpu()),
                }
            )
        episode_results.append(
            {
                "reset": str(reset_path.resolve()),
                "instruction": instruction,
                "seed": episode_seed,
                "rounds": round_results,
            }
        )
    print(
        json.dumps(
            {
                "policy_state_dict": (
                    str(Path(args.policy_state_dict).resolve())
                    if args.policy_state_dict
                    else None
                ),
                "episodes": episode_results,
                "aggregate": {
                    "num_episodes": len(episode_results),
                    "num_rounds": len(all_reward_means),
                    "official_reward_mean": float(np.mean(all_reward_means)),
                    "official_reward_min_round_mean": float(np.min(all_reward_means)),
                    "official_reward_max_round_mean": float(np.max(all_reward_means)),
                },
                "note": "public-data local smoke only; not an organizer hidden held-out score",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
