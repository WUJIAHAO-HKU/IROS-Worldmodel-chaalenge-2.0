#!/usr/bin/env python3
"""Run the patched mirror-transfer path on two public raw observations only."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import openpi.models.model as openpi_model
from audit_v269_offline_policy_trainfit import make_overlay
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.openpi_action_model import track2_right_route_mask


def decode(value) -> np.ndarray:
    with Image.open(io.BytesIO(value.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def instruction(reset_dir: Path, episode: int) -> str:
    reset = np.load(reset_dir / f"episode{episode}.npy", allow_pickle=True)
    for item in reset.flat:
        if isinstance(item, dict) and item.get("instruction"):
            return str(item["instruction"])
    raise RuntimeError(f"episode {episode} lacks instruction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    episodes = [(1, 70, "left"), (20, 70, "right")]
    images, states, prompts = [], [], []
    for episode, frame, _ in episodes:
        with h5py.File(args.source / "data" / f"episode{episode}.hdf5", "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
            use_frame = min(frame, len(actions) - 1)
            states.append(actions[max(use_frame - 1, 0)])
            images.append(decode(item["observation/head_camera/rgb"][use_frame]))
        prompts.append(instruction(args.reset_dir, episode))

    env_obs = {
        "main_images": torch.from_numpy(np.stack(images)),
        "wrist_images": None,
        "extra_view_images": None,
        "states": torch.from_numpy(np.stack(states)),
        "task_descriptions": prompts,
    }
    with tempfile.TemporaryDirectory(prefix="v288_smoke_overlay_") as temporary:
        overlay = make_overlay(Path(temporary), args.official, args.checkpoint, "v169")
        cfg = OmegaConf.create({
            "model_path": str(overlay),
            "openpi": {
                "config_name": "pi05_aloha_robotwin_head_adjust_bottle",
                "num_images_in_input": 1,
                "noise_level": 0.3,
                "action_chunk": 8,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 14,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "use_dsrl": False,
                "track2_eval_mirror_transfer": True,
                "track2_eval_mirror_route_margin": 0.0,
            },
        })
        model = get_model(cfg).cuda().eval()
        device_obs = {
            key: value.cuda() if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }
        with torch.inference_mode():
            torch.manual_seed(28800)
            processed = model.precision_processor(
                model.input_transform(model.obs_processor(device_obs), transpose=False)
            )
            observation = openpi_model.Observation.from_dict(processed)
            direct_outputs = model.sample_actions(observation, mode="eval", compute_values=False)
            direct_actions = model.output_transform(
                {"actions": direct_outputs["actions"], "state": observation.state}
            )["actions"]
            route_mask = track2_right_route_mask(direct_actions, device_obs["states"])
            torch.manual_seed(28800)
            selected_actions, _ = model.predict_action_batch(
                device_obs, mode="eval", compute_values=False
            )

    report = {
        "format": "strict-track2-v288-mirror-transfer-openpi-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": [episode for episode, _, _ in episodes],
        "expected_arms": [arm for _, _, arm in episodes],
        "direct_route_right": route_mask.tolist(),
        "input_image_shape": list(env_obs["main_images"].shape),
        "input_state_shape": list(env_obs["states"].shape),
        "selected_action_shape": list(selected_actions.shape),
        "selected_actions_finite": bool(torch.isfinite(selected_actions).all()),
        "selected_action_abs_max": float(selected_actions.abs().max()),
        "rules": {
            "public_training_observations_only": True,
            "simulator_started": False,
            "reserved_final128_access": False,
            "real_competition_submission": False,
        },
    }
    if report["selected_action_shape"] != [2, 8, 14] or not report["selected_actions_finite"]:
        raise RuntimeError(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
