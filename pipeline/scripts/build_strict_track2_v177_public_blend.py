#!/usr/bin/env python3
"""Build the public-validation-selected v175/v176 parent-model blend.

The operation is outcome-free: it only combines two public-data-only parent
checkpoints after a fixed public frame-prediction validation rule selected
alpha=0.5.  It never reads policy rewards, success labels, or evaluation seeds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
LEFT = JOINT / "v175_v174_public_right_highmotion_refine/best"
RIGHT = JOINT / "v176_v175_public_right_dynamics_refine/best"
OUTPUT = JOINT / "v177_v175_v176_public_balanced_blend_alpha050"
ALPHA_RIGHT = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUTPUT}")
    for checkpoint in (LEFT, RIGHT):
        if not (checkpoint / "model.pt").is_file():
            raise SystemExit(f"missing parent checkpoint: {checkpoint}")

    left_payload = torch.load(LEFT / "model.pt", map_location="cpu", weights_only=True)
    right_payload = torch.load(RIGHT / "model.pt", map_location="cpu", weights_only=True)
    if left_payload.get("format") != "track2-autoregressive-unet-v1" or right_payload.get("format") != "track2-autoregressive-unet-v1":
        raise SystemExit("parents must be autoregressive UNet checkpoints")
    left_state, right_state = left_payload["state_dict"], right_payload["state_dict"]
    if left_state.keys() != right_state.keys():
        raise SystemExit("parent state dictionaries have different keys")
    state = {
        key: value.lerp(right_state[key], ALPHA_RIGHT)
        if value.is_floating_point()
        else value.clone()
        for key, value in left_state.items()
    }

    OUTPUT.mkdir(parents=True)
    torch.save({"format": "track2-autoregressive-unet-v1", "state_dict": state}, OUTPUT / "model.pt")
    for filename in ("action_normalization.npz", "track2_autoregressive_unet_config.npz"):
        shutil.copy2(LEFT / filename, OUTPUT / filename)
    manifest = {
        "backend": "autoregressive_unet",
        "format": "track2-autoregressive-unet-v1",
        "model_version": "track2-v177-public-balanced-blend-alpha050",
        "construction": "parameter_linear_interpolation",
        "alpha_v175": 0.5,
        "alpha_v176": ALPHA_RIGHT,
        "parent_checkpoints": {
            "v175_step400": str(LEFT.resolve()),
            "v176_step100": str(RIGHT.resolve()),
        },
        "parent_model_sha256": {
            "v175_step400": sha256(LEFT / "model.pt"),
            "v176_step100": sha256(RIGHT / "model.pt"),
        },
        "selection": {
            "data": "declared public validation episodes only",
            "outcome_free": True,
            "threshold": 0.03,
            "metric": "mean(global_rollout_mae, global_high_motion_rollout_mae, right_rollout_mae, right_high_motion_rollout_mae)",
            "selected_metric": 0.06218926887959242,
            "candidate_metrics": {
                "alpha_0.25": 0.06222132686525583,
                "alpha_0.50": 0.06218926887959242,
                "alpha_0.75": 0.06222913879901171,
            },
        },
        "prohibited_inputs": ["success labels", "reward outputs", "development-evaluation results", "final-evaluation seeds or results"],
    }
    (OUTPUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT / "model.pt"), "selection": manifest["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
