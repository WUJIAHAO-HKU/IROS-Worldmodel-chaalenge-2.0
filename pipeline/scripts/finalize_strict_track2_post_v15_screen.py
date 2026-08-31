#!/usr/bin/env python3
"""Finalize the preregistered screen for the selected post-V15 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-visual", required=True)
    parser.add_argument("--official-visual", required=True)
    parser.add_argument("--mixed-reward", required=True)
    parser.add_argument("--official-reward", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "mixed_visual": args.mixed_visual,
            "official_visual": args.official_visual,
            "mixed_reward": args.mixed_reward,
            "official_reward": args.official_reward,
            "preregistration": args.preregistration,
        }.items()
    }
    reports = {name: json.loads(path.read_text()) for name, path in paths.items() if name != "preregistration"}
    mv, ov = reports["mixed_visual"], reports["official_visual"]
    mr, reward = reports["mixed_reward"], reports["official_reward"]
    checks = {
        "official_reward_mae_improvement_at_least_3_percent": reward["groups"]["overall"]["improvement"]["reward_mae_improvement_percent"] >= 3.0,
        "official_visual_rgb_nonregression": ov["improvement"]["overall"]["rgb_mae_improvement_percent"] >= 0.0,
        "official_left_visual_nonregression": ov["improvement"]["left"]["rgb_mae_improvement_percent"] >= 0.0,
        "official_right_visual_nonregression": ov["improvement"]["right"]["rgb_mae_improvement_percent"] >= 0.0,
        "official_left_reward_nonregression": reward["groups"]["left"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "official_right_reward_nonregression": reward["groups"]["right"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "synthetic_visual_nonregression": mv["improvement"]["synthetic"]["rgb_mae_improvement_percent"] >= 0.0,
        "synthetic_reward_nonregression": mr["groups"]["synthetic"]["improvement"]["reward_mae_improvement_percent"] >= 0.0,
        "deployed_residual_within_8_of_255": max(
            mv["maximum_observed_absolute_residual_255"],
            ov["maximum_observed_absolute_residual_255"],
        ) <= 8.0001,
    }
    result = {
        "format": "strict-track2-post-v15-parent-screen-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "selected_strength": ov["output_strength"],
        "official_reward_mae_improvement_percent": reward["groups"]["overall"]["improvement"]["reward_mae_improvement_percent"],
        "official_reward_rmse_improvement_percent": reward["groups"]["overall"]["improvement"]["reward_rmse_improvement_percent"],
        "official_visual_rgb_improvement_percent": ov["improvement"]["overall"]["rgb_mae_improvement_percent"],
        "official_left_reward_mae_improvement_percent": reward["groups"]["left"]["improvement"]["reward_mae_improvement_percent"],
        "official_right_reward_mae_improvement_percent": reward["groups"]["right"]["improvement"]["reward_mae_improvement_percent"],
        "maximum_observed_absolute_residual_255": max(
            mv["maximum_observed_absolute_residual_255"],
            ov["maximum_observed_absolute_residual_255"],
        ),
        "reports": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "inference_uses_target_or_reward": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
