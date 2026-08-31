#!/usr/bin/env python3
"""Three-way recursive gate: v432 step50 vs step25 and immutable parent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v432_step25_shortgate import (
    PHASES,
    evaluate_checkpoint,
    fixed_selection,
    load_samples,
    score_reward,
    sha256,
)
from train_v423_mirror_augmented_autoregressive_unet import WindowDataset


EXPECTED_STEP25_MODEL_SHA256 = "dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f"


def phase_metrics(result: dict) -> dict[str, dict[str, float]]:
    return {
        phase: {
            key: float(np.mean([row[key] for row in result["rows"] if row["phase"] == phase]))
            for key in (
                "first8_rgb_mae", "recursive32_rgb_mae", "temporal_delta_error",
                "reward_prediction_mae", "endpoint_reward_prediction_mae",
            )
        }
        for phase in PHASES
    }


def metric_ratios(numerator: dict, denominator: dict) -> dict[str, dict[str, float]]:
    return {
        phase: {
            key: float(numerator[phase][key] / max(denominator[phase][key], 1e-12))
            for key in numerator[phase]
        }
        for phase in PHASES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("parent", "step25", "step50", "windows", "source-data", "split", "reward-checkpoint", "t5-model", "original-preregistration", "step25-gate", "continuation-authorization", "replay-verification", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.original_preregistration.read_text())
    short_gate = json.loads(args.step25_gate.read_text())
    authority = json.loads(args.continuation_authorization.read_text())
    replay_verification = json.loads(args.replay_verification.read_text())
    if prereg.get("format") != "strict-track2-v432-public-mirror-prompt-terminal-preregistration-v1":
        raise RuntimeError("wrong original v432 preregistration")
    if short_gate.get("format") != "strict-track2-v432-step25-shortgate-v1" or not short_gate.get("passed"):
        raise RuntimeError("step25 short gate is absent or failed")
    if authority.get("format") != "strict-track2-v432-step50-continuation-authorization-v1":
        raise RuntimeError("wrong continuation authorization")
    if authority.get("continuation", {}).get("method") != "full deterministic replay from original parent":
        raise RuntimeError("step50 was not authorized as a full deterministic replay")
    if replay_verification.get("format") != "strict-track2-v432-replay-step25-identity-v1":
        raise RuntimeError("wrong replay-step25 verification format")
    if replay_verification.get("passed") is not True:
        raise RuntimeError("replayed step25 did not match the existing accepted checkpoint")
    if replay_verification.get("expected_model_sha256") != EXPECTED_STEP25_MODEL_SHA256:
        raise RuntimeError("replay verification expected the wrong step25 hash")
    if replay_verification.get("observed_model_sha256") != EXPECTED_STEP25_MODEL_SHA256:
        raise RuntimeError("replay verification did not observe the exact step25 hash")
    observed_step25_hash = sha256(args.step25 / "model.pt")
    if observed_step25_hash != EXPECTED_STEP25_MODEL_SHA256:
        raise RuntimeError("step25 model hash differs from the passed short-gate checkpoint")

    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    validation_right = [int(value) for value in split["validation_episodes"] if arms[int(value)] == "right"]
    dataset = WindowDataset(args.windows, validation_right, rollout_horizon=32)
    selection = fixed_selection(dataset, args.source_data)
    samples = load_samples(dataset, selection, prompts)

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    target_frames = np.concatenate([sample["target_frames"] for sample in samples], axis=0)
    target_prompts = [sample["prompt"] for sample in samples for _ in range(32)]
    target_reward = score_reward(reward, target_frames, target_prompts, device, args.reward_batch_size)
    results = {
        "parent": evaluate_checkpoint("parent", args.parent, samples, target_reward, reward, device, args.reward_batch_size),
        "step25": evaluate_checkpoint("step25", args.step25, samples, target_reward, reward, device, args.reward_batch_size),
        "step50": evaluate_checkpoint("step50", args.step50, samples, target_reward, reward, device, args.reward_batch_size),
    }
    metrics = results["parent"]["metrics"].keys()
    ratios_parent = {
        key: float(results["step50"]["metrics"][key] / max(results["parent"]["metrics"][key], 1e-12))
        for key in metrics
    }
    ratios_step25 = {
        key: float(results["step50"]["metrics"][key] / max(results["step25"]["metrics"][key], 1e-12))
        for key in metrics
    }
    phase_values = {name: phase_metrics(result) for name, result in results.items()}
    phase_ratios_parent = metric_ratios(phase_values["step50"], phase_values["parent"])
    phase_ratios_step25 = metric_ratios(phase_values["step50"], phase_values["step25"])
    checks = {
        "exact_32_samples": len(samples) == 32,
        "phase_8_each": all(sum(row["phase"] == phase for row in selection) == 8 for phase in PHASES),
        "absolute_first8_gate": ratios_parent["first8_rgb_mae"] <= 0.998,
        "absolute_recursive32_gate": ratios_parent["recursive32_rgb_mae"] <= 1.002,
        "absolute_temporal_gate": ratios_parent["temporal_delta_error"] <= 1.0,
        "absolute_reward_gate": ratios_parent["reward_prediction_mae"] <= 0.98,
        "absolute_endpoint_gate": ratios_parent["endpoint_reward_prediction_mae"] <= 0.98,
        "step25_first8_nonregression": ratios_step25["first8_rgb_mae"] <= 1.002,
        "step25_recursive32_nonregression": ratios_step25["recursive32_rgb_mae"] <= 1.002,
        "step25_temporal_nonregression": ratios_step25["temporal_delta_error"] <= 1.002,
        "step25_reward_nonregression": ratios_step25["reward_prediction_mae"] <= 1.0,
        "step25_endpoint_nonregression": ratios_step25["endpoint_reward_prediction_mae"] <= 1.0,
        # These phase gates are frozen before step50 training.  They explicitly
        # repair the regressions hidden by step25's aggregate PASS and may not
        # be relaxed after observing step50.
        "postgrasp_recursive32_ratio_le_1p002": phase_ratios_parent["postgrasp"]["recursive32_rgb_mae"] <= 1.002,
        "endpoint_recursive32_ratio_le_1p002": phase_ratios_parent["endpoint"]["recursive32_rgb_mae"] <= 1.002,
        "postgrasp_temporal_ratio_le_1p002": phase_ratios_parent["postgrasp"]["temporal_delta_error"] <= 1.002,
        "endpoint_temporal_ratio_le_1p002": phase_ratios_parent["endpoint"]["temporal_delta_error"] <= 1.002,
        "early_reward_ratio_le_1p05": phase_ratios_parent["early"]["reward_prediction_mae"] <= 1.05,
        "grasp_reward_ratio_le_1p10": phase_ratios_parent["grasp"]["reward_prediction_mae"] <= 1.10,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v432-step50-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "selection": selection,
        "results": results,
        "step50_over_parent": ratios_parent,
        "step50_over_step25": ratios_step25,
        "phase_metrics": phase_values,
        "phase_step50_over_parent": phase_ratios_parent,
        "phase_step50_over_step25": phase_ratios_step25,
        "checks": checks,
        "decision": "recursive gate passed; still no RL authority" if passed else "reject step50; preserve step25 as frozen candidate",
        "evidence_sha256": {
            "original_preregistration": sha256(args.original_preregistration),
            "step25_gate": sha256(args.step25_gate),
            "continuation_authorization": sha256(args.continuation_authorization),
            "replay_verification": sha256(args.replay_verification),
            "parent_model": sha256(args.parent / "model.pt"),
            "step25_model": sha256(args.step25 / "model.pt"),
            "step50_model": sha256(args.step50 / "model.pt"),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_holdout_only": True,
            "outcomes_read": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "step50_over_parent": ratios_parent, "step50_over_step25": ratios_step25, "phase_step50_over_parent": phase_ratios_parent, "phase_step50_over_step25": phase_ratios_step25, "checks": checks}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
