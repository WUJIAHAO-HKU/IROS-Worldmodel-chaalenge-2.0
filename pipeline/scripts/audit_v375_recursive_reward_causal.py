#!/usr/bin/env python3
"""Paired all-offset recursive RGB/reward gate for frozen v375 versus v355."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


def instantiate(name, args):
    if name == "baseline":
        manifest = json.loads((args.baseline_checkpoint_dir / "arm_routed_autoregressive_manifest.json").read_text())
        return Track2ArmRoutedAutoregressiveUNet(
            args.baseline_checkpoint_dir / manifest["left_expert"],
            args.baseline_checkpoint_dir / manifest["right_expert"], args.device,
        )
    return Track2V375BoundedCartesianPhase(args.candidate_checkpoint_dir, args.library_index, args.device)


def metric(aggregates, model, split, key):
    return float(aggregates[model][split]["all"][key]["mean"])


def ratio(candidate, baseline):
    return float(candidate / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("baseline-checkpoint-dir", "candidate-checkpoint-dir", "library-index", "windows", "instruction-map", "reward-checkpoint", "t5-model", "preregistration", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v375-bounded-cartesian-phase-pilot-preregistration-v1":
        raise RuntimeError("wrong v375 preregistration")
    if not prereg["selection"]["checks"] or not all(prereg["selection"]["checks"].values()):
        raise RuntimeError("v375 bounded selection did not pass")
    mapping = json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    original = shared.instantiate
    shared.instantiate = instantiate
    try:
        rows = {name: shared.run_model(name, args, mapping["episode_to_instruction"], reward) for name in ("baseline", "candidate")}
    finally:
        shared.instantiate = original
    if [row["key"] for row in rows["baseline"]] != [row["key"] for row in rows["candidate"]] or len(rows["candidate"]) != 512:
        raise RuntimeError("recursive rows misaligned")
    aggregates = {name: shared.aggregate(value, set()) for name, value in rows.items()}
    keys = (
        "teacher_next_context_rgb_mae", "recursive_next_context_rgb_mae",
        "teacher_temporal_delta_error", "recursive_temporal_delta_error",
        "teacher_reward_absolute_error", "recursive_reward_absolute_error",
    )
    comparisons = {
        split: {key: ratio(metric(aggregates, "candidate", split, key), metric(aggregates, "baseline", split, key)) for key in keys}
        for split in RIGHT_EPISODES
    }
    reward_gain = {
        split: metric(aggregates, "candidate", split, "recursive_terminal_reward") - metric(aggregates, "baseline", split, "recursive_terminal_reward")
        for split in RIGHT_EPISODES
    }
    hit_gain = {
        split: float(aggregates["candidate"][split]["all"]["recursive_reward_hit_rate_at_0p9"] - aggregates["baseline"][split]["all"]["recursive_reward_hit_rate_at_0p9"])
        for split in RIGHT_EPISODES
    }
    checks = {"exact_512_rows": len(rows["candidate"]) == 512}
    for split in RIGHT_EPISODES:
        prefix = "validation" if split == "validation" else "local"
        checks.update({
            f"{prefix}_teacher_rgb_ratio_le_0p98": comparisons[split]["teacher_next_context_rgb_mae"] <= 0.98,
            f"{prefix}_recursive_rgb_ratio_le_0p95": comparisons[split]["recursive_next_context_rgb_mae"] <= 0.95,
            f"{prefix}_teacher_temporal_ratio_le_1p00": comparisons[split]["teacher_temporal_delta_error"] <= 1.00,
            f"{prefix}_recursive_temporal_ratio_le_1p00": comparisons[split]["recursive_temporal_delta_error"] <= 1.00,
            f"{prefix}_recursive_reward_error_improves": comparisons[split]["recursive_reward_absolute_error"] < 1.00,
            f"{prefix}_recursive_reward_nonzero_gain": reward_gain[split] >= 0.02 or hit_gain[split] >= 0.05,
        })
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v375-recursive-reward-causal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": prereg["model_version"],
        "coverage": "512 public right holdout windows in eight recursive alignment chains",
        "aggregates": aggregates, "v375_over_v355": comparisons,
        "recursive_reward_mean_gain": reward_gain, "recursive_reward_hit_rate_gain": hit_gain,
        "checks": checks, "passed": passed, "authorizes_fixed_official_rl": passed,
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "baseline_manifest": sha256(args.baseline_checkpoint_dir / "arm_routed_autoregressive_manifest.json"),
            "candidate_manifest": sha256(args.candidate_checkpoint_dir / "cartesian_phase_manifest.json"),
            "library": sha256(args.library_index), "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_world_model_holdout_only": True, "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False, "official_reward_modified": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v375_over_v355": comparisons, "reward_gain": reward_gain, "hit_gain": hit_gain, "checks": checks, "passed": passed}, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
