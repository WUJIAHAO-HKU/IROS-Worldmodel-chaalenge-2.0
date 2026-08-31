#!/usr/bin/env python3
"""Paired 512-window recursive screen of frozen v164-step50 versus v209."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet


def instantiate(name, args):
    root = args.baseline_checkpoint_dir if name == "baseline" else args.candidate_checkpoint_dir
    manifest = json.loads((root / "arm_routed_autoregressive_manifest.json").read_text())
    return Track2ArmRoutedAutoregressiveUNet(
        root / manifest["left_expert"], root / manifest["right_expert"], args.device
    )


def metric(aggregates, model, split, key):
    return float(aggregates[model][split]["all"][key]["mean"])


def ratio(candidate, baseline):
    return float(candidate / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "baseline-checkpoint-dir", "candidate-checkpoint-dir", "windows",
        "instruction-map", "reward-checkpoint", "t5-model", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v392-v164s50-parametric-recursive-preregistration-v1":
        raise RuntimeError("wrong v392 preregistration")
    mapping = json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    original = shared.instantiate
    shared.instantiate = instantiate
    try:
        rows = {
            name: shared.run_model(name, args, mapping["episode_to_instruction"], reward)
            for name in ("baseline", "candidate")
        }
    finally:
        shared.instantiate = original
    if len(rows["candidate"]) != 512 or [r["key"] for r in rows["baseline"]] != [r["key"] for r in rows["candidate"]]:
        raise RuntimeError("recursive rows misaligned")
    aggregates = {name: shared.aggregate(value, set()) for name, value in rows.items()}
    keys = (
        "teacher_next_context_rgb_mae", "recursive_next_context_rgb_mae",
        "teacher_temporal_delta_error", "recursive_temporal_delta_error",
        "recursive_reward_absolute_error",
    )
    comparisons = {
        split: {
            key: ratio(metric(aggregates, "candidate", split, key), metric(aggregates, "baseline", split, key))
            for key in keys
        }
        for split in RIGHT_EPISODES
    }
    checks = {"exact_512_rows": len(rows["candidate"]) == 512}
    for split in RIGHT_EPISODES:
        prefix = "validation" if split == "validation" else "local"
        value = comparisons[split]
        checks.update({
            f"{prefix}_teacher_rgb_le1p00": value["teacher_next_context_rgb_mae"] <= 1.00,
            f"{prefix}_recursive_rgb_le1p00": value["recursive_next_context_rgb_mae"] <= 1.00,
            f"{prefix}_teacher_temporal_le1p02": value["teacher_temporal_delta_error"] <= 1.02,
            f"{prefix}_recursive_temporal_le1p02": value["recursive_temporal_delta_error"] <= 1.02,
            f"{prefix}_recursive_reward_error_le1p02": value["recursive_reward_absolute_error"] <= 1.02,
        })
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v392-v164s50-parametric-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": prereg["fixed_candidate"],
        "comparisons_v164s50_over_v209": comparisons,
        "aggregates": aggregates,
        "checks": checks,
        "passed": passed,
        "authorizes_full_v390_integration_audit": passed,
        "authorizes_rollout_or_policy_update": False,
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "baseline_manifest": sha256(args.baseline_checkpoint_dir / "arm_routed_autoregressive_manifest.json"),
            "candidate_manifest": sha256(args.candidate_checkpoint_dir / "arm_routed_autoregressive_manifest.json"),
        },
        "guards": {
            "public_holdout_only": True, "policy_modified": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons, "checks": checks, "passed": passed}, indent=2), flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
