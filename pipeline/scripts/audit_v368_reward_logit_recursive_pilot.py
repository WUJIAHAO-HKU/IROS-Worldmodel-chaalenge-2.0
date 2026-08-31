#!/usr/bin/env python3
"""Compare a v368 right expert with the frozen v354 public recursive baseline."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.autoregressive_unet_runtime import Track2AutoregressiveUNet


METRICS = (
    "teacher_next_context_rgb_mae",
    "recursive_next_context_rgb_mae",
    "teacher_temporal_delta_error",
    "recursive_temporal_delta_error",
    "teacher_reward_absolute_error",
    "recursive_reward_absolute_error",
)


def mean(aggregates: dict, split: str, metric: str) -> float:
    value = aggregates[split]["all"][metric]["mean"]
    if value is None:
        raise RuntimeError(f"empty aggregate {split}/{metric}")
    return float(value)


def ratio(candidate: float, baseline: float) -> float:
    return float(candidate / max(baseline, 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--reward-checkpoint", required=True, type=Path)
    parser.add_argument("--t5-model", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preregistration = json.loads(args.preregistration.read_text())
    if preregistration.get("format") != (
        "strict-track2-v368-public-right-reward-logit-recursive-pilot-preregistration-v1"
    ):
        raise RuntimeError("wrong preregistration")
    if args.checkpoint_step not in preregistration["selection"]["candidate_steps"]:
        raise RuntimeError("checkpoint was not preregistered")

    baseline_document = json.loads(args.baseline_report.read_text())
    baseline_rows = baseline_document["rows"]["candidate"]
    if len(baseline_rows) != 512:
        raise RuntimeError("frozen v354 baseline report does not contain 512 rows")
    mapping = json.loads(args.instruction_map.read_text())["episode_to_instruction"]

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)

    original_instantiate = shared.instantiate

    def instantiate(_name, _args):
        return Track2AutoregressiveUNet(args.candidate_checkpoint, args.device)

    shared.instantiate = instantiate
    try:
        candidate_rows = shared.run_model("candidate", args, mapping, reward)
    finally:
        shared.instantiate = original_instantiate
    del reward
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    if [row["key"] for row in baseline_rows] != [row["key"] for row in candidate_rows]:
        raise RuntimeError("candidate rows do not align with frozen baseline")
    baseline = shared.aggregate(baseline_rows, set())
    candidate = shared.aggregate(candidate_rows, set())
    comparisons = {
        split: {
            metric: ratio(mean(candidate, split, metric), mean(baseline, split, metric))
            for metric in METRICS
        }
        for split in RIGHT_EPISODES
    }
    validation = comparisons["validation"]
    checks = {
        "exact_512_aligned_rows": len(candidate_rows) == 512,
        "validation_recursive_rgb_ratio_le_1p01": (
            validation["recursive_next_context_rgb_mae"] <= 1.01
        ),
        "validation_recursive_temporal_ratio_le_1p01": (
            validation["recursive_temporal_delta_error"] <= 1.01
        ),
        "validation_recursive_reward_mae_ratio_le_0p95": (
            validation["recursive_reward_absolute_error"] <= 0.95
        ),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v368-public-right-reward-logit-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": args.checkpoint_step,
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "baseline": "frozen v354 right expert rows from v356",
        "coverage": "512 public right holdout windows in eight recursive alignment chains",
        "decision_split": "validation (episodes 7 and 18)",
        "confirmation_only_split": "local_test (episodes 6 and 22)",
        "aggregates": {"baseline": baseline, "candidate": candidate},
        "candidate_over_v354": comparisons,
        "checks": checks,
        "passed": passed,
        "authorizes_trainmode_rollout32_only": passed,
        "evidence_sha256": {
            "candidate_model": sha256(args.candidate_checkpoint / "model.pt"),
            "baseline_report": sha256(args.baseline_report),
            "preregistration": sha256(args.preregistration),
        },
        "guards": {
            "public_holdout_only": True,
            "hidden_or_final_data": False,
            "official_batch16_outcomes_used": False,
            "real_submission": False,
        },
        "rows": candidate_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checkpoint_step": args.checkpoint_step, "ratios": comparisons, "checks": checks, "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
