#!/usr/bin/env python3
"""Public recursive gate for the final strong-dose v369 U-Net pilot."""

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


def metric(aggregates: dict, split: str, name: str) -> float:
    value = aggregates[split]["all"][name]["mean"]
    if value is None:
        raise RuntimeError(f"empty metric {split}/{name}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--reward-checkpoint", required=True, type=Path)
    parser.add_argument("--t5-model", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v369-final-strong-reward-logit-pilot-preregistration-v1":
        raise RuntimeError("wrong v369 preregistration")
    if args.checkpoint_step not in prereg["training"]["candidate_steps"]:
        raise RuntimeError("checkpoint was not preregistered")

    baseline_doc = json.loads(args.baseline_report.read_text())
    baseline_rows = baseline_doc["rows"]["candidate"]
    mapping = json.loads(args.instruction_map.read_text())["episode_to_instruction"]
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    original = shared.instantiate
    shared.instantiate = lambda _name, _args: Track2AutoregressiveUNet(
        args.candidate_checkpoint, args.device
    )
    try:
        candidate_rows = shared.run_model("candidate", args, mapping, reward)
    finally:
        shared.instantiate = original
    del reward
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if len(candidate_rows) != 512 or [r["key"] for r in baseline_rows] != [r["key"] for r in candidate_rows]:
        raise RuntimeError("recursive rows are incomplete or misaligned")

    baseline = shared.aggregate(baseline_rows, set())
    candidate = shared.aggregate(candidate_rows, set())
    comparisons = {
        split: {
            name: float(metric(candidate, split, name) / max(metric(baseline, split, name), 1e-12))
            for name in METRICS
        }
        for split in RIGHT_EPISODES
    }
    decision = comparisons["validation"]
    checks = {
        "exact_512_aligned_rows": True,
        "validation_recursive_rgb_ratio_le_1p01": decision["recursive_next_context_rgb_mae"] <= 1.01,
        "validation_recursive_temporal_ratio_le_1p01": decision["recursive_temporal_delta_error"] <= 1.01,
        "validation_recursive_reward_mae_ratio_le_0p95": decision["recursive_reward_absolute_error"] <= 0.95,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v369-final-strong-reward-logit-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": args.checkpoint_step,
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "coverage": "512 public right holdout windows in eight recursive alignment chains",
        "decision_split": "validation episodes 7 and 18",
        "confirmation_only_split": "local_test episodes 6 and 22",
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
    print(json.dumps({"step": args.checkpoint_step, "ratios": comparisons, "checks": checks, "passed": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
