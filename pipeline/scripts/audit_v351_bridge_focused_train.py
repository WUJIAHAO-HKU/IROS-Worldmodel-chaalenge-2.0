#!/usr/bin/env python3
"""Focused recursive train gate for the frozen v350 residual bridge."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_v342_focused_train_recursive as shared
from audit_v310_full_mirror_causal_gate import sha256
from wam_pipeline.v350_learned_residual_bridge_runtime import Track2V350LearnedResidualBridge


def ratio(a, b):
    return float(a / max(b, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "bridge-profile", "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "v340-train-report", "v350-fit-report", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--feature-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if json.loads(args.preregistration.read_text()).get("format") != "strict-track2-v351-bridge-focused-train-preregistration-v1":
        raise RuntimeError("wrong v351 preregistration")
    if json.loads(args.v350_fit_report.read_text()).get("passed") is not True:
        raise RuntimeError("v350 bridge fit failed")
    baseline_report = json.loads(args.v340_train_report.read_text())
    if baseline_report.get("passed") is not True:
        raise RuntimeError("v340 baseline gate failed")
    mapping = json.loads(args.instruction_map.read_text())
    episodes = [int(e) for e in mapping["train_episodes"] if mapping["arm_by_episode"][str(e)] == "right"]
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    os.environ["WAM_V350_BRIDGE_PROFILE"] = str(args.bridge_profile)
    original = shared.Track2V342TemporalBlendedPublicReanchor
    shared.Track2V342TemporalBlendedPublicReanchor = Track2V350LearnedResidualBridge
    try:
        candidate = shared.run_candidate(args, episodes, mapping["episode_to_instruction"], reward)
    finally:
        shared.Track2V342TemporalBlendedPublicReanchor = original
    baseline = [row for row in baseline_report["rows"]["v334"] if row["alignment"] in shared.ALIGNMENTS]
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("v351 focused rows misaligned")
    direct = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {"v334": shared.aggregate(baseline, direct), "v350": shared.aggregate(candidate, direct)}
    comparisons = {
        group: {metric: ratio(aggregates["v350"][group][metric]["mean"], aggregates["v334"][group][metric]["mean"])
                for metric in ("recursive_rgb_mae", "recursive_temporal_error", "recursive_reward_error")}
        for group in ("all", "direct")
    }
    teacher_rgb_delta = max(abs(a["teacher_rgb_mae"] - b["teacher_rgb_mae"]) for a, b in zip(candidate, baseline, strict=True))
    teacher_temporal_delta = max(abs(a["teacher_temporal_error"] - b["teacher_temporal_error"]) for a, b in zip(candidate, baseline, strict=True))
    checks = {
        "v350_bridge_fit_gate": True, "exact_484_rows": len(candidate) == 484,
        "teacher_precheck_reanchors_zero": not any(r["teacher_reanchor_precheck"] for r in candidate),
        "teacher_runtime_reanchors_zero": not any(r["teacher_reanchor_runtime"] for r in candidate),
        "teacher_rgb_metric_max_delta_le_0p01": teacher_rgb_delta <= 0.01,
        "teacher_temporal_metric_max_delta_le_0p01": teacher_temporal_delta <= 0.01,
        "direct_reanchors_ge_20": len(direct) >= 20,
        "direct_rgb_ratio_le_0p90": comparisons["direct"]["recursive_rgb_mae"] <= 0.90,
        "direct_temporal_ratio_le_0p90": comparisons["direct"]["recursive_temporal_error"] <= 0.90,
        "direct_reward_error_ratio_le_1p10": comparisons["direct"]["recursive_reward_error"] <= 1.10,
        "direct_reward_mean_ge_0p90": aggregates["v350"]["direct"]["recursive_reward"]["mean"] >= 0.90,
        "direct_reward_hit_rate_ge_0p85": aggregates["v350"]["direct"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
        "all_rgb_ratio_le_1p02": comparisons["all"]["recursive_rgb_mae"] <= 1.02,
        "all_temporal_ratio_le_1p02": comparisons["all"]["recursive_temporal_error"] <= 1.02,
        "all_reward_error_ratio_le_1p02": comparisons["all"]["recursive_reward_error"] <= 1.02,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v351-bridge-focused-train-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v350-learned-residual-bridge-v339-v334",
        "direct_reanchor_count": len(direct),
        "teacher_metric_max_delta": {"rgb": teacher_rgb_delta, "temporal": teacher_temporal_delta},
        "aggregates": aggregates, "v350_over_v334": comparisons,
        "checks": checks, "passed": passed, "authorizes_public_holdout_gate": passed,
        "evidence_sha256": {"preregistration": sha256(args.preregistration),
                            "v350_fit_report": sha256(args.v350_fit_report),
                            "bridge_profile": sha256(args.bridge_profile),
                            "v340_train_report": sha256(args.v340_train_report)},
        "guards": {"public_train_only": True, "official_batch16_outcomes_read": False,
                   "hidden_or_final_data": False, "real_submission": False},
        "rows": {"v350": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"direct_reanchor_count": len(direct), "v350_over_v334": comparisons,
                      "teacher_metric_max_delta": report["teacher_metric_max_delta"],
                      "checks": checks, "passed": passed}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
