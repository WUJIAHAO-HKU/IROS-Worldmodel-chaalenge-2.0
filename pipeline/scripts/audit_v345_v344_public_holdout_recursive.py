#!/usr/bin/env python3
"""Final all-offset public-WM holdout gate for frozen v344 semantics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_v343_v342_public_holdout_recursive as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.v344_conservative_temporal_reanchor_runtime import (
    Track2V344ConservativeTemporalReanchor,
)


def ratio(a: float, b: float) -> float:
    return float(a / max(b, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "recursive-ood-gate",
        "windows", "instruction-map", "reward-checkpoint", "t5-model",
        "v335-baseline-report", "v335-causal-report", "v344-train-report",
        "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    parser.add_argument("--feature-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v345-v344-public-holdout-recursive-preregistration-v1":
        raise RuntimeError("wrong v345 preregistration")
    focused = json.loads(args.v344_train_report.read_text())
    if focused.get("passed") is not True:
        raise RuntimeError("v344 focused train gate failed")
    causal = json.loads(args.v335_causal_report.read_text())
    if causal.get("passed") is not True:
        raise RuntimeError("v334 teacher causal gate failed")
    baseline_report = json.loads(args.v335_baseline_report.read_text())
    baseline = baseline_report["rows"]["v335"]
    mapping = json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    original = shared.Track2V342TemporalBlendedPublicReanchor
    shared.Track2V342TemporalBlendedPublicReanchor = Track2V344ConservativeTemporalReanchor
    try:
        candidate = shared.run_candidate(args, mapping["episode_to_instruction"], reward)
    finally:
        shared.Track2V342TemporalBlendedPublicReanchor = original
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("holdout baseline/candidate rows misaligned")
    direct = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {
        "v334": shared.aggregate(baseline, direct),
        "v344": shared.aggregate(candidate, direct),
    }
    comparisons = {}
    for split in RIGHT_EPISODES:
        comparisons[split] = {
            group: {
                metric: ratio(
                    aggregates["v344"][split][group][metric]["mean"],
                    aggregates["v334"][split][group][metric]["mean"],
                )
                for metric in (
                    "recursive_next_context_rgb_mae", "recursive_temporal_delta_error",
                    "recursive_reward_absolute_error",
                )
            }
            for group in ("all", "direct")
        }
    counts = {
        split: sum(row["recursive_reanchor"] and row["split"] == split for row in candidate)
        for split in RIGHT_EPISODES
    }
    teacher_rgb_delta = max(
        abs(a["teacher_next_context_rgb_mae"] - b["teacher_next_context_rgb_mae"])
        for a, b in zip(candidate, baseline, strict=True)
    )
    teacher_temporal_delta = max(
        abs(a["teacher_temporal_delta_error"] - b["teacher_temporal_delta_error"])
        for a, b in zip(candidate, baseline, strict=True)
    )
    checks = {
        "v344_focused_train_gate": True,
        "v334_teacher_causal_gate": True,
        "exact_512_rows": len(candidate) == 512,
        "teacher_reanchor_count_zero": not any(row["teacher_reanchor"] for row in candidate),
        "teacher_rgb_metric_max_delta_le_0p01": teacher_rgb_delta <= 0.01,
        "teacher_temporal_metric_max_delta_le_0p01": teacher_temporal_delta <= 0.01,
        "validation_direct_reanchors_ge_10": counts["validation"] >= 10,
        "local_direct_reanchors_ge_10": counts["local_test"] >= 10,
    }
    for split in RIGHT_EPISODES:
        prefix = "validation" if split == "validation" else "local"
        checks.update({
            f"{prefix}_direct_rgb_ratio_le_0p90": comparisons[split]["direct"]["recursive_next_context_rgb_mae"] <= 0.90,
            f"{prefix}_direct_temporal_ratio_le_0p90": comparisons[split]["direct"]["recursive_temporal_delta_error"] <= 0.90,
            f"{prefix}_direct_reward_error_ratio_le_1p10": comparisons[split]["direct"]["recursive_reward_absolute_error"] <= 1.10,
            f"{prefix}_direct_reward_mean_ge_0p90": aggregates["v344"][split]["direct"]["recursive_terminal_reward"]["mean"] >= 0.90,
            f"{prefix}_direct_reward_hit_rate_ge_0p85": aggregates["v344"][split]["direct"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
            f"{prefix}_all_rgb_ratio_le_1p02": comparisons[split]["all"]["recursive_next_context_rgb_mae"] <= 1.02,
            f"{prefix}_all_temporal_ratio_le_1p02": comparisons[split]["all"]["recursive_temporal_delta_error"] <= 1.02,
            f"{prefix}_all_reward_error_ratio_le_1p02": comparisons[split]["all"]["recursive_reward_absolute_error"] <= 1.02,
        })
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v345-v344-public-holdout-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v344-conservative-temporal-reanchor-alpha0p75-v339-v334",
        "direct_reanchor_counts": counts,
        "teacher_metric_max_delta": {"rgb": teacher_rgb_delta, "temporal": teacher_temporal_delta},
        "aggregates": aggregates, "v344_over_v334": comparisons,
        "checks": checks, "passed": passed,
        "authorizes_service_acceptance_only": passed,
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "v344_train_report": sha256(args.v344_train_report),
            "v335_baseline_report": sha256(args.v335_baseline_report),
            "v335_causal_report": sha256(args.v335_causal_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_world_model_holdout_only": True,
            "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": {"v344": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct_reanchor_counts": counts,
        "teacher_metric_max_delta": report["teacher_metric_max_delta"],
        "v344_over_v334": comparisons, "checks": checks, "passed": passed,
    }, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
