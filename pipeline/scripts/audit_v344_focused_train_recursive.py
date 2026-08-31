#!/usr/bin/env python3
"""Focused public-train gate for the conservative v344 alpha."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_v342_focused_train_recursive as shared
from audit_v310_full_mirror_causal_gate import sha256
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
        "v340-train-report", "preregistration", "output",
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
    if prereg.get("format") != "strict-track2-v344-focused-train-recursive-preregistration-v1":
        raise RuntimeError("wrong v344 preregistration")
    baseline_report = json.loads(args.v340_train_report.read_text())
    if baseline_report.get("passed") is not True:
        raise RuntimeError("v340 full train gate failed")
    mapping = json.loads(args.instruction_map.read_text())
    episodes = [
        int(episode) for episode in mapping["train_episodes"]
        if mapping["arm_by_episode"][str(episode)] == "right"
    ]
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    original = shared.Track2V342TemporalBlendedPublicReanchor
    shared.Track2V342TemporalBlendedPublicReanchor = Track2V344ConservativeTemporalReanchor
    try:
        candidate = shared.run_candidate(args, episodes, mapping["episode_to_instruction"], reward)
    finally:
        shared.Track2V342TemporalBlendedPublicReanchor = original
    baseline = [
        row for row in baseline_report["rows"]["v334"]
        if row["alignment"] in shared.ALIGNMENTS
    ]
    if [row["key"] for row in baseline] != [row["key"] for row in candidate]:
        raise RuntimeError("v344 focused rows misaligned")
    direct = {row["key"] for row in candidate if row["recursive_reanchor"]}
    aggregates = {
        "v334": shared.aggregate(baseline, direct),
        "v344": shared.aggregate(candidate, direct),
    }
    comparisons = {
        group: {
            metric: ratio(
                aggregates["v344"][group][metric]["mean"],
                aggregates["v334"][group][metric]["mean"],
            )
            for metric in ("recursive_rgb_mae", "recursive_temporal_error", "recursive_reward_error")
        }
        for group in ("all", "direct")
    }
    teacher_rgb_delta = max(
        abs(a["teacher_rgb_mae"] - b["teacher_rgb_mae"])
        for a, b in zip(candidate, baseline, strict=True)
    )
    teacher_temporal_delta = max(
        abs(a["teacher_temporal_error"] - b["teacher_temporal_error"])
        for a, b in zip(candidate, baseline, strict=True)
    )
    checks = {
        "exact_484_rows": len(candidate) == 484,
        "teacher_precheck_reanchors_zero": not any(row["teacher_reanchor_precheck"] for row in candidate),
        "teacher_runtime_reanchors_zero": not any(row["teacher_reanchor_runtime"] for row in candidate),
        "teacher_rgb_metric_max_delta_le_0p01": teacher_rgb_delta <= 0.01,
        "teacher_temporal_metric_max_delta_le_0p01": teacher_temporal_delta <= 0.01,
        "direct_reanchors_ge_20": len(direct) >= 20,
        "direct_rgb_ratio_le_0p90": comparisons["direct"]["recursive_rgb_mae"] <= 0.90,
        "direct_temporal_ratio_le_0p90": comparisons["direct"]["recursive_temporal_error"] <= 0.90,
        "direct_reward_error_ratio_le_1p10": comparisons["direct"]["recursive_reward_error"] <= 1.10,
        "direct_reward_mean_ge_0p90": aggregates["v344"]["direct"]["recursive_reward"]["mean"] >= 0.90,
        "direct_reward_hit_rate_ge_0p85": aggregates["v344"]["direct"]["recursive_reward_hit_rate_at_0p9"] >= 0.85,
        "all_rgb_ratio_le_1p02": comparisons["all"]["recursive_rgb_mae"] <= 1.02,
        "all_temporal_ratio_le_1p02": comparisons["all"]["recursive_temporal_error"] <= 1.02,
        "all_reward_error_ratio_le_1p02": comparisons["all"]["recursive_reward_error"] <= 1.02,
    }
    report = {
        "format": "strict-track2-v344-focused-train-recursive-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v344-conservative-temporal-reanchor-alpha0p75-v339-v334",
        "alignments": list(shared.ALIGNMENTS), "rows_found": len(candidate),
        "direct_reanchor_count": len(direct),
        "teacher_metric_max_delta": {"rgb": teacher_rgb_delta, "temporal": teacher_temporal_delta},
        "aggregates": aggregates, "v344_over_v334": comparisons,
        "checks": checks, "passed": all(checks.values()),
        "authorizes_public_holdout_gate": all(checks.values()),
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "v340_train_report": sha256(args.v340_train_report),
            "recursive_ood_gate": sha256(args.recursive_ood_gate),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_train_only": True, "official_batch16_outcomes_read": False,
            "hidden_or_final_data": False, "real_submission": False,
        },
        "rows": {"v344": candidate},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct_reanchor_count": len(direct), "teacher_metric_max_delta": report["teacher_metric_max_delta"],
        "v344_over_v334": comparisons, "checks": checks, "passed": report["passed"],
    }, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
