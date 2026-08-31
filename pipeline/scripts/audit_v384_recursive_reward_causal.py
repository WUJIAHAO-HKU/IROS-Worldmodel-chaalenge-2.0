#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v384_action_phase_clean_reanchor_runtime import Track2V384ActionPhaseCleanReanchor


def instantiate(name, args):
    cls = Track2V326BlendedPhaseTerminal if name == "baseline" else Track2V384ActionPhaseCleanReanchor
    path = args.baseline_checkpoint_dir if name == "baseline" else args.candidate_checkpoint_dir
    return cls(path, args.library_index, args.device)


def mean(aggregates, model, split, metric):
    return float(aggregates[model][split]["all"][metric]["mean"])


def ratio(value, baseline):
    return value / max(baseline, 1e-9)


def classification(rows, split):
    subset = [row for row in rows if row["split"] == split]
    positive = [row for row in subset if row["gt_terminal_reward"] >= 0.9]
    negative = [row for row in subset if row["gt_terminal_reward"] < 0.9]
    true_positive = sum(row["recursive_terminal_reward"] >= 0.9 for row in positive)
    false_positive = sum(row["recursive_terminal_reward"] >= 0.9 for row in negative)
    predicted_positive = true_positive + false_positive
    return {
        "rows": len(subset),
        "ground_truth_positive": len(positive),
        "ground_truth_negative": len(negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "recall_at_0p9": true_positive / max(len(positive), 1),
        "false_positive_rate_at_0p9": false_positive / max(len(negative), 1),
        "precision_at_0p9": true_positive / max(predicted_positive, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "baseline-checkpoint-dir", "candidate-checkpoint-dir", "library-index", "windows",
        "instruction-map", "reward-checkpoint", "t5-model", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v384-action-phase-clean-reanchor-preregistration-v1":
        raise RuntimeError("wrong preregistration")
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
        raise RuntimeError("row mismatch")
    aggregates = {name: shared.aggregate(values, set()) for name, values in rows.items()}
    comparisons = {}
    classifications = {}
    teacher_exact = {}
    checks = {"exact_512": len(rows["candidate"]) == 512}
    for split in RIGHT_EPISODES:
        comparisons[split] = {
            "recursive_rgb_ratio": ratio(
                mean(aggregates, "candidate", split, "recursive_next_context_rgb_mae"),
                mean(aggregates, "baseline", split, "recursive_next_context_rgb_mae"),
            ),
            "recursive_temporal_ratio": ratio(
                mean(aggregates, "candidate", split, "recursive_temporal_delta_error"),
                mean(aggregates, "baseline", split, "recursive_temporal_delta_error"),
            ),
            "recursive_reward_error_ratio": ratio(
                mean(aggregates, "candidate", split, "recursive_reward_absolute_error"),
                mean(aggregates, "baseline", split, "recursive_reward_absolute_error"),
            ),
            "recursive_reward_gain": (
                mean(aggregates, "candidate", split, "recursive_terminal_reward")
                - mean(aggregates, "baseline", split, "recursive_terminal_reward")
            ),
        }
        classifications[split] = classification(rows["candidate"], split)
        teacher_exact[split] = all(
            base["teacher_next_context_sha256"] == candidate["teacher_next_context_sha256"]
            for base, candidate in zip(rows["baseline"], rows["candidate"], strict=True)
            if candidate["split"] == split
        )
        prefix = "validation" if split == "validation" else "local"
        values = comparisons[split]
        cls = classifications[split]
        checks.update({
            f"{prefix}_teacher_exact": teacher_exact[split],
            f"{prefix}_recursive_rgb_le1p05": values["recursive_rgb_ratio"] <= 1.05,
            f"{prefix}_recursive_temporal_le1p05": values["recursive_temporal_ratio"] <= 1.05,
            f"{prefix}_recursive_reward_error_improves": values["recursive_reward_error_ratio"] < 1.0,
            f"{prefix}_positive_recall_ge0p50": cls["recall_at_0p9"] >= 0.50,
            f"{prefix}_false_positive_rate_le0p10": cls["false_positive_rate_at_0p9"] <= 0.10,
        })
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v384-recursive-reward-causal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": prereg["model_version"],
        "aggregates": aggregates,
        "comparisons_vs_v326": comparisons,
        "success_classification": classifications,
        "teacher_exact": teacher_exact,
        "checks": checks,
        "passed": passed,
        "authorizes_rollout_only_preflight": passed,
        "authorizes_policy_update": False,
        "guards": {
            "public_holdout_only": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "comparisons": comparisons,
        "success_classification": classifications,
        "teacher_exact": teacher_exact,
        "checks": checks,
        "passed": passed,
    }, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
