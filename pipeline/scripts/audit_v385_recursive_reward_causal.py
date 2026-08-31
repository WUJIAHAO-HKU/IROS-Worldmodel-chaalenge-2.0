#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v385_native_batch_clean_reanchor_runtime import Track2V385NativeBatchCleanReanchor


def instantiate(name, args):
    cls = Track2V326BlendedPhaseTerminal if name == "baseline" else Track2V385NativeBatchCleanReanchor
    path = args.baseline_checkpoint_dir if name == "baseline" else args.candidate_checkpoint_dir
    return cls(path, args.library_index, args.device)


def mean(aggregates, model, split, metric):
    return float(aggregates[model][split]["all"][metric]["mean"])


def classification(rows, split):
    subset = [row for row in rows if row["split"] == split]
    positive = [row for row in subset if row["gt_terminal_reward"] >= 0.9]
    negative = [row for row in subset if row["gt_terminal_reward"] < 0.9]
    tp = sum(row["recursive_terminal_reward"] >= 0.9 for row in positive)
    fp = sum(row["recursive_terminal_reward"] >= 0.9 for row in negative)
    return {
        "rows": len(subset), "positive": len(positive), "negative": len(negative),
        "true_positive": tp, "false_positive": fp,
        "recall": tp / max(len(positive), 1),
        "false_positive_rate": fp / max(len(negative), 1),
        "precision": tp / max(tp + fp, 1),
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
    if prereg.get("format") != "strict-track2-v385-native-batch-clean-reanchor-preregistration-v1":
        raise RuntimeError("wrong preregistration")
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
    if len(rows["candidate"]) != 512 or [r["key"] for r in rows["baseline"]] != [r["key"] for r in rows["candidate"]]:
        raise RuntimeError("row mismatch")
    aggregates = {name: shared.aggregate(values, set()) for name, values in rows.items()}
    comparisons, classifications, teacher_exact = {}, {}, {}
    checks = {"exact_512": len(rows["candidate"]) == 512}
    for split in RIGHT_EPISODES:
        baseline_cls = classification(rows["baseline"], split)
        candidate_cls = classification(rows["candidate"], split)
        classifications[split] = {"baseline": baseline_cls, "candidate": candidate_cls}
        comparisons[split] = {
            "recursive_rgb_ratio": mean(aggregates, "candidate", split, "recursive_next_context_rgb_mae") / max(mean(aggregates, "baseline", split, "recursive_next_context_rgb_mae"), 1e-9),
            "recursive_temporal_ratio": mean(aggregates, "candidate", split, "recursive_temporal_delta_error") / max(mean(aggregates, "baseline", split, "recursive_temporal_delta_error"), 1e-9),
            "recursive_reward_error_ratio": mean(aggregates, "candidate", split, "recursive_reward_absolute_error") / max(mean(aggregates, "baseline", split, "recursive_reward_absolute_error"), 1e-9),
            "recursive_reward_gain": mean(aggregates, "candidate", split, "recursive_terminal_reward") - mean(aggregates, "baseline", split, "recursive_terminal_reward"),
        }
        teacher_exact[split] = all(
            base["teacher_next_context_sha256"] == candidate["teacher_next_context_sha256"]
            for base, candidate in zip(rows["baseline"], rows["candidate"], strict=True)
            if candidate["split"] == split
        )
        prefix = "validation" if split == "validation" else "local"
        value = comparisons[split]
        checks.update({
            f"{prefix}_teacher_exact": teacher_exact[split],
            f"{prefix}_recursive_rgb_le1p05": value["recursive_rgb_ratio"] <= 1.05,
            f"{prefix}_recursive_temporal_le1p05": value["recursive_temporal_ratio"] <= 1.05,
            f"{prefix}_recursive_reward_error_improves": value["recursive_reward_error_ratio"] < 1.0,
            f"{prefix}_positive_recall_ge0p50": candidate_cls["recall"] >= 0.50,
            f"{prefix}_positive_recall_nonregression": candidate_cls["recall"] >= baseline_cls["recall"],
            f"{prefix}_false_positive_rate_nonregression": candidate_cls["false_positive_rate"] <= baseline_cls["false_positive_rate"],
        })
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v385-recursive-reward-causal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "candidate": prereg["model_version"],
        "aggregates": aggregates, "comparisons_vs_v326": comparisons,
        "success_classification": classifications, "teacher_exact": teacher_exact,
        "checks": checks, "passed": passed,
        "authorizes_rollout_only_preflight": passed, "authorizes_policy_update": False,
        "guards": {"public_holdout_only": True, "runtime_reads_reward_or_outcomes": False, "policy_modified": False, "hidden_or_final_data": False, "real_submission": False},
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons, "success_classification": classifications, "teacher_exact": teacher_exact, "checks": checks, "passed": passed}, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
