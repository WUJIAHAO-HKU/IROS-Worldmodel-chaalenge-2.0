#!/usr/bin/env python3
"""Fail-closed contract audit for the V447 train-only identifiability probe."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


PREREG_FORMAT = "strict-track2-v447-right-action-identifiability-preregistration-v1"
REPORT_FORMAT = "strict-track2-v447-right-action-identifiability-probe-v1"
AUDIT_FORMAT = "strict-track2-v447-right-action-identifiability-audit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    report = json.loads(args.report.read_text())
    source = args.probe.read_text()
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    folds = report.get("folds", [])
    episodes = report.get("episodes", {})
    gates = report.get("gates", {})
    support = report.get("action_support", {})
    evidence = prereg.get("evidence_sha256", {})
    held = [episode for fold in folds for episode in fold.get("holdout_episodes", [])]
    computed_fold_passes = sum(
        float(row.get("action_error_over_context_error", 2)) <= 0.95
        and float(row.get("action_error_over_phase_shuffle_error", 2)) <= 0.97
        for row in folds
    )
    computed_episode_passes = sum(
        float(row.get("action_error_over_context_error", 2)) <= 0.95
        and float(row.get("action_error_over_phase_shuffle_error", 2)) <= 0.97
        for row in episodes.values()
    )
    expected_pass = computed_fold_passes >= 4 and computed_episode_passes >= 12
    checks = {
        "preregistration_format": prereg.get("format") == PREREG_FORMAT,
        "report_format": report.get("format") == REPORT_FORMAT,
        "probe_hash_bound": evidence.get("probe") == sha256(args.probe),
        "report_preregistration_bound": report.get("sha256", {}).get("preregistration") == sha256(args.preregistration),
        "public_train40_right15": len(prereg.get("data", {}).get("train_episodes", [])) == 40 and len(prereg.get("data", {}).get("right_train_episodes", [])) == 15,
        "validation10_excluded": len(prereg.get("data", {}).get("validation_episodes_excluded", [])) == 10 and set(prereg["data"]["right_train_episodes"]).isdisjoint(prereg["data"]["validation_episodes_excluded"]),
        "coverage_120": report.get("coverage", {}).get("rows") == 120 and all(value == 8 for value in report.get("coverage", {}).get("rows_per_episode", {}).values()),
        "five_folds_fit12_hold3": len(folds) == 5 and all(len(row.get("fit_episodes", [])) == 12 and len(row.get("holdout_episodes", [])) == 3 and row.get("fit_rows") == 96 and row.get("holdout_rows") == 24 for row in folds),
        "fold_partition_exact": len(held) == 15 and sorted(held) == sorted(prereg["data"]["right_train_episodes"]),
        "fit_only_projectors": all(row.get("fit_only_projectors") is True for row in folds),
        "same_capacity_64": all(row.get("input_dimension_equal") == {"context_only": 64, "action": 64} and row.get("same_ridge_lambda") == 1.0 and row.get("output_latent_dimension") == 16 for row in folds),
        "phase_shuffle_complete": all(all(len(value) == 3 for value in row.get("phase_shuffle_groups", {}).values()) and len(row.get("phase_shuffle_groups", {})) == 8 for row in folds),
        "fold_gate_recomputed": gates.get("passing_folds") == computed_fold_passes,
        "episode_gate_recomputed": gates.get("passing_episodes") == computed_episode_passes,
        "decision_recomputed": report.get("passed") is expected_pass,
        "action_rank_reported": all(int(row.get("action_numerical_rank", 0)) > 0 and int(row.get("action_rank95", 0)) > 0 for row in folds) and int(support.get("global_numerical_rank", 0)) > 0,
        "paired_support_reported": all(key in support for key in ("exact_same_context_different_action_pairs", "near_context_different_action_pair_count", "failure_target_rows", "intervention_or_counterfactual_target_rows")),
        "probe_has_no_reward_import": not any("reward" in value.lower() for value in imports),
        "probe_has_no_torch_or_gpu": not any(value == "torch" or value.startswith("torch.") for value in imports),
        "float_no_round_cap_gate": report.get("guards", {}).get("float_latent_no_round_or_cap") is True and report.get("guards", {}).get("runtime_gate_used") is False,
        "no_v169_or_reward": report.get("guards", {}).get("v169_used") is False and report.get("guards", {}).get("reward_used") is False,
        "outcome_descriptive_only": report.get("guards", {}).get("outcome_used_for_fit_or_gate") is False and report.get("guards", {}).get("outcome_labels_counted_descriptive_only") is True,
        "no_dev_final_policy": report.get("guards", {}).get("validation_development_or_final_used") is False and report.get("guards", {}).get("policy_loaded_or_modified") is False and report.get("guards", {}).get("policy_updates") == 0,
        "no_submission_or_authority": report.get("guards", {}).get("real_submission") is False and report.get("guards", {}).get("s1_authorized") is False and report.get("guards", {}).get("rl_authorized") is False and report.get("guards", {}).get("formal_authorized") is False,
    }
    contract_passed = all(value is True for value in checks.values())
    sufficient = contract_passed and expected_pass
    audit = {
        "format": AUDIT_FORMAT,
        "contract_passed": contract_passed,
        "passed": sufficient,
        "decision": "authorize_one_trainonly_dynamics_latent_preregistration" if sufficient else "current_right15_insufficient_require_paired_interventions",
        "checks": checks,
        "identifiability": {
            "passing_folds": computed_fold_passes,
            "passing_episodes": computed_episode_passes,
            "required_folds": 4,
            "required_episodes": 12,
        },
        "action_support": support,
        "sha256": {
            "preregistration": sha256(args.preregistration),
            "report": sha256(args.report),
            "probe": sha256(args.probe),
        },
        "guards": {
            "model_trained_or_packaged": False,
            "service_started": False,
            "reward_used": False,
            "policy_updates": 0,
            "real_submission": False,
            "s1_authorized": False,
            "rl_authorized": False,
            "formal_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"contract_passed": contract_passed, "passed": sufficient, "decision": audit["decision"]}, indent=2))
    return 0 if sufficient else 2


if __name__ == "__main__":
    raise SystemExit(main())
