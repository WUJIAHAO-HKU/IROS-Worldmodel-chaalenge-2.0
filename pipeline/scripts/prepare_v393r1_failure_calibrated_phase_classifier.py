#!/usr/bin/env python3
"""Preregister one public-failure-calibrated terminal phase classifier."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v393r1_public_failure_calibrated_phase_seed1554_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "trainer": ROOT / "pipeline/scripts/train_v393r1_failure_calibrated_phase_classifier.py",
        "feature_contract": ROOT / "pipeline/wam_pipeline/v389_public_recursive_phase_gate.py",
        "public_success_split": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "public_failure_split": J / "onpolicy_windows_full128_stride4/split_manifest.json",
        "public_failure_seed_manifest": J / "onpolicy_train_seeds128/manifest.json",
        "phase_labels": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "v389_training_report": J / "v389r1_public_recursive_phase_classifier_seed1550_20260823/training_report.json",
        "v391_route_diagnosis": O / "runs/v391_v169_v390_route_trace32_seed1497_20260823/audit/route_analysis.json",
        "aborted_v393_preregistration": J / "v393_public_failure_calibrated_phase_seed1554_20260823/preregistration.json",
        "aborted_v393_log": J / "v393_public_failure_calibrated_phase_seed1554_20260823/training.log",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    old = json.loads(sources["v389_training_report"].read_text())
    diagnosis = json.loads(sources["v391_route_diagnosis"].read_text())
    failures = json.loads(sources["public_failure_split"].read_text())
    if not old.get("passed") or old["selection"]["threshold"] != 0.7:
        raise RuntimeError("v389 source evidence drift")
    if diagnosis["successes"] != 11 or diagnosis["sequential_survivors"]["continuous_phase_ready"] != 8:
        raise RuntimeError("v391 bottleneck evidence drift")
    right_train = [r for r in failures["episodes"] if r["split"] == "train" and r["arm"] == "right"]
    right_validation = [r for r in failures["episodes"] if r["split"] == "validation" and r["arm"] == "right"]
    if len(right_train) != 47 or len(right_validation) != 7:
        raise RuntimeError("public failure split drift")
    if any(r["capture_success"] is not False for r in right_train + right_validation):
        raise RuntimeError("right failure calibration set contains a success")

    RUN.mkdir(parents=True)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v393r1-failure-calibrated-phase-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "add independent public Pi0.5 right-arm failure negatives so phase coverage can rise without relaxing false-positive control",
        "implementation_retry": "v393 was stopped before any model or metric output after a frozen action-only precheck proved its route-row minima structurally impossible (observed train/validation 28/3 versus 32/8); v393r1 changes only those feasibility checks and parallelizes deterministic file loading",
        "fixed_training": {
            "positive_source": "same 15 public success-demo right episodes and frozen v323 onsets as v389",
            "recursive_context_sources": ["v326", "v355", "v378"],
            "negative_source": "47 public RoboTwin trajectories from official unmodified Pi0.5, all right-arm failures",
            "independent_test": "7 disjoint public Pi0.5 right-arm failure trajectories; never used for fitting or operating-point selection",
            "features": "unchanged v389 request-only RGB/action feature contract",
            "grouping": "15-fold episode-grouped out-of-fold across all training episodes",
            "c_values": [0.001, 0.01, 0.1],
            "thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999],
        },
        "fixed_selection": {
            "generated_route_negative_specificity_min": 0.99,
            "each_generated_source_route_negative_specificity_min": 0.985,
            "teacher_negative_specificity_min": 0.98,
            "onpolicy_failure_train_all_rows": 2256,
            "onpolicy_failure_train_all_negative_specificity_min": 0.99,
            "generated_positive_recall_min": 0.50,
            "each_generated_source_positive_recall_min": 0.40,
            "tie_break": "maximum minimum generated-source recall, then generated recall, specificity, higher threshold, smaller C",
        },
        "post_selection_gate": {
            "independent_failure_all_rows": 336,
            "independent_failure_all_negative_specificity_min": 0.99,
            "independent_failure_route_rows_exact": 3,
            "independent_failure_route_false_positives": 0,
            "all_required": True,
        },
        "next_authority": "passing training/test only authorizes one frozen v394 integration and 512 fidelity audit; never direct rollout or RL",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "public_world_model_data_only": True,
            "public_simulator_outcomes_used_only_as_world_model_negative_supervision": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False,
            "official_batch16_or_hidden_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "preregistration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
