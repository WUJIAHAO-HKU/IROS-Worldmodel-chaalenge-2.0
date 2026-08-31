#!/usr/bin/env python3
"""Preregister the action-domain terminal rescue experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v396_action_domain_terminal_rescue_seed1556_20260823"
RUN = J / NAME
REG = O / "run_registry" / NAME


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if RUN.exists() or REG.exists():
        raise FileExistsError("refusing overwrite")
    sources = {
        "trainer": ROOT / "pipeline/scripts/train_v396_action_domain_terminal_rescue.py",
        "feature_contract": ROOT / "pipeline/wam_pipeline/v396_action_phase_gate.py",
        "public_success_split": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "public_failure_split": J / "onpolicy_windows_full128_stride4/split_manifest.json",
        "phase_labels": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "v389_report": J / "v389r1_public_recursive_phase_classifier_seed1550_20260823/training_report.json",
        "v393_report": J / "v393r1_public_failure_calibrated_phase_seed1554_20260823/training_report.json",
        "v395_diagnosis": O / "run_registry/v395_v169_v394_route_trace32_seed1497_20260823/route_analysis.json",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    RUN.mkdir(parents=True)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v396-action-domain-terminal-rescue-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "preserve the v389 visual phase route while adding a domain-invariant action-only terminal rescue; avoid v393 RGB-domain collapse",
        "fixed_training": {
            "positive_and_early_negative_source": "15 declared public success-demo right episodes with frozen v323 onset labels",
            "failure_negative_source": "47 public official-unmodified-Pi0.5 right failure episodes",
            "independent_test": "7 disjoint public right failure episodes",
            "features": "4+8 request actions only; no RGB, reward, outcome, seed, request identity, or evaluation metadata",
            "c_values": [0.001, 0.01, 0.1, 1.0],
            "thresholds": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999],
            "grouping": "15-fold episode-grouped out-of-fold",
        },
        "fixed_selection": {
            "success_all_early_specificity_min": 0.98,
            "success_route_early_specificity_min": 0.99,
            "success_route_terminal_recall_min": 0.50,
            "failure_all_specificity_min": 0.99,
            "failure_route_specificity_min": 0.96,
            "tie_break": "maximum success-route terminal recall, then failure-route specificity, success-route specificity, higher threshold, smaller C",
        },
        "post_selection_gate": {"independent_failure_rows": 336, "route_rows": 3, "all_specificity_min": 0.99, "route_false_positives": 0},
        "next_authority": "passing classifier authorizes one OR-rescue integration with unchanged v390 route and one 512 fidelity audit; never direct RL",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {"public_world_model_data_only": True, "runtime_reads_reward_or_outcomes": False, "policy_modified": False, "official_batch16_or_hidden_data": False, "real_submission": False},
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "preregistration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
