#!/usr/bin/env python3
"""Preregister v389 public-train recursive visual/action phase training."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v389r1_public_recursive_phase_classifier_seed1550_20260823"
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
        "trainer": ROOT / "pipeline/scripts/train_v389_public_recursive_phase_classifier.py",
        "feature_contract": ROOT / "pipeline/wam_pipeline/v389_public_recursive_phase_gate.py",
        "public_split": J / "v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json",
        "public_library_manifest": J / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.manifest.json",
        "phase_labels": J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz",
        "phase_training_report": J / "v323_public_terminal_phase_gate_seed1493_20260822/audit/training_report.json",
        "action_gate": J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz",
        "v326_runtime": ROOT / "pipeline/wam_pipeline/v326_blended_phase_terminal_runtime.py",
        "v355_manifest": J / "v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json",
        "v378_manifest": J / "v378_source_routed_blended_cartesian_seed1541_20260823/release/source_routed_blend_manifest.json",
        "v387_diagnosis": O / "runs/v387_v169_v385_route_trace32_seed1497_20260823/audit/route_analysis.json",
        "v388_rejection": J / "v388_no_phase_clean_reanchor_seed1549_20260823/audit/recursive_reward_causal.json",
        "aborted_v389_preregistration": J / "v389_public_recursive_phase_classifier_seed1550_20260823/preregistration.json",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    diagnosis = json.loads(sources["v387_diagnosis"].read_text())
    rejection = json.loads(sources["v388_rejection"].read_text())
    if diagnosis["requests"] != 800 or diagnosis["sequential_survivors"]["phase_ready"] != 1:
        raise RuntimeError("v387 phase-bottleneck evidence drift")
    if rejection.get("passed") is not False:
        raise RuntimeError("v388 removal was not rejected")

    RUN.mkdir(parents=True)
    REG.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v389-public-recursive-phase-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "replace the brittle nearest-row onset boolean with a continuous RGB/action phase probability while preserving a high-specificity terminal route",
        "implementation_retry": "v389 was stopped before any output or metric because recursive action loading unnecessarily decompressed unused real RGB; v389r1 changes only that I/O path and progress logging",
        "fixed_training": {
            "episodes": "exact 15 declared public-train right episodes",
            "recursive_context_sources": ["v326", "v355", "v378"],
            "teacher_contexts": "all 1926 public-train right windows",
            "recursive_alignments": list(range(8)),
            "label": "current public window start >= frozen v323 sustained-success onset for that public-training episode",
            "features": "request-only RGB spatial/quality/temporal summaries plus raw/delta/summary 4+8 actions",
            "grouping": "leave-one-public-train-episode-out",
            "c_values": [0.001, 0.01, 0.1],
            "thresholds": [0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999],
            "runtime_route_scope": "right + source-generated + post-grasp + action probability >=0.99 + no frozen failure signature",
        },
        "fixed_selection": {
            "generated_route_negative_specificity_min": 0.99,
            "each_generated_source_route_negative_specificity_min": 0.985,
            "generated_route_positive_recall_min": 0.50,
            "each_generated_source_route_positive_recall_min": 0.40,
            "teacher_all_negative_specificity_min": 0.98,
            "tie_break": "maximum minimum generated-source recall, then generated recall, specificity, higher threshold, smaller C",
            "no_heldout_or_policy_outcome_selection": True,
        },
        "next_authority": "a passing classifier only authorizes one frozen v390 integration and the existing validation/local recursive fidelity audit; it does not authorize RL",
        "evidence_sha256": {name: sha256(path) for name, path in sources.items()},
        "guards": {
            "public_train_only": True,
            "recursive_labels_use_public_demo_timeline_only": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_or_simulator_outcomes_read": False,
            "public_holdout_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(payload, indent=2) + "\n"
    (RUN / "preregistration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "registry": str(REG)}, indent=2))


if __name__ == "__main__":
    main()
