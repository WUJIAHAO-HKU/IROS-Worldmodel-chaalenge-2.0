#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v385_native_batch_clean_reanchor_seed1548_20260823"
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
    source = J / "v209_v202_v208_public_arm_routed_release"
    source_training = J / "v381_public_recursive_source_gate_seed1544_20260823/training_report.json"
    source_gate = J / "v381_public_recursive_source_gate_seed1544_20260823/recursive_source_gate.npz"
    action_gate = J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz"
    phase_gate = J / "v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz"
    runtime = ROOT / "pipeline/wam_pipeline/v385_native_batch_clean_reanchor_runtime.py"
    backends = ROOT / "pipeline/wam_pipeline/backends.py"
    library = J / "v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for path in (source, source_training, source_gate, action_gate, phase_gate, runtime, backends, library):
        if not path.exists():
            raise FileNotFoundError(path)
    training = json.loads(source_training.read_text())
    if not training.get("passed") or training["selection"]["teacher_specificity"] != 1.0:
        raise RuntimeError("source gate is not eligible")
    RUN.mkdir(parents=True)
    (RUN / "audit").mkdir()
    REG.mkdir(parents=True)
    release = RUN / "release"
    release.mkdir()
    for name in ("left_expert", "right_expert", "arm_routed_autoregressive_manifest.json"):
        path = source / name
        os.symlink(path.resolve(), release / name, target_is_directory=path.is_dir())
    os.symlink(source_gate.resolve(), release / "source_gate.npz")
    manifest = {
        "format": "strict-track2-v385-native-batch-clean-reanchor-release-v1",
        "model_version": "track2-v385-v326-nativebatch-actionphase-clean-reanchor",
        "source_gate": "source_gate.npz",
        "source_gate_sha256": sha256(source_gate),
        "source_threshold": float(training["selection"]["threshold"]),
        "renderer": "absolute eight-frame public clean terminal sequence selected by future right endpoint",
        "route": "identical to v384; native parent predict_batch is mandatory",
        "reward_or_outcomes_used": False,
        "hidden_or_final_data": False,
    }
    (release / "native_batch_clean_reanchor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prereg = {
        "format": "strict-track2-v385-native-batch-clean-reanchor-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["model_version"],
        "release": str(release),
        "change_from_v384": "implementation-only batch equivalence fix; routing and rendering semantics unchanged",
        "fixed_candidate": {
            "parent": "frozen v326 native batch path",
            "source_threshold": manifest["source_threshold"],
            "action_probability_min": 0.99,
            "selection": "action-only phase row and future endpoint",
            "no_parameter_sweep": True,
        },
        "fixed_recursive_gate": {
            "exact_rows": 512,
            "teacher_bit_exact_v326": True,
            "recursive_rgb_ratio_max_both": 1.05,
            "recursive_temporal_ratio_max_both": 1.05,
            "recursive_reward_error_ratio_lt_both": 1.0,
            "ground_truth_positive_recall_at_0p9_min_both": 0.50,
            "ground_truth_positive_recall_nonregression_vs_v326": True,
            "ground_truth_negative_false_positive_rate_nonregression_vs_v326": True,
            "all_required": True,
        },
        "gate_rationale": "v384 showed the absolute 10% false-positive cap is infeasible for the frozen v326 validation baseline itself (20/196); v385 requires exact non-regression instead of weakening candidate fidelity",
        "evidence_sha256": {
            "v384_report": sha256(J / "v384_action_phase_clean_reanchor_seed1547_20260823/audit/recursive_reward_causal.json"),
            "source_training": sha256(source_training),
            "source_gate": sha256(source_gate),
            "action_gate": sha256(action_gate),
            "phase_gate": sha256(phase_gate),
            "runtime": sha256(runtime),
            "backends": sha256(backends),
            "library": sha256(library),
        },
        "guards": {
            "public_world_model_data_only": True,
            "runtime_reads_reward_or_outcomes": False,
            "policy_modified": False,
            "official_reward_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    text = json.dumps(prereg, indent=2) + "\n"
    (RUN / "release_registration.json").write_text(text)
    (REG / "preregistration.json").write_text(text)
    print(json.dumps({"run": str(RUN), "release": str(release)}, indent=2))


if __name__ == "__main__":
    main()
