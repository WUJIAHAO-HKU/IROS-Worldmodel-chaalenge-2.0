#!/usr/bin/env python3
"""Preregister sparse terminal-only repair after v366's two fixed failures."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
RUN = J / "v367_sparse_terminal_overlay_v355_seed1531_20260822"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if RUN.exists():
        raise FileExistsError("refusing overwrite of v367")
    paths = {
        "v366_gate": J / "v366_serial_consistent_v355_v326_seed1530_20260822/causal_gate_report.json",
        "runtime": ROOT / "pipeline/wam_pipeline/v367_sparse_terminal_overlay_runtime.py",
        "backend_factory": ROOT / "pipeline/wam_pipeline/backends.py",
        "contract": ROOT / "pipeline/scripts/test_v367_sparse_terminal_overlay.py",
        "audit": ROOT / "pipeline/scripts/audit_v367_sparse_terminal_overlay_gate.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    prior = json.loads(paths["v366_gate"].read_text())
    if prior.get("passed") is not False:
        raise RuntimeError("v367 requires rejected v366")
    RUN.mkdir(parents=True)
    payload = {
        "format": "strict-track2-v367-sparse-terminal-overlay-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "pure v355 except phase-qualified right success injection alpha1 and three fixed failure-terminal suppressions",
        "baseline": "pure frozen v355",
        "removed_v326_behavior": "all pregrasp and ordinary nonterminal retrieval/blending",
        "fixed_thresholds": {"validation_positive_hit_rate_min": 0.90, "local_positive_hit_rate_min": 0.90, "validation_reward_mae_ratio_max": 1.0, "local_reward_mae_ratio_max": 1.0, "validation_rgb_mae_ratio_max": 1.01, "local_rgb_mae_ratio_max": 1.03, "counterfactual_hit_rate_max_validation": 0.10, "counterfactual_hit_rate_max_local": 0.15, "all_checks_required": True},
        "sequence": ["numeric contract", "full public causal/reward audit", "HTTP acceptance", "training-mode rollout-only128"],
        "evidence_sha256": {key: sha(path) for key, path in paths.items()},
        "guards": {"public_data_only": True, "runtime_reward_access": False, "policy_modified": False, "hidden_or_final_data": False, "real_submission": False},
    }
    (RUN / "release_registration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(RUN)


if __name__ == "__main__":
    main()
