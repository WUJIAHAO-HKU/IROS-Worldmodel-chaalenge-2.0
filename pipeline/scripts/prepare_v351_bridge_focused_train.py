#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v351_bridge_focused_train_seed1520_20260822"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True, exist_ok=True)
    output = RUN / "release_registration.json"
    if output.exists(): raise FileExistsError(output)
    fit = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v350_public_residual_bridge_fit_seed1519_20260822"
    sources = {
        "runtime": ROOT / "pipeline/wam_pipeline/v350_learned_residual_bridge_runtime.py",
        "audit": ROOT / "pipeline/scripts/audit_v351_bridge_focused_train.py",
        "profile": fit / "residual_bridge_profile.npz", "fit_report": fit / "fit_report.json",
        "v340_train": ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v340_train_recursive_reanchor_seed1509_20260822/train_recursive_gate_report.json",
    }
    payload = {"format": "strict-track2-v351-bridge-focused-train-preregistration-v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "candidate": "frozen v350 residual bridge; no coefficient changes",
               "scope": "public-train alignments 0 and 4; exactly 484 recursive windows",
               "checks": "same frozen focused thresholds; all required",
               "authorizes": "one public WM holdout gate only",
               "guards": {"public_train_only": True, "no_official_batch16_outcomes": True,
                          "no_hidden_or_final_data": True, "no_real_submission": True},
               "source_sha256": {name: sha(path) for name, path in sources.items()}}
    output.write_text(json.dumps(payload, indent=2) + "\n"); print(output); return 0
if __name__ == "__main__": raise SystemExit(main())
