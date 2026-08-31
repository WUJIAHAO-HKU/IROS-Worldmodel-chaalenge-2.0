#!/usr/bin/env python3
"""Package the passed v432-step25 checkpoint for parent-only tracing."""

from __future__ import annotations

import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
O = ROOT / "artifacts/strict_track2_official_20260810"
TAG = "v436_v432_step25_parent_diagnostic_release"
RELEASE = J / TAG
REG = O / "run_registry" / TAG
LEFT = J / "v209_v202_v208_public_arm_routed_release/left_expert"
PARENT = J / "v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
EXPECTED_CANDIDATE_SHA256 = "dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--step25-gate", required=True, type=Path)
    args = ap.parse_args()
    gate = json.loads(args.step25_gate.read_text())
    guards = gate.get("guards", {})
    if (
        gate.get("format") != "strict-track2-v432-step25-shortgate-v1"
        or gate.get("passed") is not True
        or guards.get("public_holdout_only") is not True
        or guards.get("hidden_or_final_data") is not False
        or guards.get("real_submission") is not False
        or guards.get("policy_updates") != 0
    ):
        raise RuntimeError("v432 step25 public-only short gate did not pass")
    if args.candidate.name != "checkpoint_step_000025":
        raise RuntimeError("v436 requires the preregistered v432 step25 checkpoint")
    if sha(args.candidate / "model.pt") != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("v432 step25 candidate hash mismatch")
    if gate.get("evidence_sha256", {}).get("candidate_model") != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("step25 gate is not bound to the expected candidate")
    if not all((LEFT / "model.pt").is_file() for _ in [0]) or not (PARENT / "model.pt").is_file() or not (args.candidate / "model.pt").is_file():
        raise FileNotFoundError("missing v436 diagnostic expert")
    if RELEASE.exists() or REG.exists(): raise FileExistsError(TAG)
    RELEASE.mkdir(parents=True); REG.mkdir(parents=True)
    for name, source in (("left_expert", LEFT), ("parent_right", PARENT), ("candidate_right", args.candidate.resolve())):
        os.symlink(source.resolve(), RELEASE / name, target_is_directory=True)
    manifest = {
        "format": "track2-v436-v432-step25-diagnostic-release-v1",
        "classification": "parent world-model diagnostic only",
        "candidate_stage": "passed v432 step25 diagnostic candidate",
        "formal_candidate_authorized": False,
        "left_expert": "left_expert", "parent_right": "parent_right", "candidate_right": "candidate_right",
        "model_sha256": {"left": sha(LEFT/"model.pt"), "parent_right": sha(PARENT/"model.pt"), "candidate_right": sha(args.candidate/"model.pt")},
        "step25_gate": str(args.step25_gate.resolve()), "step25_gate_sha256": sha(args.step25_gate),
        "guards": {"policy_updates": 0, "hidden_or_final_data": False, "real_submission": False},
    }
    mp = RELEASE / "v436_diagnostic_manifest.json"; mp.write_text(json.dumps(manifest, indent=2)+"\n")
    (REG/"registration.json").write_text(json.dumps({"format":"strict-track2-v436-release-registration-v1","created_at":datetime.now(timezone.utc).isoformat(),"release":str(RELEASE),"manifest_sha256":sha(mp),"formal_candidate_authorized":False},indent=2)+"\n")
    print(RELEASE); return 0

if __name__ == "__main__": raise SystemExit(main())
