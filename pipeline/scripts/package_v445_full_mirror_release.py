#!/usr/bin/env python3
"""Package passed v445 S0 without starting S1, service, or RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "s0-audit"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--v169-release", type=Path, default=J / "v169_instruction_arm_routed_release")
    parser.add_argument("--v169-library", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=J / "v445_v169_full_mirror_release")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    audit = json.loads(args.s0_audit.read_text())
    if prereg.get("format") != "strict-track2-v445-full-mirror-preregistration-v1":
        raise RuntimeError("wrong v445 preregistration")
    if audit.get("format") != "strict-track2-v445-full-mirror-s0-static-contract-v1" or audit.get("passed") is not True:
        raise RuntimeError("v445 S0 failed")
    runtime = ROOT / "pipeline/wam_pipeline/v445_v169_full_mirror_runtime.py"
    v169_manifest = args.v169_release / "v169_arm_routed_manifest.json"
    if prereg.get("evidence_sha256", {}).get("runtime") != sha256(runtime):
        raise RuntimeError("v445 runtime differs from preregistration")
    if audit.get("sha256", {}).get("runtime") != sha256(runtime):
        raise RuntimeError("v445 S0/runtime binding mismatch")
    args.output.mkdir(parents=True)
    os.symlink(args.v169_release.resolve(), args.output / "v169_release", target_is_directory=True)
    os.symlink(args.v169_library.resolve(), args.output / "v169_library", target_is_directory=True)
    os.symlink(args.preregistration.resolve(), args.output / "preregistration.json")
    os.symlink(args.s0_audit.resolve(), args.output / "s0_audit.json")
    manifest = {
        "format": "track2-v445-v169-full-mirror-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "frozen v169 parent diagnostic; no policy/RL authority",
        "v169_release": "v169_release", "v169_library": "v169_library",
        "runtime_class": "wam_pipeline.v445_v169_full_mirror_runtime.Track2V445V169FullMirror",
        "gate_inputs": ["history_actions", "future_actions", "instruction"],
        "formula": prereg["formula"], "s1_gate": prereg["s1_gate"],
        "official_reward_runtime_used": False,
        "sha256": {
            "v169_manifest": sha256(v169_manifest), "runtime": sha256(runtime),
            "preregistration": sha256(args.preregistration), "s0_audit": sha256(args.s0_audit),
        },
        "guards": {
            "left_and_nonexplicit_output": "bit-exact same-request v169",
            "seed_selection": False, "action_generation": False,
            "policy_modified": False, "official_reward_modified": False,
            "policy_updates": 0, "hidden_or_final_data": False,
            "real_submission": False, "rl_authorized": False, "s1_automatically_started": False,
        },
    }
    manifest_path = args.output / "v445_full_mirror_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(args.output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
