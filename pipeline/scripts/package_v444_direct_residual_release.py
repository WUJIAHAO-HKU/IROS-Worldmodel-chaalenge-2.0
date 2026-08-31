#!/usr/bin/env python3
"""Package passed v444 step25 as a parent diagnostic; never start S1/RL."""

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
    for name in ("checkpoint", "training-report", "preregistration", "s0-audit"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--v169-release", type=Path, default=J / "v169_instruction_arm_routed_release")
    parser.add_argument("--v169-library", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=J / "v444_v169_direct_residual_release")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    report = json.loads(args.training_report.read_text())
    audit = json.loads(args.s0_audit.read_text())
    if prereg.get("format") != "strict-track2-v444-direct-residual-preregistration-v1":
        raise RuntimeError("wrong v444 preregistration")
    if report.get("format") != "strict-track2-v444-step25-kill-report-v1" or report.get("passed") is not True:
        raise RuntimeError("v444 training kill gate failed")
    if audit.get("format") != "strict-track2-v444-step25-s0-killgate-v1" or audit.get("passed") is not True:
        raise RuntimeError("v444 S0 audit failed")
    if audit.get("sha256", {}).get("checkpoint") != sha256(args.checkpoint):
        raise RuntimeError("v444 S0/checkpoint binding mismatch")
    runtime = ROOT / "pipeline/wam_pipeline/v444_v169_direct_residual_runtime.py"
    v169_manifest = args.v169_release / "v169_arm_routed_manifest.json"
    for path in (runtime, v169_manifest, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if prereg.get("evidence_sha256", {}).get("runtime") != sha256(runtime):
        raise RuntimeError("v444 runtime differs from preregistration")
    args.output.mkdir(parents=True)
    os.symlink(args.v169_release.resolve(), args.output / "v169_release", target_is_directory=True)
    os.symlink(args.v169_library.resolve(), args.output / "v169_library", target_is_directory=True)
    os.symlink(args.checkpoint.resolve(), args.output / "model_step25.pt")
    os.symlink(args.preregistration.resolve(), args.output / "preregistration.json")
    os.symlink(args.training_report.resolve(), args.output / "training_report.json")
    os.symlink(args.s0_audit.resolve(), args.output / "s0_audit.json")
    manifest = {
        "format": "track2-v444-v169-direct-residual-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "parent world-model diagnostic only; no policy/RL authority",
        "checkpoint": "model_step25.pt", "v169_release": "v169_release", "v169_library": "v169_library",
        "runtime_class": "wam_pipeline.v444_v169_direct_residual_runtime.Track2V444V169DirectResidual",
        "gate_inputs": ["history_actions", "future_actions", "instruction"],
        "formula": {
            "baseline": "frozen original v169 RGB",
            "residual": "direct train12 learned clip(target-v169, -8, 8)",
            "output": "round_clip_uint8(B + g * R)",
            "gate": "v442 close-only explicit-right/action-close gate",
            "protected_frames_exact_zero": [0, 1, 6, 7],
        },
        "kill_gate": audit["metrics"],
        "official_reward_runtime_used": False,
        "sha256": {
            "checkpoint": sha256(args.checkpoint), "v169_manifest": sha256(v169_manifest),
            "runtime": sha256(runtime), "preregistration": sha256(args.preregistration),
            "training_report": sha256(args.training_report), "s0_audit": sha256(args.s0_audit),
        },
        "guards": {
            "left_and_g0_output": "bit-exact original v169",
            "protected_frames_0_1_6_7_output": "bit-exact original v169",
            "candidate_minus_v169_pixel_abs_max": 8,
            "policy_modified": False, "official_reward_modified": False,
            "policy_updates": 0, "hidden_or_final_data": False,
            "real_submission": False, "rl_authorized": False, "s1_authorized": False,
        },
    }
    manifest_path = args.output / "v444_direct_residual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(args.output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
