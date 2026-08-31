#!/usr/bin/env python3
"""Package the audited v439 hybrid core without starting any service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
DEFAULT_RELEASE = J / "v439_v169_action_causal_projection_release"
MU = [-0.330, -1.425, -1.563, 1.617, 0.494, 0.779]
ALPHA = [0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--projection-index", required=True, type=Path)
    parser.add_argument("--data-contract", required=True, type=Path)
    parser.add_argument("--v169-release", type=Path, default=J / "v169_instruction_arm_routed_release")
    parser.add_argument("--v169-library", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--v436-release", type=Path, default=J / "v436_v432_step25_parent_diagnostic_release")
    parser.add_argument("--output", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    contract = json.loads(args.data_contract.read_text())
    index = json.loads(args.projection_index.read_text())
    if prereg.get("format") != "strict-track2-v439-action-causal-projection-preregistration-v1":
        raise RuntimeError("wrong v439 preregistration")
    if contract.get("format") != "strict-track2-v439-projection-data-contract-v1" or contract.get("passed") is not True:
        raise RuntimeError("v439 data contract did not pass")
    if index.get("format") != "strict-track2-v439-public-right-action-projection-index-v1":
        raise RuntimeError("wrong v439 projection index")
    if prereg.get("formula", {}).get("mu_right6d") != MU or prereg.get("formula", {}).get("alpha_8") != ALPHA:
        raise RuntimeError("v439 preregistered formula drift")
    if index.get("projection", {}).get("mu_right6d") != MU or index.get("projection", {}).get("alpha_8") != ALPHA:
        raise RuntimeError("v439 index formula drift")
    if contract.get("sha256", {}).get("projection_index") != sha256(args.projection_index):
        raise RuntimeError("v439 contract/index binding mismatch")
    runtime = ROOT / "pipeline/wam_pipeline/v439_v169_action_causal_projection_runtime.py"
    if prereg.get("evidence_sha256", {}).get("runtime") != sha256(runtime):
        raise RuntimeError("v439 runtime differs from preregistration")
    v169_manifest = args.v169_release / "v169_arm_routed_manifest.json"
    v436_manifest = args.v436_release / "v436_diagnostic_manifest.json"
    for path in (args.projection_index, args.data_contract, runtime, v169_manifest, v436_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    v436 = json.loads(v436_manifest.read_text())
    if v436.get("format") != "track2-v436-v432-step25-diagnostic-release-v1":
        raise RuntimeError("wrong v439 teacher release")

    args.output.mkdir(parents=True)
    os.symlink(args.v169_release.resolve(), args.output / "v169_release", target_is_directory=True)
    os.symlink(args.v169_library.resolve(), args.output / "v169_library", target_is_directory=True)
    os.symlink(args.v436_release.resolve(), args.output / "v436_release", target_is_directory=True)
    os.symlink(args.projection_index.resolve(), args.output / "projection_index.json")
    os.symlink(args.preregistration.resolve(), args.output / "preregistration.json")
    os.symlink(args.data_contract.resolve(), args.output / "data_contract.json")
    manifest = {
        "format": "track2-v439-v169-action-causal-projection-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "world-model diagnostic candidate; no RL authority",
        "v169_release": "v169_release",
        "v169_library": "v169_library",
        "v436_release": "v436_release",
        "projection_index": "projection_index.json",
        "runtime_class": "wam_pipeline.v439_v169_action_causal_projection_runtime.Track2V439V169ActionCausalProjection",
        "gate_inputs": ["history_actions", "future_actions", "instruction"],
        "formula": {
            "mu_right6d": MU,
            "alpha_8": ALPHA,
            "teacher_delta_clip": [-8, 8],
            "teacher_delta_color": "signed RGB channels preserved independently",
            "baseline": "original v169 RGB",
            "right_gripper_gate": "close in history-last/future or already closed, then held below 0.5",
        },
        "official_reward_runtime_used": False,
        "sha256": {
            "projection_index": sha256(args.projection_index),
            "v169_manifest": sha256(v169_manifest),
            "v436_manifest": sha256(v436_manifest),
            "runtime": sha256(runtime),
            "preregistration": sha256(args.preregistration),
            "data_contract": sha256(args.data_contract),
        },
        "guards": {
            "left_output": "bit-exact original v169",
            "g0_output": "bit-exact original v169",
            "alpha_zero_frames": [1, 2, 7, 8],
            "policy_modified": False,
            "official_reward_modified": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
            "rl_authorized": False,
        },
    }
    manifest_path = args.output / "v439_action_causal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(args.output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
