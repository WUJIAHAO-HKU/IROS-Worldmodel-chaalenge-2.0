#!/usr/bin/env python3
"""Package a passed v442 S0 core; never start service, trace, or RL."""

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
    for name in ("preregistration", "alignment-index", "s0-audit"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--v169-release", type=Path, default=J / "v169_instruction_arm_routed_release")
    parser.add_argument("--v169-library", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--v436-release", type=Path, default=J / "v436_v432_step25_parent_diagnostic_release")
    parser.add_argument("--output", type=Path, default=J / "v442_v169_close_aligned_projection_release")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    index = json.loads(args.alignment_index.read_text())
    s0 = json.loads(args.s0_audit.read_text())
    if prereg.get("format") != "strict-track2-v442-close-trainonly-preregistration-v1":
        raise RuntimeError("wrong v442 preregistration")
    if index.get("format") != "strict-track2-v442-close-trainonly-rgb-alignment-index-v1":
        raise RuntimeError("wrong v442 alignment index")
    if s0.get("format") != "strict-track2-v442-close-s0-static-contract-v1" or s0.get("passed") is not True:
        raise RuntimeError("v442 S0 did not pass")
    if s0.get("sha256", {}).get("alignment_index") != sha256(args.alignment_index):
        raise RuntimeError("v442 S0/index binding mismatch")
    runtime = ROOT / "pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py"
    v169_manifest = args.v169_release / "v169_arm_routed_manifest.json"
    v436_manifest = args.v436_release / "v436_diagnostic_manifest.json"
    for path in (runtime, v169_manifest, v436_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if prereg.get("evidence_sha256", {}).get("runtime") != sha256(runtime):
        raise RuntimeError("v442 runtime differs from preregistration")
    args.output.mkdir(parents=True)
    os.symlink(args.v169_release.resolve(), args.output / "v169_release", target_is_directory=True)
    os.symlink(args.v169_library.resolve(), args.output / "v169_library", target_is_directory=True)
    os.symlink(args.v436_release.resolve(), args.output / "v436_release", target_is_directory=True)
    os.symlink(args.alignment_index.resolve(), args.output / "alignment_index.json")
    os.symlink(args.preregistration.resolve(), args.output / "preregistration.json")
    os.symlink(args.s0_audit.resolve(), args.output / "s0_audit.json")
    manifest = {
        "format": "track2-v442-v169-close-aligned-projection-release-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "parent world-model diagnostic only; no RL authority",
        "v169_release": "v169_release", "v169_library": "v169_library",
        "v436_release": "v436_release", "alignment_index": "alignment_index.json",
        "runtime_class": "wam_pipeline.v442_v169_close_aligned_projection_runtime.Track2V442V169CloseAlignedProjection",
        "gate_inputs": ["history_actions", "future_actions", "instruction"],
        "train_episodes": index["train_episodes"],
        "formula": {
            "baseline": "original v169 RGB",
            "teacher_delta": "per-channel clip(v432-step25 - v354, -8, 8)",
            "beta": "train-only LOEO phase/frame/RGB coefficient with abs<=1",
            "output": "round_clip_uint8(B + g * beta * D)",
            "phases": ["close"],
            "gate": "explicit-right + right7D-path>left7D-path + history-last-open + future-close-then-held; no mu",
        },
        "s1_action_coverage_contract": {
            "fixed_right_dev_samples": 32,
            "decisions": 128,
            "active_exact": 8,
            "distinct_samples_exact": 8,
            "active_phase": "grasp only",
            "chunk_pattern": [1, 0, 0, 0],
            "other_phase_or_chunk_active_exact": 0,
            "repeat_active_exact": 0,
            "intervention_fraction_exact": 0.0625,
        },
        "official_reward_runtime_used": False,
        "sha256": {
            "alignment_index": sha256(args.alignment_index),
            "v169_manifest": sha256(v169_manifest),
            "v436_manifest": sha256(v436_manifest),
            "runtime": sha256(runtime),
            "preregistration": sha256(args.preregistration),
            "s0_audit": sha256(args.s0_audit),
        },
        "guards": {
            "left_and_g0_output": "bit-exact original v169",
            "candidate_minus_v169_pixel_abs_max": 8,
            "policy_modified": False, "official_reward_modified": False,
            "policy_updates": 0, "hidden_or_final_data": False,
            "real_submission": False, "rl_authorized": False,
        },
    }
    manifest_path = args.output / "v442_close_aligned_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(args.output), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

