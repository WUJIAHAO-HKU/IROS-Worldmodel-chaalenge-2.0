#!/usr/bin/env python3
"""Preregister the frozen v439 hybrid formula; never launch a service."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
O = ROOT / "artifacts/strict_track2_official_20260810"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
V169 = J / "v169_instruction_arm_routed_release"
V436 = J / "v436_v432_step25_parent_diagnostic_release"
EXPECTED_V432_SHA256 = "dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f"
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
    parser.add_argument("--projection-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.projection_index.read_text())
    if index.get("format") != "strict-track2-v439-public-right-action-projection-index-v1":
        raise RuntimeError("wrong v439 projection index")
    projection = index.get("projection", {})
    if projection.get("mu_right6d") != MU or projection.get("alpha_8") != ALPHA:
        raise RuntimeError("v439 formula differs from the frozen constants")
    if len(index.get("train_episodes", [])) != 40 or len(index.get("right_train_episodes", [])) != 15:
        raise RuntimeError("v439 index is not bound to public train40/right15")
    if set(index["train_episodes"]) & set(index.get("validation_episodes_excluded", [])):
        raise RuntimeError("v439 projection index crossed the public holdout boundary")
    if index.get("guards", {}).get("reward_read") is not False or index.get("guards", {}).get("outcome_read") is not False:
        raise RuntimeError("v439 projection index used reward or outcome")

    v169_manifest = V169 / "v169_arm_routed_manifest.json"
    v436_manifest = V436 / "v436_diagnostic_manifest.json"
    v436 = json.loads(v436_manifest.read_text())
    candidate = V436 / v436["candidate_right"] / "model.pt"
    parent = V436 / v436["parent_right"] / "model.pt"
    if v436.get("format") != "track2-v436-v432-step25-diagnostic-release-v1":
        raise RuntimeError("wrong v432-step25 teacher release")
    if sha256(candidate) != EXPECTED_V432_SHA256 or v436["model_sha256"]["candidate_right"] != EXPECTED_V432_SHA256:
        raise RuntimeError("v439 is not bound to the passed v432 step25 teacher")
    core_files = {
        "index_builder": ROOT / "pipeline/scripts/build_v439_public_right_projection_index.py",
        "data_contract_auditor": ROOT / "pipeline/scripts/audit_v439_projection_data_contract.py",
        "runtime": ROOT / "pipeline/wam_pipeline/v439_v169_action_causal_projection_runtime.py",
        "packager": ROOT / "pipeline/scripts/package_v439_action_causal_release.py",
        "projection_index": args.projection_index,
        "v169_manifest": v169_manifest,
        "v436_manifest": v436_manifest,
        "v432_candidate_model": candidate,
        "v354_parent_model": parent,
    }
    for path in core_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v439-action-causal-projection-preregistration-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "classification": "world-model diagnostic candidate; no RL authority",
        "formula": {
            "baseline": "B = original v169 RGB",
            "teacher_delta": "D = per-RGB-channel clip(C_v432step25 - P_v354, -8, +8)",
            "output": "Y = uint8(round(clip(B + g * alpha_t * D, 0, 255)))",
            "mu_right6d": MU,
            "alpha_8": ALPHA,
            "gate_all_required": [
                "explicit right-arm prompt and no left-arm prompt",
                "right 6D joint path strictly greater than left 6D joint path",
                "right gripper closes in history-last/future or is already closed, then remains held below 0.5",
                "(future-last right6D - history-last right6D) dot mu strictly positive",
            ],
            "bitexact_fallback": "left or g=0 returns B for all frames; frames 1,2,7,8 are B even when g=1",
        },
        "models": {
            "fallback": "original v169 arm-routed world model",
            "teacher_parent": "frozen v354 step150",
            "teacher_candidate": "passed v432 step25",
            "teacher_candidate_model_sha256": EXPECTED_V432_SHA256,
        },
        "data": {
            "projection_index": str(args.projection_index.resolve()),
            "public_train40_only": True,
            "public_right_train_episodes": 15,
            "public_holdout_used_for_construction": False,
            "reward_or_outcome_used_for_construction": False,
        },
        "runtime": {
            "returns": "eight uint8 RGB frames",
            "official_reward_loaded_or_called": False,
            "seed_used_by_gate": False,
            "request_identity_used_by_gate": False,
            "left_teacher_inference": False,
        },
        "evidence_sha256": {key: sha256(path) for key, path in core_files.items()},
        "guards": {
            "official_pi05_modified": False,
            "official_reward_modified": False,
            "official_reward_runtime_used": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
