#!/usr/bin/env python3
"""Authorize only a deterministic v432 replay from the parent through step50."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_STEP25_MODEL_SHA256 = "dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("original-preregistration", "step25-gate", "step25-checkpoint", "trainer", "auditor", "launcher", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.original_preregistration.read_text())
    gate = json.loads(args.step25_gate.read_text())
    if prereg.get("format") != "strict-track2-v432-public-mirror-prompt-terminal-preregistration-v1":
        raise RuntimeError("wrong original v432 preregistration")
    if prereg["training"]["steps_max_after_gate"] != 50:
        raise RuntimeError("original preregistration did not authorize step50")
    if gate.get("format") != "strict-track2-v432-step25-shortgate-v1" or not gate.get("passed"):
        raise RuntimeError("v432 step25 short gate did not pass")
    model = args.step25_checkpoint / "model.pt"
    normalization = args.step25_checkpoint / "action_normalization.npz"
    manifest = args.step25_checkpoint / "training_manifest.json"
    for path in (model, normalization, manifest, args.trainer, args.auditor, args.launcher):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(model) != EXPECTED_STEP25_MODEL_SHA256:
        raise RuntimeError("step25 model differs from the already-passed short-gate checkpoint")
    payload = {
        "format": "strict-track2-v432-step50-continuation-authorization-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authority": "the original v432 preregistration permits a deterministic 50-step replay after the passed step25 gate",
        "original_preregistration": str(args.original_preregistration),
        "step25_gate": str(args.step25_gate),
        "continuation": {
            "method": "full deterministic replay from original parent",
            "from_global_step": 0,
            "to_global_step": 50,
            "optimizer_steps": 50,
            "initialization": prereg["training"]["initialization"],
            "existing_passed_step25_checkpoint": str(args.step25_checkpoint),
            "expected_replay_step25_model_sha256": EXPECTED_STEP25_MODEL_SHA256,
            "accepted_output_checkpoint": "checkpoint_step_000050 only after exact replay-step25 identity",
            "all_training_hyperparameters_unchanged": True,
            "sampler": "one uninterrupted seed1582 draw stream across steps 1..50",
            "optimizer_semantics": "one uninterrupted AdamW instance across steps 1..50",
            "replay_step25_mismatch_action": "stop before promotion and before step50 audit",
            "step25_anchor_only": True,
        },
        "evidence_sha256": {
            "original_preregistration": sha256(args.original_preregistration),
            "step25_gate": sha256(args.step25_gate),
            "step25_model": sha256(model),
            "step25_normalization": sha256(normalization),
            "step25_manifest": sha256(manifest),
            "trainer": sha256(args.trainer),
            "auditor": sha256(args.auditor),
            "launcher": sha256(args.launcher),
        },
        "guards": {
            "new_model_selection_data": False,
            "official_pi05_modified": False,
            "official_reward_modified": False,
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
