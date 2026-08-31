#!/usr/bin/env python3
"""Preregister the stronger public-data-only v221 right-arm co-training run."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    output, base_ckpt, dataset, worker, launcher, v220_report = map(
        pathlib.Path, sys.argv[1:]
    )
    prior = json.loads(v220_report.read_text())
    assert prior["counts"]["right_success"] == 0
    assert prior["counts"]["left_success"] == 4
    info = json.loads((dataset / "meta" / "info.json").read_text())
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "v221_v169_rightsft_kl_h8_step24_seed1421",
        "rules": {
            "real_submission": False,
            "hidden_or_reserved_evaluation_access": False,
            "training_data": "official/public successful demonstrations only",
            "selection_signal": "v220 public local batch00 gate only",
        },
        "rationale": {
            "v220_public_gate": prior["counts"],
            "interpretation": "one conservative SFT update preserved/improved left but was insufficient to activate right-arm success",
            "change": "increase supervised exposure and optimizer steps while retaining KL to the same v169 initial policy",
        },
        "initial_policy": {
            "name": "v169_parent_step5_seed1243",
            "historical_public_score": "51/128",
            "path": str(base_ckpt),
            "sha256": digest(base_ckpt),
        },
        "sft": {
            "dataset": str(dataset),
            "source": "public clean50 train40 right-arm subset",
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "batch_size": 2,
            "num_workers": 0,
            "loss_weight": 0.1,
            "estimated_supervised_examples": 24 * 8 * 2,
        },
        "rl": {
            "parent_world_model": "v218 route-aware public model",
            "max_updates": 24,
            "episode_steps": 8,
            "rollout_epochs": 1,
            "group_size": 4,
            "total_envs": 8,
            "actor_lr": 1e-5,
            "kl_beta_to_initial_v169": 0.1,
        },
        "training_acceptance": {
            "valid_terminal_checkpoint": True,
            "all_24_updates_logged": True,
            "finite_sft_loss_every_update": True,
            "no_oom_or_uncaught_exception": True,
        },
        "next_gate_if_training_passes": {
            "public_batch": "batch_00 (4 left, 12 right)",
            "thresholds": {
                "right_success": 1,
                "left_success": 2,
                "total_success": 3,
                "grasp_once": 12,
            },
        },
        "source_hashes": {
            "fsdp_actor_worker.py": digest(worker),
            "run_strict_track2_conservative_kl.sh": digest(launcher),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print("V221_RIGHT_SFT_PREREGISTERED")


if __name__ == "__main__":
    main()
