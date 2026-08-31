#!/usr/bin/env python3
"""Create the immutable public-data preregistration for the v220 SFT smoke."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = pathlib.Path(sys.argv[1])
    base_checkpoint = pathlib.Path(sys.argv[2])
    dataset = pathlib.Path(sys.argv[3])
    worker = pathlib.Path(sys.argv[4])
    launcher = pathlib.Path(sys.argv[5])
    info = json.loads((dataset / "meta" / "info.json").read_text())

    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "v220_v169_rightsft_smoke_h8_step1_seed1420",
        "purpose": "gradient-path and checkpoint smoke; not a final candidate",
        "rules": {
            "real_submission": False,
            "hidden_or_reserved_evaluation_access": False,
            "training_data": "official/public successful demonstrations only",
            "selection_data": "none for this smoke",
        },
        "initial_policy": {
            "name": "v169_parent_step5_seed1243",
            "historical_public_score": "51/128",
            "historical_arm_breakdown": {"left": "51/62", "right": "0/66"},
            "path": str(base_checkpoint),
            "sha256": sha256(base_checkpoint),
        },
        "sft_dataset": {
            "path": str(dataset),
            "source": "aloha-agilex_clean_50 public train40, action-motion-classified right arm subset",
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "arm": "right",
            "batch_size": 1,
            "num_workers": 0,
            "loss_weight": 0.02,
        },
        "rl": {
            "parent_world_model": "v218 route-aware public model",
            "max_updates": 1,
            "episode_steps": 8,
            "rollout_epochs": 1,
            "group_size": 4,
            "total_envs": 8,
            "actor_lr": 2e-6,
            "kl_beta_to_initial_v169": 0.1,
        },
        "smoke_acceptance": {
            "required_metrics": ["sft_loss", "weighted_sft_loss", "actor/kl_loss"],
            "valid_terminal_checkpoint": True,
            "no_oom_or_uncaught_exception": True,
            "policy_parameters_changed": True,
            "public_simulator_success_gate": "not part of smoke; run only after training acceptance",
        },
        "source_hashes": {
            "fsdp_actor_worker.py": sha256(worker),
            "run_strict_track2_conservative_kl.sh": sha256(launcher),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("V220_RIGHT_SFT_SMOKE_PREREGISTERED")


if __name__ == "__main__":
    main()
