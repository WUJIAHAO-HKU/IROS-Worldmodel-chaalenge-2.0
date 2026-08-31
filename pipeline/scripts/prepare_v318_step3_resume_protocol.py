#!/usr/bin/env python3
"""Register the operationally exact v318 step-3 recovery protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
    off = root / "artifacts/strict_track2_official_20260810"
    name = "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
    reg = off / "run_registry" / name
    output = reg / "global_step3_resume_protocol_preregistration_v8.json"
    if output.exists():
        raise FileExistsError(output)
    tools = {
        "resume_launcher": root / "pipeline/scripts/launch_v318_resume_from_step3.sh",
        "continuation_wrapper": root / "pipeline/scripts/run_v318_step3_continuation_gate.sh",
        "fixed_training_runner": root / "pipeline/scripts/run_strict_track2_conservative_kl.sh",
    }
    for path in tools.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    payload = {
        "format": "strict-track2-v318-step3-operational-resume-v8",
        "supersedes": {
            "path": str(reg / "global_step3_resume_protocol_preregistration_v7.json"),
            "reason": "v7 passed relative checkpoint paths into a temporary overlay, creating broken symlinks; v8 uses absolute paths before any offline diagnostic result was produced",
        },
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "reason": "original recovery loop exhausted while v317 service was intentionally stopped for the preregistered gate; checkpoint is intact",
        "resume_from": "global_step_3 model plus optimizer",
        "remaining_budget": "steps 3 through 9; final global_step_10 checkpoint",
        "invariants": {
            "same_run_directory": True,
            "same_model_and_optimizer_state": True,
            "same_official_initial_reference": True,
            "same_actor_seed": 1471,
            "same_env_seed": 0,
            "same_lr": 2e-5,
            "same_kl_beta": 0.01,
            "same_rollouts_per_update": 128,
            "same_horizon": 200,
            "same_max_steps": 10,
            "algorithm_changed": False,
            "public112_or_final_outcomes_used": False,
            "real_submission": False,
        },
        "tools": {name_: {"path": str(path), "sha256": sha(path)} for name_, path in tools.items()},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
