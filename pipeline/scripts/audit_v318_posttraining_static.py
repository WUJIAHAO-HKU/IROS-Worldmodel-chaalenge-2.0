#!/usr/bin/env python3
"""Fail closed on cross-file identity drift in the v318 post-training pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
OFF = ROOT / "artifacts/strict_track2_official_20260810"
NAME = "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822"
VARIANT = "v319_v318_v317_rtx5090_step10_seed1471"
FROZEN = "v320_v318_v317_rtx5090_step10_seed1471"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    reg = OFF / "run_registry" / NAME
    prereg_path = reg / "posttraining_pipeline_preregistration.json"
    output = reg / "posttraining_pipeline_static_audit.json"
    if output.exists():
        raise FileExistsError(output)
    prereg = json.loads(prereg_path.read_text())
    runner = ROOT / "pipeline/scripts/run_v318_posttraining_pipeline.sh"
    wrapper = ROOT / "pipeline/scripts/run_frozen_v320_v318_final128_once.sh"
    freeze = ROOT / "pipeline/scripts/freeze_v320_v318_candidate.py"
    batch = ROOT / "pipeline/scripts/prepare_v318_batch00_gate.py"
    texts = {name: path.read_text() for name, path in {"runner": runner, "wrapper": wrapper, "freeze": freeze, "batch": batch}.items()}
    checks = {
        "prereg_variant": prereg["variant"] == VARIANT,
        "prereg_final_target": prereg["final128_gate"] == {"count": 128, "one_unique_frozen_candidate": True, "selection_after_final": False, "successes_min": 85},
        "all_registered_tool_hashes_hold": all(
            Path(item["path"]).is_file() and sha(Path(item["path"])) == item["sha256"]
            for item in prereg["tools"].values()
        ),
        "runner_identity": NAME in texts["runner"] and VARIANT in texts["runner"] and FROZEN in texts["runner"],
        "runner_uses_v317_release": "v317_v315_native_batch_causal_gate_seed1490_20260822" in texts["runner"],
        "runner_uses_v317_runtime": "v317_batched_sparse_failure_terminal_runtime.py" in texts["runner"],
        "runner_uses_v317_authorization": "--world-model-authorization" in texts["runner"],
        "runner_stops_v317_service": "restart_v317_services.sh\" stop" in texts["runner"],
        "wrapper_identity": VARIANT in texts["wrapper"] and FROZEN in texts["wrapper"],
        "wrapper_is_local_only": "contest_submission':False" in texts["wrapper"] and "TRACK2_FINAL128_RUN_BASELINE=false" in texts["wrapper"],
        "freeze_binds_v317_authorization": "authorization.get(\"launch_permission\") is True" in texts["freeze"],
        "freeze_binds_runtime_hash": "runtime_evidence.get(\"sha256\") == sha256(args.parent_runtime)" in texts["freeze"],
        "batch_gate_requires_v317": "track2-v317-batched-sparse-failure-terminal-v315" in texts["batch"],
        "no_old_candidate_identity_in_runner": all(token not in texts["runner"] for token in ("v309_v308_v301", "v311_v308_v301", "V308_TRAINING_ACCEPTED")),
        "guards_hold": prereg["guards"] == {
            "hidden_or_final_outcomes_used_for_training_or_selection": False,
            "participant_component": "world-model RGB service only",
            "policy_mirroring": False,
            "public_policy_outcomes_read_at_registration": False,
            "real_submission": False,
            "sft_co_training": False,
        },
    }
    report = {
        "format": "strict-track2-v318-posttraining-static-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "preregistration": {"path": str(prereg_path), "sha256": sha(prereg_path)},
        "checks": checks,
        "passed": all(checks.values()),
        "outcomes_read": False,
        "real_submission": False,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
