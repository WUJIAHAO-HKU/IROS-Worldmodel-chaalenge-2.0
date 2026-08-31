#!/usr/bin/env python3
"""Preregister the outcome-free v308 evaluation and unique-freeze pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    base = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
    off = base / "artifacts/strict_track2_official_20260810"
    scripts = base / "pipeline/scripts"
    name = "v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821"
    run = off / "runs" / name
    reg = off / "run_registry" / name
    output = reg / "posttraining_pipeline_preregistration.json"
    if output.exists():
        raise FileExistsError(output)
    variant = "v309_v308_v301_rtx5090_step10_seed1471"
    dev = off / "real_robotwin_eval/public_unseen_train_dev112_seed1403"
    dev_out = off / "real_robotwin_eval/public_unseen_train_dev112_metrics_seed1403"
    freeze_dir = off / "frozen_candidates/v311_v308_v301_rtx5090_step10_seed1471"
    final_out = off / "real_robotwin_eval/frozen_v311_v308_v301_rtx5090_step10_seed1471_final128"
    seed_bundle = off / "real_robotwin_eval/final128_effective_seed_bundle_manifest.json"
    training_prereg = reg / "preregistration.json"
    public_preflight = off / "runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v304_public112_static_preflight.json"
    final_preflight = off / "runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v304_final128_static_preflight.json"
    numeric = off / "runs/v304_v301_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/audit/v303_numeric_gate_clarification.json"
    hardware_numeric = off / "diagnostics/v308_v301_rtx5090_batch_equivalence_envtf32off_20260821.json"
    service_acceptance = off / "diagnostics/v308_rtx5090_service_acceptance_20260821.json"
    required = (training_prereg, public_preflight, final_preflight, numeric, hardware_numeric, service_acceptance, dev / "manifest.json", dev / "batch_00.json", seed_bundle)
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    training = json.loads(training_prereg.read_text())
    pub = json.loads(public_preflight.read_text())
    final = json.loads(final_preflight.read_text())
    clarification = json.loads(numeric.read_text())
    hw = json.loads(hardware_numeric.read_text())
    service = json.loads(service_acceptance.read_text())
    seeds = json.loads(seed_bundle.read_text())
    assert training["guards"]["official_rule_compliant"] is True
    assert pub["passed"] is True and pub["outcomes_read"] is False
    assert final["passed"] is True and final["outcomes_read"] is False
    assert clarification["passed"] is True and clarification["runtime_or_training_changed"] is False
    assert hw["max_absolute_pixel_change"] <= 2 and hw["mean_absolute_pixel_change"] <= 0.05 and hw["speedup"] >= 1.5
    assert service["passed"] is True
    assert seeds["result_data_used"] is False and seeds["seed_count"] == 128 and seeds["unique_seed_count"] == 128
    assert not (dev_out / variant).exists() and not freeze_dir.exists() and not final_out.exists()
    tool_names = [
        "prepare_v308_posttraining_pipeline.py", "watch_v308_to_posttraining.sh",
        "run_v308_posttraining_pipeline.sh", "run_frozen_v311_v308_final128_once.sh",
        "prepare_v305_v304_batch00_gate.py", "audit_v228_public_right_gate.py",
        "audit_v276_v274_public112_gate.py", "freeze_v307_v304_candidate.py",
        "prepare_frozen_final128_prereg.py", "run_strict_track2_dev_eval.sh",
        "run_strict_track2_final128_eval.sh", "summarize_frozen_final128.py",
        "verify_v304_posttraining_tools.py",
    ]
    tools = {}
    for name_ in tool_names:
        path = scripts / name_
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        tools[name_] = {"path": str(path), "sha256": sha(path)}
    record = {
        "format": "strict-track2-v304-posttraining-pipeline-preregistration-v2",
        "schema_reused_for": "v308 RTX5090 continuation",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "training_preregistration": {"path": str(training_prereg), "sha256": sha(training_prereg)},
        "stage_sequence": ["complete all 10 preregistered RL updates", "pass full training metric and checkpoint audit", "evaluate public batch00 only", "if and only if batch00 passes evaluate remaining public112", "if and only if public112 passes freeze exactly one candidate", "preregister and execute exactly one local final128 evaluation"],
        "batch00_gate": {"count": 16, "left_count": 4, "right_count": 12, "thresholds": {"total_success": 10, "left_success": 3, "right_success": 6, "grasp_once": 14, "right_grasp": 10}},
        "public112_gate": {"count": 112, "thresholds": {"total_success": 75, "left_success_rate": 0.60, "right_success_rate": 0.60, "grasp_at_least_success": True}},
        "final128_gate": {"count": 128, "one_unique_frozen_candidate": True, "selection_after_final": False, "successes_min": 85},
        "public_manifest": {"path": str(dev / "manifest.json"), "sha256": sha(dev / "manifest.json")},
        "public_batch00": {"path": str(dev / "batch_00.json"), "sha256": sha(dev / "batch_00.json")},
        "final_seed_bundle": {"path": str(seed_bundle), "sha256": sha(seed_bundle), "result_data_used": False},
        "static_evidence": {
            "public112_preflight": {"path": str(public_preflight), "sha256": sha(public_preflight)},
            "final128_preflight": {"path": str(final_preflight), "sha256": sha(final_preflight)},
            "v303_numeric_gate_clarification": {"path": str(numeric), "sha256": sha(numeric)},
            "rtx5090_numeric_gate": {"path": str(hardware_numeric), "sha256": sha(hardware_numeric)},
            "rtx5090_service_acceptance": {"path": str(service_acceptance), "sha256": sha(service_acceptance)},
        },
        "tools": tools,
        "guards": {"hidden_or_final_outcomes_used_for_training_or_selection": False, "public_policy_outcomes_read_at_registration": False, "real_submission": False, "participant_component": "world-model RGB service only", "policy_mirroring": False, "sft_co_training": False},
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
