#!/usr/bin/env python3
"""Audit a preregistered Track 2 KL smoke run using training-only metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovery_receipt_valid(
    path: Path, expected_sha256: str, valid_steps: list[int], selected_step: int
) -> bool:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        return False
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        receipt.get("valid_steps") == valid_steps
        and receipt.get("invalid") == []
        and receipt.get("selected", {}).get("step") == selected_step
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merged_scalar_series(
    accumulators: list[EventAccumulator], tag: str
) -> tuple[list[int], list[float]]:
    """Merge restarted event files, keeping the latest value for each step."""
    latest: dict[int, tuple[float, float]] = {}
    for accumulator in accumulators:
        if tag not in set(accumulator.Tags().get("scalars", [])):
            continue
        for item in accumulator.Scalars(tag):
            step = int(item.step)
            candidate = (float(item.wall_time), float(item.value))
            if step not in latest or candidate[0] >= latest[step][0]:
                latest[step] = candidate
    steps = sorted(latest)
    return steps, [latest[step][1] for step in steps]


def main() -> int:
    args = parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    gates = preregistration["acceptance_gates"]
    event_files = sorted((args.run / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("no training event files found")

    accumulators = [
        EventAccumulator(str(path), size_guidance={"scalars": 0})
        for path in event_files
    ]
    for accumulator in accumulators:
        accumulator.Reload()
    tags = set().union(
        *(set(accumulator.Tags().get("scalars", [])) for accumulator in accumulators)
    )
    required_tags = {
        "train/actor/kl_loss",
        "train/actor/kl_beta",
        "train/actor/approx_kl",
        "train/actor/clip_fraction",
        "train/actor/grad_norm",
    }
    missing_tags = sorted(required_tags - tags)

    expected_steps = int(preregistration["frozen_training"]["max_steps"])
    amendment_path = (
        args.preregistration.parent / "model_only_step1_recovery_amendment.json"
    )
    amendment = json.loads(amendment_path.read_text()) if amendment_path.is_file() else None
    retention_evidence_path = (
        args.preregistration.parent
        / "checkpoint_retention_evidence_amendment.json"
    )
    retention_evidence = (
        json.loads(retention_evidence_path.read_text())
        if retention_evidence_path.is_file()
        else None
    )
    missing_step3_evidence_mode = None
    recovery_start_step = 0
    allowed_missing_metric_steps: list[int] = []
    amendment_valid = amendment is None
    if amendment is not None:
        compliance = amendment.get("compliance", {})
        recovery = amendment.get("recovery", {})
        incident = amendment.get("incident", {})
        runtime_fixes = amendment.get("runtime_fixes", {})
        recovery_start_step = int(recovery.get("start_step", -1))
        amendment_valid = (
            recovery_start_step == 1
            and incident.get("completed_update_step") == 1
            and incident.get("model_checkpoint_available") is True
            and incident.get("optimizer_checkpoint_available") is False
            and recovery.get("model_structure", {}).get(
                "torch_load_mmap_weights_only"
            )
            is True
            and recovery.get("model_structure", {}).get("state_dict_keys") == 821
            and compliance.get("policy_or_training_data_changed") is False
            and compliance.get("hidden_or_final_outcomes_read") is False
            and compliance.get("real_submission_performed") is False
            and compliance.get("official_evaluation_used_for_selection") is False
        )
        if runtime_fixes.get(
            "missing_logged_step3_due_post_checkpoint_worker_release"
        ) is True:
            checkpoint4_root = (
                args.run
                / "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
                / "checkpoints/global_step_4/actor"
            )
            checkpoint4_model = checkpoint4_root / "model_state_dict/full_weights.pt"
            checkpoint4_optimizer = checkpoint4_root / "optimizer_recovery.pt"
            live_checkpoint4_evidence_valid = (
                checkpoint4_model.is_file()
                and checkpoint4_model.stat().st_size == 8_529_316_588
                and zipfile.is_zipfile(checkpoint4_model)
                and checkpoint4_optimizer.is_file()
                and checkpoint4_optimizer.stat().st_size == 3_253_939_504
                and zipfile.is_zipfile(checkpoint4_optimizer)
                and runtime_fixes.get("resumed_from_step4_model_and_optimizer")
                is True
                and runtime_fixes.get("numeric_gate_thresholds_changed") is False
            )
            receipt_chain_evidence_valid = False
            if retention_evidence is not None:
                receipts = retention_evidence.get("recovery_receipts", {})
                supporting_log = retention_evidence.get("supporting_log", {})
                compliance_evidence = retention_evidence.get("compliance", {})
                supporting_log_path = (
                    args.preregistration.parent / supporting_log.get("file", "")
                )
                supporting_log_text = (
                    supporting_log_path.read_text(errors="replace")
                    if supporting_log_path.is_file()
                    else ""
                )
                receipt_specs = (
                    ("attempt13", [3, 4], 4),
                    ("attempt15", [3, 4], 4),
                    ("attempt16", [4, 6], 6),
                    ("attempt20", [6, 9], 9),
                )
                receipt_chain_evidence_valid = (
                    retention_evidence.get("schema_version")
                    == "strict-track2-checkpoint-retention-evidence-v1"
                    and retention_evidence.get("run_name")
                    == args.run.name
                    and preregistration["recovery_protocol"].get(
                        "keep_last_checkpoints"
                    )
                    == 2
                    and retention_evidence.get("missing_metric_step") == 3
                    and retention_evidence.get("pruned_checkpoint_step") == 4
                    and compliance_evidence.get("numeric_gate_thresholds_changed")
                    is False
                    and compliance_evidence.get("hidden_or_final_outcomes_read")
                    is False
                    and compliance_evidence.get("real_submission_performed") is False
                    and supporting_log_path.is_file()
                    and sha256_file(supporting_log_path)
                    == supporting_log.get("sha256")
                    and "global_step_4/actor/model_state_dict/full_weights.pt"
                    in supporting_log_text
                    and "global_step_4/actor/optimizer_recovery.pt"
                    in supporting_log_text
                    and "Global Step:    5/10" in supporting_log_text
                    and "Global Step:    6/10" in supporting_log_text
                    and all(
                        key in receipts
                        and recovery_receipt_valid(
                            args.preregistration.parent / receipts[key]["file"],
                            receipts[key]["sha256"],
                            valid_steps,
                            selected_step,
                        )
                        for key, valid_steps, selected_step in receipt_specs
                    )
                )
            missing_step3_evidence_valid = (
                live_checkpoint4_evidence_valid or receipt_chain_evidence_valid
            )
            if live_checkpoint4_evidence_valid:
                missing_step3_evidence_mode = "live_checkpoint4_model_and_optimizer"
            elif receipt_chain_evidence_valid:
                missing_step3_evidence_mode = "hashed_recovery_receipt_chain"
            amendment_valid = amendment_valid and missing_step3_evidence_valid
            if missing_step3_evidence_valid:
                allowed_missing_metric_steps = [3]
    # The runner logs with the loop index `_step`, then increments global_step
    # before saving. Therefore scalar step N corresponds to checkpoint N + 1.
    expected_metric_steps = list(range(recovery_start_step, expected_steps))
    required_metric_steps = [
        step for step in expected_metric_steps if step not in allowed_missing_metric_steps
    ]
    metric_series = {
        tag: merged_scalar_series(accumulators, tag) for tag in sorted(required_tags)
    }
    metric_steps = {tag: series[0] for tag, series in metric_series.items()}
    metrics = {tag: series[1] for tag, series in metric_series.items()}
    checkpoint = (
        args.run
        / "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05"
        / "checkpoints"
        / f"global_step_{expected_steps}"
        / "actor"
        / "model_state_dict"
        / "full_weights.pt"
    )
    finite_metrics = all(
        metric_steps[tag] == required_metric_steps
        and len(values) == len(required_metric_steps)
        and all(math.isfinite(value) for value in values)
        for tag, values in metrics.items()
    )
    approx_kl_gate = gates.get(
        "action_dim_normalized_approx_kl_abs_max",
        gates.get("action_dim_normalized_approx_kl_max"),
    )
    if approx_kl_gate is None:
        raise KeyError("missing action-dimension-normalized approx-KL gate")
    checks = {
        "required_tags_present": not missing_tags,
        "operational_recovery_amendment_valid": amendment_valid,
        "all_metrics_finite_and_complete": finite_metrics,
        "approx_kl_within_gate": bool(metrics["train/actor/approx_kl"])
        and max(abs(value) for value in metrics["train/actor/approx_kl"])
        <= float(approx_kl_gate),
        "clip_fraction_within_gate": bool(metrics["train/actor/clip_fraction"])
        and max(metrics["train/actor/clip_fraction"])
        <= float(gates["action_dim_normalized_clip_fraction_max"]),
        "gradient_norm_within_gate": bool(metrics["train/actor/grad_norm"])
        and max(metrics["train/actor/grad_norm"])
        <= float(gates["gradient_norm_max"]),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_zip_integrity": checkpoint.is_file()
        and zipfile.is_zipfile(checkpoint),
    }
    if gates.get("second_update_kl_loss_nonnegative", False):
        kl_values = metrics["train/actor/kl_loss"]
        kl_steps = metric_steps["train/actor/kl_loss"]
        checks["second_update_kl_loss_nonnegative"] = (
            1 in kl_steps and kl_values[kl_steps.index(1)] >= 0.0
        )
    if gates.get("all_update_kl_loss_finite_nonnegative", False):
        kl_values = metrics["train/actor/kl_loss"]
        checks["all_update_kl_loss_finite_nonnegative"] = (
            len(kl_values) == len(required_metric_steps)
            and all(math.isfinite(value) and value >= 0.0 for value in kl_values)
        )
    extrema = {
        tag: {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
        for tag, values in metrics.items()
    }
    report = {
        "format": "strict-track2-kl-smoke-audit-v1",
        "run": str(args.run),
        "preregistration": str(args.preregistration),
        "event_file": str(event_files[-1]),
        "event_files": [str(path) for path in event_files],
        "recovery_amendment": str(amendment_path) if amendment is not None else None,
        "checkpoint_retention_evidence": (
            str(retention_evidence_path) if retention_evidence is not None else None
        ),
        "missing_step3_evidence_mode": missing_step3_evidence_mode,
        "expected_metric_steps": expected_metric_steps,
        "allowed_missing_metric_steps": allowed_missing_metric_steps,
        "required_metric_steps": required_metric_steps,
        "metric_to_checkpoint_step": {
            str(step): step + 1 for step in expected_metric_steps
        },
        "metric_steps": metric_steps,
        "missing_tags": missing_tags,
        "metrics": metrics,
        "metric_extrema": extrema,
        "checkpoint": str(checkpoint),
        "checks": checks,
        "passed": all(checks.values()),
        "selection_inputs": (
            "preregistered training metrics, operational recovery amendment, "
            "and checkpoint integrity only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
