#!/usr/bin/env python3
"""S0 audit for v441 train-only alignment, runtime purity, and RGB bounds."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v441_v169_postclose_aligned_projection_runtime import (
    PHASES,
    apply_aligned_projection,
    gate_decision,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("alignment-index", "preregistration", "split", "runtime", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.alignment_index.read_text())
    prereg = json.loads(args.preregistration.read_text())
    split = json.loads(args.split.read_text())
    tree = ast.parse(args.runtime.read_text(), filename=str(args.runtime))
    imports = [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    ] + [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    evidence = index.get("loeo", {}).get("evidence", {})
    beta = {phase: np.asarray(index.get("beta", {}).get(phase), dtype=np.float64) for phase in PHASES}
    component_checks = {}
    for phase in PHASES:
        active = np.asarray(evidence.get(phase, {}).get("active"), dtype=bool)
        agreement = np.asarray(evidence.get(phase, {}).get("sign_agreement"), dtype=np.float64)
        improvement = np.asarray(evidence.get(phase, {}).get("heldout_improvement_fraction"), dtype=np.float64)
        component_checks[f"{phase}_shape"] = beta[phase].shape == active.shape == agreement.shape == improvement.shape == (8, 3)
        component_checks[f"{phase}_has_active"] = int(active.sum()) >= 1
        component_checks[f"{phase}_active_sign_stable"] = bool(np.all(agreement[active] >= 0.80))
        component_checks[f"{phase}_active_heldout_improves"] = bool(np.all(improvement[active] >= 0.80))
        component_checks[f"{phase}_inactive_zero"] = bool(np.all(beta[phase][~active] == 0.0))
        component_checks[f"{phase}_bounded_finite"] = bool(np.isfinite(beta[phase]).all() and np.abs(beta[phase]).max() <= 1.0)

    history = np.zeros((4, 14), dtype=np.float32); history[:, 13] = 1.0
    held_history = history.copy(); held_history[:, 13] = 0.25
    held_future = np.zeros((8, 14), dtype=np.float32); held_future[:, 13] = 0.25
    held_future[:, 7:13] = np.linspace(0, 1, 8)[:, None] * np.asarray([-0.330, -1.425, -1.563, 1.617, 0.494, 0.779])[None]
    postclose = gate_decision(held_history, held_future, "Use the right arm to adjust the bottle.")
    left = gate_decision(held_history, held_future, "Use the left arm to adjust the bottle.")
    baseline = np.arange(8 * 4 * 5 * 3, dtype=np.uint16).reshape(8, 4, 5, 3).astype(np.uint8)
    parent = np.full_like(baseline, 100); candidate = np.full_like(baseline, 108)
    disabled = apply_aligned_projection(baseline, candidate, parent, None)
    bounded = apply_aligned_projection(baseline, candidate, parent, np.ones((8, 3)))
    checks = {
        "index_format": index.get("format") == "strict-track2-v441-postclose-trainonly-rgb-alignment-index-v1",
        "preregistration_format": prereg.get("format") == "strict-track2-v441-postclose-trainonly-preregistration-v1",
        "exact_train40": index.get("train_episodes") == train and len(train) == 40,
        "exact_right15": len(index.get("right_train_episodes", [])) == 15,
        "validation10_excluded": index.get("validation_episodes_excluded") == validation and len(validation) == 10,
        "train_validation_disjoint": not bool(set(train) & set(validation)),
        "exact_60_samples": len(index.get("samples", [])) == 60,
        "no_dev_sweep": index.get("guards", {}).get("validation_or_dev_used") is False and index.get("guards", {}).get("coefficient_dev_sweep") is False,
        "no_reward_calibration": index.get("guards", {}).get("reward_read") is False,
        "runtime_no_reward_import": not any("reward" in value.lower() for value in imports),
        "runtime_no_compute_reward": "compute_reward" not in calls,
        "postclose_gate_fixture": postclose.get("gate") is True and postclose.get("phase") == "postclose",
        "left_g0_fixture": left.get("gate") is False,
        "g0_bitexact": np.array_equal(disabled, baseline),
        "rgb_residual_bound_8": int(np.abs(bounded.astype(np.int16) - baseline.astype(np.int16)).max()) <= 8,
        **component_checks,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v441-postclose-s0-static-contract-v1",
        "passed": passed,
        "checks": checks,
        "fixtures": {"postclose": postclose, "left": left},
        "sha256": {
            "alignment_index": sha256(args.alignment_index),
            "preregistration": sha256(args.preregistration),
            "split": sha256(args.split),
            "runtime": sha256(args.runtime),
        },
        "guards": {
            "service_started": False, "policy_updates": 0,
            "hidden_or_final_data": False, "real_submission": False,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

