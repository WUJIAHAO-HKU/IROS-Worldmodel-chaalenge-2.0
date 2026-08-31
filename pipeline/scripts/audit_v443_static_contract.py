#!/usr/bin/env python3
"""S0 audit for v443 train-only spatial alignment and protected frames."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision
from wam_pipeline.v443_v169_close_spatial_projection_runtime import PROTECTED_FRAMES, apply_spatial_projection


FORMAT = "strict-track2-v443-close-s0-static-contract-v1"
INDEX_FORMAT = "strict-track2-v443-close-trainonly-spatial-alignment-index-v1"
PREREG_FORMAT = "strict-track2-v443-close-trainonly-preregistration-v1"
MIN_STABLE_FOLDS = 12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("alignment-index", "preregistration", "split", "runtime", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.alignment_index, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        beta = np.asarray(archive["beta_map"], dtype=np.float32)
        sign_count = np.asarray(archive["sign_agreement_count"], dtype=np.uint8)
        improvement_count = np.asarray(archive["heldout_improvement_count"], dtype=np.uint8)
    prereg = json.loads(args.preregistration.read_text())
    split = json.loads(args.split.read_text())
    tree = ast.parse(args.runtime.read_text(), filename=str(args.runtime))
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    active = beta != 0.0
    active_pixels = int(np.any(active, axis=(0, 3)).sum()) if beta.ndim == 4 else -1

    history = np.zeros((4, 14), dtype=np.float32); history[:, 13] = 1.0
    close_future = np.zeros((8, 14), dtype=np.float32); close_future[:4, 13] = 1.0; close_future[4:, 13] = 0.25
    close = gate_decision(history, close_future, "Use the right arm to adjust the bottle.")
    held_history = history.copy(); held_history[:, 13] = 0.25
    held_future = close_future.copy(); held_future[:, 13] = 0.25
    repeated = gate_decision(held_history, held_future, "Use the right arm to adjust the bottle.")
    left = gate_decision(history, close_future, "Use the left arm to adjust the bottle.")
    valid_shape = beta.ndim == 4 and beta.shape[0] == 8 and beta.shape[-1] == 3
    if valid_shape:
        baseline = np.full(beta.shape, 120, dtype=np.uint8)
        parent = np.full(beta.shape, 100, dtype=np.uint8)
        candidate = np.full(beta.shape, 108, dtype=np.uint8)
        disabled = apply_spatial_projection(baseline, candidate, parent, None)
        enabled = apply_spatial_projection(baseline, candidate, parent, beta)
        max_delta = int(np.abs(enabled.astype(np.int16) - baseline.astype(np.int16)).max(initial=0))
        protected_exact = np.array_equal(enabled[list(PROTECTED_FRAMES)], baseline[list(PROTECTED_FRAMES)])
    else:
        disabled = enabled = np.asarray([]); max_delta = 999; protected_exact = False
    checks = {
        "index_format": metadata.get("format") == INDEX_FORMAT,
        "preregistration_format": prereg.get("format") == PREREG_FORMAT,
        "exact_train40": metadata.get("train_episodes") == train and len(train) == 40,
        "exact_right15": len(metadata.get("right_train_episodes", [])) == 15,
        "validation10_excluded": metadata.get("validation_episodes_excluded") == validation and len(validation) == 10,
        "train_validation_disjoint": not bool(set(train) & set(validation)),
        "exact_15_samples": len(metadata.get("samples", [])) == 15,
        "spatial_shape": valid_shape and sign_count.shape == beta.shape and improvement_count.shape == beta.shape,
        "bounded_finite": bool(np.isfinite(beta).all() and np.abs(beta).max(initial=0.0) <= 1.0),
        "protected_beta_exact_zero": valid_shape and bool(np.all(beta[list(PROTECTED_FRAMES)] == 0.0)),
        "active_pixels_min": active_pixels >= 64 and active_pixels == metadata.get("loeo", {}).get("active_pixels"),
        "active_sign_stable": bool(np.all(sign_count[active] >= MIN_STABLE_FOLDS)),
        "active_heldout_improves": bool(np.all(improvement_count[active] >= MIN_STABLE_FOLDS)),
        "aggregate_loeo_improves": 0.0 <= float(metadata.get("loeo", {}).get("aggregate_sse_ratio", 2.0)) < 1.0,
        "no_morphology_or_smoothing": metadata.get("loeo", {}).get("morphology_or_smoothing") is False,
        "no_dev_sweep": metadata.get("guards", {}).get("validation_or_dev_used") is False and metadata.get("guards", {}).get("coefficient_dev_sweep") is False,
        "no_reward_calibration": metadata.get("guards", {}).get("reward_read") is False,
        "runtime_no_reward_import": not any("reward" in value.lower() for value in imports),
        "runtime_no_compute_reward": "compute_reward" not in calls,
        "close_gate_fixture": close.get("gate") is True and close.get("first_close_index") == 4,
        "postclose_repeat_disabled": repeated.get("gate") is False,
        "left_g0_fixture": left.get("gate") is False,
        "g0_bitexact": valid_shape and np.array_equal(disabled, baseline),
        "protected_output_bitexact": protected_exact,
        "rgb_residual_bound_8": max_delta <= 8,
    }
    passed = all(checks.values())
    report = {
        "format": FORMAT, "passed": passed, "checks": checks,
        "metrics": {"active_pixels": active_pixels, "active_components": int(active.sum()), "aggregate_loeo_sse_ratio": metadata.get("loeo", {}).get("aggregate_sse_ratio")},
        "fixtures": {"close": close, "repeated_postclose": repeated, "left": left},
        "sha256": {
            "alignment_index": sha256(args.alignment_index), "preregistration": sha256(args.preregistration),
            "split": sha256(args.split), "runtime": sha256(args.runtime),
        },
        "guards": {
            "service_started": False, "policy_updates": 0, "hidden_or_final_data": False,
            "real_submission": False, "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
