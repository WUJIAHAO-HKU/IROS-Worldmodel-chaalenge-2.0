#!/usr/bin/env python3
"""Check online v216 request math against the frozen v216 first chunk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_pipeline.v216_public_knn_blend_runtime import Track2V216PublicKNNBlend


BASE = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT = BASE / "artifacts/strict_track2_joint_augmentation_20260810"
V209 = JOINT / "v209_v202_v208_public_arm_routed_release"
V214 = JOINT / "v214_public_right_knn_action_visual_diagnostic_seed1413"
V216 = JOINT / "v216_public_right_parametric_knn_blend_sweep_seed1415"
PUBLIC = BASE / "artifacts/adjust_bottle_windows_full"


def main() -> None:
    baseline_path = V216 / "audit/public_success_baseline.npz"
    candidate_path = V216 / "audit/alpha070_public_success_candidate.npz"
    with np.load(baseline_path, allow_pickle=False) as values:
        paths = values["path"].astype(str)
        seeds = values["synthetic_seed"].astype(np.int64)
        instructions = values["instruction"].astype(str)
        right = values["arm_right"].astype(bool)
    position = int(np.flatnonzero(right)[0])
    with np.load(PUBLIC / paths[position], allow_pickle=False) as values:
        context = values["context_frames"].copy()
        history = values["history_actions"].copy()
        future = values["future_actions"].copy()
    runtime = Track2V216PublicKNNBlend(
        V209, V214 / "library/public_right_knn.npz", "cuda"
    )
    actual = runtime.predict(
        context, history, future, int(seeds[position]), instructions[position]
    )
    with np.load(candidate_path, allow_pickle=False) as values:
        expected = values["candidate"][position, :8].copy()
    manifest = json.loads(
        (V214 / "audit/public_success_candidate.manifest.json").read_text()
    )
    expected_library_path = manifest["selections"][position][0]["library_path"]
    actual_library_path = str(runtime.paths[runtime.last_retrieval_index])
    report = {
        "query": paths[position],
        "route": runtime.last_arm_route,
        "selected_index": runtime.last_retrieval_index,
        "selected_path_matches": actual_library_path == expected_library_path,
        "actual_selected_path": actual_library_path,
        "expected_selected_path": expected_library_path,
        "pixel_exact": bool(np.array_equal(actual, expected)),
        "pixel_mae": float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean()),
        "pixel_max_abs": int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()),
    }
    print(json.dumps(report, indent=2))
    if not report["selected_path_matches"] or not report["pixel_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
