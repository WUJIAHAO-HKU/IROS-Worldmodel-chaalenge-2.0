#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wam_pipeline.v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN = ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v375_bounded_cartesian_phase_pilot_seed1538_20260823"


def main() -> None:
    window = ROOT / "artifacts/adjust_bottle_windows_full/episode6_00064.npz"
    split = json.loads((ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json").read_text())
    with np.load(window, allow_pickle=False) as data:
        context = data["context_frames"].copy(); history = data["history_actions"].copy(); future = data["future_actions"].copy()
    model = Track2V375BoundedCartesianPhase(
        RUN / "release",
        ROOT / "artifacts/strict_track2_joint_augmentation_20260810/v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz",
        "cuda",
    )
    instruction = split["episode_to_instruction"]["6"]
    start = time.perf_counter(); first = model.predict(context, history, future, 1538, instruction); latency = time.perf_counter() - start
    second = model.predict(context, history, future, 1538, instruction)
    report = {
        "format": "strict-track2-v375-runtime-smoke-v1",
        "shape": list(first.shape), "dtype": str(first.dtype), "latency_seconds_first": latency,
        "deterministic": bool(np.array_equal(first, second)),
        "pixel_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "arm_route": model.last_arm_route, "base_index": model.last_base_index,
        "selected_frames": model.last_selected_frames,
    }
    checks = {
        "shape": first.shape == (8, 256, 256, 3), "dtype": first.dtype == np.uint8,
        "deterministic": report["deterministic"], "right_route": report["arm_route"] == "right",
        "monotonic_frames": all(a <= b for a, b in zip(report["selected_frames"], report["selected_frames"][1:])),
    }
    report["checks"] = checks; report["passed"] = all(checks.values())
    (RUN / "audit/runtime_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
