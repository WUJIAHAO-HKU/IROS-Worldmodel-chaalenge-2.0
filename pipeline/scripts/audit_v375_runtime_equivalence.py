#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.run / "audit/runtime_equivalence.json"
    if output.exists():
        raise FileExistsError("refusing overwrite")
    split = json.loads(args.split.read_text())
    arms = {int(k): v for k, v in split["arm_by_episode"].items()}
    episodes = sorted(ep for ep in map(int, split["validation_episodes"]) if arms[ep] == "right")
    paths = [path for ep in episodes for path in sorted(args.windows.glob(f"episode{ep}_*.npz"))]
    if len(paths) != 512:
        raise RuntimeError(f"expected 512 windows, got {len(paths)}")
    model = Track2V375BoundedCartesianPhase(args.run / "release", args.library, args.device)
    candidate_errors, copy_errors = [], []
    for begin in range(0, len(paths), 8):
        batch = paths[begin : begin + 8]
        contexts, histories, futures, targets, instructions = [], [], [], [], []
        for path in batch:
            ep = int(path.name.split("_")[0][7:])
            with np.load(path, allow_pickle=False) as data:
                contexts.append(data["context_frames"].copy()); histories.append(data["history_actions"].copy())
                futures.append(data["future_actions"].copy()); targets.append(data["target_frames"].copy())
            instructions.append(split["episode_to_instruction"][str(ep)])
        contexts = np.stack(contexts); histories = np.stack(histories); futures = np.stack(futures); targets = np.stack(targets)
        prediction = model.predict_batch(contexts, histories, futures, np.arange(begin, begin + len(batch)), instructions)
        candidate_errors.extend(np.abs(prediction.astype(np.float32) - targets.astype(np.float32)).mean(axis=(1, 2, 3, 4)))
        copies = np.repeat(contexts[:, -1:, :, :, :], 8, axis=1)
        copy_errors.extend(np.abs(copies.astype(np.float32) - targets.astype(np.float32)).mean(axis=(1, 2, 3, 4)))
        print(json.dumps({"completed": begin + len(batch), "total": len(paths)}), flush=True)
    candidate_errors = np.asarray(candidate_errors); copy_errors = np.asarray(copy_errors)
    ratio = float(candidate_errors.mean() / copy_errors.mean())
    checks = {
        "exact_512_windows": len(paths) == 512,
        "finite": bool(np.isfinite(candidate_errors).all() and np.isfinite(copy_errors).all()),
        "runtime_delta_rgb_ratio_le_0p75": ratio <= 0.75,
        "both_calibration_and_test_halves_improve": bool(
            candidate_errors[:251].mean() < copy_errors[:251].mean()
            and candidate_errors[251:].mean() < copy_errors[251:].mean()
        ),
    }
    report = {
        "format": "strict-track2-v375-runtime-equivalence-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes, "windows": len(paths), "candidate_rgb_mae": float(candidate_errors.mean()),
        "copy_last_rgb_mae": float(copy_errors.mean()), "ratio": ratio,
        "checks": checks, "passed": all(checks.values()),
        "guards": {"public_holdout_only": True, "reward_or_outcomes_used": False, "policy_modified": False, "real_submission": False},
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
