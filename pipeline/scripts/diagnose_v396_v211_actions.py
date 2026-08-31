#!/usr/bin/env python3
"""Outcome-free action rescue coverage on the frozen v211 request capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_pipeline.v396_action_phase_gate import PublicActionPhaseGate


ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CAPTURE = ROOT / "artifacts/strict_track2_official_20260810/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"


def causal_action_features(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    anchor = history[-1, 7:13]
    relative = future[:, 7:13] - anchor
    sequence = np.concatenate((anchor[None], future[:, 7:13]), axis=0)
    delta = np.diff(sequence, axis=0)
    path = np.linalg.norm(delta, axis=1)
    net = float(np.linalg.norm(relative[-1]))
    summary = np.asarray([path.sum(), net, net / max(float(path.sum()), 1e-8), path.mean(), path.max(), np.linalg.norm(np.diff(delta, axis=0), axis=1).mean()], dtype=np.float32)
    gripper = np.concatenate((history[-1:, 13], future[:, 13])).astype(np.float32)
    return np.concatenate((relative.reshape(-1), delta.reshape(-1), gripper, summary))


def main() -> None:
    gate = PublicActionPhaseGate(J / "v396_action_domain_terminal_rescue_seed1556_20260823/action_phase_gate.npz")
    with np.load(J / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz", allow_pickle=False) as p:
        mean, scale, coef = (p[k].astype(np.float32) for k in ("feature_mean", "feature_scale", "coefficient"))
        intercept = float(p["intercept"].item())
    rows = []
    for path in sorted(CAPTURE.glob("rollout_*.npz")):
        with np.load(path, allow_pickle=False) as p:
            for history, future in zip(p["history_actions"], p["future_actions"], strict=True):
                feature = causal_action_features(history, future)
                logit = float(((feature - mean) / scale) @ coef + intercept)
                action_probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))
                post_grasp = bool(history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75)
                sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
                path_length = float(np.linalg.norm(np.diff(sequence, axis=0), axis=1).sum())
                failure = bool((history[-1, 13] <= 0.5 and float(future[:, 13].mean()) > 0.5) or path_length <= 1e-6 or (float(future[:, 13].mean()) <= 0.5 and action_probability < 0.01))
                eligible = bool(post_grasp and action_probability >= 0.99 and not failure)
                rows.append((eligible, gate.probability(history, future)))
    eligible_probability = np.asarray([p for eligible, p in rows if eligible])
    print(json.dumps({
        "contract": "outcome-free actions-only coverage diagnostic",
        "rows": len(rows), "runtime_action_eligible": int(len(eligible_probability)),
        "action_phase_ready": int((eligible_probability >= gate.threshold).sum()),
        "threshold": gate.threshold,
        "eligible_probability_quantiles": {str(q): float(np.quantile(eligible_probability, q)) for q in (0, .1, .25, .5, .75, .9, .99, 1)},
        "guards": {"outcomes_read": False, "rgb_read": False, "hidden_or_final_data": False},
    }, indent=2))


if __name__ == "__main__":
    main()
