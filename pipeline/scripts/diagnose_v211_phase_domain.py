#!/usr/bin/env python3
"""Read-only domain diagnostic on frozen public-policy request captures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wam_pipeline.v312_causal_terminal_mirror_runtime import action_features
from wam_pipeline.v337_public_recursive_ood_gate import PublicRecursiveOODGate
from wam_pipeline.v389_public_recursive_phase_gate import PublicRecursivePhaseGate


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CAPTURE = (
    ROOT
    / "artifacts/strict_track2_official_20260810/run_registry"
    / "v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit"
)
ACTION_GATE = ART / "v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz"
SOURCE_GATE = ART / "v390_continuous_phase_clean_reanchor_seed1551_20260823/release/source_gate.npz"
OLD_PHASE = ART / "v389r1_public_recursive_phase_classifier_seed1550_20260823/recursive_phase_gate.npz"
NEW_PHASE = ART / "v393r1_public_failure_calibrated_phase_seed1554_20260823/failure_calibrated_phase_gate.npz"


def sigmoid(logit: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "q01", "q10", "q50", "q90", "q99", "max")}
    a = np.asarray(values, dtype=np.float64)
    q = np.quantile(a, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return dict(zip(("min", "q01", "q10", "q50", "q90", "q99", "max"), map(float, q), strict=True))


def main() -> None:
    with np.load(ACTION_GATE, allow_pickle=False) as payload:
        action_mean = payload["feature_mean"].astype(np.float32)
        action_scale = payload["feature_scale"].astype(np.float32)
        action_coef = payload["coefficient"].astype(np.float32)
        action_intercept = float(payload["intercept"].item())
        action_threshold = float(payload["threshold"].item())

    source = PublicRecursiveOODGate(SOURCE_GATE)
    old = PublicRecursivePhaseGate(OLD_PHASE)
    new = PublicRecursivePhaseGate(NEW_PHASE)
    rows: list[dict[str, float]] = []

    for path in sorted(CAPTURE.glob("rollout_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            contexts = payload["context_frames"]
            histories = payload["history_actions"]
            futures = payload["future_actions"]
            for context, history, future in zip(contexts, histories, futures, strict=True):
                feature = action_features(history, future)
                action_p = sigmoid(float(((feature - action_mean) / action_scale) @ action_coef + action_intercept))
                rows.append(
                    {
                        "action": action_p,
                        "source": source.probability(context),
                        "old_phase": old.probability(context, history, future),
                        "new_phase": new.probability(context, history, future),
                    }
                )

    action_rows = [row for row in rows if row["action"] >= action_threshold]
    source_rows = [row for row in action_rows if row["source"] >= source.threshold]
    result = {
        "contract": "outcome-free read-only public-policy request-domain diagnostic",
        "capture_files": len(list(CAPTURE.glob("rollout_*.npz"))),
        "rows": len(rows),
        "thresholds": {
            "action": action_threshold,
            "source": source.threshold,
            "old_phase": old.threshold,
            "new_phase": new.threshold,
        },
        "sequential_counts": {
            "action": len(action_rows),
            "action_and_source": len(source_rows),
            "old_full": sum(row["old_phase"] >= old.threshold for row in source_rows),
            "new_full": sum(row["new_phase"] >= new.threshold for row in source_rows),
        },
        "all_rows_probability_quantiles": {
            key: quantiles([row[key] for row in rows])
            for key in ("action", "source", "old_phase", "new_phase")
        },
        "action_source_rows_phase_quantiles": {
            "old_phase": quantiles([row["old_phase"] for row in source_rows]),
            "new_phase": quantiles([row["new_phase"] for row in source_rows]),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
