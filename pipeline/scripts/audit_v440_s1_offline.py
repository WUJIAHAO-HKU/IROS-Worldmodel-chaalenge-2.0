#!/usr/bin/env python3
"""Offline one-shot public-development S1 gate for v440.

Recursive performance is measured against an independently rolled v169
trajectory.  Request-local exactness and the emitted residual cap are measured
only against the v169 baseline returned for that same hybrid request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REQUIRED = {
    "independent_v169_frames", "same_request_v169_frames", "v354_parent_frames",
    "hybrid_frames", "shuffle_frames", "target_frames",
    "independent_v169_reward", "same_request_v169_reward", "v354_parent_reward",
    "hybrid_reward", "target_reward", "episode", "arm", "phase", "gate",
    "swap_gate", "s1",
}
HOLDOUT10 = {2, 5, 6, 7, 9, 16, 18, 22, 39, 48}
RIGHT_DEV = {6, 7, 18, 22}
PROTECTED_IN_CHUNK = (0, 1, 6, 7)
LINEAGE = "v440"
STATIC_FORMAT = "strict-track2-v440-s0-static-contract-v1"
REPORT_FORMAT = "strict-track2-v440-s1-offline-gate-v1"
STRUCTURAL_GATE_CONTRACT = "right_s1_intervention_fraction_0p20_to_0p80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def ratio(value: float, reference: float) -> float:
    return float(value / max(reference, 1e-12))


def frame_metrics(frames: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "first8_rgb_mae": mae(frames[:, :8], target[:, :8]),
        "recursive32_rgb_mae": mae(frames, target),
    }


def reward_metrics(reward: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "reward_prediction_mae": float(np.abs(reward - target).mean()),
        "endpoint_reward_prediction_mae": float(np.abs(reward[:, -1] - target[:, -1]).mean()),
        "final_reward_mean": float(reward[:, -1].mean()),
    }


def exact(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def protected_indices(horizon: int) -> list[int]:
    if horizon % 8:
        raise ValueError(f"horizon must be a multiple of eight, got {horizon}")
    return [offset + item for offset in range(0, horizon, 8) for item in PROTECTED_IN_CHUNK]


def chunk_exact_counts(
    hybrid: np.ndarray, reference: np.ndarray, gate: np.ndarray, select_g0: bool
) -> tuple[int, int]:
    matched = total = 0
    for sample_index in range(len(hybrid)):
        for chunk_index in range(4):
            is_active = bool(gate[sample_index, chunk_index])
            if (select_g0 and is_active) or (not select_g0 and not is_active):
                continue
            begin = 8 * chunk_index
            total += 1
            matched += int(exact(
                hybrid[sample_index, begin : begin + 8],
                reference[sample_index, begin : begin + 8],
            ))
    return matched, total


def intervention_structure_checks(
    gate: np.ndarray, s1: np.ndarray, phase: np.ndarray
) -> dict[str, bool]:
    """Default postclose structure gate; lineages may preregister another."""
    del phase
    fraction = float(gate[s1].mean())
    return {
        "right_intervention_fraction_ge_0p20": fraction >= 0.20,
        "right_intervention_fraction_le_0p80": fraction <= 0.80,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, type=Path)
    parser.add_argument("--static-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    static = json.loads(args.static_report.read_text(encoding="utf-8"))
    if static.get("format") != STATIC_FORMAT:
        raise RuntimeError(f"wrong {LINEAGE} static report")

    with np.load(args.npz, allow_pickle=False) as values:
        missing = sorted(REQUIRED - set(values.files))
        if missing:
            raise RuntimeError(f"S1 input is missing arrays: {missing}")
        arrays = {key: np.asarray(values[key]) for key in REQUIRED}

    independent = arrays["independent_v169_frames"]
    local = arrays["same_request_v169_frames"]
    parent = arrays["v354_parent_frames"]
    hybrid = arrays["hybrid_frames"]
    shuffle = arrays["shuffle_frames"]
    target = arrays["target_frames"]
    frames = (independent, local, parent, hybrid, shuffle, target)
    frame_shape_ok = (
        all(item.shape == independent.shape for item in frames)
        and independent.ndim == 5 and independent.shape[1] == 32
        and independent.shape[-1] == 3
        and all(item.dtype == np.uint8 for item in frames)
    )
    if not frame_shape_ok:
        raise RuntimeError("expected six aligned uint8 [N,32,H,W,3] frame arrays")
    count = len(independent)
    reward_shape = (count, 32)
    reward_keys = (
        "independent_v169_reward", "same_request_v169_reward", "v354_parent_reward",
        "hybrid_reward", "target_reward",
    )
    if any(arrays[key].shape != reward_shape for key in reward_keys):
        raise RuntimeError(f"reward arrays must be {reward_shape}")
    if any(arrays[key].shape != (count,) for key in ("episode", "arm", "phase", "s1")):
        raise RuntimeError("episode/arm/phase/s1 must be one-dimensional and aligned")
    if any(arrays[key].shape != (count, 4) for key in ("gate", "swap_gate")):
        raise RuntimeError("gate/swap_gate must be request-level [N,4] arrays")

    arm = arrays["arm"].astype("U")
    phase = arrays["phase"].astype("U")
    gate = arrays["gate"].astype(bool)
    swap_gate = arrays["swap_gate"].astype(bool)
    s1 = arrays["s1"].astype(bool)
    episode_values = arrays["episode"].astype(np.int64)
    episodes = {int(value) for value in episode_values.tolist()}
    left = arm == "left"
    right = arm == "right"
    g0_probe = phase == "g0_probe"
    protected = protected_indices(hybrid.shape[1])
    if not s1.any() or not np.all(right[s1]):
        raise RuntimeError("S1 metric rows must be a nonempty right-arm-only subset")

    metrics = {}
    for name, prediction, rewards in (
        ("independent_v169", independent, arrays["independent_v169_reward"]),
        ("v354_parent", parent, arrays["v354_parent_reward"]),
        ("hybrid", hybrid, arrays["hybrid_reward"]),
    ):
        metrics[name] = {
            **frame_metrics(prediction[s1], target[s1]),
            **reward_metrics(
                rewards[s1].astype(np.float64), arrays["target_reward"][s1].astype(np.float64)
            ),
        }
    performance_ratios = {
        key: ratio(metrics["hybrid"][key], metrics["independent_v169"][key])
        for key in (
            "first8_rgb_mae", "recursive32_rgb_mae", "reward_prediction_mae",
            "endpoint_reward_prediction_mae",
        )
    }
    performance_ratios["final_reward_hybrid_over_independent_v169"] = ratio(
        metrics["hybrid"]["final_reward_mean"],
        metrics["independent_v169"]["final_reward_mean"],
    )
    true_action_mae = metrics["hybrid"]["recursive32_rgb_mae"]
    shuffle_action_mae = mae(shuffle[s1], target[s1])
    true_over_shuffle = ratio(true_action_mae, shuffle_action_mae)
    intervention_fraction = float(gate[s1].mean())

    local_delta = hybrid.astype(np.int16) - local.astype(np.int16)
    g0_local_exact, g0_total = chunk_exact_counts(hybrid, local, gate, select_g0=True)
    g0_independent_exact, _ = chunk_exact_counts(hybrid, independent, gate, select_g0=True)
    local_protected_exact = exact(hybrid[:, protected], local[:, protected])
    independent_protected_exact = exact(hybrid[:, protected], independent[:, protected])
    gated = np.repeat(gate, 8, axis=1)
    local_reward_delta = arrays["hybrid_reward"] - arrays["same_request_v169_reward"]

    checks = {
        "static_contract_passed": static.get("passed") is True,
        "samples_exact_48": count == 48,
        "s1_exact_32_samples": int(s1.sum()) == 32,
        "s1_phase_8_each": all(
            int(np.sum(s1 & (phase == name))) == 8
            for name in ("early", "grasp", "postgrasp", "endpoint")
        ),
        "fixed_public_holdout_dev_only": episodes <= HOLDOUT10 and bool(episodes),
        "s1_exact_public_right_dev_episodes": {
            int(value) for value in episode_values[s1].tolist()
        } == RIGHT_DEV,
        "left_exact_12_probes": int(left.sum()) == 12,
        "g0_exact_4_probes": int(g0_probe.sum()) == 4,
        "both_arms_covered": bool(left.any() and right.any()),
        "left_bitexact_same_request_v169": bool(
            left.any() and exact(hybrid[left], local[left])
        ),
        "g0_request_chunk_bitexact_same_request_v169": bool(
            g0_total > 0 and g0_local_exact == g0_total
        ),
        "t1_t2_t7_t8_each_chunk_bitexact_same_request_v169": local_protected_exact,
        "emitted_uint8_delta_from_same_request_bounded_by_cap8": bool(
            np.max(np.abs(local_delta), initial=0) <= 8
        ),
        "action_swap_changes_gate": bool(np.any(gate != swap_gate)),
        "swap_disables_at_least_one_original_intervention": bool(np.any(gate & ~swap_gate)),
        "first8_hybrid_over_independent_v169_le_0p998": (
            performance_ratios["first8_rgb_mae"] <= 0.998
        ),
        "recursive32_hybrid_over_independent_v169_le_1p002": (
            performance_ratios["recursive32_rgb_mae"] <= 1.002
        ),
        "reward_mae_nonregression": performance_ratios["reward_prediction_mae"] <= 1.0,
        "endpoint_reward_mae_nonregression": (
            performance_ratios["endpoint_reward_prediction_mae"] <= 1.0
        ),
        "final_reward_mean_ge_0p95_independent_v169": (
            metrics["hybrid"]["final_reward_mean"]
            >= 0.95 * metrics["independent_v169"]["final_reward_mean"]
        ),
        "true_action_vs_shuffle_target_mae_improves_0p2pct": true_over_shuffle <= 0.998,
        **intervention_structure_checks(gate, s1, phase),
        "all_rewards_finite": all(np.isfinite(arrays[key]).all() for key in reward_keys),
    }
    passed = all(checks.values())
    report = {
        "format": REPORT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "baseline_contract": {
            "recursive_performance_reference": "independent_v169",
            "exactness_and_cap_reference": "same_request_v169",
            "teacher_parent": "v354_parent",
            "silent_fallback_to_independent_for_exactness": False,
            "structural_gate_contract": STRUCTURAL_GATE_CONTRACT,
        },
        "coverage": {
            "samples": count,
            "s1_samples": int(s1.sum()),
            "episodes": sorted(episodes),
            "left_probes": int(left.sum()),
            "g0_swap_probes": int(g0_probe.sum()),
            "requests": int(gate.size),
            "gate_active_requests": int(gate.sum()),
            "right_intervention_fraction": intervention_fraction,
            "action_swap_gate_change_fraction": float((gate != swap_gate).mean()),
        },
        "three_way_performance_metrics": metrics,
        "hybrid_over_independent_v169": performance_ratios,
        "same_request_local_telemetry": {
            "g0_chunk_exact": g0_local_exact,
            "g0_chunks": g0_total,
            "max_abs_emitted_uint8_delta": int(np.max(np.abs(local_delta), initial=0)),
            "gated_mean_abs_emitted_uint8_delta": float(
                np.abs(local_delta[gated]).mean() if gated.any() else 0.0
            ),
            "gated_mean_reward_score_delta": float(
                local_reward_delta[gated].mean() if gated.any() else 0.0
            ),
        },
        "comparator_diagnostics_not_gate_inputs": {
            "g0_chunks_exact_against_independent_v169": g0_independent_exact,
            "g0_chunks_total": g0_total,
            "protected_exact_against_same_request_v169": local_protected_exact,
            "protected_exact_against_independent_v169": independent_protected_exact,
        },
        "action_causality": {
            "true_action_target_recursive32_mae": true_action_mae,
            "shuffle_action_target_recursive32_mae": shuffle_action_mae,
            "true_over_shuffle": true_over_shuffle,
        },
        "checks": checks,
        "decision": "may prepare zero-update trace only" if passed else "reject before rollout/RL",
        "evidence_sha256": {
            "s1_npz": sha256(args.npz), "static_report": sha256(args.static_report),
        },
        "guards": {
            "models_loaded_by_auditor": False,
            "gate_receives_reward": False,
            "gate_receives_seed": False,
            "gate_receives_request_id": False,
            "development_gate_runs_authorized": 1,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
