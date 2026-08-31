#!/usr/bin/env python3
"""Extend the frozen v326 contract with closed-loop context-coherence checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import test_v324_phase_guarded_terminal as inherited
from audit_v310_full_mirror_causal_gate import make_counterfactual
from wam_pipeline.v317_batched_sparse_failure_terminal_runtime import (
    Track2V317BatchedSparseFailureTerminal,
)
from wam_pipeline.v326_blended_phase_terminal_runtime import (
    Track2V326BlendedPhaseTerminal,
)
from wam_pipeline.v328_coherent_phase_trajectory_runtime import (
    Track2V328CoherentPhaseTrajectory,
)


def temporal_delta_error(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted_delta = np.diff(prediction.astype(np.float32), axis=0)
    target_delta = np.diff(target.astype(np.float32), axis=0)
    return float(np.abs(predicted_delta - target_delta).mean())


def main() -> int:
    inherited.Track2V324PhaseGuardedTerminal = Track2V328CoherentPhaseTrajectory
    inherited_result = inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if not output.is_file():
        return inherited_result
    report = json.loads(output.read_text())
    checkpoint = Path(sys.argv[sys.argv.index("--checkpoint-dir") + 1])
    library = Path(sys.argv[sys.argv.index("--library-index") + 1])
    action_gate = Path(sys.argv[sys.argv.index("--action-gate") + 1])
    phase_gate = Path(sys.argv[sys.argv.index("--phase-gate") + 1])
    windows = Path(sys.argv[sys.argv.index("--windows") + 1])
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"

    path = windows / "episode22_00100.npz"
    with np.load(path, allow_pickle=False) as values:
        context = values["context_frames"].astype(np.uint8)
        history = values["history_actions"].astype(np.float32)
        future = values["future_actions"].astype(np.float32)
        target = values["target_frames"].astype(np.uint8)
    seed = inherited.seed(path)
    prompt = "adjust bottle"
    candidate_runtime = Track2V328CoherentPhaseTrajectory(
        checkpoint, library, device, action_gate, phase_gate
    )
    baseline_runtime = Track2V317BatchedSparseFailureTerminal(
        checkpoint, library, device, action_gate
    )
    v326_runtime = Track2V326BlendedPhaseTerminal(
        checkpoint, library, device, action_gate, phase_gate
    )
    probability = candidate_runtime._probability(history, future)
    signature = candidate_runtime._signature(history, future, probability)
    coherent = candidate_runtime._coherent_phase_override(
        context, history, future, probability, signature
    )
    candidate = candidate_runtime.predict(context, history, future, seed, prompt)
    baseline = baseline_runtime.predict(context, history, future, seed, prompt)
    v326 = v326_runtime.predict(context, history, future, seed, prompt)

    variants = [
        make_counterfactual(future, history, name)
        for name in ("open_gripper", "static_transport", "reverse_transport")
    ]
    candidate_counterfactuals = candidate_runtime.predict_batch(
        np.repeat(context[None], 3, axis=0),
        np.repeat(history[None], 3, axis=0),
        np.stack(variants),
        np.repeat(seed, 3),
        [prompt] * 3,
    )
    baseline_counterfactuals = baseline_runtime.predict_batch(
        np.repeat(context[None], 3, axis=0),
        np.repeat(history[None], 3, axis=0),
        np.stack(variants),
        np.repeat(seed, 3),
        [prompt] * 3,
    )

    next_target = target[-5:]
    candidate_rgb_mae = float(
        np.abs(candidate[-5:].astype(np.int16) - next_target.astype(np.int16)).mean()
    )
    baseline_rgb_mae = float(
        np.abs(baseline[-5:].astype(np.int16) - next_target.astype(np.int16)).mean()
    )
    candidate_delta_error = temporal_delta_error(candidate[-5:], next_target)
    baseline_delta_error = temporal_delta_error(baseline[-5:], next_target)
    added_checks = {
        "valid_transition_coherent_override_true": coherent,
        "valid_transition_preterminal_changes_v317": bool(
            np.any(candidate[:-1] != baseline[:-1])
        ),
        "valid_transition_terminal_bit_exact_v326": bool(
            np.array_equal(candidate[-1], v326[-1])
        ),
        "valid_next_context_rgb_mae_strictly_improves_v317": (
            candidate_rgb_mae < baseline_rgb_mae
        ),
        "valid_next_context_temporal_delta_error_strictly_improves_v317": (
            candidate_delta_error < baseline_delta_error
        ),
        "all_counterfactual_full_sequences_bit_exact_v317": bool(
            np.array_equal(candidate_counterfactuals, baseline_counterfactuals)
        ),
    }
    report["format"] = "strict-track2-v328-coherent-phase-trajectory-contract-v1"
    report["candidate"] = "track2-v328-coherent-phase-trajectory-v326-v317"
    report["inherited_contract"] = "strict-track2-v324-phase-guarded-terminal-contract-v1"
    report["coherent_transition"] = {
        "window": path.name,
        "gate_probability": probability,
        "failure_signature": signature,
        "candidate_next_context_rgb_mae": candidate_rgb_mae,
        "v317_next_context_rgb_mae": baseline_rgb_mae,
        "candidate_next_context_temporal_delta_error": candidate_delta_error,
        "v317_next_context_temporal_delta_error": baseline_delta_error,
        "preterminal_max_change_vs_v317": int(
            np.abs(candidate[:-1].astype(np.int16) - baseline[:-1].astype(np.int16)).max()
        ),
    }
    report["checks"].update(added_checks)
    report["passed"] = all(report["checks"].values())
    report["guards"]["closed_loop_context_definition"] = (
        "the next RLinf request consumes prediction frames 3..7"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v328_passed": report["passed"], "added_checks": added_checks,
                      "coherent_transition": report["coherent_transition"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
