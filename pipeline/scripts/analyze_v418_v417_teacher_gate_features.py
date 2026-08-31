#!/usr/bin/env python3
"""Extract causal action/phase features for v417 public holdout diagnosis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wam_pipeline.v417_reward_monotone_successor_runtime import (
    Track2V417RewardMonotoneSuccessor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--library-index", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--v417-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.v417_report.read_text())
    runtime = Track2V417RewardMonotoneSuccessor(
        args.checkpoint_dir, args.library_index, args.device
    )
    rows = []
    for source in report["rows"]["candidate"]:
        path = args.windows / f"episode{source['episode']}_{source['start']:05d}.npz"
        with np.load(path, allow_pickle=False) as values:
            context = values["context_frames"].astype(np.uint8)
            history = values["history_actions"].astype(np.float32)
            future = values["future_actions"].astype(np.float32)
        arm = runtime.parent.active_arm(history, future, "")
        source_probability = float(runtime.source_gate.probability(context))
        post_grasp = bool(runtime._post_grasp(history, future))
        action_probability = float(runtime._probability(history, future))
        signature = runtime._signature(history, future, action_probability)
        base, distance = runtime._nearest_clean(context, history, future)
        target = runtime._reward_monotone_row(base)
        phase = runtime._action_phase_base(history, future)
        phase_probability = float(
            runtime.continuous_phase_gate.probability(context, history, future)
        )
        hard_phase = bool(runtime._phase(phase)[0])
        rows.append(
            {
                "key": source["key"],
                "split": source["split"],
                "episode": source["episode"],
                "start": source["start"],
                "gt_terminal_reward": source["gt_terminal_reward"],
                "v417_recursive_terminal_reward": source["recursive_terminal_reward"],
                "arm": arm,
                "source_probability": source_probability,
                "post_grasp": post_grasp,
                "action_probability": action_probability,
                "failure_signature": signature,
                "base_start": int(runtime.row_start[base]),
                "base_distance": float(distance),
                "target_start": int(runtime.row_start[target]) if target is not None else None,
                "target_reward": float(runtime.v417_terminal_reward[target]) if target is not None else None,
                "phase_start": int(runtime.row_start[phase]),
                "phase_probability": phase_probability,
                "hard_phase_ready": hard_phase,
            }
        )
    gates = []
    validation = [row for row in rows if row["split"] == "validation"]
    for action_min in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        for base_start_min in (0, 64, 72, 80, 88, 92, 96, 100):
            selected = [
                row for row in validation
                if row["arm"] == "right"
                and row["post_grasp"]
                and row["failure_signature"] is None
                and row["target_start"] is not None
                and row["action_probability"] >= action_min
                and row["base_start"] >= base_start_min
            ]
            positive = sum(row["gt_terminal_reward"] >= 0.9 for row in selected)
            negative = sum(row["gt_terminal_reward"] < 0.9 for row in selected)
            gates.append(
                {
                    "action_probability_min": action_min,
                    "base_start_min": base_start_min,
                    "validation_routes": len(selected),
                    "validation_positive": positive,
                    "validation_negative": negative,
                }
            )
    payload = {
        "format": "strict-track2-v418-v417-teacher-causal-gate-features-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "gate_grid": gates,
        "guards": {
            "official_public_holdout_windows_only": True,
            "teacher_context_features_only": True,
            "simulator_outcomes_used": False,
            "hidden_or_final_data": False,
            "policy_modified": False,
            "real_submission": False,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "viable_validation_gates": [g for g in gates if g["validation_negative"] <= 4 and g["validation_positive"] >= 8]}, indent=2))


if __name__ == "__main__":
    main()
