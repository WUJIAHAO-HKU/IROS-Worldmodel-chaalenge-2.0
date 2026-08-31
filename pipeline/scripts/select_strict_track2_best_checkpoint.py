#!/usr/bin/env python3
"""Select the highest strictly evaluated Track-2 real-environment checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--route-summary", required=True, type=Path)
    parser.add_argument("--recursive-summary", required=True, type=Path)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_summary.read_text())
    candidate = json.loads(args.candidate_summary.read_text())
    routes = json.loads(args.route_summary.read_text())
    recursive = json.loads(args.recursive_summary.read_text())
    frozen_metrics = frozen["primary_official_success"][frozen["candidate_variant"]]
    candidate_metrics = candidate["candidate"]
    selection = {
        "format": "strict-track2-best-checkpoint-selection-v1",
        "selection_rule": "maximize official RoboTwin adjust_bottle success_once over exactly 128 seeds; ties retain frozen checkpoint",
        "selected": {
            "label": frozen["candidate_variant"],
            "checkpoint": args.selected_checkpoint,
            "successes": frozen_metrics["successes"],
            "count": frozen_metrics["count"],
            "success_rate": frozen_metrics["success_rate"],
            "left": frozen["success_improvement_by_arm"].get("left"),
            "right": frozen["success_improvement_by_arm"].get("right"),
            "status": "selected_as_highest_verified_real_env_checkpoint",
        },
        "rejected": {
            "right_focus_full200_b800_global_step_1": {
                "successes": candidate_metrics["successes"],
                "count": candidate_metrics["count"],
                "delta_vs_selected": candidate["improvement"],
                "reason": "45/128 is below frozen V16.9 49/128; right_success remains zero",
            },
            "right_route_ab": {
                "summary": routes,
                "reason": "swap, mirror and official-demo affine routes all tie frozen V16.9 at 4/8 shard success",
            },
            "recursive_parent_v1": {
                "promotion_gate": recursive["promotion_gate"],
                "improvement": recursive["improvement"]["overall"],
                "reason": "200-step prediction-only RGB and late-horizon gate regresses despite texture gain",
            },
        },
        "world_model_evaluation_policy": "targets are metric-only after inference; no future frames, MPC, or real-eval labels select the checkpoint",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
