#!/usr/bin/env python3
"""Fail-closed full-window causal and non-regression audit for v315."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import (
    COUNTERFACTUALS,
    RIGHT_EPISODES,
    episode_number,
    make_counterfactual,
    score_terminal,
    sha256,
)
from audit_v314_transition_causal_gate import all_window_summary, transition_summary
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v315_sparse_failure_terminal_runtime import (
    Track2V315SparseFailureTerminal,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--library-index", required=True, type=Path)
    parser.add_argument("--action-gate", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--reward-checkpoint", required=True, type=Path)
    parser.add_argument("--t5-model", required=True, type=Path)
    parser.add_argument("--contract-report", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract_report.read_text())
    training = json.loads(args.training_report.read_text())
    if contract.get("passed") is not True or training.get("passed") is not True:
        raise RuntimeError("prerequisite failed")
    split = json.loads(args.split_manifest.read_text())
    instructions = json.loads(args.instruction_map.read_text())["episode_to_instruction"]
    for name, episodes in RIGHT_EPISODES.items():
        if not set(episodes).issubset(set(split[f"{name}_episodes"])):
            raise RuntimeError("split boundary mismatch")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    baseline = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    candidate = Track2V315SparseFailureTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )

    all_rows = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        paths = [
            path
            for episode in episodes
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
        ]
        rows = []
        frames = []
        prompts = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                context = np.asarray(data["context_frames"], dtype=np.uint8)
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
                target = np.asarray(data["target_frames"], dtype=np.uint8)
            episode = episode_number(path)
            instruction = str(instructions[str(episode)])
            seed = int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)
            direct = baseline.predict(context, history, future, seed, instruction)
            predicted = candidate.predict(context, history, future, seed, instruction)
            positive_probability = candidate.last_gate_probability
            positive_accepted = candidate.last_gate_accepted
            positive_signature = candidate.last_failure_signature
            counter_predictions = {}
            counter_gate = {}
            for name in COUNTERFACTUALS:
                altered = make_counterfactual(future, history, name)
                counter_predictions[name] = candidate.predict(
                    context, history, altered, seed, instruction
                )
                counter_gate[name] = {
                    "probability": candidate.last_gate_probability,
                    "accepted": candidate.last_gate_accepted,
                    "failure_signature": candidate.last_failure_signature,
                }
            rows.append(
                {
                    "window": path.name,
                    "episode": episode,
                    "direct_terminal_rgb_mae": float(
                        np.abs(direct[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                    ),
                    "candidate_terminal_rgb_mae": float(
                        np.abs(predicted[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                    ),
                    "positive_terminal_parent_max_difference": int(
                        np.abs(predicted[-1].astype(np.int16) - direct[-1].astype(np.int16)).max()
                    ),
                    "positive_gate_probability": positive_probability,
                    "positive_gate_accepted": positive_accepted,
                    "positive_failure_signature": positive_signature,
                    "counterfactual_gate": counter_gate,
                }
            )
            for value in (context[-1:], target, direct, predicted):
                frames.append(value[-1])
                prompts.append(instruction)
            for name in COUNTERFACTUALS:
                frames.append(counter_predictions[name][-1])
                prompts.append(instruction)
        scores = score_terminal(
            reward, np.stack(frames), prompts, device, args.reward_batch_size
        ).reshape(len(rows), 4 + len(COUNTERFACTUALS))
        for row, values in zip(rows, scores, strict=True):
            row["context_terminal_reward"] = float(values[0])
            row["gt_terminal_reward"] = float(values[1])
            row["direct_terminal_reward"] = float(values[2])
            row["candidate_terminal_reward"] = float(values[3])
            row["counterfactual_rewards"] = {
                name: float(values[4 + index])
                for index, name in enumerate(COUNTERFACTUALS)
            }
        all_rows[split_name] = rows
        print(f"V315_SCORED {split_name} {len(rows)}", flush=True)

    all_summary = {name: all_window_summary(rows) for name, rows in all_rows.items()}
    transitions = {name: transition_summary(rows) for name, rows in all_rows.items()}
    transition_parent_exact = {}
    for split_name, rows in all_rows.items():
        selected = [
            row for row in rows
            if row["context_terminal_reward"] <= 0.10 and row["gt_terminal_reward"] >= 0.90
        ]
        transition_parent_exact[split_name] = bool(selected) and max(
            row["positive_terminal_parent_max_difference"] for row in selected
        ) == 0

    validation = transitions["validation"]
    local_test = transitions["local_test"]
    checks = {
        "contract_gate": contract.get("passed") is True,
        "training_gate": training.get("passed") is True,
        "validation_transition_count": validation["windows"] >= 12,
        "local_test_transition_count": local_test["windows"] >= 12,
        "validation_positive_parent_terminal_exact": transition_parent_exact["validation"],
        "local_test_positive_parent_terminal_exact": transition_parent_exact["local_test"],
        "validation_failure_consistency": validation["counterfactual_hit_rate_at_0p9"] <= 0.10,
        "local_test_failure_consistency": local_test["counterfactual_hit_rate_at_0p9"] <= 0.15,
        "validation_pairwise_ranking": validation["pairwise_win_rate"] >= 0.90,
        "local_test_pairwise_ranking": local_test["pairwise_win_rate"] >= 0.85,
        "validation_margin": validation["mean_margin"] is not None and validation["mean_margin"] >= 0.70,
        "local_test_margin": local_test["mean_margin"] is not None and local_test["mean_margin"] >= 0.40,
        "validation_positive_route": validation["positive_gate_accept_rate"] >= 0.90,
        "local_test_positive_route": local_test["positive_gate_accept_rate"] >= 0.85,
        "validation_counterfactual_suppression": validation["counterfactual_gate_reject_rate"] >= 0.90,
        "local_test_counterfactual_suppression": local_test["counterfactual_gate_reject_rate"] >= 0.85,
        "validation_all_window_rgb_nonregression": all_summary["validation"]["terminal_rgb"]["mae_ratio"] <= 1.0,
        "local_test_all_window_rgb_nonregression": all_summary["local_test"]["terminal_rgb"]["mae_ratio"] <= 1.0,
        "validation_all_window_reward_nonregression": all_summary["validation"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.0,
        "local_test_all_window_reward_nonregression": all_summary["local_test"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.05,
        "left_non_regression": contract["checks"]["left_parent_bit_exact"] is True,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v315-sparse-failure-terminal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v315-sparse-failure-terminal-v271",
        "transition_definition": "context official reward <=0.10 and public GT terminal reward >=0.90",
        "all_window_summaries": all_summary,
        "transition_summaries": transitions,
        "transition_positive_parent_terminal_exact": transition_parent_exact,
        "checks": checks,
        "passed": passed,
        "authorization": {
            "intended_effect_observed": validation["pairwise_win_rate"] > 0.5,
            "right_terminal_ranking": checks["validation_pairwise_ranking"]
            and checks["local_test_pairwise_ranking"]
            and checks["validation_margin"]
            and checks["local_test_margin"],
            "left_non_regression": checks["left_non_regression"],
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "action_gate": sha256(args.action_gate),
            "contract_report": sha256(args.contract_report),
            "training_report": sha256(args.training_report),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "runtime_reads_reward_or_outcome": False,
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "public_episode_disjoint_data_only": True,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": passed,
        "checks": checks,
        "all_window_summaries": all_summary,
        "transition_summaries": transitions,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
