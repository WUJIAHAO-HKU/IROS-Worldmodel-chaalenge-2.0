#!/usr/bin/env python3
"""Reward-transition causal audit for v314 on episode-disjoint public demos."""

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
    evenly_spaced,
    episode_number,
    make_counterfactual,
    score_terminal,
    sha256,
)
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v314_transition_causal_terminal_runtime import (
    Track2V314TransitionCausalTerminal,
)


def corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def all_window_summary(rows: list[dict]) -> dict:
    direct_rgb = np.asarray([row["direct_terminal_rgb_mae"] for row in rows])
    candidate_rgb = np.asarray([row["candidate_terminal_rgb_mae"] for row in rows])
    gt = np.asarray([row["gt_terminal_reward"] for row in rows])
    direct = np.asarray([row["direct_terminal_reward"] for row in rows])
    candidate = np.asarray([row["candidate_terminal_reward"] for row in rows])
    direct_mae = float(np.abs(direct - gt).mean())
    candidate_mae = float(np.abs(candidate - gt).mean())
    return {
        "windows": len(rows),
        "terminal_rgb": {
            "direct_mae": float(direct_rgb.mean()),
            "candidate_mae": float(candidate_rgb.mean()),
            "mae_ratio": float(candidate_rgb.mean() / max(direct_rgb.mean(), 1e-12)),
        },
        "terminal_reward_fidelity": {
            "direct_mae": direct_mae,
            "candidate_mae": candidate_mae,
            "mae_ratio": float(candidate_mae / max(direct_mae, 1e-12)),
            "direct_correlation": corr(direct, gt),
            "candidate_correlation": corr(candidate, gt),
        },
    }


def transition_summary(rows: list[dict]) -> dict:
    selected = [
        row
        for row in rows
        if row["context_terminal_reward"] <= 0.10
        and row["gt_terminal_reward"] >= 0.90
    ]
    positive = np.asarray([row["candidate_terminal_reward"] for row in selected])
    counter = np.asarray(
        [[row["counterfactual_rewards"][name] for name in COUNTERFACTUALS] for row in selected]
    )
    gate_positive = np.asarray([row["positive_gate_accepted"] for row in selected], dtype=bool)
    gate_negative = np.asarray(
        [
            row["counterfactual_gate"][name]["accepted"]
            for row in selected
            for name in COUNTERFACTUALS
        ],
        dtype=bool,
    )
    pairwise = positive[:, None] > counter if len(selected) else np.empty((0, 3), dtype=bool)
    margin = positive[:, None] - counter if len(selected) else np.empty((0, 3))
    return {
        "windows": len(selected),
        "positive_reward_mean": float(positive.mean()) if len(selected) else None,
        "positive_hit_rate_at_0p9": float((positive >= 0.9).mean()) if len(selected) else 0.0,
        "counterfactual_reward_mean": float(counter.mean()) if len(selected) else None,
        "counterfactual_hit_rate_at_0p9": float((counter >= 0.9).mean()) if len(selected) else 1.0,
        "pairwise_win_rate": float(pairwise.mean()) if len(selected) else 0.0,
        "strict_all_three_win_rate": float(pairwise.all(axis=1).mean()) if len(selected) else 0.0,
        "mean_margin": float(margin.mean()) if len(selected) else None,
        "positive_gate_accept_rate": float(gate_positive.mean()) if len(selected) else 0.0,
        "counterfactual_gate_reject_rate": float((~gate_negative).mean()) if len(selected) else 0.0,
        "windows_used": [row["window"] for row in selected],
    }


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
    candidate = Track2V314TransitionCausalTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )

    all_rows = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        paths = evenly_spaced(
            [
                path
                for episode in episodes
                for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
            ],
            64,
        )
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
                    "positive_gate_probability": positive_probability,
                    "positive_gate_accepted": positive_accepted,
                    "counterfactual_gate": counter_gate,
                }
            )
            for value in (context[-1:], target, direct, predicted):
                frames.append(value[-1])
                prompts.append(instruction)
            for name in COUNTERFACTUALS:
                frames.append(counter_predictions[name][-1])
                prompts.append(instruction)
        scores = score_terminal(reward, np.stack(frames), prompts, device, 32).reshape(
            len(rows), 4 + len(COUNTERFACTUALS)
        )
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
        print(f"V314_SCORED {split_name} {len(rows)}", flush=True)

    all_summary = {name: all_window_summary(rows) for name, rows in all_rows.items()}
    transitions = {name: transition_summary(rows) for name, rows in all_rows.items()}
    validation = transitions["validation"]
    local_test = transitions["local_test"]
    checks = {
        "contract_gate": contract.get("passed") is True,
        "training_gate": training.get("passed") is True,
        "validation_transition_count": validation["windows"] >= 12,
        "local_test_transition_count": local_test["windows"] >= 12,
        "validation_positive_recall": validation["positive_hit_rate_at_0p9"] >= 0.75,
        "local_test_positive_recall": local_test["positive_hit_rate_at_0p9"] >= 0.70,
        "validation_failure_consistency": validation["counterfactual_hit_rate_at_0p9"] <= 0.10,
        "local_test_failure_consistency": local_test["counterfactual_hit_rate_at_0p9"] <= 0.15,
        "validation_pairwise_ranking": validation["pairwise_win_rate"] >= 0.90,
        "local_test_pairwise_ranking": local_test["pairwise_win_rate"] >= 0.85,
        "validation_margin": validation["mean_margin"] is not None and validation["mean_margin"] >= 0.70,
        "local_test_margin": local_test["mean_margin"] is not None and local_test["mean_margin"] >= 0.60,
        "validation_positive_gate": validation["positive_gate_accept_rate"] >= 0.90,
        "local_test_positive_gate": local_test["positive_gate_accept_rate"] >= 0.85,
        "validation_negative_gate": validation["counterfactual_gate_reject_rate"] >= 0.90,
        "local_test_negative_gate": local_test["counterfactual_gate_reject_rate"] >= 0.85,
        "validation_all_window_rgb_nonregression": all_summary["validation"]["terminal_rgb"]["mae_ratio"] <= 1.05,
        "local_test_all_window_rgb_nonregression": all_summary["local_test"]["terminal_rgb"]["mae_ratio"] <= 1.05,
        "validation_all_window_reward_nonregression": all_summary["validation"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.0,
        "local_test_all_window_reward_nonregression": all_summary["local_test"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.05,
        "left_non_regression": contract["checks"]["left_parent_bit_exact"] is True,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v314-transition-causal-terminal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v314-transition-causal-terminal-v271",
        "transition_definition": "context official reward <=0.10 and public GT terminal reward >=0.90",
        "all_window_summaries": all_summary,
        "transition_summaries": transitions,
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
    print(json.dumps({"passed": passed, "checks": checks, "all_window_summaries": all_summary, "transition_summaries": transitions}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
