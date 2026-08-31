#!/usr/bin/env python3
"""Held-out visual, reward, and counterfactual gate for v312."""

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
    summarize,
)
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v312_causal_terminal_mirror_runtime import (
    Track2V312CausalTerminalMirror,
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
        raise RuntimeError("v312 prerequisite gate failed")

    split = json.loads(args.split_manifest.read_text())
    instruction_map = json.loads(args.instruction_map.read_text())["episode_to_instruction"]
    for name, episodes in RIGHT_EPISODES.items():
        if not set(episodes).issubset(set(split[f"{name}_episodes"])):
            raise RuntimeError(f"{name} boundary mismatch")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    baseline = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    candidate = Track2V312CausalTerminalMirror(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )

    all_rows: dict[str, list[dict]] = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        paths = [
            path
            for episode in episodes
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
        ]
        paths = evenly_spaced(paths, 64)
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
            instruction = str(instruction_map[str(episode)])
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
            row = {
                "window": path.name,
                "episode": episode,
                "direct_terminal_rgb_mae": float(
                    np.abs(direct[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                ),
                "mirror_terminal_rgb_mae": float(
                    np.abs(predicted[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                ),
                "positive_gate_probability": positive_probability,
                "positive_gate_accepted": positive_accepted,
                "counterfactual_gate": counter_gate,
            }
            rows.append(row)
            for value in (target, direct, predicted):
                frames.append(value[-1])
                prompts.append(instruction)
            for name in COUNTERFACTUALS:
                frames.append(counter_predictions[name][-1])
                prompts.append(instruction)
        scores = score_terminal(
            reward, np.stack(frames), prompts, device, args.reward_batch_size
        ).reshape(len(rows), 3 + len(COUNTERFACTUALS))
        for row, values in zip(rows, scores, strict=True):
            row["gt_terminal_reward"] = float(values[0])
            row["direct_terminal_reward"] = float(values[1])
            row["mirror_terminal_reward"] = float(values[2])
            row["counterfactual_rewards"] = {
                name: float(values[3 + index])
                for index, name in enumerate(COUNTERFACTUALS)
            }
        all_rows[split_name] = rows
        print(f"V312_SCORED {split_name} {len(rows)}", flush=True)

    summaries = {name: summarize(rows) for name, rows in all_rows.items()}
    gate_summary = {}
    for name, rows in all_rows.items():
        positives = np.asarray([row["positive_gate_accepted"] for row in rows], dtype=bool)
        negatives = np.asarray(
            [
                row["counterfactual_gate"][kind]["accepted"]
                for row in rows
                for kind in COUNTERFACTUALS
            ],
            dtype=bool,
        )
        gate_summary[name] = {
            "positive_accept_rate": float(positives.mean()),
            "negative_reject_rate": float((~negatives).mean()),
        }

    thresholds = {
        "validation_terminal_rgb_relative_change_max": -0.05,
        "local_test_terminal_rgb_relative_change_max": -0.03,
        "validation_reward_mae_ratio_max": 1.0,
        "local_test_reward_mae_ratio_max": 1.05,
        "validation_pairwise_win_rate_min": 0.75,
        "local_test_pairwise_win_rate_min": 0.65,
        "validation_normalized_margin_min": 0.05,
        "validation_positive_accept_rate_min": 0.70,
        "local_test_positive_accept_rate_min": 0.60,
        "validation_negative_reject_rate_min": 0.80,
        "local_test_negative_reject_rate_min": 0.70,
        "candidate_reward_change_min": 1e-6,
    }
    validation = summaries["validation"]
    local_test = summaries["local_test"]
    checks = {
        "contract_gate": contract.get("passed") is True,
        "training_gate": training.get("passed") is True,
        "validation_terminal_rgb_improves": validation["terminal_rgb"]["relative_change"] <= -0.05,
        "local_test_terminal_rgb_improves": local_test["terminal_rgb"]["relative_change"] <= -0.03,
        "validation_reward_fidelity_nonregression": validation["terminal_reward_fidelity"]["mae_ratio"] <= 1.0,
        "local_test_reward_fidelity_nonregression": local_test["terminal_reward_fidelity"]["mae_ratio"] <= 1.05,
        "validation_counterfactual_ranking": validation["counterfactual_ranking"]["pairwise_win_rate"] >= 0.75,
        "local_test_counterfactual_ranking": local_test["counterfactual_ranking"]["pairwise_win_rate"] >= 0.65,
        "validation_counterfactual_margin": validation["counterfactual_ranking"]["normalized_mean_margin"] >= 0.05,
        "validation_positive_gate": gate_summary["validation"]["positive_accept_rate"] >= 0.70,
        "local_test_positive_gate": gate_summary["local_test"]["positive_accept_rate"] >= 0.60,
        "validation_negative_gate": gate_summary["validation"]["negative_reject_rate"] >= 0.80,
        "local_test_negative_gate": gate_summary["local_test"]["negative_reject_rate"] >= 0.70,
        "intended_terminal_reward_effect_observed": validation["terminal_reward_fidelity"]["mean_absolute_candidate_change"] >= 1e-6,
        "left_non_regression": contract["checks"]["left_parent_bit_exact"] is True,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v312-causal-terminal-runtime-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v312-public-action-gated-full-terminal-mirror-v271",
        "thresholds": thresholds,
        "summaries": summaries,
        "action_gate_generalization": gate_summary,
        "checks": checks,
        "passed": passed,
        "authorization": {
            "intended_effect_observed": checks["intended_terminal_reward_effect_observed"],
            "right_terminal_ranking": checks["validation_counterfactual_ranking"]
            and checks["local_test_counterfactual_ranking"]
            and checks["validation_counterfactual_margin"],
            "left_non_regression": checks["left_non_regression"],
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "action_gate": sha256(args.action_gate),
            "contract_report": sha256(args.contract_report),
            "training_report": sha256(args.training_report),
            "split_manifest": sha256(args.split_manifest),
            "instruction_map": sha256(args.instruction_map),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "participant_component": "world-model RGB predictor only",
            "policy_modified": False,
            "official_reward_modified": False,
            "official_rl_algorithm_or_budget_modified": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "checks": checks, "summaries": summaries, "action_gate_generalization": gate_summary}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
