#!/usr/bin/env python3
"""Public-only causal, phase, and non-regression audit for v324."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import COUNTERFACTUALS, RIGHT_EPISODES, episode_number, make_counterfactual, score_terminal, sha256
from audit_v314_transition_causal_gate import all_window_summary, transition_summary
from wam_pipeline.v290_right_closed_mirror_runtime import route_right
from wam_pipeline.v317_batched_sparse_failure_terminal_runtime import Track2V317BatchedSparseFailureTerminal
from wam_pipeline.v324_phase_guarded_terminal_runtime import Track2V324PhaseGuardedTerminal


def predict_requests(runtime, requests: list[dict], batch_size: int) -> list[np.ndarray]:
    output = []
    for begin in range(0, len(requests), batch_size):
        batch = requests[begin : begin + batch_size]
        predicted = runtime.predict_batch(
            np.stack([item["context"] for item in batch]),
            np.stack([item["history"] for item in batch]),
            np.stack([item["future"] for item in batch]),
            np.asarray([item["seed"] for item in batch], dtype=np.int64),
            [item["instruction"] for item in batch],
        )
        output.extend(predicted)
    return output


def route(runtime, history, future) -> dict:
    right = route_right(history, future)
    if not right:
        return {"probability": None, "accepted": True, "failure_signature": None}
    probability = runtime._probability(history, future)
    signature = runtime._signature(history, future, probability)
    return {
        "probability": probability,
        "accepted": signature is None,
        "failure_signature": signature,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint-dir", "library-index", "action-gate", "windows", "split-manifest", "instruction-map", "reward-checkpoint", "t5-model", "contract-report", "training-report", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--phase-gate", required=True, type=Path)
    parser.add_argument("--phase-training-report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract_report.read_text())
    training = json.loads(args.training_report.read_text())
    phase_training = json.loads(args.phase_training_report.read_text())
    if (
        contract.get("passed") is not True
        or training.get("passed") is not True
        or phase_training.get("passed") is not True
    ):
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
    baseline = Track2V317BatchedSparseFailureTerminal(
        args.checkpoint_dir, args.library_index, args.device, args.action_gate
    )
    candidate = Track2V324PhaseGuardedTerminal(
        args.checkpoint_dir,
        args.library_index,
        args.device,
        args.action_gate,
        args.phase_gate,
    )

    all_rows = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        records = []
        for episode in episodes:
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz")):
                with np.load(path, allow_pickle=False) as data:
                    context = np.asarray(data["context_frames"], dtype=np.uint8)
                    history = np.asarray(data["history_actions"], dtype=np.float32)
                    future = np.asarray(data["future_actions"], dtype=np.float32)
                    target = np.asarray(data["target_frames"], dtype=np.uint8)
                instruction = str(instructions[str(episode)])
                seed = int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)
                records.append({"path": path, "episode": episode, "context": context, "history": history, "future": future, "target": target, "instruction": instruction, "seed": seed})

        positive_requests = [dict(item) for item in records]
        direct_predictions = predict_requests(baseline, positive_requests, args.inference_batch_size)
        requests = []
        for item in records:
            requests.append({**item, "kind": "positive"})
            for name in COUNTERFACTUALS:
                requests.append({**item, "future": make_counterfactual(item["future"], item["history"], name), "kind": name})
        predictions = predict_requests(candidate, requests, args.inference_batch_size)

        rows = []
        frames = []
        prompts = []
        for index, (item, direct) in enumerate(zip(records, direct_predictions, strict=True)):
            group_requests = requests[index * 4 : index * 4 + 4]
            group_predictions = predictions[index * 4 : index * 4 + 4]
            positive = group_predictions[0]
            positive_route = route(candidate, item["history"], item["future"])
            counter_gate = {
                request["kind"]: route(candidate, request["history"], request["future"])
                for request in group_requests[1:]
            }
            row = {
                "window": item["path"].name,
                "episode": item["episode"],
                "direct_terminal_rgb_mae": float(np.abs(direct[-1].astype(np.int16) - item["target"][-1].astype(np.int16)).mean()),
                "candidate_terminal_rgb_mae": float(np.abs(positive[-1].astype(np.int16) - item["target"][-1].astype(np.int16)).mean()),
                "positive_terminal_parent_max_difference": int(np.abs(positive[-1].astype(np.int16) - direct[-1].astype(np.int16)).max()),
                "positive_gate_probability": positive_route["probability"],
                "positive_gate_accepted": positive_route["accepted"],
                "positive_failure_signature": positive_route["failure_signature"],
                "counterfactual_gate": counter_gate,
            }
            rows.append(row)
            for value in (item["context"][-1:], item["target"], direct, positive):
                frames.append(value[-1])
                prompts.append(item["instruction"])
            for prediction in group_predictions[1:]:
                frames.append(prediction[-1])
                prompts.append(item["instruction"])
        scores = score_terminal(reward, np.stack(frames), prompts, device, args.reward_batch_size).reshape(len(rows), 4 + len(COUNTERFACTUALS))
        for row, values in zip(rows, scores, strict=True):
            row["context_terminal_reward"] = float(values[0])
            row["gt_terminal_reward"] = float(values[1])
            row["direct_terminal_reward"] = float(values[2])
            row["candidate_terminal_reward"] = float(values[3])
            row["counterfactual_rewards"] = {name: float(values[4 + i]) for i, name in enumerate(COUNTERFACTUALS)}
        all_rows[split_name] = rows
        print(f"V317_SCORED {split_name} {len(rows)}", flush=True)

    all_summary = {name: all_window_summary(rows) for name, rows in all_rows.items()}
    transitions = {name: transition_summary(rows) for name, rows in all_rows.items()}
    parent_max = {}
    transition_gain = {}
    for split_name, rows in all_rows.items():
        selected = [row for row in rows if row["context_terminal_reward"] <= 0.10 and row["gt_terminal_reward"] >= 0.90]
        parent_max[split_name] = max(row["positive_terminal_parent_max_difference"] for row in selected) if selected else None
        transition_gain[split_name] = (
            float(np.mean([row["candidate_terminal_reward"] for row in selected]))
            - float(np.mean([row["direct_terminal_reward"] for row in selected]))
            if selected
            else None
        )
    validation, local_test = transitions["validation"], transitions["local_test"]
    checks = {
        "contract_gate": contract.get("passed") is True,
        "training_gate": training.get("passed") is True,
        "phase_training_gate": phase_training.get("passed") is True,
        "validation_transition_count": validation["windows"] >= 12,
        "local_test_transition_count": local_test["windows"] >= 12,
        "validation_positive_reward_mean_ge_0p98": validation["positive_reward_mean"] >= 0.98,
        "local_test_positive_reward_mean_ge_0p90": local_test["positive_reward_mean"] >= 0.90,
        "validation_positive_hit_rate_ge_0p90": validation["positive_hit_rate_at_0p9"] >= 0.90,
        "local_test_positive_hit_rate_ge_0p90": local_test["positive_hit_rate_at_0p9"] >= 0.90,
        "validation_transition_gain_nonregression": transition_gain["validation"] is not None and transition_gain["validation"] >= -0.02,
        "local_test_transition_gain_ge_0p20": transition_gain["local_test"] is not None and transition_gain["local_test"] >= 0.20,
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
        "validation_all_window_rgb_ratio_le_1p01": all_summary["validation"]["terminal_rgb"]["mae_ratio"] <= 1.01,
        "local_test_all_window_rgb_ratio_le_1p01": all_summary["local_test"]["terminal_rgb"]["mae_ratio"] <= 1.01,
        "validation_all_window_reward_nonregression": all_summary["validation"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.0,
        "local_test_all_window_reward_nonregression": all_summary["local_test"]["terminal_reward_fidelity"]["mae_ratio"] <= 1.0,
        "left_non_regression": contract["checks"]["left_single_parent_bit_exact"] is True,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v324-phase-guarded-terminal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v324-phase-guarded-terminal-v317",
        "deployment_inference_batch_size": args.inference_batch_size,
        "transition_definition": "context official reward <=0.10 and public GT terminal reward >=0.90",
        "all_window_summaries": all_summary,
        "transition_summaries": transitions,
        "transition_positive_parent_terminal_max_difference": parent_max,
        "transition_reward_gain_vs_v317": transition_gain,
        "checks": checks,
        "passed": passed,
        "authorization": {
            "intended_effect_observed": checks["local_test_transition_gain_ge_0p20"],
            "right_terminal_ranking": all(
                checks[name]
                for name in (
                    "validation_positive_reward_mean_ge_0p98",
                    "local_test_positive_reward_mean_ge_0p90",
                    "validation_positive_hit_rate_ge_0p90",
                    "local_test_positive_hit_rate_ge_0p90",
                    "validation_pairwise_ranking",
                    "local_test_pairwise_ranking",
                    "validation_margin",
                    "local_test_margin",
                )
            ),
            "left_non_regression": checks["left_non_regression"],
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "action_gate": sha256(args.action_gate),
            "phase_gate": sha256(args.phase_gate),
            "contract_report": sha256(args.contract_report),
            "training_report": sha256(args.training_report),
            "phase_training_report": sha256(args.phase_training_report),
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
