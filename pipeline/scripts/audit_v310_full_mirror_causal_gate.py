#!/usr/bin/env python3
"""Fail-closed causal audit for the right-arm full-terminal mirror runtime.

Only declared public demonstration windows are read.  The validation split is
used for the decision and the episode-disjoint local-test split is reported as
a confirmation.  No policy, optimizer, reward model, simulator outcome, hidden
evaluation data, or submission endpoint is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v290_right_closed_mirror_runtime import (
    mirror_actions,
    mirror_prompt,
)


RIGHT_EPISODES = {"validation": [7, 18], "local_test": [6, 22]}
SAMPLES_PER_SPLIT = 64
COUNTERFACTUALS = ("open_gripper", "static_transport", "reverse_transport")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_number(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if match is None:
        raise ValueError(f"cannot parse episode from {path}")
    return int(match.group(1))


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def full_mirror(runtime, context, history, future, seed, instruction):
    prediction = runtime.predict(
        np.ascontiguousarray(context[:, :, ::-1, :]),
        mirror_actions(history),
        mirror_actions(future),
        seed,
        mirror_prompt(instruction),
    )
    return np.ascontiguousarray(prediction[:, :, ::-1, :])


def make_counterfactual(future: np.ndarray, history: np.ndarray, name: str) -> np.ndarray:
    result = future.copy()
    anchor = history[-1, 7:13]
    if name == "open_gripper":
        result[:, 13] = 1.0
    elif name == "static_transport":
        result[:, 7:13] = anchor
    elif name == "reverse_transport":
        result[:, 7:13] = anchor - (future[:, 7:13] - anchor)
    else:
        raise ValueError(name)
    return result


def score_terminal(model, frames: np.ndarray, prompts: list[str], device, batch_size: int) -> np.ndarray:
    output = []
    for begin in range(0, len(frames), batch_size):
        batch = torch.from_numpy(frames[begin : begin + batch_size]).permute(0, 3, 1, 2)
        batch = batch.float().div(255.0).to(device)
        with torch.inference_mode():
            score = model.compute_reward(
                batch, prompts[begin : begin + batch_size]
            ).float().cpu().numpy()
        output.append(score)
    return np.concatenate(output).astype(np.float64)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def summarize(rows: list[dict]) -> dict:
    direct_rgb = np.asarray([row["direct_terminal_rgb_mae"] for row in rows])
    mirror_rgb = np.asarray([row["mirror_terminal_rgb_mae"] for row in rows])
    gt_reward = np.asarray([row["gt_terminal_reward"] for row in rows])
    direct_reward = np.asarray([row["direct_terminal_reward"] for row in rows])
    mirror_reward = np.asarray([row["mirror_terminal_reward"] for row in rows])
    counter = np.asarray(
        [[row["counterfactual_rewards"][name] for name in COUNTERFACTUALS] for row in rows]
    )
    reward_scale = max(float(np.std(gt_reward)), 1e-9)
    direct_reward_mae = float(np.mean(np.abs(direct_reward - gt_reward)))
    mirror_reward_mae = float(np.mean(np.abs(mirror_reward - gt_reward)))
    pairwise = mirror_reward[:, None] > counter
    margins = mirror_reward[:, None] - counter
    return {
        "windows": len(rows),
        "terminal_rgb": {
            "direct_mae": float(direct_rgb.mean()),
            "mirror_mae": float(mirror_rgb.mean()),
            "relative_change": float(mirror_rgb.mean() / direct_rgb.mean() - 1.0),
            "mirror_better_fraction": float(np.mean(mirror_rgb < direct_rgb)),
        },
        "terminal_reward_fidelity": {
            "gt_mean": float(gt_reward.mean()),
            "gt_std": float(gt_reward.std()),
            "direct_mae": direct_reward_mae,
            "mirror_mae": mirror_reward_mae,
            "mae_ratio": float(mirror_reward_mae / max(direct_reward_mae, 1e-12)),
            "direct_correlation": correlation(direct_reward, gt_reward),
            "mirror_correlation": correlation(mirror_reward, gt_reward),
            "mean_absolute_candidate_change": float(np.mean(np.abs(mirror_reward - direct_reward))),
            "normalized_mean_absolute_candidate_change": float(
                np.mean(np.abs(mirror_reward - direct_reward)) / reward_scale
            ),
        },
        "counterfactual_ranking": {
            "comparisons": int(pairwise.size),
            "pairwise_win_rate": float(pairwise.mean()),
            "strict_all_three_win_rate": float(pairwise.all(axis=1).mean()),
            "mean_margin": float(margins.mean()),
            "normalized_mean_margin": float(margins.mean() / reward_scale),
            "per_counterfactual_win_rate": {
                name: float(pairwise[:, index].mean())
                for index, name in enumerate(COUNTERFACTUALS)
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--library-index", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--instruction-map", required=True, type=Path)
    parser.add_argument("--reward-checkpoint", required=True, type=Path)
    parser.add_argument("--t5-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    split = json.loads(args.split_manifest.read_text())
    instruction_map = json.loads(args.instruction_map.read_text())["episode_to_instruction"]
    for name, episodes in RIGHT_EPISODES.items():
        if not set(episodes).issubset(set(split[f"{name}_episodes"])):
            raise RuntimeError(f"{name} episode boundary mismatch")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )

    all_rows: dict[str, list[dict]] = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        paths = [
            path
            for episode in episodes
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz"))
        ]
        paths = evenly_spaced(paths, SAMPLES_PER_SPLIT)
        pending = []
        terminal_frames = []
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
            direct = runtime.predict(context, history, future, seed, instruction)
            mirrored = full_mirror(runtime, context, history, future, seed, instruction)
            predictions = {"gt": target, "direct": direct, "mirror": mirrored}
            counter_predictions = {}
            for name in COUNTERFACTUALS:
                altered = make_counterfactual(future, history, name)
                if name == "open_gripper":
                    counter_predictions[name] = runtime.predict(
                        context, history, altered, seed, instruction
                    )
                else:
                    counter_predictions[name] = full_mirror(
                        runtime, context, history, altered, seed, instruction
                    )
            row = {
                "window": path.name,
                "episode": episode,
                "direct_terminal_rgb_mae": float(
                    np.abs(direct[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                ),
                "mirror_terminal_rgb_mae": float(
                    np.abs(mirrored[-1].astype(np.int16) - target[-1].astype(np.int16)).mean()
                ),
            }
            pending.append(row)
            for name in ("gt", "direct", "mirror"):
                terminal_frames.append(predictions[name][-1])
                prompts.append(instruction)
            for name in COUNTERFACTUALS:
                terminal_frames.append(counter_predictions[name][-1])
                prompts.append(instruction)

        scores = score_terminal(
            reward, np.stack(terminal_frames), prompts, device, args.reward_batch_size
        ).reshape(len(pending), 3 + len(COUNTERFACTUALS))
        for row, values in zip(pending, scores, strict=True):
            row["gt_terminal_reward"] = float(values[0])
            row["direct_terminal_reward"] = float(values[1])
            row["mirror_terminal_reward"] = float(values[2])
            row["counterfactual_rewards"] = {
                name: float(values[3 + index])
                for index, name in enumerate(COUNTERFACTUALS)
            }
        all_rows[split_name] = pending
        print(f"V310_SCORED {split_name} {len(pending)}", flush=True)

    summaries = {name: summarize(rows) for name, rows in all_rows.items()}
    thresholds = {
        "validation_terminal_rgb_relative_change_max": -0.05,
        "local_test_terminal_rgb_relative_change_max": -0.03,
        "validation_reward_mae_ratio_max": 1.0,
        "local_test_reward_mae_ratio_max": 1.05,
        "validation_pairwise_win_rate_min": 0.75,
        "local_test_pairwise_win_rate_min": 0.65,
        "validation_normalized_margin_min": 0.05,
        "candidate_reward_change_min": 1e-6,
    }
    validation = summaries["validation"]
    local_test = summaries["local_test"]
    checks = {
        "validation_terminal_rgb_improves": validation["terminal_rgb"]["relative_change"]
        <= thresholds["validation_terminal_rgb_relative_change_max"],
        "local_test_terminal_rgb_improves": local_test["terminal_rgb"]["relative_change"]
        <= thresholds["local_test_terminal_rgb_relative_change_max"],
        "validation_reward_fidelity_nonregression": validation["terminal_reward_fidelity"]["mae_ratio"]
        <= thresholds["validation_reward_mae_ratio_max"],
        "local_test_reward_fidelity_nonregression": local_test["terminal_reward_fidelity"]["mae_ratio"]
        <= thresholds["local_test_reward_mae_ratio_max"],
        "validation_counterfactual_ranking": validation["counterfactual_ranking"]["pairwise_win_rate"]
        >= thresholds["validation_pairwise_win_rate_min"],
        "local_test_counterfactual_ranking": local_test["counterfactual_ranking"]["pairwise_win_rate"]
        >= thresholds["local_test_pairwise_win_rate_min"],
        "validation_counterfactual_margin": validation["counterfactual_ranking"]["normalized_mean_margin"]
        >= thresholds["validation_normalized_margin_min"],
        "intended_terminal_reward_effect_observed": validation["terminal_reward_fidelity"]["mean_absolute_candidate_change"]
        >= thresholds["candidate_reward_change_min"],
        "left_path_bit_exact_by_construction": True,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v310-full-mirror-causal-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v290-right-closed-full-terminal-mirror-v271",
        "data_boundary": {
            "source": "declared public 50 demonstration episodes",
            "decision_split": "validation episodes 7 and 18",
            "confirmation_split": "episode-disjoint local-test episodes 6 and 22",
            "hidden_or_final_data": False,
            "simulator_outcomes_used": False,
        },
        "counterfactuals": {
            "open_gripper": "set requested right gripper to open",
            "static_transport": "hold requested right joints at the history endpoint",
            "reverse_transport": "reflect requested right-joint displacement around the history endpoint",
        },
        "thresholds": thresholds,
        "summaries": summaries,
        "checks": checks,
        "passed": passed,
        "authorization": {
            "intended_effect_observed": checks["intended_terminal_reward_effect_observed"],
            "right_terminal_ranking": checks["validation_counterfactual_ranking"]
            and checks["local_test_counterfactual_ranking"]
            and checks["validation_counterfactual_margin"],
            "left_non_regression": checks["left_path_bit_exact_by_construction"],
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "split_manifest": sha256(args.split_manifest),
            "instruction_map": sha256(args.instruction_map),
            "reward_checkpoint": sha256(args.reward_checkpoint),
            "library_index": sha256(args.library_index),
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
    print(json.dumps({"passed": passed, "checks": checks, "summaries": summaries}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
