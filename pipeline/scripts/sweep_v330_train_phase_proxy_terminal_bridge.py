#!/usr/bin/env python3
"""Training-only proxy sweep for a temporally bridged v326 terminal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import score_terminal, sha256
from sweep_v329_train_only_terminal_bridge import (
    LAMBDAS,
    blend,
    rgb_mae,
    seed,
    temporal_delta_error,
)
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)
from wam_pipeline.v290_right_closed_mirror_runtime import mirror_actions, mirror_prompt
from wam_pipeline.v326_blended_phase_terminal_runtime import REPAIR_ALPHA
from wam_pipeline.v328_coherent_phase_trajectory_runtime import (
    Track2V328CoherentPhaseTrajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "windows",
        "instruction-map", "reward-checkpoint", "t5-model", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v330-train-phase-proxy-terminal-bridge-preregistration-v1":
        raise RuntimeError("wrong v330 preregistration")
    if tuple(prereg["fixed_sweep"]["lambdas"]) != LAMBDAS:
        raise RuntimeError("lambda sweep does not match preregistration")
    mapping = json.loads(args.instruction_map.read_text())
    train_episodes = [
        int(episode) for episode in mapping["train_episodes"]
        if mapping["arm_by_episode"][str(episode)] == "right"
    ]
    if len(train_episodes) != 15 or mapping.get("hidden_or_final_evaluation_data") is not False:
        raise RuntimeError("public training boundary is invalid")
    instructions = mapping["episode_to_instruction"]
    runtime = Track2V328CoherentPhaseTrajectory(
        args.checkpoint_dir,
        args.library_index,
        args.device,
        args.action_gate,
        args.phase_gate,
    )

    requests = []
    scanned = 0
    for episode in train_episodes:
        paths = sorted(
            args.windows.glob(f"episode{episode}_*.npz"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        for path in paths:
            start = int(path.stem.split("_")[1])
            if start % 8:
                continue
            scanned += 1
            with np.load(path, allow_pickle=False) as values:
                context = values["context_frames"].astype(np.uint8)
                history = values["history_actions"].astype(np.float32)
                future = values["future_actions"].astype(np.float32)
                target = values["target_frames"].astype(np.uint8)
            if not runtime._post_grasp(history, future):
                continue
            probability = runtime._probability(history, future)
            if probability < 0.99 or runtime._signature(history, future, probability) is not None:
                continue
            base, _ = runtime._nearest_clean(context, history, future)
            phase_ready, phase_episode, phase_start, phase_onset = runtime._phase(base)
            if not phase_ready:
                continue
            repair_row, repair_episode = runtime._success_terminal_for_action_context(
                context, future
            )
            requests.append({
                "path": path,
                "episode": episode,
                "start": start,
                "context": context,
                "history": history,
                "future": future,
                "target": target,
                "instruction": str(instructions[str(episode)]),
                "seed": seed(path),
                "probability": probability,
                "phase_episode": phase_episode,
                "phase_start": phase_start,
                "phase_onset": phase_onset,
                "repair_row": repair_row,
                "repair_episode": repair_episode,
            })
    if len(requests) != 46:
        raise RuntimeError(f"frozen proxy expected 46 requests, found {len(requests)}")

    rows = []
    reward_frames = []
    reward_prompts = []
    for begin in range(0, len(requests), args.inference_batch_size):
        batch = requests[begin : begin + args.inference_batch_size]
        contexts = np.stack([item["context"] for item in batch])
        histories = np.stack([item["history"] for item in batch])
        futures = np.stack([item["future"] for item in batch])
        seeds = np.asarray([item["seed"] for item in batch], dtype=np.int64)
        prompts = [item["instruction"] for item in batch]
        parent = runtime.parent.predict_batch(contexts, histories, futures, seeds, prompts)
        repair_targets = np.stack([runtime._target(item["repair_row"]) for item in batch])
        repaired = np.stack([
            runtime._blend(parent_prediction, target_prediction, future, REPAIR_ALPHA)
            for parent_prediction, target_prediction, future in zip(
                parent, repair_targets, futures, strict=True
            )
        ])
        mirrored = Track2V271EndpointCalibratedTerminal.predict_batch(
            runtime,
            np.ascontiguousarray(contexts[:, :, :, ::-1, :]),
            mirror_actions(histories),
            mirror_actions(futures),
            seeds,
            [mirror_prompt(prompt) for prompt in prompts],
        )
        mirrored = np.ascontiguousarray(mirrored[:, :, :, ::-1, :])
        for item, repaired_prediction, mirrored_prediction in zip(
            batch, repaired, mirrored, strict=True
        ):
            target_context = item["target"][-5:]
            variants = {}
            for weight in LAMBDAS:
                terminal = blend(mirrored_prediction[-1], repaired_prediction[-1], weight)
                next_context = np.concatenate((mirrored_prediction[3:7], terminal[None]), axis=0)
                variants[str(weight)] = {
                    "terminal": terminal,
                    "next_context_rgb_mae": rgb_mae(next_context, target_context),
                    "next_context_temporal_delta_error": temporal_delta_error(next_context, target_context),
                    "terminal_rgb_mae": rgb_mae(terminal, item["target"][-1]),
                }
            offset = len(reward_frames)
            reward_frames.extend((item["context"][-1], item["target"][-1]))
            reward_prompts.extend((item["instruction"], item["instruction"]))
            for weight in LAMBDAS:
                reward_frames.append(variants[str(weight)].pop("terminal"))
                reward_prompts.append(item["instruction"])
            rows.append({
                "window": item["path"].name,
                "episode": item["episode"],
                "start": item["start"],
                "gate_probability": item["probability"],
                "phase_episode": item["phase_episode"],
                "phase_start": item["phase_start"],
                "phase_onset": item["phase_onset"],
                "repair_episode": item["repair_episode"],
                "reward_frame_offset": offset,
                "variants": variants,
            })

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    scores = score_terminal(
        reward,
        np.stack(reward_frames),
        reward_prompts,
        torch.device(args.device),
        args.reward_batch_size,
    )
    for row in rows:
        offset = row.pop("reward_frame_offset")
        row["context_terminal_reward"] = float(scores[offset])
        row["gt_terminal_reward"] = float(scores[offset + 1])
        for index, weight in enumerate(LAMBDAS):
            value = float(scores[offset + 2 + index])
            variant = row["variants"][str(weight)]
            variant["terminal_reward"] = value
            variant["reward_absolute_error"] = abs(value - row["gt_terminal_reward"])

    summaries = {}
    for weight in LAMBDAS:
        values = [row["variants"][str(weight)] for row in rows]
        summaries[str(weight)] = {
            key: float(np.mean([value[key] for value in values]))
            for key in (
                "next_context_rgb_mae", "next_context_temporal_delta_error",
                "terminal_rgb_mae", "terminal_reward", "reward_absolute_error",
            )
        }
        summaries[str(weight)]["terminal_reward_hit_rate_at_0p9"] = float(
            np.mean([value["terminal_reward"] >= 0.90 for value in values])
        )
    baseline = summaries["1.0"]
    eligible = []
    for weight in LAMBDAS:
        value = summaries[str(weight)]
        value["next_context_rgb_ratio_vs_v326"] = value["next_context_rgb_mae"] / max(
            baseline["next_context_rgb_mae"], 1e-9
        )
        value["next_context_temporal_ratio_vs_v326"] = value[
            "next_context_temporal_delta_error"
        ] / max(baseline["next_context_temporal_delta_error"], 1e-9)
        value["reward_error_ratio_vs_v326"] = value["reward_absolute_error"] / max(
            baseline["reward_absolute_error"], 1e-9
        )
        value["eligible"] = bool(
            value["terminal_reward"] >= 0.90
            and value["terminal_reward_hit_rate_at_0p9"] >= 0.85
            and value["next_context_rgb_ratio_vs_v326"] <= 1.02
            and value["next_context_temporal_ratio_vs_v326"] <= 0.95
            and value["reward_error_ratio_vs_v326"] <= 1.10
        )
        if value["eligible"]:
            eligible.append(weight)
    selected = min(
        eligible,
        key=lambda weight: (
            summaries[str(weight)]["next_context_temporal_ratio_vs_v326"],
            summaries[str(weight)]["next_context_rgb_ratio_vs_v326"],
            -weight,
        ),
    ) if eligible else None
    report = {
        "format": "strict-track2-v330-train-phase-proxy-terminal-bridge-sweep-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "declared public-training right-arm phase proxy only",
        "authorizing": False,
        "train_episodes": train_episodes,
        "stride": 8,
        "scanned_windows": scanned,
        "phase_proxy_windows": len(rows),
        "proxy_difference_from_runtime_gate": (
            "baseline_alpha==0 is deliberately omitted only for train tuning; all action, "
            "failure, post-grasp, and public phase gates remain frozen"
        ),
        "lambdas": list(LAMBDAS),
        "summaries": summaries,
        "selection_rule": prereg["selection_rule"],
        "selected_lambda": selected,
        "selection_succeeded": selected is not None,
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "action_gate": sha256(args.action_gate),
            "phase_gate": sha256(args.phase_gate),
            "instruction_map": sha256(args.instruction_map),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_training_episodes_only": True,
            "public_holdout_or_evaluation_outcomes": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "phase_proxy_windows": len(rows),
        "summaries": summaries,
        "selected_lambda": selected,
        "selection_succeeded": selected is not None,
    }, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
