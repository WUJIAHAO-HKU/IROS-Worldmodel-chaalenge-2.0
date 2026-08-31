#!/usr/bin/env python3
"""Compare visual-only and action-aware public terminal episode lookup.

This is an offline world-model diagnostic.  It reads only declared public
demonstration windows and the frozen official reward model; it never reads a
policy checkpoint, simulator outcome, hidden/final data, or submission state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, score_terminal
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)


def summarize(rows: list[dict]) -> dict:
    gt = np.asarray([row["gt_reward"] for row in rows])
    current = np.asarray([row["visual_terminal_reward"] for row in rows])
    action = np.asarray([row["action_episode_terminal_reward"] for row in rows])
    transition = np.asarray(
        [row["context_reward"] <= 0.10 and row["gt_reward"] >= 0.90 for row in rows]
    )
    selected = lambda values: values[transition]
    return {
        "windows": len(rows),
        "episode_mapping_changed": int(
            sum(row["visual_episode"] != row["action_episode"] for row in rows)
        ),
        "all_window": {
            "visual_reward_mae": float(np.abs(current - gt).mean()),
            "action_episode_reward_mae": float(np.abs(action - gt).mean()),
            "reward_mae_ratio": float(
                np.abs(action - gt).mean() / max(float(np.abs(current - gt).mean()), 1e-12)
            ),
            "visual_rgb_mae": float(np.mean([row["visual_rgb_mae"] for row in rows])),
            "action_episode_rgb_mae": float(
                np.mean([row["action_episode_rgb_mae"] for row in rows])
            ),
        },
        "transition": {
            "windows": int(transition.sum()),
            "visual_reward_mean": float(selected(current).mean()),
            "action_episode_reward_mean": float(selected(action).mean()),
            "mean_improvement": float(selected(action).mean() - selected(current).mean()),
            "visual_hit_rate_at_0p9": float((selected(current) >= 0.90).mean()),
            "action_episode_hit_rate_at_0p9": float((selected(action) >= 0.90).mean()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir",
        "library-index",
        "windows",
        "split-manifest",
        "instruction-map",
        "reward-checkpoint",
        "t5-model",
        "preregistration",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v321-action-episode-terminal-diagnostic-preregistration-v1":
        raise RuntimeError("wrong preregistration")
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
    runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    all_rows: dict[str, list[dict]] = {}
    for split_name, episodes in RIGHT_EPISODES.items():
        rows = []
        frames = []
        prompts = []
        for episode in episodes:
            for path in sorted(args.windows.glob(f"episode{episode}_*.npz")):
                with np.load(path, allow_pickle=False) as values:
                    context = np.asarray(values["context_frames"], dtype=np.uint8)
                    history = np.asarray(values["history_actions"], dtype=np.float32)
                    future = np.asarray(values["future_actions"], dtype=np.float32)
                    target = np.asarray(values["target_frames"], dtype=np.uint8)
                base, _ = runtime._nearest_clean(context, history, future)
                visual_row = runtime._terminal_for_context(context)
                action_episode = int(runtime.row_episode[base])
                action_row = int(runtime.terminal_row[action_episode])
                visual = runtime._target(visual_row)[-1]
                action = runtime._target(action_row)[-1]
                prompt = str(instructions[str(episode)])
                row = {
                    "window": path.name,
                    "episode": episode,
                    "visual_episode": int(runtime.row_episode[visual_row]),
                    "action_episode": action_episode,
                    "base_start": int(runtime.row_start[base]),
                    "visual_rgb_mae": float(
                        np.abs(visual.astype(np.int16) - target[-1].astype(np.int16)).mean()
                    ),
                    "action_episode_rgb_mae": float(
                        np.abs(action.astype(np.int16) - target[-1].astype(np.int16)).mean()
                    ),
                }
                rows.append(row)
                frames.extend((context[-1], target[-1], visual, action))
                prompts.extend((prompt, prompt, prompt, prompt))
        scores = score_terminal(
            reward, np.stack(frames), prompts, device, args.reward_batch_size
        ).reshape(len(rows), 4)
        for row, values in zip(rows, scores, strict=True):
            row["context_reward"] = float(values[0])
            row["gt_reward"] = float(values[1])
            row["visual_terminal_reward"] = float(values[2])
            row["action_episode_terminal_reward"] = float(values[3])
        all_rows[split_name] = rows
        print(f"V321_DIAGNOSTIC_SCORED {split_name} {len(rows)}", flush=True)

    summaries = {name: summarize(rows) for name, rows in all_rows.items()}
    validation = summaries["validation"]
    confirmation = summaries["local_test"]
    checks = {
        "validation_transition_count_ge_12": validation["transition"]["windows"] >= 12,
        "confirmation_transition_count_ge_12": confirmation["transition"]["windows"] >= 12,
        "validation_reward_mae_ratio_le_0p95": validation["all_window"]["reward_mae_ratio"] <= 0.95,
        "validation_rgb_mae_nonregression": validation["all_window"]["action_episode_rgb_mae"] <= 1.05 * validation["all_window"]["visual_rgb_mae"],
        "validation_transition_mean_nonregression": validation["transition"]["mean_improvement"] >= -0.02,
        "confirmation_reward_mae_nonregression": confirmation["all_window"]["action_episode_reward_mae"] <= confirmation["all_window"]["visual_reward_mae"],
        "confirmation_transition_mean_improvement_ge_0p20": confirmation["transition"]["mean_improvement"] >= 0.20,
        "confirmation_transition_hit_rate_ge_0p80": confirmation["transition"]["action_episode_hit_rate_at_0p9"] >= 0.80,
    }
    report = {
        "format": "strict-track2-v321-action-episode-terminal-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_mechanism": "select terminal episode from public action+visual nearest row instead of context-only nearest row",
        "summaries": summaries,
        "checks": checks,
        "passed": all(checks.values()),
        "selection_boundary": "thresholds fixed before scores; validation decides and local_test only confirms",
        "guards": {
            "participant_component": "world-model diagnostic only",
            "public_demonstration_windows_only": True,
            "policy_or_policy_checkpoint_read": False,
            "simulator_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": checks, "summaries": summaries}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
