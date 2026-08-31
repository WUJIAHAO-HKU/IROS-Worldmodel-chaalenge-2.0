#!/usr/bin/env python3
"""Build a public-train-only terminal phase/quality table for right demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import score_terminal
from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal,
)


SUCCESS_THRESHOLD = 0.90
SUSTAINED_ROWS = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir",
        "library-index",
        "instruction-map",
        "reward-checkpoint",
        "t5-model",
        "preregistration",
        "output",
        "report",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing overwrite")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v323-public-terminal-phase-gate-preregistration-v1":
        raise RuntimeError("wrong preregistration")
    instruction_map = json.loads(args.instruction_map.read_text())["episode_to_instruction"]

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    runtime = Track2V271EndpointCalibratedTerminal(
        args.checkpoint_dir, args.library_index, args.device
    )
    rows = runtime.clean_rows.tolist()
    frames = [runtime._target(int(row))[-1] for row in rows]
    prompts = [
        str(instruction_map[str(int(runtime.row_episode[row]) - 20000)])
        for row in rows
    ]
    scores = score_terminal(
        reward, np.stack(frames), prompts, device, args.reward_batch_size
    )

    episode_values = []
    onset_values = []
    terminal_reward_values = []
    eligible_values = []
    episode_reports = []
    for episode in sorted(runtime.terminal_row):
        positions = [i for i, row in enumerate(rows) if int(runtime.row_episode[row]) == episode]
        starts = np.asarray([int(runtime.row_start[rows[i]]) for i in positions], dtype=np.int64)
        values = np.asarray([float(scores[i]) for i in positions], dtype=np.float64)
        order = np.argsort(starts)
        starts, values = starts[order], values[order]
        onset = None
        for index in range(0, len(values) - SUSTAINED_ROWS + 1):
            if np.all(values[index : index + SUSTAINED_ROWS] >= SUCCESS_THRESHOLD):
                onset = int(starts[index])
                break
        terminal_row = int(runtime.terminal_row[episode])
        terminal_position = rows.index(terminal_row)
        terminal_reward = float(scores[terminal_position])
        eligible = onset is not None and terminal_reward >= SUCCESS_THRESHOLD
        episode_values.append(int(episode))
        onset_values.append(int(onset if onset is not None else np.iinfo(np.int32).max))
        terminal_reward_values.append(terminal_reward)
        eligible_values.append(eligible)
        episode_reports.append(
            {
                "episode": int(episode),
                "rows": len(values),
                "onset_start": onset,
                "terminal_reward": terminal_reward,
                "eligible": eligible,
                "success_rows": int((values >= SUCCESS_THRESHOLD).sum()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode=np.asarray(episode_values, dtype=np.int64),
        onset_start=np.asarray(onset_values, dtype=np.int32),
        terminal_reward=np.asarray(terminal_reward_values, dtype=np.float32),
        eligible=np.asarray(eligible_values, dtype=np.bool_),
        success_threshold=np.asarray(SUCCESS_THRESHOLD, dtype=np.float32),
        sustained_rows=np.asarray(SUSTAINED_ROWS, dtype=np.int32),
        format=np.asarray("strict-track2-v323-public-terminal-phase-gate-v1"),
    )
    artifact_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checks = {
        "clean_public_rows_1926": len(rows) == 1926,
        "right_train_episodes_15": len(episode_values) == 15,
        "eligible_episodes_ge_12": int(sum(eligible_values)) >= 12,
        "all_eligible_onsets_finite": all(
            onset_values[i] < np.iinfo(np.int32).max
            for i, value in enumerate(eligible_values)
            if value
        ),
        "all_eligible_terminal_rewards_ge_0p9": all(
            terminal_reward_values[i] >= SUCCESS_THRESHOLD
            for i, value in enumerate(eligible_values)
            if value
        ),
    }
    report = {
        "format": "strict-track2-v323-public-terminal-phase-gate-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.output),
        "artifact_sha256": artifact_hash,
        "episodes": episode_reports,
        "checks": checks,
        "passed": all(checks.values()),
        "guards": {
            "public_train40_right_demonstrations_only": True,
            "runtime_reward_access": False,
            "policy_or_simulator_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
