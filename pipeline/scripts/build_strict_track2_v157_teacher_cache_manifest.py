#!/usr/bin/env python3
"""Freeze an indexed, arm-audited manifest over local V15.7 rollout caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_arms(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    actions = np.concatenate((history, future), axis=1)
    delta = np.abs(np.diff(actions, axis=1))
    left = delta[:, :, :7].mean(axis=(1, 2))
    right = delta[:, :, 7:].mean(axis=(1, 2))
    return right > left


def task_level_arms(paths: list[Path]) -> tuple[np.ndarray, dict]:
    """Infer one stable task arm per environment over the complete cached rollout."""
    left_energy = None
    right_energy = None
    prompt_vote = None
    prompt_conflicts = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as values:
            history = values["history_actions"]
            future = values["future_actions"]
            instructions = json.loads(str(values["instructions_json"]))
        actions = np.concatenate((history, future), axis=1)
        delta = np.abs(np.diff(actions, axis=1))
        batch_left = delta[:, :, :7].sum(axis=(1, 2))
        batch_right = delta[:, :, 7:].sum(axis=(1, 2))
        if left_energy is None:
            left_energy = np.zeros_like(batch_left, dtype=np.float64)
            right_energy = np.zeros_like(batch_right, dtype=np.float64)
            prompt_vote = np.full(len(instructions), -1, dtype=np.int8)
        if len(batch_left) != len(left_energy):
            raise SystemExit("teacher rollout batch size changed between chunks")
        left_energy += batch_left
        right_energy += batch_right
        for index, instruction in enumerate(instructions):
            lower = instruction.lower()
            vote = 1 if "right arm" in lower else 0 if "left arm" in lower else -1
            if vote >= 0:
                if prompt_vote[index] >= 0 and prompt_vote[index] != vote:
                    prompt_conflicts += 1
                prompt_vote[index] = vote
    assert left_energy is not None and right_energy is not None and prompt_vote is not None
    motion_right = right_energy > left_energy
    task_right = np.where(prompt_vote >= 0, prompt_vote.astype(bool), motion_right)
    return task_right, {
        "protocol": "one stable arm per environment across every cached chunk; explicit left/right-arm prompt wins, otherwise aggregate 14-D action motion over the complete cache",
        "prompt_labeled": int((prompt_vote >= 0).sum()),
        "motion_labeled": int((prompt_vote < 0).sum()),
        "prompt_conflicts": int(prompt_conflicts),
        "task_left": int((~task_right).sum()),
        "task_right": int(task_right.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--composed-config", type=Path, required=True)
    parser.add_argument("--run-preregistration", type=Path, required=True)
    parser.add_argument("--world-model-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-files", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    paths = sorted(args.audit_dir.glob("rollout_*.npz"))
    if args.maximum_files is not None:
        if args.maximum_files < 1:
            raise SystemExit("--maximum-files must be positive")
        paths = paths[: args.maximum_files]
    if not paths:
        raise SystemExit("teacher audit directory has no rollout caches")
    stable_right, arm_label_audit = task_level_arms(paths)
    indices = []
    records = []
    totals = {"samples": 0, "left": 0, "right": 0, "bytes": 0}
    for path in paths:
        match = re.fullmatch(r"rollout_(\d{6})\.npz", path.name)
        if match is None:
            raise SystemExit(f"unexpected rollout cache name: {path.name}")
        index = int(match.group(1))
        indices.append(index)
        with np.load(path, allow_pickle=False) as values:
            context = values["context_frames"]
            history = values["history_actions"]
            future = values["future_actions"]
            predicted = values["predicted_frames"]
            seeds = values["seeds"]
            instructions = json.loads(str(values["instructions_json"]))
            batch = int(context.shape[0])
            if context.shape != (batch, 5, 256, 256, 3) or context.dtype != np.uint8:
                raise SystemExit(f"invalid context tensor in {path}")
            if history.shape != (batch, 4, 14) or future.shape != (batch, 8, 14):
                raise SystemExit(f"invalid actions in {path}")
            if predicted.shape != (batch, 8, 256, 256, 3) or predicted.dtype != np.uint8:
                raise SystemExit(f"invalid V15.7 target tensor in {path}")
            if seeds.shape != (batch,) or len(instructions) != batch:
                raise SystemExit(f"invalid request metadata in {path}")
            if batch != len(stable_right):
                raise SystemExit("teacher rollout batch differs from task-level arm labels")
            right = stable_right
        size = path.stat().st_size
        record = {
            "index": index,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": size,
            "samples": batch,
            "left_samples": int((~right).sum()),
            "right_samples": int(right.sum()),
            "sample_indices_by_arm": {
                "left": np.flatnonzero(~right).tolist(),
                "right": np.flatnonzero(right).tolist(),
            },
        }
        records.append(record)
        totals["samples"] += batch
        totals["left"] += record["left_samples"]
        totals["right"] += record["right_samples"]
        totals["bytes"] += size
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise SystemExit("teacher rollout indices are not contiguous")
    screen = json.loads(args.world_model_screen.read_text())
    if screen.get("passed") is not True:
        raise SystemExit("V15.7 teacher screen is not passing")
    manifest = {
        "format": "strict-track2-v157-local-policy-teacher-cache-v1",
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256(Path(__file__).resolve()),
        "purpose": "single-pass world-model student training only",
        "source": "local fixed-Pi0.5 strict Track2 training rollout, never hidden evaluation traffic",
        "teacher_model_version": "track2-v15.7-hybrid-action-gated-reward-safe-blend12",
        "teacher_screen": str(args.world_model_screen.resolve()),
        "teacher_screen_sha256": sha256(args.world_model_screen),
        "composed_official_config": str(args.composed_config.resolve()),
        "composed_official_config_sha256": sha256(args.composed_config),
        "run_preregistration": str(args.run_preregistration.resolve()),
        "run_preregistration_sha256": sha256(args.run_preregistration),
        "policy_or_reward_training_target": False,
        "participant_action_selection": False,
        "development22_or_final128_used": False,
        "rollout_index_range": [indices[0], indices[-1]],
        "arm_label_audit": arm_label_audit,
        "totals": totals,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), **totals}, indent=2))


if __name__ == "__main__":
    main()
