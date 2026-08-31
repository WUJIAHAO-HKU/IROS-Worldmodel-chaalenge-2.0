#!/usr/bin/env python3
"""Convert audited on-policy RoboTwin captures into Track 2 training windows.

The collection runtime records one real simulator successor observation after
every executed action.  This converter first reconstructs complete episodes and
only then creates API-aligned five-context/eight-future windows.  Splits are
episode-disjoint and stratified by active arm; no policy success metric is used
to choose the validation episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


CONTEXT_FRAMES = 5
FUTURE_FRAMES = 8
ACTION_DIM = 14


@dataclass(frozen=True)
class Episode:
    seed: int
    arm_right: bool
    success: bool
    frames: np.ndarray
    actions: np.ndarray
    raw_hashes: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_episode(frames: np.ndarray, size: int) -> np.ndarray:
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    return np.stack([
        np.asarray(
            Image.fromarray(frame, mode="RGB").resize((size, size), bilinear)
        ).copy()
        for frame in frames
    ])


def load_episode(directory: Path, expected_chunks: int) -> Episode:
    files = sorted(directory.glob("chunk_*.npz"))
    expected_names = [f"chunk_{index:03d}.npz" for index in range(expected_chunks)]
    if [path.name for path in files] != expected_names:
        raise ValueError(f"{directory}: incomplete/non-contiguous chunks")

    seed = int(directory.name.removeprefix("seed_"))
    initial_frames: list[np.ndarray] = []
    initial_states: list[np.ndarray] = []
    successor_frames: list[np.ndarray] = []
    successor_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    arm_values: list[bool] = []
    successes: list[bool] = []
    hashes: list[tuple[str, str]] = []
    terminal_seen = False
    for chunk_index, path in enumerate(files):
        hashes.append((str(path), sha256_file(path)))
        with np.load(path, allow_pickle=False) as data:
            required = {
                "format", "seed", "chunk_index", "arm_right", "initial_frame",
                "initial_state", "actions", "successor_frames", "successor_states",
                "executed_actions", "success",
            }
            if not required.issubset(data.files):
                raise ValueError(f"{path}: missing {sorted(required.difference(data.files))}")
            if int(data["seed"]) != seed or int(data["chunk_index"]) != chunk_index:
                raise ValueError(f"{path}: seed/chunk metadata mismatch")
            executed = int(data["executed_actions"])
            action = np.asarray(data["actions"], dtype=np.float32)
            successors = np.asarray(data["successor_frames"], dtype=np.uint8)
            states = np.asarray(data["successor_states"], dtype=np.float32)
            initial = np.asarray(data["initial_frame"], dtype=np.uint8)
            initial_state = np.asarray(data["initial_state"], dtype=np.float32)
            success = bool(data["success"])
            if not 1 <= executed <= FUTURE_FRAMES:
                raise ValueError(f"{path}: invalid executed_actions={executed}")
            if action.shape != (FUTURE_FRAMES, ACTION_DIM):
                raise ValueError(f"{path}: actions have shape {action.shape}")
            if successors.shape != (FUTURE_FRAMES, *initial.shape):
                raise ValueError(f"{path}: successor frame shape mismatch")
            if initial.shape[-1] != 3 or initial.dtype != np.uint8:
                raise ValueError(f"{path}: initial frame is not HWC uint8 RGB")
            if initial_state.shape != (ACTION_DIM,) or states.shape != (FUTURE_FRAMES, ACTION_DIM):
                raise ValueError(f"{path}: state shape mismatch")
            if not np.isfinite(action).all() or not np.isfinite(initial_state).all() or not np.isfinite(states).all():
                raise ValueError(f"{path}: non-finite action/state")
            arm_values.append(bool(data["arm_right"]))
            successes.append(success)
            # RLinf keeps calling the environment through the fixed 200-step
            # rollout after task success. RoboTwin then accepts one action and
            # immediately terminates every later chunk. Those post-terminal
            # transitions are not part of the episode and must not be trained.
            if terminal_seen:
                if not success:
                    raise ValueError(f"{path}: non-success chunk after terminal success")
                continue
            if initial_frames:
                if not np.array_equal(successor_frames[-1][-1], initial):
                    raise ValueError(f"{directory}: RGB discontinuity before chunk {chunk_index}")
                if not np.array_equal(successor_states[-1][-1], initial_state):
                    raise ValueError(f"{directory}: state discontinuity before chunk {chunk_index}")
            initial_frames.append(initial.copy())
            initial_states.append(initial_state.copy())
            successor_frames.append(successors[:executed].copy())
            successor_states.append(states[:executed].copy())
            actions.append(action[:executed].copy())
            terminal_seen = success or executed < FUTURE_FRAMES

    if len(set(arm_values)) != 1:
        raise ValueError(f"{directory}: active arm changes within an episode")
    frames = np.concatenate((initial_frames[0][None], np.concatenate(successor_frames)), axis=0)
    action_array = np.concatenate(actions)
    if len(frames) != len(action_array) + 1:
        raise AssertionError("an episode must have exactly one more frame than action")
    return Episode(
        seed=seed,
        arm_right=arm_values[0],
        success=any(successes),
        frames=frames,
        actions=action_array,
        raw_hashes=tuple(hashes),
    )


def stratified_split(episodes: list[Episode], validation_count: int, split_seed: int) -> tuple[list[int], list[int]]:
    if validation_count <= 0 or validation_count >= len(episodes):
        raise ValueError("validation-count must be between 1 and number of episodes - 1")
    by_arm = {False: [], True: []}
    for episode in episodes:
        by_arm[episode.arm_right].append(episode.seed)
    if not by_arm[False] or not by_arm[True]:
        raise ValueError("both left- and right-arm episodes are required")

    def rank(seed: int) -> bytes:
        return hashlib.sha256(f"{split_seed}:{seed}".encode()).digest()

    right_count = round(validation_count * len(by_arm[True]) / len(episodes))
    right_count = max(1, min(right_count, len(by_arm[True]) - 1))
    left_count = validation_count - right_count
    if left_count <= 0 or left_count >= len(by_arm[False]):
        raise ValueError("cannot create a non-empty arm-stratified validation split")
    validation = (
        sorted(by_arm[False], key=rank)[:left_count]
        + sorted(by_arm[True], key=rank)[:right_count]
    )
    validation_set = set(validation)
    train = sorted(episode.seed for episode in episodes if episode.seed not in validation_set)
    return train, sorted(validation)


def atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--expected-chunks", type=int, default=25)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.stride <= 0 or args.resize <= 0 or args.expected_chunks <= 0:
        raise ValueError("stride, resize, and expected-chunks must be positive")

    capture_root = Path(args.capture_root).resolve()
    output = Path(args.output).resolve()
    seed_manifest_path = Path(args.seed_manifest).resolve()
    seed_manifest = json.loads(seed_manifest_path.read_text())
    if "selected_seeds" in seed_manifest:
        seed_values = seed_manifest["selected_seeds"]
    else:
        seed_values = seed_manifest.get("adjust_bottle", {}).get("success_seeds")
    if not isinstance(seed_values, list):
        raise ValueError("seed manifest must contain selected_seeds or adjust_bottle.success_seeds")
    expected_seeds = sorted(int(value) for value in seed_values)
    directories = list(capture_root.glob("batch_*/chunks/seed_*"))
    if not directories and capture_root.name.startswith("batch_"):
        directories = list(capture_root.glob("chunks/seed_*"))
    directories = sorted(directories, key=lambda p: int(p.name[5:]))
    actual_seeds = [int(path.name[5:]) for path in directories]
    if actual_seeds != expected_seeds:
        missing = sorted(set(expected_seeds).difference(actual_seeds))
        extra = sorted(set(actual_seeds).difference(expected_seeds))
        raise ValueError(f"capture seeds do not match manifest; missing={missing}, extra={extra}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    episodes = [load_episode(path, args.expected_chunks) for path in directories]
    train_seeds, validation_seeds = stratified_split(episodes, args.validation_count, args.split_seed)
    split_by_seed = {seed: "train" for seed in train_seeds}
    split_by_seed.update({seed: "validation" for seed in validation_seeds})

    raw_hash_manifest: list[dict[str, object]] = []
    output_hash_manifest: list[dict[str, object]] = []
    episode_records: list[dict[str, object]] = []
    split_episode_ids = {"train_episodes": [], "validation_episodes": []}
    window_counts = Counter()
    arm_counts = Counter()
    success_counts = Counter()
    for episode in episodes:
        episode_id = 10000 + episode.seed
        split = split_by_seed[episode.seed]
        split_episode_ids[f"{split}_episodes"].append(episode_id)
        arm = "right" if episode.arm_right else "left"
        arm_counts[f"{split}_{arm}"] += 1
        success_counts[f"{split}_{'success' if episode.success else 'failure'}"] += 1
        raw_hash_manifest.extend(
            {"seed": episode.seed, "path": path, "sha256": digest}
            for path, digest in episode.raw_hashes
        )
        frames = resize_episode(episode.frames, args.resize)
        maximum_start = len(frames) - (CONTEXT_FRAMES + FUTURE_FRAMES)
        starts = list(range(0, maximum_start + 1, args.stride))
        for start in starts:
            path = output / f"episode{episode_id}_{start:05d}.npz"
            digest = atomic_npz(
                path,
                context_frames=frames[start : start + CONTEXT_FRAMES],
                history_actions=episode.actions[start : start + CONTEXT_FRAMES - 1],
                future_actions=episode.actions[
                    start + CONTEXT_FRAMES - 1 : start + CONTEXT_FRAMES - 1 + FUTURE_FRAMES
                ],
                target_frames=frames[start + CONTEXT_FRAMES : start + CONTEXT_FRAMES + FUTURE_FRAMES],
                source=np.asarray(f"onpolicy-robotwin-adjust_bottle-seed-{episode.seed}"),
                start=np.asarray(start, dtype=np.int64),
                synthetic_seed=np.asarray(episode.seed, dtype=np.int64),
                arm_right=np.asarray(episode.arm_right),
                capture_success=np.asarray(episode.success),
            )
            output_hash_manifest.append({"path": path.name, "sha256": digest})
        window_counts[split] += len(starts)
        episode_records.append({
            "episode_id": episode_id,
            "seed": episode.seed,
            "split": split,
            "arm": arm,
            "capture_success": episode.success,
            "frames": len(frames),
            "actions": len(episode.actions),
            "windows": len(starts),
        })

    raw_root = hashlib.sha256(
        "".join(record["sha256"] for record in raw_hash_manifest).encode()
    ).hexdigest()
    output_root = hashlib.sha256(
        "".join(record["sha256"] for record in output_hash_manifest).encode()
    ).hexdigest()
    split_manifest = {
        "format": "strict-track2-onpolicy-synthetic-window-split-v1",
        "data_use": "offline world-model training and validation only",
        "policy_source": "official unmodified Pi0.5 baseline",
        "environment": "public RoboTwin adjust_bottle simulator",
        "selection_uses_policy_success": False,
        "episode_id_offset": 10000,
        "stride": args.stride,
        "resize": args.resize,
        "train_episodes": sorted(split_episode_ids["train_episodes"]),
        "validation_episodes": sorted(split_episode_ids["validation_episodes"]),
        "train_windows": window_counts["train"],
        "validation_windows": window_counts["validation"],
        "arm_episode_counts": dict(sorted(arm_counts.items())),
        "capture_outcome_counts_for_audit_only": dict(sorted(success_counts.items())),
        "seed_manifest": str(seed_manifest_path),
        "seed_manifest_sha256": sha256_file(seed_manifest_path),
        "raw_capture_merkle_sha256": raw_root,
        "output_window_merkle_sha256": output_root,
        "episodes": episode_records,
    }
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (output / "raw_capture_hashes.json").write_text(json.dumps(raw_hash_manifest, indent=2) + "\n")
    (output / "window_hashes.json").write_text(json.dumps(output_hash_manifest, indent=2) + "\n")
    if list(output.glob("*.tmp")):
        raise AssertionError("temporary files remain after conversion")
    print(json.dumps(split_manifest, indent=2))


if __name__ == "__main__":
    main()
