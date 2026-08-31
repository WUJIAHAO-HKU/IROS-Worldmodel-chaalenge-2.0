#!/usr/bin/env python3
"""Export contiguous multi-chunk GT and parent rollouts for reward auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_multichunk_reward_aligned_autoregressive_unet import (
    CHUNK_FRAMES,
    MultiChunkWindows,
)
from wam_pipeline.autoregressive_unet_runtime import Track2AutoregressiveUNet


def predict_sequence(
    runtime: Track2AutoregressiveUNet,
    context: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    chunks: int,
) -> np.ndarray:
    predictions = []
    for chunk in range(chunks):
        actions = future[chunk * CHUNK_FRAMES : (chunk + 1) * CHUNK_FRAMES]
        prediction = runtime.predict(context, history, actions, 0, None)
        predictions.append(prediction)
        context = np.concatenate((context, prediction), axis=0)[-5:]
        history = actions[-4:]
    return np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--instruction-map")
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline-left", required=True)
    parser.add_argument("--baseline-right", required=True)
    parser.add_argument("--candidate-left", required=True)
    parser.add_argument(
        "--candidate-right",
        help="Defaults to the frozen right baseline for a left-only experiment",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.chunks < 2 or args.max_sequences < 1:
        raise SystemExit("chunks must be >=2 and max-sequences must be positive")

    split = json.loads(Path(args.split_manifest).read_text())
    instruction_by_seed = None
    if args.instruction_map:
        document = json.loads(Path(args.instruction_map).read_text())
        raw = document.get("seed_to_instruction", document)
        instruction_by_seed = {int(seed): str(value) for seed, value in raw.items()}
    instruction_by_episode = {
        int(episode): str(value)
        for episode, value in split.get("episode_to_instruction", {}).items()
    }
    dataset = MultiChunkWindows(
        Path(args.windows),
        split[args.episodes_key],
        chunks=args.chunks,
        chunk_stride=CHUNK_FRAMES,
        arm_filter="all",
        instruction_by_seed=instruction_by_seed,
    )
    indices = list(range(len(dataset)))
    if args.max_sequences < len(indices):
        indices = np.linspace(
            0, len(dataset) - 1, args.max_sequences, dtype=int
        ).tolist()

    baseline_left = Track2AutoregressiveUNet(args.baseline_left, args.device)
    baseline_right = Track2AutoregressiveUNet(args.baseline_right, args.device)
    candidate_left = Track2AutoregressiveUNet(args.candidate_left, args.device)
    candidate_right = (
        Track2AutoregressiveUNet(args.candidate_right, args.device)
        if args.candidate_right
        else baseline_right
    )
    records: dict[str, list] = {
        "context_last": [],
        "target": [],
        "baseline": [],
        "candidate": [],
        "arm_right": [],
        "capture_success": [],
        "path": [],
        "synthetic_seed": [],
        "episode_id": [],
        "instruction": [],
        "start": [],
    }
    for position, index in enumerate(indices, 1):
        context_t, history_t, future_t, target_t, arm_t, success_t, instruction_t = dataset[index]
        context = context_t.numpy()
        history = history_t.numpy()
        future = future_t.numpy()
        target = target_t.numpy()
        arm_right = bool(arm_t)
        baseline_runtime = baseline_right if arm_right else baseline_left
        candidate_runtime = candidate_right if arm_right else candidate_left
        baseline = predict_sequence(
            baseline_runtime, context.copy(), history.copy(), future, args.chunks
        )
        if arm_right and args.candidate_right is None:
            candidate = baseline.copy()
        else:
            candidate = predict_sequence(
                candidate_runtime, context.copy(), history.copy(), future, args.chunks
            )
        first_path = dataset.sequences[index][0]
        episode = int(dataset.episode_id[index])
        with np.load(first_path, allow_pickle=False) as values:
            synthetic_seed = int(values["synthetic_seed"]) if "synthetic_seed" in values.files else -1
        instruction = str(instruction_t) or instruction_by_episode.get(episode, "")
        if not instruction:
            raise ValueError(f"missing exact instruction for {first_path}")
        records["context_last"].append(context[-1])
        records["target"].append(target)
        records["baseline"].append(baseline)
        records["candidate"].append(candidate)
        records["arm_right"].append(np.asarray(arm_right, dtype=np.bool_))
        records["capture_success"].append(
            np.asarray(bool(success_t), dtype=np.bool_)
        )
        records["path"].append(first_path.name)
        records["synthetic_seed"].append(np.asarray(synthetic_seed, dtype=np.int64))
        records["episode_id"].append(np.asarray(episode, dtype=np.int64))
        records["instruction"].append(instruction)
        records["start"].append(
            np.asarray(dataset.start[index], dtype=np.int64)
        )
        if position == 1 or position % 8 == 0 or position == len(indices):
            print(json.dumps({"exported": position, "total": len(indices)}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            key: np.asarray(value) if key == "path" else np.stack(value)
            for key, value in records.items()
        },
    )
    manifest = {
        "format": "strict-track2-multichunk-reward-cache-v1",
        "output": str(output.resolve()),
        "sequence_count": len(indices),
        "chunks": args.chunks,
        "frame_count_per_sequence": args.chunks * CHUNK_FRAMES,
        "predicted_context_between_chunks": True,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "instruction_map": str(Path(args.instruction_map).resolve()) if args.instruction_map else None,
        "exact_instruction_per_sequence": True,
        "episodes_key": args.episodes_key,
        "baseline_left": str(Path(args.baseline_left).resolve()),
        "baseline_right": str(Path(args.baseline_right).resolve()),
        "candidate_left": str(Path(args.candidate_left).resolve()),
        "candidate_right": (
            str(Path(args.candidate_right).resolve())
            if args.candidate_right
            else "frozen_baseline_right"
        ),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
