#!/usr/bin/env python3
"""Generate the immutable, one-shot public-development S1 inputs for v440.

The independent v169 trajectory is retained for performance comparisons.  A
second v169 prediction is captured from ``predict_batch_with_baseline`` for
each hybrid request and is used only for local exactness and residual-bound
audits.  The v354 teacher parent is named explicitly throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import generate_v439_s1_offline_inputs as common
from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.v440_v169_trainonly_aligned_projection_runtime import (
    Track2V440V169TrainOnlyAlignedProjection,
)


INFERENCE_SEED = 1586
LINEAGE = "v440"
INPUT_FORMAT = "strict-track2-v440-s1-offline-inputs-v1"
RUNTIME_CLASS = Track2V440V169TrainOnlyAlignedProjection


def stable_seed(episode: int, start: int, chunk_index: int) -> int:
    payload = f"{LINEAGE}-s1/episode{episode}/start{start}/chunk{chunk_index}/seed{INFERENCE_SEED}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**31)


def _rollout_block(runtime, samples: list[dict], shuffled_future: np.ndarray) -> dict:
    prompts = [str(row["prompt"]) for row in samples]
    episode = np.asarray([int(row["episode"]) for row in samples], dtype=np.int64)
    start = np.asarray([int(row["start"]) for row in samples], dtype=np.int64)
    actual_future = np.stack([row["future_actions"] for row in samples]).astype(np.float32)
    initial_context = np.stack([row["context_frames"] for row in samples]).astype(np.uint8)
    initial_history = np.stack([row["history_actions"] for row in samples]).astype(np.float32)

    state_names = ("independent_v169", "v354_parent", "hybrid", "shuffle")
    contexts = {name: initial_context.copy() for name in state_names}
    histories = {name: initial_history.copy() for name in state_names}
    predictions = {
        name: [] for name in (*state_names, "same_request_v169")
    }
    gates: list[np.ndarray] = []
    swap_gates: list[np.ndarray] = []
    for chunk_index in range(common.HORIZON // common.CHUNK):
        begin, end = chunk_index * common.CHUNK, (chunk_index + 1) * common.CHUNK
        actual = actual_future[:, begin:end]
        shuffled = shuffled_future[:, begin:end]
        seeds = np.asarray(
            [stable_seed(int(ep), int(offset), chunk_index) for ep, offset in zip(episode, start)],
            dtype=np.int64,
        )
        hybrid_history_before = histories["hybrid"].copy()
        independent_v169 = runtime.v169.predict_batch(
            contexts["independent_v169"], histories["independent_v169"], actual, seeds, prompts
        )
        v354_parent = runtime.parent.predict_batch(
            contexts["v354_parent"], histories["v354_parent"], actual, seeds, prompts
        )
        same_request_v169, hybrid, decisions = runtime.predict_batch_with_baseline(
            contexts["hybrid"], histories["hybrid"], actual, seeds, prompts
        )
        shuffle = runtime.predict_batch(
            contexts["shuffle"], histories["shuffle"], shuffled, seeds, prompts
        )
        chunk_frames = {
            "independent_v169": independent_v169,
            "same_request_v169": same_request_v169,
            "v354_parent": v354_parent,
            "hybrid": hybrid,
            "shuffle": shuffle,
        }
        for name, frames in chunk_frames.items():
            frames = np.asarray(frames, dtype=np.uint8)
            if frames.shape[:2] != (len(samples), common.CHUNK):
                raise RuntimeError(f"unexpected {name} chunk shape: {frames.shape}")
            predictions[name].append(frames)
        for name, actions in (
            ("independent_v169", actual),
            ("v354_parent", actual),
            ("hybrid", actual),
            ("shuffle", shuffled),
        ):
            contexts[name], histories[name] = common.update_state(
                contexts[name], histories[name], chunk_frames[name], actions
            )
        gates.append(np.asarray([bool(row["gate"]) for row in decisions], dtype=np.bool_))
        swap_gates.append(np.asarray([
            bool(runtime.gate_decision(
                common.swap_arms(history), common.swap_arms(future), prompt
            )["gate"])
            for history, future, prompt in zip(hybrid_history_before, actual, prompts)
        ], dtype=np.bool_))
    return {
        **{f"{name}_frames": np.concatenate(chunks, axis=1) for name, chunks in predictions.items()},
        "gate": np.stack(gates, axis=1),
        "swap_gate": np.stack(swap_gates, axis=1),
    }


def rollout(runtime, samples: list[dict], shuffled_future: np.ndarray, inference_batch_size: int) -> dict:
    if inference_batch_size != 4:
        raise ValueError(f"{LINEAGE} S1 is preregistered for inference_batch_size=4")
    blocks = []
    for begin in range(0, len(samples), inference_batch_size):
        end = min(begin + inference_batch_size, len(samples))
        blocks.append(_rollout_block(runtime, samples[begin:end], shuffled_future[begin:end]))
    keys = set(blocks[0])
    if any(set(block) != keys for block in blocks):
        raise RuntimeError(f"{LINEAGE} S1 inference blocks returned inconsistent keys")
    return {key: np.concatenate([block[key] for block in blocks], axis=0) for key in sorted(keys)}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "release", "windows", "source-data", "split", "reward-checkpoint", "t5-model", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inference_batch_size != 4:
        raise SystemExit(f"{LINEAGE} S1 requires --inference-batch-size 4")

    split = json.loads(args.split.read_text())
    train = sorted(int(value) for value in split.get("train_episodes", []))
    validation = sorted(int(value) for value in split.get("validation_episodes", []))
    if train != common.TRAIN40 or validation != common.HOLDOUT10 or set(train).intersection(validation):
        raise RuntimeError(f"{LINEAGE} requires the frozen public train40/holdout10 split")
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    validation_right = [episode for episode in validation if arms[episode] == "right"]
    validation_left = [episode for episode in validation if arms[episode] == "left"]
    if validation_right != common.RIGHT_DEV or not validation_left:
        raise RuntimeError(f"{LINEAGE} public development arm split drift")

    # Selection, contract probes, and phase shuffle are fixed before reward is loaded.
    right_dataset = WindowDataset(args.windows, validation_right, rollout_horizon=common.HORIZON)
    left_dataset = WindowDataset(args.windows, validation_left, rollout_horizon=common.HORIZON)
    s1_rows = common.right_s1(right_dataset, args.source_data, prompts)
    samples = (
        s1_rows
        + common.left_contract_probes(left_dataset, prompts)
        + common.right_g0_swap_probes(s1_rows)
    )
    shuffled_future = common.phase_shuffle(samples)

    runtime = RUNTIME_CLASS(args.release, args.device)
    generated = rollout(runtime, samples, shuffled_future, args.inference_batch_size)
    target_frames = np.stack([row["target_frames"] for row in samples]).astype(np.uint8)
    prompts_list = [str(row["prompt"]) for row in samples]

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    reward_names = ("independent_v169", "same_request_v169", "v354_parent", "hybrid")
    reward_arrays = {
        f"{name}_reward": common.score_reward(
            reward, generated[f"{name}_frames"], prompts_list, device, args.reward_batch_size
        )
        for name in reward_names
    }
    target_reward = common.score_reward(
        reward, target_frames, prompts_list, device, args.reward_batch_size
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        independent_v169_frames=generated["independent_v169_frames"],
        same_request_v169_frames=generated["same_request_v169_frames"],
        v354_parent_frames=generated["v354_parent_frames"],
        hybrid_frames=generated["hybrid_frames"],
        shuffle_frames=generated["shuffle_frames"],
        target_frames=target_frames,
        independent_v169_reward=reward_arrays["independent_v169_reward"],
        same_request_v169_reward=reward_arrays["same_request_v169_reward"],
        v354_parent_reward=reward_arrays["v354_parent_reward"],
        hybrid_reward=reward_arrays["hybrid_reward"],
        target_reward=target_reward,
        episode=np.asarray([row["episode"] for row in samples], dtype=np.int64),
        arm=np.asarray([row["arm"] for row in samples]),
        phase=np.asarray([row["phase"] for row in samples]),
        gate=generated["gate"],
        swap_gate=generated["swap_gate"],
        s1=np.asarray([row["s1"] for row in samples], dtype=np.bool_),
    )
    print(json.dumps({
        "format": INPUT_FORMAT,
        "output": str(args.output),
        "samples": len(samples),
        "s1_samples": sum(bool(row["s1"]) for row in samples),
        "left_probes": sum(row["arm"] == "left" for row in samples),
        "g0_swap_probes": sum(row["phase"] == "g0_probe" for row in samples),
        "requests": int(generated["gate"].size),
        "inference_batch_size": args.inference_batch_size,
        "reward_batch_size": args.reward_batch_size,
        "gate_active": int(generated["gate"].sum()),
        "baseline_semantics": {
            "performance": "independent_v169",
            "local_exact_and_cap": "same_request_v169",
            "teacher_parent": "v354_parent",
        },
        "guards": {
            "outcomes_read": False, "hidden_or_final_data": False,
            "policy_updates": 0, "development_run_count": 1,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
