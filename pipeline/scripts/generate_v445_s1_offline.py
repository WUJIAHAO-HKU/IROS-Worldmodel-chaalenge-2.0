#!/usr/bin/env python3
"""One-shot frozen public-development S1 generator for v445 full mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import generate_v439_s1_offline_inputs as common
from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.v445_v169_full_mirror_runtime import Track2V445V169FullMirror


FORMAT = "strict-track2-v445-full-mirror-s1-offline-inputs-v1"
INFERENCE_SEED = 1596


def stable_seed(episode: int, start: int, chunk: int) -> int:
    digest = hashlib.sha256(f"v445-s1/episode{episode}/start{start}/chunk{chunk}/seed{INFERENCE_SEED}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def rollout(runtime, samples: list[dict], action_variants: dict[str, np.ndarray], batch_size: int) -> dict:
    if batch_size != 4:
        raise ValueError("v445 S1 requires inference batch size 4")
    outputs = []
    for block_begin in range(0, len(samples), batch_size):
        block = samples[block_begin:block_begin + batch_size]
        prompts = [str(row["prompt"]) for row in block]
        episodes = np.asarray([int(row["episode"]) for row in block])
        starts = np.asarray([int(row["start"]) for row in block])
        actual_future = np.stack([row["future_actions"] for row in block]).astype(np.float32)
        variant_block = {name: value[block_begin:block_begin + len(block)] for name, value in action_variants.items()}
        initial_context = np.stack([row["context_frames"] for row in block]).astype(np.uint8)
        initial_history = np.stack([row["history_actions"] for row in block]).astype(np.float32)
        state_names = ("v169", "mirror", "shuffle", "open", "static", "reverse")
        contexts = {name: initial_context.copy() for name in state_names}
        histories = {name: initial_history.copy() for name in state_names}
        frames = {name: [] for name in state_names}
        gates = []
        for chunk in range(4):
            begin, end = chunk * 8, (chunk + 1) * 8
            actual = actual_future[:, begin:end]
            actions = {name: value[:, begin:end] for name, value in variant_block.items()}
            seeds = np.asarray([stable_seed(int(ep), int(start), chunk) for ep, start in zip(episodes, starts)], dtype=np.int64)
            v169 = runtime.v169.predict_batch(contexts["v169"], histories["v169"], actual, seeds, prompts)
            mirror = runtime.predict_batch(contexts["mirror"], histories["mirror"], actual, seeds, prompts)
            chunk_frames = {"v169": v169, "mirror": mirror}
            for name in ("shuffle", "open", "static", "reverse"):
                chunk_frames[name] = runtime.predict_batch(contexts[name], histories[name], actions[name], seeds, prompts)
            for name, value in chunk_frames.items():
                value = np.asarray(value, dtype=np.uint8)
                if value.shape[:2] != (len(block), 8):
                    raise RuntimeError(f"unexpected v445 {name} shape {value.shape}")
                frames[name].append(value)
            for name, action in (("v169", actual), ("mirror", actual), *((name, actions[name]) for name in ("shuffle", "open", "static", "reverse"))):
                contexts[name], histories[name] = common.update_state(contexts[name], histories[name], chunk_frames[name], action)
            gates.append(np.asarray([runtime.gate_decision(h, f, p)["gate"] for h, f, p in zip(histories["mirror"], actual, prompts)], dtype=np.bool_))
        outputs.append({
            **{f"{name}_frames": np.concatenate(value, axis=1) for name, value in frames.items()},
            "gate": np.stack(gates, axis=1),
        })
    return {key: np.concatenate([block[key] for block in outputs], axis=0) for key in outputs[0]}


def contract_probes(runtime, right_rows: list[dict], left_rows: list[dict]) -> dict:
    if len(right_rows) < 2 or len(left_rows) < 2:
        raise RuntimeError("v445 requires two right and two left contract probes")
    rows = [right_rows[0], left_rows[0], right_rows[1], left_rows[1]]
    context = np.stack([row["context_frames"] for row in rows])
    history = np.stack([row["history_actions"] for row in rows])
    future = np.stack([row["future_actions"][:8] for row in rows])
    prompts = [str(row["prompt"]) for row in rows]
    seeds = np.asarray([stable_seed(int(row["episode"]), int(row["start"]), 7) for row in rows], dtype=np.int64)
    baseline, batch, _ = runtime.predict_batch_with_baseline(context, history, future, seeds, prompts)
    left_exact = bool(np.array_equal(batch[[1, 3]], baseline[[1, 3]]))
    serial = np.stack([runtime.predict(context[i], history[i], future[i], int(seeds[i]), prompts[i]) for i in (0, 2)])
    serial_exact = bool(np.array_equal(serial, batch[[0, 2]]))
    permutation = np.asarray([2, 0, 3, 1])
    permuted = runtime.predict_batch(context[permutation], history[permutation], future[permutation], seeds[permutation], [prompts[i] for i in permutation])
    restored = np.empty_like(permuted); restored[permutation] = permuted
    permutation_exact = bool(np.array_equal(restored, batch))
    return {"mixed_left_exact": left_exact, "right_serial_batch_exact": serial_exact, "batch_permutation_exact": permutation_exact}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("release", "windows", "source-data", "split", "reward-checkpoint", "t5-model", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inference_batch_size != 4 or args.reward_batch_size != 32:
        raise SystemExit("v445 S1 frozen batch sizes are inference=4/reward=32")
    split = json.loads(args.split.read_text())
    train = sorted(int(value) for value in split["train_episodes"])
    holdout = sorted(int(value) for value in split["validation_episodes"])
    if train != common.TRAIN40 or holdout != common.HOLDOUT10 or set(train) & set(holdout):
        raise RuntimeError("v445 frozen split drift")
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    right = [episode for episode in holdout if arms[episode] == "right"]
    left = [episode for episode in holdout if arms[episode] == "left"]
    if right != common.RIGHT_DEV or not left:
        raise RuntimeError("v445 public-development arm split drift")
    right_dataset = WindowDataset(args.windows, right, rollout_horizon=32)
    left_dataset = WindowDataset(args.windows, left, rollout_horizon=32)
    s1_rows = common.right_s1(right_dataset, args.source_data, prompts)
    samples = s1_rows + common.left_contract_probes(left_dataset, prompts)
    actual_future = np.stack([row["future_actions"] for row in samples]).astype(np.float32)
    shuffled_future = common.phase_shuffle(samples)
    open_future = actual_future.copy(); open_future[:, :, 13] = 1.0
    static_future = np.stack([np.repeat(row["history_actions"][-1:], 32, axis=0) for row in samples]).astype(np.float32)
    reverse_future = actual_future[:, ::-1].copy()
    action_variants = {"shuffle": shuffled_future, "open": open_future, "static": static_future, "reverse": reverse_future}
    runtime = Track2V445V169FullMirror(args.release, args.device)
    probes = contract_probes(runtime, s1_rows, samples[len(s1_rows):])
    generated = rollout(runtime, samples, action_variants, args.inference_batch_size)
    target = np.stack([row["target_frames"] for row in samples]).astype(np.uint8)
    prompt_list = [str(row["prompt"]) for row in samples]

    # Reward is loaded only after fixed selection, shuffle, and RGB rollout.
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    reward_names = ("mirror", "shuffle", "open", "static", "reverse")
    rewards = {name: common.score_reward(reward, generated[f"{name}_frames"], prompt_list, device, args.reward_batch_size) for name in reward_names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format=np.asarray(FORMAT),
        v169_frames=generated["v169_frames"], mirror_frames=generated["mirror_frames"],
        shuffle_frames=generated["shuffle_frames"], open_frames=generated["open_frames"],
        static_frames=generated["static_frames"], reverse_frames=generated["reverse_frames"], target_frames=target,
        mirror_reward=rewards["mirror"], shuffle_reward=rewards["shuffle"], open_reward=rewards["open"],
        static_reward=rewards["static"], reverse_reward=rewards["reverse"],
        episode=np.asarray([row["episode"] for row in samples], dtype=np.int64),
        arm=np.asarray([row["arm"] for row in samples]), phase=np.asarray([row["phase"] for row in samples]),
        s1=np.asarray([row["s1"] for row in samples], dtype=np.bool_), gate=generated["gate"],
        contract_mixed_left_exact=np.asarray(probes["mixed_left_exact"]),
        contract_right_serial_batch_exact=np.asarray(probes["right_serial_batch_exact"]),
        contract_batch_permutation_exact=np.asarray(probes["batch_permutation_exact"]),
    )
    print(json.dumps({
        "format": FORMAT, "output": str(args.output), "samples": len(samples),
        "s1_samples": sum(bool(row["s1"]) for row in samples), "development_run_count": 1, "contract_probes": probes,
        "guards": {"reward_loaded_after_fixed_rollout": True, "outcomes_read": False, "policy_updates": 0},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
