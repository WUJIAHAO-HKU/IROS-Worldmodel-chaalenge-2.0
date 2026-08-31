#!/usr/bin/env python3
"""Generate immutable public-development inputs for the v439 S1 auditor.

The generator binds v169, v354, v432-step25-through-v439 and the hybrid to the
same requests.  Selection and action shuffling are completed before the frozen
official reward model is loaded.  Reward, inference seed and request identity
are never accepted by ``gate_decision``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_v432_step25_shortgate import fixed_selection
from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.v439_v169_action_causal_projection_runtime import (
    Track2V439V169ActionCausalProjection,
)


TRAIN40 = [
    0, 1, 3, 4, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41,
    42, 43, 44, 45, 46, 47, 49,
]
HOLDOUT10 = [2, 5, 6, 7, 9, 16, 18, 22, 39, 48]
RIGHT_DEV = [6, 7, 18, 22]
PHASES = ("early", "grasp", "postgrasp", "endpoint")
CHUNK = 8
HORIZON = 32
HISTORY = 4
INFERENCE_SEED = 1584


def stable_seed(episode: int, start: int, chunk_index: int) -> int:
    payload = f"v439-s1/episode{episode}/start{start}/chunk{chunk_index}/seed{INFERENCE_SEED}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**31)


def swap_arms(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != 14:
        raise ValueError(f"expected joint14 actions, got {values.shape}")
    output = np.empty_like(values)
    output[..., :7] = values[..., 7:14]
    output[..., 7:14] = values[..., :7]
    return output


def load_sample(dataset: WindowDataset, row: dict, prompts: dict[int, str]) -> dict:
    arrays = dataset.load_arrays(int(row["dataset_index"]))
    episode = int(row["episode"])
    prompt = str(prompts[episode])
    sample = {**row, **arrays, "prompt": prompt}
    expected = "right arm" if row["arm"] == "right" else "left arm"
    opposite = "left arm" if expected == "right arm" else "right arm"
    lowered = prompt.lower()
    if expected not in lowered or opposite in lowered:
        raise RuntimeError(f"arm-inconsistent public prompt for episode {episode}: {prompt!r}")
    if sample["future_actions"].shape != (HORIZON, 14):
        raise RuntimeError(f"expected future[32,14], got {sample['future_actions'].shape}")
    if sample["history_actions"].shape != (HISTORY, 14):
        raise RuntimeError(f"expected history[4,14], got {sample['history_actions'].shape}")
    return sample


def right_s1(dataset: WindowDataset, source_data: Path, prompts: dict[int, str]) -> list[dict]:
    selection = fixed_selection(dataset, source_data)
    if len(selection) != 32:
        raise RuntimeError(f"expected exact 32 right S1 rows, got {len(selection)}")
    rows = []
    for row in selection:
        rows.append(load_sample(dataset, {**row, "arm": "right", "s1": True}, prompts))
    if {row["episode"] for row in rows} != set(RIGHT_DEV):
        raise RuntimeError("right S1 episode drift")
    if any(sum(row["phase"] == phase for row in rows) != 8 for phase in PHASES):
        raise RuntimeError("right S1 must contain exactly eight rows per phase")
    return rows


def left_contract_probes(dataset: WindowDataset, prompts: dict[int, str]) -> list[dict]:
    by_episode: dict[int, list[int]] = {}
    for index, path in enumerate(dataset.paths):
        episode = int(path.name.split("_")[0][7:])
        by_episode.setdefault(episode, []).append(index)
    rows = []
    for episode, indices in sorted(by_episode.items()):
        ordered = sorted(indices, key=lambda index: int(dataset.paths[index].stem.split("_")[1]))
        chosen = [ordered[0], ordered[-1]] if len(ordered) > 1 else [ordered[0]]
        for index in chosen:
            start = int(dataset.paths[index].stem.split("_")[1])
            row = {
                "dataset_index": index, "episode": episode, "start": start,
                "phase": "left_probe", "arm": "left", "s1": False,
                "path": str(dataset.paths[index]),
            }
            rows.append(load_sample(dataset, row, prompts))
    if not rows:
        raise RuntimeError("no left bit-exact contract probes were selected")
    return rows


def right_g0_swap_probes(rows: list[dict]) -> list[dict]:
    probes = []
    for phase in PHASES:
        source = next(row for row in rows if row["phase"] == phase)
        probe = dict(source)
        probe.update({"phase": "g0_probe", "s1": False})
        probe["history_actions"] = swap_arms(source["history_actions"])
        probe["future_actions"] = swap_arms(source["future_actions"])
        probes.append(probe)
    return probes


def phase_shuffle(samples: list[dict]) -> np.ndarray:
    """Rotate true future actions by one inside each frozen S1 phase stratum."""
    shuffled = [np.asarray(row["future_actions"], dtype=np.float32).copy() for row in samples]
    for phase in PHASES:
        indices = [index for index, row in enumerate(samples) if row["s1"] and row["phase"] == phase]
        if len(indices) != 8:
            raise RuntimeError(f"shuffle requires eight S1 rows for {phase}, got {len(indices)}")
        for position, index in enumerate(indices):
            source = indices[(position + 1) % len(indices)]
            shuffled[index] = np.asarray(samples[source]["future_actions"], dtype=np.float32).copy()
    # Contract probes are deliberately not shuffled; they never enter the
    # true-vs-shuffle S1 metric.
    return np.stack(shuffled)


def update_state(context: np.ndarray, history: np.ndarray, frames: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    next_context = np.concatenate((context, frames), axis=1)[:, -5:].copy()
    next_history = np.concatenate((history, actions), axis=1)[:, -HISTORY:].copy()
    return next_context, next_history


def _rollout_block(runtime, samples: list[dict], shuffled_future: np.ndarray) -> dict:
    prompts = [str(row["prompt"]) for row in samples]
    episode = np.asarray([int(row["episode"]) for row in samples], dtype=np.int64)
    start = np.asarray([int(row["start"]) for row in samples], dtype=np.int64)
    actual_future = np.stack([row["future_actions"] for row in samples]).astype(np.float32)
    initial_context = np.stack([row["context_frames"] for row in samples]).astype(np.uint8)
    initial_history = np.stack([row["history_actions"] for row in samples]).astype(np.float32)

    contexts = {name: initial_context.copy() for name in ("v169", "baseline", "hybrid", "shuffle")}
    histories = {name: initial_history.copy() for name in contexts}
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in contexts}
    gates = []
    swap_gates = []
    for chunk_index in range(HORIZON // CHUNK):
        begin, end = chunk_index * CHUNK, (chunk_index + 1) * CHUNK
        actual = actual_future[:, begin:end]
        shuffled = shuffled_future[:, begin:end]
        seeds = np.asarray(
            [stable_seed(int(ep), int(offset), chunk_index) for ep, offset in zip(episode, start)],
            dtype=np.int64,
        )
        hybrid_history_before = histories["hybrid"].copy()
        v169 = runtime.v169.predict_batch(
            contexts["v169"], histories["v169"], actual, seeds, prompts
        )
        baseline = runtime.parent.predict_batch(
            contexts["baseline"], histories["baseline"], actual, seeds, prompts
        )
        _, hybrid, decisions = runtime.predict_batch_with_baseline(
            contexts["hybrid"], histories["hybrid"], actual, seeds, prompts
        )
        shuffle = runtime.predict_batch(
            contexts["shuffle"], histories["shuffle"], shuffled, seeds, prompts
        )
        for name, frames, actions in (
            ("v169", v169, actual), ("baseline", baseline, actual),
            ("hybrid", hybrid, actual), ("shuffle", shuffle, shuffled),
        ):
            frames = np.asarray(frames, dtype=np.uint8)
            if frames.shape[:2] != (len(samples), CHUNK):
                raise RuntimeError(f"unexpected {name} chunk shape: {frames.shape}")
            predictions[name].append(frames)
            contexts[name], histories[name] = update_state(
                contexts[name], histories[name], frames, actions
            )
        gates.append(np.asarray([bool(row["gate"]) for row in decisions], dtype=np.bool_))
        swap_gates.append(np.asarray([
            bool(runtime.gate_decision(swap_arms(history), swap_arms(future), prompt)["gate"])
            for history, future, prompt in zip(hybrid_history_before, actual, prompts)
        ], dtype=np.bool_))
    return {
        **{f"{name}_frames": np.concatenate(chunks, axis=1) for name, chunks in predictions.items()},
        "gate": np.stack(gates, axis=1),
        "swap_gate": np.stack(swap_gates, axis=1),
    }


def rollout(runtime, samples: list[dict], shuffled_future: np.ndarray, inference_batch_size: int) -> dict:
    """Run fixed-size sample blocks and restore the immutable selection order."""
    if inference_batch_size != 4:
        raise ValueError("v439 S1 is preregistered for inference_batch_size=4")
    blocks: list[dict[str, np.ndarray]] = []
    for begin in range(0, len(samples), inference_batch_size):
        end = min(begin + inference_batch_size, len(samples))
        blocks.append(
            _rollout_block(runtime, samples[begin:end], shuffled_future[begin:end])
        )
    keys = set(blocks[0])
    if any(set(block) != keys for block in blocks):
        raise RuntimeError("v439 S1 inference blocks returned inconsistent keys")
    return {key: np.concatenate([block[key] for block in blocks], axis=0) for key in sorted(keys)}


def score_reward(model, frames: np.ndarray, prompts: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    count, horizon = frames.shape[:2]
    flat = frames.reshape(count * horizon, *frames.shape[2:])
    texts = [prompt for prompt in prompts for _ in range(horizon)]
    values = []
    for begin in range(0, len(flat), batch_size):
        end = min(begin + batch_size, len(flat))
        tensor = torch.from_numpy(np.ascontiguousarray(flat[begin:end])).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.inference_mode():
            result = model.compute_reward(tensor, task_descriptions=texts[begin:end])
        values.extend(result.detach().float().cpu().tolist())
    return np.asarray(values, dtype=np.float32).reshape(count, horizon)


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
    if args.inference_batch_size != 4:
        raise SystemExit("v439 S1 requires --inference-batch-size 4")

    split = json.loads(args.split.read_text())
    train = sorted(int(value) for value in split.get("train_episodes", []))
    validation = sorted(int(value) for value in split.get("validation_episodes", []))
    if train != TRAIN40 or validation != HOLDOUT10 or set(train).intersection(validation):
        raise RuntimeError("v439 requires the frozen public train40/holdout10 split")
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    validation_right = [episode for episode in validation if arms[episode] == "right"]
    validation_left = [episode for episode in validation if arms[episode] == "left"]
    if validation_right != RIGHT_DEV or not validation_left:
        raise RuntimeError("v439 public development arm split drift")

    # All selection, probes and shuffle pairing are frozen before loading the
    # official reward model. No outcome or reward is available to this code.
    right_dataset = WindowDataset(args.windows, validation_right, rollout_horizon=HORIZON)
    left_dataset = WindowDataset(args.windows, validation_left, rollout_horizon=HORIZON)
    s1_rows = right_s1(right_dataset, args.source_data, prompts)
    samples = s1_rows + left_contract_probes(left_dataset, prompts) + right_g0_swap_probes(s1_rows)
    shuffled_future = phase_shuffle(samples)

    runtime = Track2V439V169ActionCausalProjection(args.release, args.device)
    generated = rollout(runtime, samples, shuffled_future, args.inference_batch_size)
    target_frames = np.stack([row["target_frames"] for row in samples]).astype(np.uint8)
    prompts_list = [str(row["prompt"]) for row in samples]

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    reward_arrays = {
        f"{name}_reward": score_reward(reward, generated[f"{name}_frames"], prompts_list, device, args.reward_batch_size)
        for name in ("v169", "baseline", "hybrid")
    }
    target_reward = score_reward(reward, target_frames, prompts_list, device, args.reward_batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        v169_frames=generated["v169_frames"],
        baseline_frames=generated["baseline_frames"],
        hybrid_frames=generated["hybrid_frames"],
        shuffle_frames=generated["shuffle_frames"],
        target_frames=target_frames,
        v169_reward=reward_arrays["v169_reward"],
        baseline_reward=reward_arrays["baseline_reward"],
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
        "format": "strict-track2-v439-s1-offline-inputs-v1",
        "output": str(args.output),
        "samples": len(samples),
        "s1_samples": sum(bool(row["s1"]) for row in samples),
        "left_probes": sum(row["arm"] == "left" for row in samples),
        "g0_swap_probes": sum(row["phase"] == "g0_probe" for row in samples),
        "requests": int(generated["gate"].size),
        "inference_batch_size": args.inference_batch_size,
        "reward_batch_size": args.reward_batch_size,
        "gate_active": int(generated["gate"].sum()),
        "guards": {"outcomes_read": False, "hidden_or_final_data": False, "policy_updates": 0},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
