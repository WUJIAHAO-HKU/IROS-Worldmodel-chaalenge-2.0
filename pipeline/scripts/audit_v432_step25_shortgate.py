#!/usr/bin/env python3
"""Paired v432 step25 short gate against its immutable initialization.

The gate uses 32 episode-disjoint public right-arm holdout sequences: two from
each of four phases in each of four episodes.  It evaluates exactly quantized
runtime RGB, a recursive 32-frame rollout, temporal deltas, and the frozen
official reward model.  The report is written before a failed gate returns 2.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.autoregressive_unet import OneStepActionUNet


PHASES = ("early", "grasp", "postgrasp", "endpoint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_unique(candidates: list[int], used: set[int], count: int = 2) -> list[int]:
    selected = []
    for index in candidates:
        if index not in used:
            selected.append(index)
            used.add(index)
        if len(selected) == count:
            return selected
    raise RuntimeError("not enough distinct phase-stratified validation sequences")


def fixed_selection(dataset: WindowDataset, source_data: Path) -> list[dict]:
    by_episode: dict[int, list[int]] = {}
    for index, path in enumerate(dataset.paths):
        episode = int(path.name.split("_")[0][7:])
        by_episode.setdefault(episode, []).append(index)
    if len(by_episode) != 4:
        raise RuntimeError(f"expected four real-right holdout episodes, got {sorted(by_episode)}")
    rows = []
    for episode, indices in sorted(by_episode.items()):
        indices = sorted(indices, key=lambda value: int(dataset.paths[value].stem.split("_")[1]))
        starts = {index: int(dataset.paths[index].stem.split("_")[1]) for index in indices}
        with h5py.File(source_data / f"episode{episode}.hdf5", "r") as item:
            actions = np.asarray(item["joint_action/vector"], dtype=np.float32)
        closed = np.flatnonzero(actions[:, 13] < 0.5)
        if not closed.size:
            raise RuntimeError(f"right gripper never closes in public holdout episode {episode}")
        first_close = int(closed[0])
        used: set[int] = set()
        ordered = {
            # Reserve endpoint first, then contact and post-contact neighborhoods.
            "endpoint": sorted(indices, key=lambda value: starts[value], reverse=True),
            "grasp": sorted(indices, key=lambda value: abs((starts[value] + 5) - first_close)),
            "postgrasp": sorted(indices, key=lambda value: abs((starts[value] + 5) - (first_close + 24))),
            "early": sorted(indices, key=lambda value: starts[value]),
        }
        chosen = {phase: choose_unique(ordered[phase], used) for phase in ("endpoint", "grasp", "postgrasp", "early")}
        for phase in PHASES:
            for index in chosen[phase]:
                rows.append({
                    "dataset_index": index,
                    "episode": episode,
                    "start": starts[index],
                    "phase": phase,
                    "first_right_close": first_close,
                    "path": str(dataset.paths[index]),
                })
    if len(rows) != 32 or {phase: sum(row["phase"] == phase for row in rows) for phase in PHASES} != {phase: 8 for phase in PHASES}:
        raise RuntimeError("fixed phase selection is not 8/8/8/8")
    return rows


def load_samples(dataset: WindowDataset, selection: list[dict], prompts: dict[int, str]) -> list[dict]:
    samples = []
    for row in selection:
        arrays = dataset.load_arrays(int(row["dataset_index"]))
        prompt = str(prompts[int(row["episode"])])
        lowered = prompt.lower()
        if "right arm" not in lowered or "left arm" in lowered:
            raise RuntimeError(f"holdout prompt is not right-arm-consistent: {prompt!r}")
        samples.append({**row, **arrays, "prompt": prompt})
    return samples


def score_reward(reward, frames: np.ndarray, prompts: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    for begin in range(0, len(frames), batch_size):
        end = begin + batch_size
        tensor = torch.from_numpy(np.ascontiguousarray(frames[begin:end])).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.inference_mode():
            result = reward.compute_reward(tensor, task_descriptions=prompts[begin:end])
        values.extend(result.detach().float().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise RuntimeError(f"unsupported model checkpoint: {checkpoint}")
    model = OneStepActionUNet().to(device).eval()
    model.load_state_dict(state["state_dict"], strict=True)
    with np.load(checkpoint / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(np.asarray(values["mean"], dtype=np.float32)).to(device)
        std = torch.from_numpy(np.asarray(values["std"], dtype=np.float32)).to(device)
    return model, mean, std


def predict32(model, mean, std, sample: dict, device: torch.device) -> np.ndarray:
    context = torch.from_numpy(np.ascontiguousarray(sample["context_frames"])).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(device)
    history = torch.from_numpy(np.ascontiguousarray(sample["history_actions"])).unsqueeze(0).to(device)
    future = torch.from_numpy(np.ascontiguousarray(sample["future_actions"])).unsqueeze(0).to(device)
    history = ((history - mean) / std).float()
    future = ((future - mean) / std).float()
    predictions = []
    with torch.inference_mode():
        for action in future.unbind(1):
            frame = model(context, torch.cat((history, action[:, None]), 1)).clamp(0, 1)
            predictions.append(frame)
            context = torch.cat((context[:, 1:], frame[:, None]), 1)
            history = torch.cat((history[:, 1:], action[:, None]), 1)
    output = torch.stack(predictions, 1).mul(255).round().to(torch.uint8)
    return output[0].permute(0, 2, 3, 1).cpu().numpy().copy()


def evaluate_checkpoint(name: str, checkpoint: Path, samples: list[dict], target_reward: np.ndarray, reward, device: torch.device, reward_batch_size: int) -> dict:
    model, mean, std = load_model(checkpoint, device)
    rows = []
    all_predictions = []
    all_prompts = []
    for sample_index, sample in enumerate(samples):
        prediction = predict32(model, mean, std, sample, device)
        target = sample["target_frames"].astype(np.uint8)
        context_last = sample["context_frames"][-1:].astype(np.int16)
        predicted_previous = np.concatenate((context_last, prediction[:-1].astype(np.int16)), axis=0)
        target_previous = np.concatenate((context_last, target[:-1].astype(np.int16)), axis=0)
        rows.append({
            "sample_index": sample_index,
            "episode": sample["episode"], "start": sample["start"], "phase": sample["phase"],
            "first8_rgb_mae": float(np.abs(prediction[:8].astype(np.int16) - target[:8].astype(np.int16)).mean()),
            "recursive32_rgb_mae": float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean()),
            "temporal_delta_error": float(np.abs(
                (prediction.astype(np.int16) - predicted_previous)
                - (target.astype(np.int16) - target_previous)
            ).mean()),
        })
        all_predictions.extend(prediction)
        all_prompts.extend([sample["prompt"]] * 32)
    predicted_reward = score_reward(reward, np.stack(all_predictions), all_prompts, device, reward_batch_size).reshape(len(samples), 32)
    target_reward = target_reward.reshape(len(samples), 32)
    for index, row in enumerate(rows):
        row["reward_prediction_mae"] = float(np.abs(predicted_reward[index] - target_reward[index]).mean())
        row["endpoint_reward_prediction_mae"] = float(abs(predicted_reward[index, -1] - target_reward[index, -1]))
        row["predicted_endpoint_reward"] = float(predicted_reward[index, -1])
        row["target_endpoint_reward"] = float(target_reward[index, -1])
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("first8_rgb_mae", "recursive32_rgb_mae", "temporal_delta_error", "reward_prediction_mae", "endpoint_reward_prediction_mae")
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"model": name, "metrics": metrics}), flush=True)
    return {"checkpoint": str(checkpoint), "metrics": metrics, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("parent", "candidate", "windows", "source-data", "split", "reward-checkpoint", "t5-model", "preregistration", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v432-public-mirror-prompt-terminal-preregistration-v1":
        raise RuntimeError("wrong v432 preregistration")
    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    validation_right = [int(value) for value in split["validation_episodes"] if arms[int(value)] == "right"]
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    dataset = WindowDataset(args.windows, validation_right, rollout_horizon=32)
    selection = fixed_selection(dataset, args.source_data)
    samples = load_samples(dataset, selection, prompts)

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    device = torch.device(args.device)
    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(device).eval().requires_grad_(False)
    target_frames = np.concatenate([sample["target_frames"] for sample in samples], axis=0)
    target_prompts = [sample["prompt"] for sample in samples for _ in range(32)]
    target_reward = score_reward(reward, target_frames, target_prompts, device, args.reward_batch_size)
    results = {
        "parent": evaluate_checkpoint("parent", args.parent, samples, target_reward, reward, device, args.reward_batch_size),
        "candidate": evaluate_checkpoint("candidate", args.candidate, samples, target_reward, reward, device, args.reward_batch_size),
    }
    parent = results["parent"]["metrics"]
    candidate = results["candidate"]["metrics"]
    ratios = {key: float(candidate[key] / max(parent[key], 1e-12)) for key in parent}
    checks = {
        "exact_32_samples": len(samples) == 32,
        "phase_8_each": all(sum(row["phase"] == phase for row in selection) == 8 for phase in PHASES),
        "first8_mae_improves_0p2pct": ratios["first8_rgb_mae"] <= 0.998,
        "recursive32_mae_regression_le_0p2pct": ratios["recursive32_rgb_mae"] <= 1.002,
        "temporal_delta_nonregression": ratios["temporal_delta_error"] <= 1.0,
        "reward_prediction_mae_improves_2pct": ratios["reward_prediction_mae"] <= 0.98,
        "endpoint_reward_mae_improves_2pct": ratios["endpoint_reward_prediction_mae"] <= 0.98,
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v432-step25-shortgate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "selection": selection,
        "results": results,
        "candidate_over_parent": ratios,
        "checks": checks,
        "decision": "may prepare step50 extension only" if passed else "reject; preserve checkpoint/report; no step50 or RL",
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "split": sha256(args.split),
            "parent_model": sha256(args.parent / "model.pt"),
            "candidate_model": sha256(args.candidate / "model.pt"),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "public_holdout_only": True,
            "outcomes_read": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "ratios": ratios, "checks": checks}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
