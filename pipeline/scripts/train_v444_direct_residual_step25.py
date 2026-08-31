#!/usr/bin/env python3
"""Train the preregistered v444 direct residual head for exactly 25 steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision
from wam_pipeline.v444_v169_direct_residual_runtime import (
    DirectResidualHead, PROTECTED_FRAMES, apply_direct_residual, instruction_tokens,
)


SEED = 1595
STEPS = 25
BATCH_SIZE = 4
INFERENCE_BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(episode: int, start: int, chunk: int = 0) -> int:
    value = hashlib.sha256(f"v444/seed{SEED}/episode{episode}/start{start}/chunk{chunk}".encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**31)


def select_close_rows(dataset: WindowDataset, episodes: list[int], prompts: dict[int, str]) -> list[dict]:
    rows = []
    counts = {episode: 0 for episode in episodes}
    for index, path in enumerate(dataset.paths):
        episode = int(path.name.split("_")[0][7:])
        history, future = dataset.load_actions(index)
        decision = gate_decision(history, future, prompts[episode])
        if decision.get("gate") and decision.get("phase") == "close":
            arrays = dataset.load_arrays(index)
            start = int(path.stem.split("_")[1])
            rows.append({"episode": episode, "start": start, "prompt": prompts[episode], **arrays})
            counts[episode] += 1
    if any(count != 8 for count in counts.values()) or len(rows) != len(episodes) * 8:
        raise RuntimeError(f"v444 requires exact eight close windows per episode: {counts}")
    return rows


def cache_v169(rows: list[dict], runtime: Track2V169ArmRoutedRuntime) -> None:
    for begin in range(0, len(rows), INFERENCE_BATCH_SIZE):
        batch = rows[begin:begin + INFERENCE_BATCH_SIZE]
        context = np.stack([row["context_frames"] for row in batch])
        history = np.stack([row["history_actions"] for row in batch])
        future = np.stack([row["future_actions"] for row in batch])
        seeds = np.asarray([stable_seed(row["episode"], row["start"]) for row in batch], dtype=np.int64)
        prompts = [row["prompt"] for row in batch]
        baseline = runtime.predict_batch(context, history, future, seeds, prompts)
        for local, row in enumerate(batch):
            row["baseline_frames"] = baseline[local].copy()


def action_statistics(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    actions = np.concatenate([np.concatenate((row["history_actions"], row["future_actions"]), axis=0) for row in rows], axis=0).astype(np.float64)
    mean = actions.mean(axis=0).astype(np.float32)
    std = actions.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std


def tensors(rows: list[dict], indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device):
    batch = [rows[int(index)] for index in indices]
    baseline_np = np.stack([row["baseline_frames"] for row in batch])
    target_np = np.stack([row["target_frames"] for row in batch])
    context_np = np.stack([row["context_frames"][-1] for row in batch])
    actions_np = np.stack([np.concatenate((row["history_actions"], row["future_actions"]), axis=0) for row in batch])
    target_residual = np.clip(target_np.astype(np.float32) - baseline_np.astype(np.float32), -8.0, 8.0)
    baseline = torch.as_tensor(baseline_np, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
    context = torch.as_tensor(context_np, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    actions = torch.as_tensor((actions_np - mean) / std, dtype=torch.float32, device=device)
    tokens = torch.as_tensor(instruction_tokens([row["prompt"] for row in batch]), device=device)
    residual = torch.as_tensor(target_residual, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
    return baseline, context, actions, tokens, residual


@torch.inference_mode()
def evaluate_rows(model, rows, mean, std, device) -> tuple[dict, dict[int, float]]:
    residual_absolute = 0.0; residual_count = 0
    rgb_absolute = 0.0; rgb_count = 0
    episode_abs: dict[int, float] = {}; episode_count: dict[int, int] = {}
    for begin in range(0, len(rows), BATCH_SIZE):
        indices = np.arange(begin, min(begin + BATCH_SIZE, len(rows)))
        baseline, context, actions, tokens, target_residual = tensors(rows, indices, mean, std, device)
        predicted = model(baseline, context, actions, tokens)
        residual_absolute += float(torch.abs(predicted - target_residual).sum().cpu())
        residual_count += target_residual.numel()
        predicted_np = predicted.permute(0, 1, 3, 4, 2).cpu().numpy()
        for local, index in enumerate(indices):
            row = rows[int(index)]
            output = apply_direct_residual(row["baseline_frames"], predicted_np[local])
            absolute = np.abs(output.astype(np.float64) - row["target_frames"].astype(np.float64))
            rgb_absolute += float(absolute.sum()); rgb_count += absolute.size
            episode_abs[row["episode"]] = episode_abs.get(row["episode"], 0.0) + float(absolute.sum())
            episode_count[row["episode"]] = episode_count.get(row["episode"], 0) + absolute.size
    metrics = {"residual_mae": residual_absolute / residual_count, "rgb_mae": rgb_absolute / rgb_count}
    return metrics, {episode: episode_abs[episode] / episode_count[episode] for episode in episode_abs}


def selected_recursive_rows(windows: Path, episodes: list[int], prompts: dict[int, str]) -> list[dict]:
    short = WindowDataset(windows, episodes, rollout_horizon=8)
    selected = []
    grouped: dict[int, list[tuple[int, int]]] = {episode: [] for episode in episodes}
    for index, path in enumerate(short.paths):
        episode = int(path.name.split("_")[0][7:])
        history, future = short.load_actions(index)
        decision = gate_decision(history, future, prompts[episode])
        if decision.get("gate"):
            start = int(path.stem.split("_")[1])
            grouped[episode].append((abs(int(decision["first_close_index"]) - 4), start))
    long = WindowDataset(windows, episodes, rollout_horizon=32)
    long_by_key = {(int(path.name.split("_")[0][7:]), int(path.stem.split("_")[1])): index for index, path in enumerate(long.paths)}
    for episode in episodes:
        candidates = sorted(grouped[episode])
        usable = [(distance, start) for distance, start in candidates if (episode, start) in long_by_key]
        if not usable:
            raise RuntimeError(f"v444 kill episode {episode} has no preregistered close window with 32-frame continuation")
        _, start = usable[0]
        arrays = long.load_arrays(long_by_key[(episode, start)])
        selected.append({"episode": episode, "start": start, "prompt": prompts[episode], **arrays})
    return selected


@torch.inference_mode()
def recursive32(model, rows, v169, mean, std, device) -> np.ndarray:
    context = np.stack([row["context_frames"] for row in rows])
    history = np.stack([row["history_actions"] for row in rows])
    all_future = np.stack([row["future_actions"] for row in rows])
    predictions = []
    for chunk in range(4):
        future = all_future[:, chunk * 8:(chunk + 1) * 8]
        seeds = np.asarray([stable_seed(row["episode"], row["start"], chunk) for row in rows], dtype=np.int64)
        prompts = [row["prompt"] for row in rows]
        baseline = v169.predict_batch(context, history, future, seeds, prompts)
        decisions = [gate_decision(h, f, p) for h, f, p in zip(history, future, prompts)]
        output = baseline.copy()
        enabled = np.asarray([decision["gate"] for decision in decisions], dtype=bool)
        if enabled.any():
            actions_np = np.concatenate((history[enabled], future[enabled]), axis=1)
            normalized = torch.as_tensor((actions_np - mean) / std, dtype=torch.float32, device=device)
            base = torch.as_tensor(baseline[enabled], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
            last = torch.as_tensor(context[enabled, -1], dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
            tokens = torch.as_tensor(instruction_tokens([prompts[i] for i in np.flatnonzero(enabled)]), device=device)
            residual = model(base, last, normalized, tokens).permute(0, 1, 3, 4, 2).cpu().numpy()
            for local, index in enumerate(np.flatnonzero(enabled)):
                output[index] = apply_direct_residual(baseline[index], residual[local])
        predictions.append(output)
        context = np.concatenate((context, output), axis=1)[:, -5:]
        history = np.concatenate((history, future), axis=1)[:, -4:]
    return np.concatenate(predictions, axis=1)


def recursive_metrics(prediction: np.ndarray, rows: list[dict]) -> dict:
    target = np.stack([row["target_frames"] for row in rows]).astype(np.float64)
    absolute = np.abs(prediction.astype(np.float64) - target)
    return {
        "overall_rgb_mae": float(absolute.mean()),
        "chunk3_rgb_mae": float(absolute[:, 16:24].mean()),
        "chunk4_rgb_mae": float(absolute[:, 24:32].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("windows", "split", "preregistration", "v169-release", "v169-library", "checkpoint", "report"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.checkpoint.exists() or args.report.exists():
        raise FileExistsError("v444 checkpoint/report already exists")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v444-direct-residual-preregistration-v1" or prereg.get("seed") != SEED:
        raise RuntimeError("wrong v444 preregistration")
    split = json.loads(args.split.read_text())
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    right = prereg["data"]["right_train_episodes"]
    fit_episodes = prereg["data"]["fit_episodes"]
    kill_episodes = prereg["data"]["kill_episodes"]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    rows = select_close_rows(WindowDataset(args.windows, right, rollout_horizon=8), right, prompts)
    fit_rows = [row for row in rows if row["episode"] in fit_episodes]
    kill_rows = [row for row in rows if row["episode"] in kill_episodes]
    if len(fit_rows) != 96 or len(kill_rows) != 24:
        raise RuntimeError("v444 fit/kill window contract drift")
    v169 = Track2V169ArmRoutedRuntime(args.v169_release, args.v169_library, args.device)
    cache_v169(rows, v169)
    mean, std = action_statistics(fit_rows)
    model = DirectResidualHead(channels=16).to(device)
    model.eval()
    init_metrics, init_episode = evaluate_rows(model, kill_rows, mean, std, device)
    recursive_rows = selected_recursive_rows(args.windows, kill_episodes, prompts)
    init_recursive = recursive_metrics(recursive32(model, recursive_rows, v169, mean, std, device), recursive_rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(SEED)
    order = np.asarray([], dtype=np.int64)
    losses = []
    model.train()
    for step in range(1, STEPS + 1):
        if len(order) < BATCH_SIZE:
            order = np.concatenate((order, rng.permutation(len(fit_rows))))
        indices, order = order[:BATCH_SIZE], order[BATCH_SIZE:]
        baseline, context, actions, tokens, target_residual = tensors(fit_rows, indices, mean, std, device)
        prediction = model(baseline, context, actions, tokens)
        loss = F.l1_loss(prediction[:, 2:6], target_residual[:, 2:6])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    step_metrics, step_episode = evaluate_rows(model, kill_rows, mean, std, device)
    step_recursive = recursive_metrics(recursive32(model, recursive_rows, v169, mean, std, device), recursive_rows)
    residual_ratio = step_metrics["residual_mae"] / init_metrics["residual_mae"]
    rgb_ratio = step_metrics["rgb_mae"] / init_metrics["rgb_mae"]
    improved_episodes = sum(step_episode[e] < init_episode[e] for e in kill_episodes)
    recursive_ratios = {key: step_recursive[key] / init_recursive[key] for key in init_recursive}
    passed = bool(
        residual_ratio <= 0.99 and rgb_ratio <= 0.99 and improved_episodes == 3
        and recursive_ratios["overall_rgb_mae"] <= 1.0
        and recursive_ratios["chunk3_rgb_mae"] <= 1.0
        and recursive_ratios["chunk4_rgb_mae"] <= 1.0
    )
    checkpoint = {
        "format": "strict-track2-v444-direct-residual-checkpoint-v1", "step": STEPS,
        "channels": 16, "model": model.state_dict(), "action_mean": mean, "action_std": std,
        "fit_episodes": fit_episodes, "kill_episodes": kill_episodes,
        "preregistration_sha256": sha256(args.preregistration),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {
        "format": "strict-track2-v444-step25-kill-report-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "step": STEPS, "fit_windows": len(fit_rows), "kill_windows": len(kill_rows),
        "fit_episodes": fit_episodes, "kill_episodes": kill_episodes,
        "init": init_metrics, "step25": step_metrics,
        "ratios": {"residual_mae": residual_ratio, "rgb_mae": rgb_ratio, **recursive_ratios},
        "kill_episode_rgb_mae_init": init_episode, "kill_episode_rgb_mae_step25": step_episode,
        "kill_episode_rgb_improved": improved_episodes,
        "recursive32_init": init_recursive, "recursive32_step25": step_recursive,
        "training_loss": {"first": losses[0], "last": losses[-1], "min": min(losses)},
        "sha256": {"checkpoint": sha256(args.checkpoint), "preregistration": sha256(args.preregistration), "split": sha256(args.split)},
        "guards": {
            "development_or_final_used": False, "reward_or_outcome_used": False,
            "policy_updates": 0, "real_submission": False, "rl_authorized": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
