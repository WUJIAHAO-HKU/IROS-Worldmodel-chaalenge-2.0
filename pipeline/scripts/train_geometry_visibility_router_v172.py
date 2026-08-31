#!/usr/bin/env python3
"""Train the v17.2 observable hard router on the supplied 50 episodes only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.geometry_visibility_router_v172 import (
    GeometryVisibilityRouterV172, build_candidates, hard_render, parameter_count,
    routing_targets,
)
from wam_pipeline.object_geometry_v170 import ActionPoseProjector, normalize_action
from train_contact_occlusion_head_v131 import active_arm


def episode(name: str) -> int:
    match = re.match(r"episode(\d+)_", name)
    if match is None: raise ValueError(name)
    return int(match.group(1))


def load_pose(path: str | Path, device: torch.device) -> dict:
    value = torch.load(path, map_location=device, weights_only=False)
    model = ActionPoseProjector(int(value["hidden"])).to(device)
    model.load_state_dict(value["model"]); model.eval()
    value["runtime_model"] = model
    for key in ("action_mean", "action_std", "target_mean", "target_std"):
        value[f"runtime_{key}"] = torch.from_numpy(value[key]).to(device)
    return value


@torch.inference_mode()
def predict_pose(checkpoint: dict, action: np.ndarray, arm: int,
                 device: torch.device) -> np.ndarray:
    value = torch.from_numpy(action.astype(np.float32)).to(device)
    arms = torch.full((len(value),), arm, dtype=torch.long, device=device)
    normalized = normalize_action(value, checkpoint["runtime_action_mean"],
                                  checkpoint["runtime_action_std"], arms)
    result = checkpoint["runtime_model"](normalized, arms)
    result = result * checkpoint["runtime_target_std"][arms] + checkpoint["runtime_target_mean"][arms]
    return result.float().cpu().numpy()


def precompute_geometry(windows: Path, names: np.ndarray, pose: dict,
                        device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, future, arms = [], [], []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as value:
            arm, action = active_arm(value["history_actions"], value["future_actions"])
            history = value["history_actions"][:, arm * 7:(arm + 1) * 7]
        source_action = np.concatenate((history[:1], history), 0)[:5]
        source.append(predict_pose(pose, source_action, arm, device))
        future.append(predict_pose(pose, action, arm, device)); arms.append(arm)
        if (index + 1) % 200 == 0:
            print(json.dumps({"geometry": index + 1, "total": len(names)}), flush=True)
    return np.stack(source), np.stack(future), np.asarray(arms, np.int64)


def make_example(index: int, time: int, parent: np.ndarray, target: np.ndarray,
                 context: np.ndarray, source_geometry: np.ndarray,
                 future_geometry: np.ndarray, arms: np.ndarray):
    candidate, support, condition = build_candidates(
        context[index], parent[index, time], source_geometry[index],
        future_geometry[index, time], int(arms[index]))
    return (parent[index, time], target[index, time], candidate, support, condition,
            time, int(arms[index]))


def tensors(batch: list[tuple], device: torch.device):
    parents = torch.from_numpy(np.stack([v[0] for v in batch]).transpose(0, 3, 1, 2).copy()).to(device).float() / 255
    targets = torch.from_numpy(np.stack([v[1] for v in batch]).transpose(0, 3, 1, 2).copy()).to(device).float() / 255
    candidates = torch.from_numpy(np.stack([v[2] for v in batch]).transpose(0, 1, 4, 2, 3).copy()).to(device).float() / 255
    supports = torch.from_numpy(np.stack([v[3] for v in batch]).copy()).to(device).float()
    conditions = torch.from_numpy(np.stack([v[4] for v in batch])).to(device).float()
    horizons = torch.tensor([(v[5] + 1) / 8 for v in batch], device=device).float()[:, None]
    arms = torch.tensor([v[6] for v in batch], device=device).float()[:, None]
    return parents, targets, candidates, supports, conditions, horizons, arms


def metric_state() -> dict:
    return {"parent": 0.0, "routed": 0.0, "oracle": 0.0, "count": 0,
            "selected": 0, "focus": 0, "positive": 0, "correct": 0}


@torch.inference_mode()
def evaluate(model, indices: list[int], parent: np.ndarray, target: np.ndarray,
             context: np.ndarray, source_geometry: np.ndarray, future_geometry: np.ndarray,
             arms: np.ndarray, device: torch.device, threshold: float,
             maximum_windows: int = 0) -> dict:
    model.eval(); state = metric_state(); by_arm = {0: metric_state(), 1: metric_state()}
    by_horizon = [metric_state() for _ in range(8)]
    chosen = indices
    if maximum_windows and len(chosen) > maximum_windows:
        positions = np.linspace(0, len(chosen) - 1, maximum_windows, dtype=int)
        chosen = [chosen[p] for p in positions]
    for order, index in enumerate(chosen):
        for time in range(8):
            batch = [make_example(index, time, parent, target, context, source_geometry,
                                  future_geometry, arms)]
            p, t, c, s, condition, horizon, arm = tensors(batch, device)
            score = model(p, c, s, condition, horizon, arm)
            output, choice = hard_render(p, c, s, score, threshold)
            label, focus, gain = routing_targets(p, c, s, t)
            valid = s.max(2).values > .5
            oracle_index = gain.argmax(1)
            oracle_candidate = c.gather(1, oracle_index[:, None, None].expand(-1, 1, 3, 256, 256)).squeeze(1)
            oracle = torch.where((gain.max(1).values > .25 / 255)[:, None], oracle_candidate, p)
            for local in (state, by_arm[int(arms[index])], by_horizon[time]):
                local["parent"] += float((p - t).abs().sum())
                local["routed"] += float((output - t).abs().sum())
                local["oracle"] += float((oracle - t).abs().sum())
                local["count"] += t.numel()
                local["selected"] += int((choice > 0).sum()); local["focus"] += int(focus.sum())
                local["positive"] += int((label > 0).sum())
                local["correct"] += int(((choice == label) & focus).sum())
        if (order + 1) % 16 == 0:
            print(json.dumps({"eval": order + 1, "total": len(chosen)}), flush=True)

    def finish(value: dict) -> dict:
        parent_mae = 255 * value["parent"] / max(value["count"], 1)
        routed_mae = 255 * value["routed"] / max(value["count"], 1)
        return {"parent_rgb_mae": parent_mae, "routed_rgb_mae": routed_mae,
                "oracle_rgb_mae": 255 * value["oracle"] / max(value["count"], 1),
                "improvement_percent": 100 * (parent_mae - routed_mae) / max(parent_mae, 1e-9),
                "selected_fraction": value["selected"] / max(value["focus"], 1),
                "positive_fraction": value["positive"] / max(value["focus"], 1),
                "focused_choice_accuracy": value["correct"] / max(value["focus"], 1)}
    return {"all": finish(state), "arms": {f"arm{k}": finish(v) for k, v in by_arm.items()},
            "horizons": [finish(value) for value in by_horizon], "windows": len(chosen)}


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--pose-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--dev-episodes", default="36,47"); parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--validation-interval", type=int, default=300)
    parser.add_argument("--validation-windows", type=int, default=16); parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(1)
    device = torch.device(args.device); windows = Path(args.windows)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; target = cache["target"]; context = cache["context"]
        names = cache["windows"].astype(str)
    pose = load_pose(args.pose_checkpoint, device)
    source_geometry, future_geometry, arms = precompute_geometry(windows, names, pose, device)
    dev_episodes = {int(v) for v in args.dev_episodes.split(",") if v}
    train_indices = [i for i, name in enumerate(names) if episode(name) not in dev_episodes]
    dev_indices = [i for i, name in enumerate(names) if episode(name) in dev_episodes]
    arm_indices = {arm: [i for i in train_indices if arms[i] == arm] for arm in (0, 1)}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model = GeometryVisibilityRouterV172(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = np.random.default_rng(args.seed); history = []; best = -1e9
    executor = ThreadPoolExecutor(max_workers=max(args.batch_size, 1))
    for step in range(1, args.steps + 1):
        # Equal arm sampling is mandatory; horizons are uniformly sampled.
        chosen = [int(generator.choice(arm_indices[(step + offset) % 2])) for offset in range(args.batch_size)]
        times = generator.integers(0, 8, len(chosen)).tolist()
        futures = [executor.submit(make_example, index, time, parent, target, context,
                                   source_geometry, future_geometry, arms)
                   for index, time in zip(chosen, times)]
        batch = [value.result() for value in futures]
        p, t, c, s, condition, horizon, arm = tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scores = model(p, c, s, condition, horizon, arm)[:, :, 0]
            labels, focus, gain = routing_targets(p, c, s, t)
            logits = torch.cat((torch.zeros_like(scores[:, :1]), scores), 1)
            pixel_loss = F.cross_entropy(logits, labels, reduction="none")
            weight = 1 + 4 * (labels > 0).float()
            classification = (pixel_loss * weight * focus).sum() / torch.clamp((weight * focus).sum(), 1)
            valid = s.max(2).values > .5
            regression_target = (gain * 64).clamp(-4, 4)
            regression = F.smooth_l1_loss(scores[valid], regression_target[valid])
            loss = classification + .20 * regression
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              "classification": float(classification), "regression": float(regression),
                              "positive_fraction": float((labels[focus] > 0).float().mean())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(model, dev_indices, parent, target, context, source_geometry,
                               future_geometry, arms, device, args.threshold, args.validation_windows)
            record = {"step": step, "metrics": metrics}; history.append(record); print(json.dumps(record), flush=True)
            # Require both arms to improve; optimize the worst-arm improvement.
            score = min(v["improvement_percent"] for v in metrics["arms"].values())
            checkpoint = {"format": "track2-geometry-visibility-router-v17.2", "step": step,
                          "state_dict": model.state_dict(), "base_channels": args.base_channels,
                          "threshold": args.threshold, "metrics": metrics,
                          "pose_checkpoint": args.pose_checkpoint, "dev_episodes": sorted(dev_episodes)}
            atomic_save(checkpoint, output / "latest.pt")
            if score > best: best = score; atomic_save(checkpoint, output / "best.pt")
    executor.shutdown()
    manifest = {"format": "track2-geometry-visibility-router-v17.2-training",
                "data_boundary": "supplied_50_episodes_only", "train_windows": len(train_indices),
                "dev_windows": len(dev_indices), "train_arm_counts": {str(k): len(v) for k, v in arm_indices.items()},
                "dev_arm_counts": np.bincount(arms[dev_indices], minlength=2).tolist(),
                "steps": args.steps, "batch_size": args.batch_size,
                "parameter_count": parameter_count(args.base_channels), "best_worst_arm_improvement_percent": best,
                "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
