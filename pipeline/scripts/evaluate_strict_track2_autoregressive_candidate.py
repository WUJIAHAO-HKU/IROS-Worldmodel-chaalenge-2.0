#!/usr/bin/env python3
"""Paired, arm-aware evaluation for an autoregressive Track 2 parent candidate."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.autoregressive_unet import OneStepActionUNet


CONTACT_REGION = (72, 232, 30, 210)


def episode_id(path: Path) -> int:
    match = re.match(r"episode(\d+)_", path.name)
    if not match:
        raise ValueError(f"invalid window filename: {path.name}")
    return int(match.group(1))


def active_arm(history: np.ndarray, future: np.ndarray) -> int:
    actions = np.concatenate((history, future))
    delta = np.abs(np.diff(actions, axis=0))
    return int(delta[:, 7:].mean() > delta[:, :7].mean())


class Windows(Dataset):
    def __init__(
        self,
        root: Path,
        episodes: list[int],
        min_start: int | None = None,
        max_start: int | None = None,
    ) -> None:
        allowed = set(episodes)
        self.paths = []
        self.starts = []
        for path in sorted(root.glob("episode*_*.npz")):
            if episode_id(path) not in allowed:
                continue
            with np.load(path, allow_pickle=False) as data:
                start = int(data["start"]) if "start" in data.files else int(path.stem.rsplit("_", 1)[1])
            if min_start is not None and start < min_start:
                continue
            if max_start is not None and start > max_start:
                continue
            self.paths.append(path)
            self.starts.append(start)
        if not self.paths:
            raise ValueError("no evaluation windows selected")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as data:
            context = data["context_frames"].copy()
            history = data["history_actions"].astype(np.float32).copy()
            future = data["future_actions"].astype(np.float32).copy()
            target = data["target_frames"].copy()
            arm = bool(data["arm_right"]) if "arm_right" in data.files else bool(active_arm(history, future))
            success = bool(data["capture_success"]) if "capture_success" in data.files else True
        return context, history, future, target, arm, success


def load_model(path: Path, device: torch.device):
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError(f"unsupported checkpoint: {path}")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    with np.load(path / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(values["mean"].astype(np.float32)).to(device)
        std = torch.from_numpy(values["std"].astype(np.float32)).to(device)
    return model, mean, std


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device).float().div(255)


def rollout(model, context, history, future):
    output = []
    for action in future.unbind(1):
        prediction = model(context, torch.cat((history, action[:, None]), 1)).clamp(0, 1)
        output.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), 1)
        history = torch.cat((history[:, 1:], action[:, None]), 1)
    return torch.stack(output, 1)


def highpass(value: torch.Tensor) -> torch.Tensor:
    shape = value.shape
    flat = value.flatten(0, 1)
    result = flat - F.avg_pool2d(flat, 3, 1, 1, count_include_pad=False)
    return result.unflatten(0, shape[:2])


def metric_rows(prediction: torch.Tensor, target: torch.Tensor, context_last: torch.Tensor) -> dict[str, torch.Tensor]:
    error = (prediction - target).abs().mean(2)
    previous_target = torch.cat((context_last[:, None], target[:, :-1]), 1)
    previous_prediction = torch.cat((context_last[:, None], prediction[:, :-1]), 1)
    target_motion = (target - previous_target).abs().mean(2, keepdim=True)
    motion_mask = F.max_pool2d(
        (target_motion > 3 / 255).flatten(0, 1).float(), 9, 1, 4
    ).unflatten(0, target_motion.shape[:2])
    static_mask = (target_motion <= 2 / 255).float()
    y0, y1, x0, x1 = CONTACT_REGION

    def masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.squeeze(2)
        return (values * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)

    return {
        "rgb_mae": 255 * error.mean((2, 3)),
        "motion_rgb_mae": 255 * masked(error, motion_mask),
        "static_rgb_mae": 255 * masked(error, static_mask),
        "texture_mae": 255 * (highpass(prediction) - highpass(target)).abs().mean((2, 3, 4)),
        "temporal_delta_mae": 255 * (
            (prediction - previous_prediction) - (target - previous_target)
        ).abs().mean((2, 3, 4)),
        "contact_rgb_mae": 255 * error[:, :, y0:y1, x0:x1].mean((2, 3)),
    }


class Accumulator:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, group: str, rows: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        self.counts[group] += int(mask.sum())
        for name, value in rows.items():
            self.values[group][name].append(value[mask.to(value.device)].cpu())

    def result(self) -> dict[str, object]:
        output: dict[str, object] = {}
        for group in sorted(self.counts):
            record: dict[str, object] = {"windows": self.counts[group]}
            for name, chunks in self.values[group].items():
                value = torch.cat(chunks)
                record[name] = float(value.mean())
                record[f"frame_{name}"] = [float(item) for item in value.mean(0)]
            output[group] = record
        return output


def gains(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for group in sorted(set(baseline).intersection(candidate)):
        before = baseline[group]
        after = candidate[group]
        record: dict[str, object] = {"windows": before["windows"]}
        for name, value in before.items():
            if name == "windows" or name.startswith("frame_"):
                continue
            base_value = float(value)
            candidate_value = float(after[name])
            record[f"{name}_improvement_percent"] = 100 * (base_value - candidate_value) / base_value
            base_frames = np.asarray(before[f"frame_{name}"], dtype=np.float64)
            candidate_frames = np.asarray(after[f"frame_{name}"], dtype=np.float64)
            record[f"frame_{name}_improvement_percent"] = (
                100 * (base_frames - candidate_frames) / base_frames
            ).tolist()
        result[group] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--episodes-key", default="validation_episodes")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    dataset: Dataset = Windows(
        Path(args.windows), split[args.episodes_key],
        min_start=args.min_start, max_start=args.max_start,
    )
    selected_indices = list(range(len(dataset)))
    if args.max_windows and args.max_windows < len(selected_indices):
        selected_indices = np.linspace(0, len(dataset) - 1, args.max_windows, dtype=int).tolist()
        dataset = Subset(dataset, selected_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    baseline_model, baseline_mean, baseline_std = load_model(Path(args.baseline), device)
    candidate_model, candidate_mean, candidate_std = load_model(Path(args.candidate), device)
    before, after = Accumulator(), Accumulator()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for raw_context, raw_history, raw_future, raw_target, arm_right, success in loader:
            context = frames(raw_context, device)
            target = frames(raw_target, device)
            raw_history = raw_history.to(device)
            raw_future = raw_future.to(device)
            baseline_prediction = rollout(
                baseline_model, context.clone(),
                (raw_history - baseline_mean) / baseline_std,
                (raw_future - baseline_mean) / baseline_std,
            )
            candidate_prediction = rollout(
                candidate_model, context.clone(),
                (raw_history - candidate_mean) / candidate_std,
                (raw_future - candidate_mean) / candidate_std,
            )
            baseline_rows = metric_rows(baseline_prediction.float(), target, context[:, -1])
            candidate_rows = metric_rows(candidate_prediction.float(), target, context[:, -1])
            arm_right = arm_right.bool()
            success = success.bool()
            masks = {
                "overall": torch.ones(len(arm_right), dtype=torch.bool),
                "left": ~arm_right,
                "right": arm_right,
                "capture_success": success,
                "capture_failure": ~success,
            }
            for group, mask in masks.items():
                before.add(group, baseline_rows, mask)
                after.add(group, candidate_rows, mask)

    baseline_result, candidate_result = before.result(), after.result()
    report = {
        "format": "strict-track2-paired-autoregressive-parent-evaluation-v1",
        "windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "episodes_key": args.episodes_key,
        "selected_window_count": len(dataset),
        "min_start": args.min_start,
        "max_start": args.max_start,
        "baseline_checkpoint": str(Path(args.baseline).resolve()),
        "candidate_checkpoint": str(Path(args.candidate).resolve()),
        "baseline": baseline_result,
        "candidate": candidate_result,
        "improvement": gains(baseline_result, candidate_result),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
