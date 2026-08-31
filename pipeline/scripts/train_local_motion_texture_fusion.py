#!/usr/bin/env python3
"""Train a small observable-input fusion model while both parent models stay frozen."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from wam_pipeline.local_fusion import LocalMotionTextureFusion


class FusionDataset(Dataset):
    def __init__(self, windows: Path, first: np.ndarray, second: np.ndarray, names: list[str], indices: list[int]):
        self.windows, self.first, self.second, self.names, self.indices = windows, first, second, names, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        with np.load(self.windows / self.names[index], allow_pickle=False) as window:
            context_last = window["context_frames"][-1:].copy()
            future_actions = window["future_actions"].copy()
            target = window["target_frames"].copy()
        return self.first[index], self.second[index], context_last, future_actions, target


def load_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"prediction", "windows"}:
            raise ValueError(f"invalid prediction cache: {path}")
        return cache["prediction"], [str(name) for name in cache["windows"]]


def episode(name: str) -> int:
    match = re.fullmatch(r"episode(\d+)_\d+\.npz", name)
    if match is None:
        raise ValueError(f"invalid window name: {name}")
    return int(match.group(1))


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    filtered = flat - functional.avg_pool2d(flat, 3, stride=1, padding=1, count_include_pad=False)
    return filtered.unflatten(0, value.shape[:2])


def fusion_loss(prediction, target, context_last, autoregressive, alpha, residual, loss_profile: str = "balanced-v1", second_parent=None, oracle_routing_weight: float = 0.0, oracle_routing_temperature: float = 0.01) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    previous_target = torch.cat([context_last, target[:, :-1]], dim=1)
    previous_prediction = torch.cat([context_last, prediction[:, :-1]], dim=1)
    target_motion = (target - previous_target).abs().mean(dim=2, keepdim=True)
    moving = target_motion >= 0.03
    motion_weights = 1.0 + 1.5 * moving.to(target.dtype)
    horizon = torch.arange(1, 9, device=target.device, dtype=target.dtype)
    horizon = (horizon / horizon.mean()).view(1, 8, 1, 1, 1)
    error = prediction - target
    pixel = (horizon * motion_weights * error.abs()).mean() + 0.05 * (horizon * motion_weights * error.square()).mean()
    coarse = functional.l1_loss(functional.avg_pool2d(prediction.flatten(0, 1), 2), functional.avg_pool2d(target.flatten(0, 1), 2))
    edge_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    edge_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    edge = 0.5 * (edge_x.mean() + edge_y.mean())
    delta_difference = (prediction - previous_prediction) - (target - previous_target)
    temporal = functional.l1_loss(functional.avg_pool2d(delta_difference.flatten(0, 1), 4), torch.zeros_like(functional.avg_pool2d(delta_difference.flatten(0, 1), 4)))
    texture_error = (highpass(prediction) - highpass(target)).abs()
    texture = texture_error.mean()
    target_dark = torch.sigmoid((0.38 - target.mean(dim=2, keepdim=True)) * 24.0)
    prediction_dark = torch.sigmoid((0.38 - prediction.mean(dim=2, keepdim=True)) * 24.0)
    dark_structure = (horizon * (prediction_dark - target_dark).abs()).mean()
    background = (~moving).to(target.dtype)
    preservation = (background * (prediction - autoregressive).abs()).sum() / (background.sum() * 3.0).clamp_min(1.0)
    gate_sparsity = (background * (1.0 - alpha)).sum() / background.sum().clamp_min(1.0)
    residual_penalty = residual.abs().mean()
    routing = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    oracle_second_fraction = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if oracle_routing_weight > 0:
        if second_parent is None:
            raise ValueError("oracle routing requires the second parent")
        with torch.no_grad():
            first_score = (autoregressive - target).abs().mean(dim=2, keepdim=True)
            second_score = (second_parent - target).abs().mean(dim=2, keepdim=True)
            first_score = first_score + 0.25 * (highpass(autoregressive) - highpass(target)).abs().mean(dim=2, keepdim=True)
            second_score = second_score + 0.25 * (highpass(second_parent) - highpass(target)).abs().mean(dim=2, keepdim=True)
            oracle_alpha = torch.sigmoid((second_score - first_score) / oracle_routing_temperature)
            oracle_second_fraction = (oracle_alpha < 0.5).to(target.dtype).mean()
        safe_alpha = alpha.float().clamp(1e-5, 1.0 - 1e-5)
        routing = -(
            oracle_alpha.float() * safe_alpha.log()
            + (1.0 - oracle_alpha.float()) * (1.0 - safe_alpha).log()
        )
        routing = (motion_weights.float() * routing).mean()
    if loss_profile == "structure-v2":
        # Preserve dark robot geometry and high-frequency markings. Moving and
        # late-horizon pixels receive the same emphasis as the RGB objective,
        # instead of allowing L1 averaging to dissolve thin black structures.
        weighted_texture = (horizon * motion_weights * texture_error).mean()
        weighted_edge = 0.5 * (
            (horizon * motion_weights[..., 1:] * edge_x).mean()
            + (horizon * motion_weights[..., 1:, :] * edge_y).mean()
        )
        total = (
            pixel
            + 0.08 * coarse
            + 0.18 * weighted_edge
            + 0.08 * temporal
            + 0.18 * weighted_texture
            + 0.12 * dark_structure
            + 0.04 * preservation
            + 0.003 * gate_sparsity
            + 0.003 * residual_penalty
            + oracle_routing_weight * routing
        )
    elif loss_profile == "balanced-v1":
        total = pixel + 0.10 * coarse + 0.05 * edge + 0.04 * temporal + 0.04 * texture + 0.08 * preservation + 0.005 * gate_sparsity + 0.01 * residual_penalty + oracle_routing_weight * routing
    else:
        raise ValueError(f"unsupported loss profile: {loss_profile}")
    return total, {"pixel": pixel, "temporal": temporal, "texture": texture, "dark_structure": dark_structure, "routing": routing, "oracle_second_fraction": oracle_second_fraction, "preservation": preservation, "alpha_mean": alpha.mean(), "residual_abs": residual_penalty}


@torch.inference_mode()
def evaluate(loader, model, device, action_mean, action_std) -> dict[str, dict[str, float]]:
    sums = {model_name: {name: 0.0 for name in ("rgb", "high_rgb", "delta", "moving_rgb", "moving_delta", "moving_laplacian", "motion_magnitude")} for model_name in ("autoregressive", "fusion")}
    counts = {name: 0.0 for name in ("rgb", "high_rgb", "delta", "moving", "motion_magnitude")}
    alpha_sum = residual_sum = sample_count = 0.0
    model.eval()
    for first, second, context, actions, target in loader:
        first, second, context, target = (frames(value, device) for value in (first, second, context, target))
        actions = ((actions.to(device, non_blocking=True).float() - action_mean) / action_std)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, alpha, residual = model(context, first, second, actions)
        previous_target = torch.cat([context, target[:, :-1]], dim=1)
        target_delta = target - previous_target
        moving = target_delta.abs().mean(dim=2, keepdim=True) >= 0.03
        high = target_delta.abs().mean(dim=(1, 2, 3, 4)) >= 0.04
        moving_channels = moving.expand(-1, -1, 3, -1, -1)
        for name, value in (("autoregressive", first), ("fusion", prediction.float())):
            previous = torch.cat([context, value[:, :-1]], dim=1)
            error = (value - target).abs()
            delta_error = ((value - previous) - target_delta).abs()
            sums[name]["rgb"] += float(error.sum())
            sums[name]["high_rgb"] += float(error[high].sum())
            sums[name]["delta"] += float(delta_error.sum())
            sums[name]["moving_rgb"] += float(error[moving_channels].sum())
            sums[name]["moving_delta"] += float(delta_error[moving_channels].sum())
            lap_error = (highpass(value) - highpass(target)).abs()
            sums[name]["moving_laplacian"] += float(lap_error[moving_channels].sum())
            predicted_motion = (value - previous).abs().mean(dim=2)
            target_motion = target_delta.abs().mean(dim=2)
            sums[name]["motion_magnitude"] += float((predicted_motion - target_motion).abs().sum())
        counts["rgb"] += target.numel()
        counts["high_rgb"] += int(high.sum()) * int(np.prod(target.shape[1:]))
        counts["delta"] += target.numel()
        counts["moving"] += int(moving.sum()) * 3
        counts["motion_magnitude"] += int(np.prod(target.shape[:2] + target.shape[-2:]))
        alpha_sum += float(alpha.float().sum())
        residual_sum += float(residual.float().abs().sum())
        sample_count += alpha.numel()
    result = {}
    for name in sums:
        result[name] = {
            "rgb_mae": sums[name]["rgb"] / counts["rgb"],
            "high_motion_rgb_mae": sums[name]["high_rgb"] / max(counts["high_rgb"], 1.0),
            "temporal_delta_mae": sums[name]["delta"] / counts["delta"],
            "moving_region_rgb_mae": sums[name]["moving_rgb"] / max(counts["moving"], 1.0),
            "moving_region_temporal_delta_mae": sums[name]["moving_delta"] / max(counts["moving"], 1.0),
            "moving_region_laplacian_mae": sums[name]["moving_laplacian"] / max(counts["moving"], 1.0),
            "motion_magnitude_mae": sums[name]["motion_magnitude"] / counts["motion_magnitude"],
        }
    result["fusion"]["alpha_mean"] = alpha_sum / sample_count
    result["fusion"]["residual_abs_mean"] = residual_sum / (sample_count * 3.0)
    return result


def selection_metric(result: dict[str, dict[str, float]]) -> float:
    baseline, candidate = result["autoregressive"], result["fusion"]
    weights = {"rgb_mae": 0.30, "high_motion_rgb_mae": 0.15, "temporal_delta_mae": 0.15, "moving_region_rgb_mae": 0.10, "moving_region_temporal_delta_mae": 0.10, "moving_region_laplacian_mae": 0.15, "motion_magnitude_mae": 0.05}
    score = sum(weight * candidate[name] / baseline[name] for name, weight in weights.items())
    for name in ("rgb_mae", "high_motion_rgb_mae"):
        score += 10.0 * max(candidate[name] / baseline[name] - 1.002, 0.0)
    return float(score)


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(output: Path, model, manifest: dict, normalization_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"format": "track2-local-motion-texture-fusion-v1", "state_dict": model.state_dict()}, output / "model.pt")
    with (output / "local_fusion_config.npz.tmp").open("wb") as handle:
        np.savez(handle, base_channels=model.base_channels, residual_scale=model.residual_scale, prediction_frames=8, action_dim=14)
    os.replace(output / "local_fusion_config.npz.tmp", output / "local_fusion_config.npz")
    shutil.copy2(normalization_path, output / "action_normalization.npz")
    temporary = output / f"training_manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "training_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--autoregressive-cache", required=True)
    parser.add_argument("--direct-flow-cache", required=True)
    parser.add_argument("--action-normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--dev-episode-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-checkpoint", help="Optional compatible fusion checkpoint used to initialize a new run.")
    parser.add_argument("--loss-profile", choices=("balanced-v1", "structure-v2"), default="balanced-v1")
    parser.add_argument("--residual-scale", type=float, default=0.05, help="Maximum learned RGB correction; set to zero for a pure convex router.")
    parser.add_argument("--oracle-routing-weight", type=float, default=0.0)
    parser.add_argument("--oracle-routing-temperature", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    first, names = load_cache(Path(args.autoregressive_cache))
    second, second_names = load_cache(Path(args.direct_flow_cache))
    if names != second_names or first.shape != second.shape:
        raise ValueError("parent prediction caches are not aligned")
    split = json.loads(Path(args.split_manifest).read_text())
    rng = np.random.default_rng(args.seed)
    dev_episodes = sorted(rng.choice(split["train_episodes"], args.dev_episode_count, replace=False).tolist())
    train_indices = [index for index, name in enumerate(names) if episode(name) not in dev_episodes]
    dev_indices = [index for index, name in enumerate(names) if episode(name) in dev_episodes]
    if not train_indices or not dev_indices:
        raise RuntimeError("fusion train/dev split is empty")
    train_dataset = FusionDataset(Path(args.windows), first, second, names, train_indices)
    dev_dataset = FusionDataset(Path(args.windows), first, second, names, dev_indices)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=2, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    normalization_path = Path(args.action_normalization)
    normalization = np.load(normalization_path, allow_pickle=False)
    device = torch.device(args.device)
    action_mean = torch.from_numpy(np.asarray(normalization["mean"], dtype=np.float32)).to(device)
    action_std = torch.from_numpy(np.asarray(normalization["std"], dtype=np.float32)).to(device)
    torch.manual_seed(args.seed)
    model = LocalMotionTextureFusion(residual_scale=args.residual_scale).to(device)
    if args.initial_checkpoint and not args.resume:
        initial_state = torch.load(Path(args.initial_checkpoint) / "model.pt", map_location="cpu", weights_only=True)
        if initial_state.get("format") != "track2-local-motion-texture-fusion-v1":
            raise ValueError("unsupported initial local fusion checkpoint")
        model.load_state_dict(initial_state["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    state_path = output / "training_state.pt"
    start_step, best_metric, history = 0, float("inf"), []
    config = {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "validation_interval": args.validation_interval, "seed": args.seed, "dev_episodes": dev_episodes, "train_sample_count": len(train_indices), "dev_sample_count": len(dev_indices), "loss_profile": args.loss_profile, "residual_scale": args.residual_scale, "oracle_routing_weight": args.oracle_routing_weight, "oracle_routing_temperature": args.oracle_routing_temperature, "initial_checkpoint": str(Path(args.initial_checkpoint).resolve()) if args.initial_checkpoint else None}
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != "track2-local-fusion-training-state-v1" or state.get("config") != config:
            raise ValueError("local fusion resume state does not match")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_metric, history = int(state["step"]), float(state["best_metric"]), list(state["history"])
    baseline_result = evaluate(dev_loader, model, device, action_mean, action_std)
    baseline_metric = selection_metric(baseline_result)
    if not history:
        history.append({"step": 0, "selection_metric": baseline_metric, **baseline_result})
        best_metric = baseline_metric
        manifest = {"format": "track2-local-motion-texture-fusion-v1", **config, "checkpoint_step": 0, "validation": history, "best_selection_metric": best_metric, "autoregressive_cache": str(Path(args.autoregressive_cache).resolve()), "direct_flow_cache": str(Path(args.direct_flow_cache).resolve())}
        save_checkpoint(output / "best", model, manifest, normalization_path)
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        first_batch, second_batch, context, actions, target = batch
        first_batch, second_batch, context, target = (frames(value, device) for value in (first_batch, second_batch, context, target))
        actions = ((actions.to(device, non_blocking=True).float() - action_mean) / action_std)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, alpha, residual = model(context, first_batch, second_batch, actions)
            loss, components = fusion_loss(prediction, target, context, first_batch, alpha, residual, args.loss_profile, second_batch, args.oracle_routing_weight, args.oracle_routing_temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), **{name: float(value.detach()) for name, value in components.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(dev_loader, model, device, action_mean, action_std)
            metric = selection_metric(result)
            record = {"step": step, "selection_metric": metric, **result}
            history.append(record)
            manifest = {"format": "track2-local-motion-texture-fusion-v1", **config, "checkpoint_step": step, "validation": history, "best_selection_metric": min(best_metric, metric), "autoregressive_cache": str(Path(args.autoregressive_cache).resolve()), "direct_flow_cache": str(Path(args.direct_flow_cache).resolve())}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, manifest, normalization_path)
            if metric < best_metric:
                best_metric = metric
                manifest["best_checkpoint_step"] = step
                save_checkpoint(output / "best", model, manifest, normalization_path)
            print(json.dumps(record), flush=True)
            model.train()
        if step % args.checkpoint_interval == 0 or step == args.steps:
            atomic_torch_save({"format": "track2-local-fusion-training-state-v1", "config": config, "step": step, "best_metric": best_metric, "history": history, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, state_path)


if __name__ == "__main__":
    main()
