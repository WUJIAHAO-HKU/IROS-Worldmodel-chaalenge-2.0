#!/usr/bin/env python3
"""Train the v10 candidate-conditioned risk router with episode-disjoint dev data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.candidate_risk_router_v10 import CandidateRiskRouterV10
from wam_pipeline.candidate_risk_router_v101 import CandidateRiskRouterV101
from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet
from wam_pipeline.protected_layered_flow_v91 import ProtectedLayeredFlowV91


class RiskDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int]) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            return (
                self.parent[index], window["context_frames"].copy(), window["history_actions"].copy(),
                window["future_actions"].copy(), window["target_frames"].copy(), name,
            )


def frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255.0)


def pool_error(candidates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    shape = candidates.shape[:3]
    error = (candidates - target[:, :, None]).abs().mean(dim=3)
    pooled = F.avg_pool2d(error.flatten(0, 2)[:, None], 4, stride=4).squeeze(1)
    return pooled.unflatten(0, shape) * 255.0


def risk_loss(result, candidates, target, context, loss_weights):
    error = pool_error(candidates, target)
    target_advantage = error[:, :, :1] - error[:, :, 1:]
    scaled_target = target_advantage.clamp(-25, 25) / CandidateRiskRouterV10.advantage_scale
    mean = result["raw_mean"]
    uncertainty = result["raw_uncertainty"]
    previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
    motion = (target - previous).abs().mean(dim=2)
    moving = F.avg_pool2d(motion.flatten(0, 1)[:, None], 4, stride=4).squeeze(1)
    moving = moving.unflatten(0, target.shape[:2]) >= 0.03
    positive = target_advantage > 0.5
    weights = (1.0 + loss_weights["positive_emphasis"] * positive
               + loss_weights["moving_emphasis"] * moving[:, :, None])
    regression_error = F.smooth_l1_loss(mean, scaled_target, beta=0.2, reduction="none")
    regression = (weights * (regression_error / uncertainty + uncertainty.log())).sum() / weights.sum()
    uncertainty_target = (mean.detach() - scaled_target).abs().clamp(0.05, 5.0)
    uncertainty_calibration = F.smooth_l1_loss(uncertainty, uncertainty_target, beta=0.2)

    positive_count = positive.sum().clamp_min(1)
    negative_count = positive.numel() - positive.sum()
    positive_weight = (negative_count / positive_count).clamp(1, 20)
    classification_raw = F.binary_cross_entropy_with_logits(
        mean, positive.to(mean.dtype), pos_weight=positive_weight, reduction="none"
    )
    classification_weight = 1.0 + loss_weights["moving_emphasis"] * moving[:, :, None]
    classification = (classification_raw * classification_weight).sum() / classification_weight.expand_as(mean).sum()
    best_other_error, best_other = error[:, :, 1:].min(dim=2)
    choice = torch.where(best_other_error + 0.5 < error[:, :, 0], best_other + 1, 0)
    logits = torch.cat((torch.zeros_like(mean[:, :, :1]), mean), dim=2)
    ranking = F.cross_entropy(logits.flatten(0, 1), choice.flatten(0, 1), reduction="none")
    ranking_weight = 1.0 + loss_weights["moving_emphasis"] * moving
    ranking = (ranking.unflatten(0, target.shape[:2]) * ranking_weight).sum() / ranking_weight.sum()
    route_weights = torch.softmax(logits / 0.5, dim=2)
    expected = (route_weights * error).sum(dim=2).mean() / CandidateRiskRouterV10.advantage_scale
    total = (
        loss_weights["regression"] * regression
        + loss_weights["uncertainty"] * uncertainty_calibration
        + loss_weights["classification"] * classification
        + loss_weights["ranking"] * ranking
        + loss_weights["expected"] * expected
    )
    return total, {
        "regression": regression, "uncertainty_calibration": uncertainty_calibration,
        "classification": classification, "ranking": ranking,
        "expected": expected, "positive_fraction": positive.float().mean(),
        "target_advantage_mean": target_advantage.mean(),
        "predicted_uncertainty_mean": result["advantage_uncertainty"].mean(),
    }


def _auc(labels: list[np.ndarray], scores: list[np.ndarray]) -> float:
    label = np.concatenate(labels).astype(bool)
    score = np.concatenate(scores)
    positive, negative = int(label.sum()), int((~label).sum())
    if not positive or not negative:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_score = score[order]
    _, starts, counts = np.unique(sorted_score, return_index=True, return_counts=True)
    sorted_ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    for start, count in zip(starts, counts):
        if count > 1:
            sorted_ranks[start:start + count] = sorted_ranks[start:start + count].mean()
    ranks[order] = sorted_ranks
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


@torch.inference_mode()
def evaluate(loader, flow_model, router, device, action_mean, action_std, active_mean, active_std):
    risk_weights = (0.0, 0.1, 0.25, 0.5, 1.0)
    margins = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    settings = [(risk, margin) for risk in risk_weights for margin in margins]
    sums = {setting: 0.0 for setting in settings}
    moving_sums = {setting: 0.0 for setting in settings}
    selections = {setting: 0 for setting in settings}
    true_positives = {setting: 0 for setting in settings}
    baseline_sum = oracle_sum = moving_baseline = moving_oracle = 0.0
    total = moving_count = block_count = beneficial_count = 0
    auc_labels, auc_scores = [], []
    flow_model.eval(); router.eval()
    for parent, context, history, future, target, _ in loader:
        parent, context, target = (frames(value, device) for value in (parent, context, target))
        actions = torch.cat((history, future), dim=1).to(device).float()
        active, arm = flow_model.active_arm_actions(actions)
        flow = flow_model(context, (actions - action_mean) / action_std,
                          (active - active_mean) / active_std, arm, parent)
        candidates = flow["candidates"]
        risk = router(context, parent, candidates[:, :, 1:], flow["refined_flow"],
                      flow["visibility_logits"], active, arm)
        error = pool_error(candidates, target)
        target_advantage = error[:, :, :1] - error[:, :, 1:]
        oracle_choice = error.argmin(dim=2)
        full_oracle = oracle_choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
        oracle = candidates.gather(2, full_oracle[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
        previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
        moving = (target - previous).abs().mean(dim=2) >= 0.03
        base_error = (parent - target).abs(); oracle_error = (oracle - target).abs()
        baseline_sum += float(base_error.sum()); oracle_sum += float(oracle_error.sum())
        moving_baseline += float((base_error * moving[:, :, None]).sum())
        moving_oracle += float((oracle_error * moving[:, :, None]).sum())
        beneficial = target_advantage.max(dim=2).values > 0.5
        beneficial_count += int(beneficial.sum())
        sample_slice = slice(None, None, 32)
        auc_labels.append(beneficial.flatten()[sample_slice].cpu().numpy())
        auc_scores.append(risk["advantage_mean"].max(dim=2).values.flatten()[sample_slice].cpu().numpy())
        for setting in settings:
            choice, _ = router.safe_choice(risk["advantage_mean"], risk["advantage_uncertainty"], *setting)
            full = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            prediction = candidates.gather(2, full[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)).squeeze(2)
            prediction_error = (prediction - target).abs()
            sums[setting] += float(prediction_error.sum())
            moving_sums[setting] += float((prediction_error * moving[:, :, None]).sum())
            selected = choice != 0
            selections[setting] += int(selected.sum())
            selected_source = (choice - 1).clamp_min(0)
            realized = target_advantage.gather(2, selected_source[:, :, None]).squeeze(2)
            true_positives[setting] += int((selected & (realized > 0.5)).sum())
        total += target.numel(); moving_count += int(moving.sum()) * 3
        block_count += error.shape[0] * error.shape[1] * 64 * 64
    best = min(settings, key=lambda setting: sums[setting])
    safe_settings = [setting for setting in settings if moving_sums[setting] <= moving_baseline]
    safe_best = min(safe_settings, key=lambda setting: sums[setting]) if safe_settings else max(settings)
    baseline = 255 * baseline_sum / total
    router_mae = 255 * sums[best] / total
    selected = selections[best]
    return {
        "baseline_rgb_mae": baseline, "block4_oracle_rgb_mae": 255 * oracle_sum / total,
        "block4_oracle_improvement_percent": 100 * (baseline_sum - oracle_sum) / baseline_sum,
        "router_rgb_mae": router_mae, "router_improvement_percent": 100 * (baseline - router_mae) / baseline,
        "best_risk_weight": best[0], "best_margin": best[1],
        "moving_baseline_rgb_mae": 255 * moving_baseline / moving_count,
        "moving_oracle_rgb_mae": 255 * moving_oracle / moving_count,
        "moving_router_rgb_mae": 255 * moving_sums[best] / moving_count,
        "selected_transport_block_fraction": selected / block_count,
        "selected_precision": true_positives[best] / max(selected, 1),
        "beneficial_recall": true_positives[best] / max(beneficial_count, 1),
        "beneficial_block_auroc": _auc(auc_labels, auc_scores),
        "safe_risk_weight": safe_best[0], "safe_margin": safe_best[1],
        "safe_router_rgb_mae": 255 * sums[safe_best] / total,
        "safe_router_improvement_percent": 100 * (baseline - 255 * sums[safe_best] / total) / baseline,
        "safe_moving_router_rgb_mae": 255 * moving_sums[safe_best] / moving_count,
        "setting_rgb_mae": {f"risk={risk},margin={margin}": 255 * sums[(risk, margin)] / total
                            for risk, margin in settings},
        "setting_moving_rgb_mae": {f"risk={risk},margin={margin}": 255 * moving_sums[(risk, margin)] / moving_count
                                   for risk, margin in settings},
    }


def load_flow(base_checkpoint: Path, head_checkpoint: Path, device: torch.device):
    with np.load(base_checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False) as config:
        base = MultiSourceActionFlowUNet(int(config["base_channels"]))
    base_state = torch.load(base_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    base.load_state_dict(base_state["state_dict"], strict=True)
    head = torch.load(head_checkpoint, map_location="cpu", weights_only=True)
    model = ProtectedLayeredFlowV91(base, int(head["base_channels"]), float(head["max_residual_flow"]))
    incompatible = model.load_state_dict(head["state_dict"], strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("base_flow_model.")]
    if missing or incompatible.unexpected_keys:
        raise ValueError(f"invalid flow head: {missing} {incompatible.unexpected_keys}")
    return model.requires_grad_(False).to(device).eval(), head["active_mean"].to(device), head["active_std"].to(device)


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--router-version", choices=("v10", "v10.1"), default="v10")
    parser.add_argument("--init-router", help="Optional v10/v10.1 risk checkpoint used for initialization.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dev-episodes", type=int, default=8)
    parser.add_argument("--evaluation-samples", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.2)
    parser.add_argument("--classification-loss-weight", type=float, default=0.5)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--expected-loss-weight", type=float, default=0.25)
    parser.add_argument("--positive-emphasis", type=float, default=4.0)
    parser.add_argument("--moving-emphasis", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    episodes = sorted({name.split("_")[0] for name in names})
    generator = np.random.default_rng(args.seed)
    shuffled = [str(value) for value in generator.permutation(episodes)]
    dev_episodes = set(shuffled[: args.dev_episodes])
    train_indices = [index for index, name in enumerate(names) if name.split("_")[0] not in dev_episodes]
    dev_indices = [index for index, name in enumerate(names) if name.split("_")[0] in dev_episodes]
    if args.evaluation_samples and len(dev_indices) > args.evaluation_samples:
        positions = np.linspace(0, len(dev_indices) - 1, args.evaluation_samples, dtype=np.int64)
        dev_indices = [dev_indices[int(position)] for position in positions]
    train_dataset = RiskDataset(Path(args.windows), parent, names, train_indices)
    dev_dataset = RiskDataset(Path(args.windows), parent, names, dev_indices)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    base_checkpoint = Path(args.base_checkpoint)
    flow_model, active_mean, active_std = load_flow(base_checkpoint, Path(args.flow_head), device)
    with np.load(base_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)
    router_class = CandidateRiskRouterV101 if args.router_version == "v10.1" else CandidateRiskRouterV10
    router = router_class(args.base_channels)
    if args.init_router:
        initialization = torch.load(args.init_router, map_location="cpu", weights_only=False)
        initial_version = str(initialization.get("router_version", "v10"))
        if args.router_version == "v10.1" and initial_version == "v10":
            router.load_v10_state_dict(initialization["state_dict"])
        elif initial_version == args.router_version:
            router.load_state_dict(initialization["state_dict"], strict=True)
        else:
            raise ValueError(f"cannot initialize {args.router_version} from {initial_version}")
        print(json.dumps({"event": "loaded_router", "path": str(Path(args.init_router).resolve()),
                          "source_version": initial_version, "source_step": initialization.get("step")}), flush=True)
    router = router.to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    history = []
    loss_weights = {
        "regression": args.regression_loss_weight, "uncertainty": args.uncertainty_loss_weight,
        "classification": args.classification_loss_weight, "ranking": args.ranking_loss_weight,
        "expected": args.expected_loss_weight, "positive_emphasis": args.positive_emphasis,
        "moving_emphasis": args.moving_emphasis,
    }
    initial = evaluate(dev_loader, flow_model, router, device, action_mean, action_std, active_mean, active_std)
    history.append({"step": 0, **initial}); print(json.dumps(history[-1]), flush=True)
    best = initial["router_rgb_mae"]
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        router.train(); flow_model.eval()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        parent_batch, context, history_actions, future, target, _ = batch
        parent_batch, context, target = (frames(value, device) for value in (parent_batch, context, target))
        actions = torch.cat((history_actions, future), dim=1).to(device).float()
        active, arm = flow_model.active_arm_actions(actions)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            flow = flow_model(context, (actions - action_mean) / action_std,
                              (active - active_mean) / active_std, arm, parent_batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            risk = router(context, parent_batch, flow["candidates"][:, :, 1:], flow["refined_flow"],
                          flow["visibility_logits"], active, arm)
            loss, parts = risk_loss(risk, flow["candidates"], target, context, loss_weights)
        loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, flow_model, router, device, action_mean, action_std, active_mean, active_std)
            history.append({"step": step, **metrics}); print(json.dumps(history[-1]), flush=True)
            value = {
                "format": "track2-candidate-risk-router-v10", "router_version": args.router_version,
                "state_dict": router.state_dict(),
                "step": step, "metrics": metrics, "base_channels": args.base_channels,
                "flow_head": str(Path(args.flow_head).resolve()), "dev_episodes": sorted(dev_episodes),
            }
            atomic_save(value, output / "latest.pt")
            if metrics["router_rgb_mae"] < best:
                best = metrics["router_rgb_mae"]; atomic_save(value, output / "best.pt")
    manifest = {
        "format": "track2-candidate-risk-router-v10-training", "steps": args.steps,
        "router_version": args.router_version,
        "init_router": str(Path(args.init_router).resolve()) if args.init_router else None,
        "train_window_count": len(train_dataset), "dev_window_count": len(dev_dataset),
        "train_episode_count": len(episodes) - len(dev_episodes), "dev_episodes": sorted(dev_episodes),
        "flow_head": str(Path(args.flow_head).resolve()), "loss_weights": loss_weights, "history": history,
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
