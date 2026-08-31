#!/usr/bin/env python3
"""Pretrain the native-resolution AR parent on randomized adjust-bottle data.

External examples teach broad motion and occlusion.  Official examples remain
in every epoch and the checkpoint selector sees only episode-disjoint official
development windows, preventing source-domain quality from selecting a model
that damages the actual Track-2 camera domain.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from wam_pipeline.external_robotwin_data_v260 import ExternalRandomizedWindowDataset, available_external_episodes
from train_autoregressive_unet import (
    WindowDataset,
    evaluate,
    frames_for_model,
    load_autoregressive_initialization,
    reconstruction_loss,
    rollout,
    save_checkpoint,
)
from wam_pipeline.autoregressive_unet import OneStepActionUNet


def fixed_action_normalization(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
    if mean.shape != (14,) or std.shape != (14,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("action normalization must contain finite mean/std [14]")
    if np.any(std <= 0):
        raise ValueError("action normalization std must be positive")
    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)


def save_state(path: Path, model, optimizer, step: int, history: list[dict], config: dict, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "track2-external-randomized-parent-training-v26.0",
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "history": history,
            "config": config,
            "best_official_dev_mae": best,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--official-windows", required=True)
    parser.add_argument("--official-split", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--external-stride", type=int, default=4)
    parser.add_argument("--official-sample-fraction", type=float, default=0.20)
    parser.add_argument("--rollout-steps", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.external_stride, args.validation_interval, args.validation_windows) < 1:
        raise SystemExit("positive counts required")
    if not 0.0 < args.official_sample_fraction <= 1.0:
        raise SystemExit("--official-sample-fraction must be in (0,1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    split = json.loads(Path(args.official_split).read_text())
    external_ids = available_external_episodes(args.external_root)
    # The published repository has 500 randomized episodes.  Reserve the last
    # 50 as an external-domain monitor; model selection never uses this split.
    external_train_ids = set(external_ids[:-50] if len(external_ids) > 50 else external_ids[:-1])
    external_dev_ids = set(external_ids) - external_train_ids
    if not external_train_ids or not external_dev_ids:
        raise SystemExit("at least two complete external episodes are required")
    external_train = ExternalRandomizedWindowDataset(
        args.external_root, episodes=external_train_ids, stride=args.external_stride
    )
    external_dev = ExternalRandomizedWindowDataset(
        args.external_root, episodes=external_dev_ids, stride=max(args.external_stride, 8)
    )
    official_train = WindowDataset(Path(args.official_windows), split["train_episodes"])
    official_dev = WindowDataset(Path(args.official_windows), split["validation_episodes"])

    mixed = ConcatDataset([external_train, official_train])
    external_weight = (1.0 - args.official_sample_fraction) / len(external_train)
    official_weight = args.official_sample_fraction / len(official_train)
    weights = torch.cat(
        [torch.full((len(external_train),), external_weight), torch.full((len(official_train),), official_weight)]
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(mixed), args.steps * args.batch_size),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        mixed, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True
    )
    official_indices = np.linspace(
        0, len(official_dev) - 1, min(len(official_dev), args.validation_windows), dtype=np.int64
    ).tolist()
    external_indices = np.linspace(
        0, len(external_dev) - 1, min(len(external_dev), args.validation_windows), dtype=np.int64
    ).tolist()
    official_loader = DataLoader(Subset(official_dev, official_indices), batch_size=args.batch_size, num_workers=2)
    external_loader = DataLoader(Subset(external_dev, external_indices), batch_size=args.batch_size, num_workers=2)

    init = Path(args.init_checkpoint)
    mean, std = fixed_action_normalization(init / "action_normalization.npz", device)
    model = OneStepActionUNet().to(device)
    load_autoregressive_initialization(model, init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    config = {
        "external_root": str(Path(args.external_root).resolve()),
        "external_episode_count": len(external_ids),
        "external_train_episode_count": len(external_train_ids),
        "external_dev_episode_count": len(external_dev_ids),
        "external_train_windows": len(external_train),
        "official_train_windows": len(official_train),
        "official_dev_windows": len(official_dev),
        "official_sample_fraction": args.official_sample_fraction,
        "rollout_steps": args.rollout_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "init_checkpoint": str(init.resolve()),
        "data_policy": "external_randomized_pretraining_allowed",
    }
    history: list[dict] = []
    best = float("inf")
    start = 0
    state_path = output / "training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != "track2-external-randomized-parent-training-v26.0" or state.get("config") != config:
            raise SystemExit("resume state does not match this run")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        history = list(state["history"])
        best = float(state["best_official_dev_mae"])

    if start == 0:
        # Make the source parent an explicit candidate.  External pretraining
        # is allowed to change the domain, but it is never allowed to become
        # "best" merely because the selector started at infinity.
        official_result = evaluate(official_loader, model, device, mean, std, 0.04)
        external_result = evaluate(external_loader, model, device, mean, std, 0.04)
        model.train()
        history.append({"step": 0, "official_dev": official_result, "external_dev": external_result})
        best = float(official_result["rollout_mae"])
        baseline_metadata = {
            "backend": "autoregressive_unet",
            "format": "track2-autoregressive-unet-v1",
            "stage": "v26.0-external-randomized-pretraining",
            "checkpoint_step": 0,
            "selection_metric": "episode-disjoint official dev rollout MAE",
            "best_official_dev_mae": best,
            "validation": history,
            **config,
        }
        save_checkpoint(output / "best", model, mean, std, baseline_metadata)
        save_checkpoint(output, model, mean, std, baseline_metadata)
        print(json.dumps({"step": 0, "official_dev": official_result, "external_dev": external_result, "best": best}), flush=True)

    stop = False
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    iterator = iter(train_loader)
    for step in range(start + 1, args.steps + 1):
        try:
            context, history_actions, future_actions, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, history_actions, future_actions, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        history_actions = ((history_actions.to(device, non_blocking=True) - mean) / std).float()
        future_actions = ((future_actions.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = rollout(model, context, history_actions, future_actions[:, : args.rollout_steps])
            truth = target[:, : args.rollout_steps]
            previous = torch.cat([context[:, -1:], truth[:, :-1]], dim=1)
            loss = reconstruction_loss(
                prediction,
                truth,
                previous,
                motion_weight=2.5,
                motion_threshold=0.025,
                horizon_loss_power=0.35,
                temporal_delta_weight=0.35,
                texture_laplacian_weight=0.25,
                temporal_delta_pool=1,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "first_mae": float(F.l1_loss(prediction[:, 0].float(), truth[:, 0]))}), flush=True)

        if step % args.validation_interval == 0 or step == args.steps:
            official_result = evaluate(official_loader, model, device, mean, std, 0.04)
            external_result = evaluate(external_loader, model, device, mean, std, 0.04)
            model.train()
            record = {"step": step, "official_dev": official_result, "external_dev": external_result}
            history.append(record)
            metric = float(official_result["rollout_mae"])
            metadata = {
                "backend": "autoregressive_unet",
                "format": "track2-autoregressive-unet-v1",
                "stage": "v26.0-external-randomized-pretraining",
                "checkpoint_step": step,
                "selection_metric": "episode-disjoint official dev rollout MAE",
                "validation": history,
                **config,
            }
            if metric < best:
                best = metric
                metadata["best_official_dev_mae"] = best
                save_checkpoint(output / "best", model, mean, std, metadata)
                save_checkpoint(output, model, mean, std, metadata)
            save_checkpoint(output / "latest", model, mean, std, metadata)
            print(json.dumps({"step": step, "official_dev": official_result, "external_dev": external_result, "best": best}), flush=True)
        if step % args.state_interval == 0 or step == args.steps or stop:
            save_state(state_path, model, optimizer, step, history, config, best)
        if stop:
            print(json.dumps({"step": step, "status": "checkpointed_for_stop"}), flush=True)
            return


if __name__ == "__main__":
    main()
