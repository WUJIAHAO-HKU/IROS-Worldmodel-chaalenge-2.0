#!/usr/bin/env python3
"""IRASim-v2 rescue training with physical actions and native Bridge geometry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wam_pipeline.irasim_lora import inject_irasim_lora


class PhysicalLatentWindows(Dataset):
    def __init__(self, cache: Path, names: list[str], action_cache: Path | None = None):
        self.latents = cache / "latents"
        self.actions = action_cache if action_cache is not None else cache / "actions"
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        stem = Path(self.names[index]).stem
        latent = np.load(self.latents / f"{stem}.npy").astype(np.float32)
        action = np.load(self.actions / f"{stem}.npy").astype(np.float32)
        if latent.shape != (13, 4, 32, 40) or action.shape != (12, 8):
            raise ValueError(f"invalid v2 cache shapes for {stem}: {latent.shape}, {action.shape}")
        return torch.from_numpy(latent), torch.from_numpy(action)


def load_pretrained_native(model, path: Path) -> dict:
    """Load Bridge weights without interpolating deterministic position embeddings."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("ema", raw.get("model", raw))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    target = model.state_dict()
    loaded, adapted, skipped = {}, [], []
    for key, value in state.items():
        if key not in target:
            skipped.append(key)
        elif value.shape == target[key].shape:
            loaded[key] = value
        elif key == "temp_embed" and value.ndim == 3 and target[key].shape[1] <= value.shape[1]:
            loaded[key] = value[:, : target[key].shape[1]].clone()
            adapted.append("temp_embed:slice16to13")
        elif key == "embed_state.fc1.weight" and value.shape[1] == 7 and target[key].shape[1] == 8:
            expanded = torch.zeros_like(target[key])
            expanded[:, :7] = value
            loaded[key] = expanded
            adapted.append("embed_state.fc1.weight:bridge7+arm_id")
        else:
            skipped.append(key)
    incompatible = model.load_state_dict(loaded, strict=False)
    allowed_missing = {key for key in target if key.startswith("embed_state.")}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing:
        raise RuntimeError(f"unexpected unmatched pretrained parameters: {sorted(unexpected_missing)[:10]}")
    return {"loaded": len(loaded), "adapted": adapted, "skipped": skipped, "source_keys": len(state)}


def curriculum_mask(step: int, curriculum_steps: int, generator: torch.Generator) -> int:
    if curriculum_steps <= 0:
        return 5
    maximum = min(5, 1 + int(4 * max(0, step - 1) / max(1, curriculum_steps)))
    return int(torch.randint(1, maximum + 1, (1,), generator=generator).item())


def diffusion_objective(model, diffusion, latent, actions, timestep, mask_frames, noise, args):
    future = latent[:, mask_frames:]
    noisy_future = diffusion.q_sample(future, timestep, noise=noise)
    noisy = torch.cat((latent[:, :mask_frames], noisy_future), dim=1)
    prediction = model(noisy, timestep, actions=actions, mask_frame_num=mask_frames)
    prediction = prediction[:, mask_frames:]
    noise_mse = (prediction.float() - noise.float()).square().flatten(1).mean(1)
    loss = noise_mse.mean()
    metrics = {"noise_mse": noise_mse.mean().detach()}
    eligible = timestep <= args.x0_max_timestep
    if eligible.any() and (args.x0_l1_weight > 0 or args.x0_gradient_weight > 0 or args.x0_temporal_weight > 0):
        predicted_x0 = diffusion._predict_xstart_from_eps(noisy_future, timestep, prediction)
        predicted_x0 = predicted_x0[eligible].float()
        target_x0 = future[eligible].float()
        x0_l1 = F.l1_loss(predicted_x0, target_x0)
        spatial_gradient = F.l1_loss(
            predicted_x0[..., 1:, :] - predicted_x0[..., :-1, :],
            target_x0[..., 1:, :] - target_x0[..., :-1, :],
        ) + F.l1_loss(
            predicted_x0[..., :, 1:] - predicted_x0[..., :, :-1],
            target_x0[..., :, 1:] - target_x0[..., :, :-1],
        )
        if predicted_x0.shape[1] > 1:
            temporal = F.l1_loss(predicted_x0[:, 1:] - predicted_x0[:, :-1], target_x0[:, 1:] - target_x0[:, :-1])
        else:
            temporal = predicted_x0.new_zeros(())
        loss = loss + args.x0_l1_weight * x0_l1
        loss = loss + args.x0_gradient_weight * spatial_gradient
        loss = loss + args.x0_temporal_weight * temporal
        metrics.update({"x0_l1": x0_l1.detach(), "x0_gradient": spatial_gradient.detach(), "x0_temporal": temporal.detach()})
    return loss, metrics


def save_checkpoint(path: Path, model, optimizer, step, config, history, load_report, save_optimizer: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    payload = {
        "format": "track2-irasim-v2-physical-native-adapter-v1",
        "step": step,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items() if key in trainable},
        "config": config,
        "history": history,
        "pretrained_load": load_report,
    }
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--irasim-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-actions", help="Optional production-mapped action cache overriding train-cache/actions")
    parser.add_argument("--validation-actions", help="Optional production-mapped action cache overriding validation-cache/actions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--limit-train-windows", type=int)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--last-spatial-blocks", type=int, default=4)
    parser.add_argument("--curriculum-steps", type=int, default=300)
    parser.add_argument("--low-timestep-fraction", type=float, default=0.5)
    parser.add_argument("--low-timestep-max", type=int, default=400)
    parser.add_argument("--x0-max-timestep", type=int, default=400)
    parser.add_argument("--x0-l1-weight", type=float, default=0.10)
    parser.add_argument("--x0-gradient-weight", type=float, default=0.05)
    parser.add_argument("--x0-temporal-weight", type=float, default=0.05)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(20260807)
    np.random.seed(20260807)
    sys.path.insert(0, str(Path(args.irasim_root).resolve()))
    from diffusion import create_mask_diffusion
    from models.irasim import IRASim_models

    windows = Path(args.windows)
    split = json.loads(Path(args.split_manifest).read_text())
    names = lambda episodes: [path.name for episode in episodes for path in sorted(windows.glob(f"episode{episode}_*.npz"))]
    train_names = names(split["train_episodes"])
    validation_names = names(split["validation_episodes"])
    if args.limit_train_windows is not None:
        train_names = train_names[: args.limit_train_windows]
    validation_names = [
        validation_names[index]
        for index in np.linspace(0, len(validation_names) - 1, min(args.validation_samples, len(validation_names)), dtype=int)
    ]
    train_data = PhysicalLatentWindows(Path(args.train_cache), train_names, Path(args.train_actions) if args.train_actions else None)
    validation_data = PhysicalLatentWindows(Path(args.validation_cache), validation_names, Path(args.validation_actions) if args.validation_actions else None)
    sampler = WeightedRandomSampler(
        np.ones(len(train_names), np.float64),
        len(train_names),
        replacement=True,
        generator=torch.Generator().manual_seed(20260807),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_data, batch_size=1, num_workers=1)

    model_args = SimpleNamespace(dataset="bridge", state_dim=8, final_frame_ada=False, gradient_checkpointing=True)
    model = IRASim_models["IRASim-XL/2"](
        input_size=(32, 40), num_frames=13, learn_sigma=False, extras=3, attention_mode="sdpa", args=model_args
    )
    load_report = load_pretrained_native(model, Path(args.pretrained))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lora_modules = inject_irasim_lora(model, args.lora_rank, args.lora_alpha)
    for module in (model.embed_state, model.mask_emb_fn, model.x_embedder, model.final_layer):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for index in range(1, len(model.blocks), 2):
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad_(True)
    spatial_indices = list(range(0, len(model.blocks), 2))[-args.last_spatial_blocks :]
    for index in spatial_indices:
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad_(True)

    device = torch.device(args.device)
    model.to(device).train()
    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    fast = [parameter for name, parameter in named_trainable if name.startswith(("embed_state", "mask_emb_fn")) or ".lora_" in name]
    fast_ids = {id(parameter) for parameter in fast}
    base = [parameter for _, parameter in named_trainable if id(parameter) not in fast_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base, "lr": args.learning_rate},
            {"params": fast, "lr": args.adapter_learning_rate},
        ],
        weight_decay=1e-4,
    )
    diffusion = create_mask_diffusion(timestep_respacing="", learn_sigma=False)
    config = {
        **vars(args),
        "num_frames": 13,
        "latent_size": [32, 40],
        "mask_frame_num_validation": 5,
        "state_dim": 8,
        "train_windows": len(train_names),
        "trainable_parameters": sum(parameter.numel() for _, parameter in named_trainable),
        "spatial_indices": spatial_indices,
        "lora_modules": lora_modules,
        "action_normalization": "none; official Bridge scaling embedded in cache",
        "action_source": "learned joint-command mapper" if args.train_actions else "oracle endpose upper bound",
    }
    print(json.dumps({"pretrained": load_report, "config": config}), flush=True)

    history, output = [], Path(args.output)
    iterator = iter(train_loader)
    mask_generator = torch.Generator().manual_seed(20260809)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accumulated = {"loss": 0.0, "noise_mse": 0.0, "x0_l1": 0.0, "x0_gradient": 0.0, "x0_temporal": 0.0}
        mask_frames = curriculum_mask(step, args.curriculum_steps, mask_generator)
        for _ in range(args.gradient_accumulation):
            try:
                latent, actions = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                latent, actions = next(iterator)
            latent = latent.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            batch = latent.shape[0]
            low = torch.rand(batch, device=device) < args.low_timestep_fraction
            timestep = torch.randint(0, diffusion.num_timesteps, (batch,), device=device)
            timestep[low] = torch.randint(0, args.low_timestep_max + 1, (int(low.sum()),), device=device)
            noise = torch.randn_like(latent[:, mask_frames:])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, metrics = diffusion_objective(model, diffusion, latent, actions, timestep, mask_frames, noise, args)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated["loss"] += float(scaled_loss.detach())
            for key, value in metrics.items():
                accumulated[key] += float(value) / args.gradient_accumulation
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named_trainable], 0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == 1:
            accumulated.update(
                {"step": step, "mask_frames": mask_frames, "gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
            )
            print(json.dumps(accumulated), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            model.eval()
            values = []
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                torch.manual_seed(20260808)
                for latent, actions in validation_loader:
                    latent, actions = latent.to(device), actions.to(device)
                    timestep = torch.randint(0, diffusion.num_timesteps, (1,), device=device)
                    noise = torch.randn_like(latent[:, 5:])
                    _, metrics = diffusion_objective(model, diffusion, latent, actions, timestep, 5, noise, args)
                    values.append(float(metrics["noise_mse"]))
            validation = {"step": step, "diffusion_loss": float(np.mean(values))}
            history.append(validation)
            print(json.dumps({"validation": validation}), flush=True)
            save_checkpoint(
                output / f"adapter_step_{step:06d}.pt",
                model,
                optimizer,
                step,
                config,
                history,
                load_report,
                args.save_optimizer,
            )
            model.train()
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_manifest.json").write_text(
        json.dumps({"format": "track2-irasim-v2-physical-native-adapter-v1", "config": config, "history": history, "pretrained_load": load_report}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
