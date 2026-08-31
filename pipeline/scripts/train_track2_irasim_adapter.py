#!/usr/bin/env python3
"""Adapt pretrained IRASim Bridge frame conditioning to Track 2 joint14 windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wam_pipeline.irasim_lora import inject_irasim_lora


class LatentWindows(Dataset):
    def __init__(self, windows: Path, latents: Path, names: list[str], mean: np.ndarray, std: np.ndarray):
        self.windows, self.latents, self.names, self.mean, self.std = windows, latents, names, mean, std

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        latent = np.load(self.latents / f"{Path(name).stem}.npy").astype(np.float32)
        with np.load(self.windows / name, allow_pickle=False) as value:
            actions = np.concatenate((value["history_actions"], value["future_actions"])).astype(np.float32)
        return torch.from_numpy(latent), torch.from_numpy((actions - self.mean) / self.std)


def action_statistics(windows: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    total, square, count = np.zeros(14, np.float64), np.zeros(14, np.float64), 0
    for name in names:
        with np.load(windows / name, allow_pickle=False) as value:
            action = np.concatenate((value["history_actions"], value["future_actions"])).astype(np.float64)
        total += action.sum(0); square += np.square(action).sum(0); count += len(action)
    mean = total / count
    return mean.astype(np.float32), np.sqrt(np.maximum(square / count - mean * mean, 1e-8)).astype(np.float32)


def adapt_pretrained(model, checkpoint: Path) -> dict:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("ema", raw.get("model", raw))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    target = model.state_dict(); loaded, adapted, skipped = {}, [], []
    for key, value in state.items():
        if key not in target:
            skipped.append(key); continue
        if value.shape == target[key].shape:
            loaded[key] = value; continue
        if key == "pos_embed" and value.ndim == 3:
            old_tokens, new_tokens = value.shape[1], target[key].shape[1]
            old_h = int(round(old_tokens ** 0.5)); old_w = old_tokens // old_h
            if old_h * old_w != old_tokens:
                old_h, old_w = 16, old_tokens // 16
            grid = value.reshape(1, old_h, old_w, value.shape[-1]).permute(0, 3, 1, 2)
            new_h = int(round(new_tokens ** 0.5)); new_w = new_tokens // new_h
            loaded[key] = functional.interpolate(grid.float(), (new_h, new_w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape_as(target[key])
            adapted.append(key); continue
        if key == "temp_embed" and value.ndim == 3:
            loaded[key] = functional.interpolate(value.permute(0, 2, 1).float(), size=target[key].shape[1], mode="linear", align_corners=False).permute(0, 2, 1)
            adapted.append(key); continue
        if key == "embed_state.fc1.weight" and value.shape[1] == 7 and target[key].shape[1] == 14:
            loaded[key] = torch.cat((value, value), dim=1) * 0.5
            adapted.append(key); continue
        skipped.append(key)
    incompatible = model.load_state_dict(loaded, strict=False)
    allowed_missing = {key for key in target if key.startswith("embed_state.")}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing:
        raise RuntimeError(f"unexpected unmatched pretrained parameters: {sorted(unexpected_missing)[:10]}")
    return {"loaded": len(loaded), "adapted": adapted, "skipped": skipped, "source_keys": len(state)}


def save_adapter(path: Path, model, optimizer, mean, std, step, config, history, load_report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    torch.save({
        "format": "track2-irasim-frame-action-adapter-v1", "step": step,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items() if key in trainable_names},
        "optimizer": optimizer.state_dict(), "action_mean": mean, "action_std": std,
        "config": config, "history": history, "pretrained_load": load_report,
    }, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--irasim-root", required=True); parser.add_argument("--pretrained", required=True)
    parser.add_argument("--windows", required=True); parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--train-latents", required=True); parser.add_argument("--validation-latents", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", default="IRASim-XL/2")
    parser.add_argument("--steps", type=int, default=2000); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=100); parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--gradient-accumulation", type=int, default=4); parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-rank", type=int, default=8); parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--train-temporal-blocks", action="store_true", help="Fully tune temporal blocks and output while retaining spatial texture weights.")
    parser.add_argument("--resume", help="Adapter checkpoint to resume after an interruption.")
    parser.add_argument("--initial-adapter", help="Load adapter weights without optimizer/step state before a stronger tuning phase.")
    args = parser.parse_args(); torch.manual_seed(20260807); np.random.seed(20260807)
    sys.path.insert(0, str(Path(args.irasim_root).resolve()))
    from diffusion import create_mask_diffusion
    from models.irasim import IRASim_models

    windows, split = Path(args.windows), json.loads(Path(args.split_manifest).read_text())
    names = lambda episodes: [p.name for e in episodes for p in sorted(windows.glob(f"episode{e}_*.npz"))]
    train_names, validation_names = names(split["train_episodes"]), names(split["validation_episodes"])
    validation_names = [validation_names[i] for i in np.linspace(0, len(validation_names)-1, min(args.validation_samples, len(validation_names)), dtype=int)]
    mean, std = action_statistics(windows, train_names)
    train_data = LatentWindows(windows, Path(args.train_latents), train_names, mean, std)
    validation_data = LatentWindows(windows, Path(args.validation_latents), validation_names, mean, std)
    weights = np.ones(len(train_names), np.float64)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(20260807))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_data, batch_size=1, num_workers=1)
    model_args = SimpleNamespace(dataset="bridge", state_dim=14, final_frame_ada=False, gradient_checkpointing=not args.no_gradient_checkpointing)
    model = IRASim_models[args.model](input_size=32, num_frames=13, learn_sigma=False, extras=3, attention_mode="sdpa", args=model_args)
    load_report = adapt_pretrained(model, Path(args.pretrained)); print(json.dumps({"pretrained": load_report}), flush=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.embed_state.parameters(): parameter.requires_grad_(True)
    for parameter in model.mask_emb_fn.parameters(): parameter.requires_grad_(True)
    lora_modules = inject_irasim_lora(model, args.lora_rank, args.lora_alpha)
    if args.train_temporal_blocks:
        for index in range(1, len(model.blocks), 2):
            for parameter in model.blocks[index].parameters(): parameter.requires_grad_(True)
        for parameter in model.final_layer.parameters(): parameter.requires_grad_(True)
    if args.initial_adapter:
        initial = torch.load(args.initial_adapter, map_location="cpu", weights_only=False)
        if initial.get("format") != "track2-irasim-frame-action-adapter-v1":
            raise ValueError("unsupported initial IRASim adapter checkpoint")
        incompatible = model.load_state_dict(initial["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected initial adapter keys: {incompatible.unexpected_keys[:5]}")
        print(json.dumps({"initial_adapter": args.initial_adapter, "source_step": initial.get("step")}), flush=True)
    device = torch.device(args.device); model.to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    diffusion = create_mask_diffusion(timestep_respacing="", learn_sigma=False)
    config = {**vars(args), "num_frames":13, "mask_frame_num":5, "state_dim":14, "train_windows":len(train_names), "trainable_parameters":sum(p.numel() for p in trainable), "lora_modules":lora_modules}
    output, history, start_step, iterator = Path(args.output), [], 0, iter(train_loader)
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("format") != "track2-irasim-frame-action-adapter-v1":
            raise ValueError("unsupported IRASim adapter resume checkpoint")
        model.load_state_dict(resume["state_dict"], strict=False)
        try:
            optimizer.load_state_dict(resume["optimizer"])
        except ValueError as error:
            # Permit forward-compatible resumes after adding a newly trainable
            # conditioning parameter; model weights and the global step remain valid.
            print(json.dumps({"optimizer_resume": "reinitialized", "reason": str(error)}), flush=True)
        history, start_step = list(resume.get("history", [])), int(resume["step"])
        print(json.dumps({"resumed_from": args.resume, "step": start_step}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        running = 0.0
        for _ in range(args.gradient_accumulation):
            try: latent, actions = next(iterator)
            except StopIteration: iterator = iter(train_loader); latent, actions = next(iterator)
            latent, actions = latent.to(device, non_blocking=True), actions.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = diffusion.training_losses(model, latent, t, {"actions": actions, "mask_frame_num":5})["loss"].mean() / args.gradient_accumulation
            loss.backward(); running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(json.dumps({"step":step,"loss":running,"gpu_mb":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            model.eval(); values=[]
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            # Every checkpoint sees exactly the same validation timesteps/noise.
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                torch.manual_seed(20260808)
                for latent, actions in validation_loader:
                    latent, actions = latent.to(device), actions.to(device); t=torch.randint(0,diffusion.num_timesteps,(1,),device=device)
                    values.append(float(diffusion.training_losses(model,latent,t,{"actions":actions,"mask_frame_num":5})["loss"].mean()))
            metrics={"step":step,"diffusion_loss":float(np.mean(values))}; history.append(metrics); print(json.dumps({"validation":metrics}),flush=True)
            save_adapter(output / f"adapter_step_{step:06d}.pt",model,optimizer,mean,std,step,config,history,load_report); model.train()
    (output / "training_manifest.json").write_text(json.dumps({"format":"track2-irasim-frame-action-adapter-v1","config":config,"history":history,"pretrained_load":load_report},indent=2)+"\n")


if __name__ == "__main__": main()
