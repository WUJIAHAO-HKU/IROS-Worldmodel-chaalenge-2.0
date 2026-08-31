#!/usr/bin/env python3
"""Train a 14-D action-conditioned Wan LoRA on exact Track 2 windows.

This trainer intentionally does not use DiffSynth's public ``RLinfNpyDataset``:
that loader overwrites its first action with a hard-coded seven-dimensional
value.  Instead, it consumes the audited 5+4+8 window definition directly.
Only the eight future latent slots contribute to the flow-matching loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from diffusers import AutoencoderKLWan, WanTransformer3DModel

from wam_pipeline.profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES
from wam_pipeline.track2_wan import (
    TRACK2_CONTEXT_LATENT_FRAMES,
    TRACK2_LATENT_FRAMES,
    TOTAL_ACTION_SLOTS,
    Track2ActionConditioner,
    Track2WanLayout,
    Track2WanWorldModel,
    build_action_slots,
    configure_wan_lora,
    extract_lora_state,
    normalize_action_slots,
    normalize_vae_latents,
    vae_latent_stats,
)
from wam_pipeline.wan_latent_cache import open_verified_cache
from wam_pipeline.wan_track2_data import WanTrack2WindowDataset


FORMAT_V1 = "track2-action-conditioned-wan-lora-v1"
FORMAT_V2 = "track2-action-conditioned-wan-trajectory-lora-v2"
FORMAT_V3 = "track2-action-conditioned-wan-latent-action-lora-v3"
FORMAT_V4 = "track2-action-conditioned-wan-trajectory-latent-lora-v4"
SUPPORTED_FORMATS = {FORMAT_V1, FORMAT_V2, FORMAT_V3, FORMAT_V4}


def checkpoint_conditioning_modes(checkpoint_format: object) -> tuple[bool, bool]:
    """Return (trajectory, latent-action) branches stored by a checkpoint."""
    return (
        checkpoint_format in {FORMAT_V2, FORMAT_V4},
        checkpoint_format in {FORMAT_V3, FORMAT_V4},
    )


class Track2WindowTorchDataset(Dataset):
    """Memory-mapped official NPY episodes surfaced as exact Tensor windows."""

    def __init__(self, root: Path, split: str, latent_cache: np.ndarray | None = None) -> None:
        self.dataset = WanTrack2WindowDataset(root, split)
        if latent_cache is not None and len(latent_cache) != len(self.dataset):
            raise ValueError(f"{split} latent cache window count does not match the official dataset")
        self.latent_cache = latent_cache

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if self.latent_cache is not None:
            return (
                torch.from_numpy(item.history_actions.copy()),
                torch.from_numpy(item.future_actions.copy()),
                torch.from_numpy(np.asarray(self.latent_cache[index]).copy()),
            )
        return (
            torch.from_numpy(item.context_frames.copy()),
            torch.from_numpy(item.history_actions.copy()),
            torch.from_numpy(item.future_actions.copy()),
            torch.from_numpy(item.target_frames.copy()),
        )


def frames_to_wan_video(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,H,W,3] RGB uint8 into Wan VAE [B,3,13,H,W] input."""
    frames = torch.cat((context, target), dim=1)
    if tuple(frames.shape[1:]) != (CONTEXT_FRAMES + PREDICTION_FRAMES, 256, 256, 3):
        raise ValueError("Track 2 Wan input must be [B,13,256,256,3]")
    return frames.permute(0, 4, 1, 2, 3).float().div(127.5).sub(1.0)


def action_statistics(dataset: Track2WindowTorchDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute physical abs14 statistics from train only, excluding the zero anchor."""
    total = np.zeros(ACTION_DIM, dtype=np.float64)
    squared = np.zeros(ACTION_DIM, dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        item = dataset.dataset[index]
        actions = np.concatenate((item.history_actions, item.future_actions), axis=0).astype(np.float64)
        total += actions.sum(axis=0)
        squared += np.square(actions).sum(axis=0)
        count += len(actions)
    if count < 1:
        raise ValueError("cannot compute action normalization from an empty dataset")
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def sample_sigma(
    batch_size: int,
    device: torch.device,
    *,
    distribution: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if distribution == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if distribution == "logit-normal":
        return torch.sigmoid(torch.randn(batch_size, device=device, generator=generator))
    raise ValueError(f"unsupported sigma distribution: {distribution}")


def future_flow_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The first two latent slots encode five observed RGB frames and get no loss."""
    return functional.mse_loss(
        prediction[:, :, TRACK2_CONTEXT_LATENT_FRAMES:].float(),
        target[:, :, TRACK2_CONTEXT_LATENT_FRAMES:].float(),
    )


@torch.no_grad()
def encode_latents(vae: AutoencoderKLWan, video: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(video.to(dtype=next(vae.parameters()).dtype)).latent_dist
    latents = posterior.mode()
    normalized = normalize_vae_latents(latents, mean, std)
    if normalized.shape[2] != TRACK2_LATENT_FRAMES:
        raise RuntimeError(
            f"Track 2 13 RGB frames must encode to {TRACK2_LATENT_FRAMES} Wan slots, got {normalized.shape[2]}"
        )
    return normalized


def load_components(
    base_model: Path,
    device: torch.device,
    dtype: torch.dtype,
    lora_rank: int,
    *,
    load_vae: bool,
    trajectory_conditioned: bool = False,
    trajectory_hidden_dim: int | None = None,
    latent_action_conditioned: bool = False,
    latent_action_hidden_dim: int | None = None,
) -> tuple[AutoencoderKLWan | None, Track2WanWorldModel, list[str]]:
    transformer = WanTransformer3DModel.from_pretrained(
        str(base_model), subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    transformer.to(device)
    vae = None
    if load_vae:
        vae = AutoencoderKLWan.from_pretrained(
            str(base_model), subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
    names = configure_wan_lora(transformer, rank=lora_rank)
    conditioner = Track2ActionConditioner(
        int(transformer.config.text_dim),
        trajectory_conditioned=trajectory_conditioned,
        trajectory_hidden_dim=trajectory_hidden_dim,
        latent_action_conditioned=latent_action_conditioned,
        latent_channels=int(transformer.config.out_channels) if latent_action_conditioned else None,
        latent_action_hidden_dim=latent_action_hidden_dim,
    ).to(device=device, dtype=dtype)
    model = Track2WanWorldModel(transformer, conditioner).to(device)
    return vae, model, names


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def trajectory_parameters(model: Track2WanWorldModel) -> list[nn.Parameter]:
    """Return v2 motion tensors separately from inherited v1 visual weights."""
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("action_conditioner.trajectory_")
    ]


def latent_action_parameters(model: Track2WanWorldModel) -> list[nn.Parameter]:
    """Return only the v3 time-aligned latent action residual parameters."""
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("action_conditioner.latent_action_")
    ]


def set_inherited_trainable(
    model: Track2WanWorldModel,
    enabled: bool,
    *,
    freeze_trajectory: bool = False,
) -> None:
    """Freeze inherited tensors while a newly added action branch warms up."""
    for name, parameter in model.named_parameters():
        # The frozen 5B Wan base must never be re-enabled here. Peft LoRA
        # tensors consistently contain ``lora_``; all non-trajectory action
        # conditioner tensors belong to the inherited abs14 path.
        inherited_action = name.startswith("action_conditioner.") and not name.startswith(
            "action_conditioner.latent_action_"
        )
        if not freeze_trajectory:
            inherited_action = inherited_action and not name.startswith("action_conditioner.trajectory_")
        inherited_lora = name.startswith("transformer.") and "lora_" in name
        if inherited_action or inherited_lora:
            parameter.requires_grad_(enabled)


def assert_frozen_wan_base(model: Track2WanWorldModel) -> None:
    """Fail before backward if any non-LoRA Wan base tensor becomes trainable."""
    invalid = [
        name
        for name, parameter in model.transformer.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if invalid:
        raise RuntimeError(f"Wan base transformer must remain frozen; found trainable {invalid[:3]}")


def model_checkpoint_path(output: Path) -> Path:
    return output / "track2_wan_lora.pt"


def deployable_state(
    *,
    model: Track2WanWorldModel,
    step: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    metadata: dict,
) -> dict:
    """Build the inference-only portion shared by latest and selected checkpoints."""
    return {
        "format": str(metadata["format"]),
        "step": int(step),
        "action_conditioner": {name: value.detach().cpu() for name, value in model.action_conditioner.state_dict().items()},
        "transformer_lora": extract_lora_state(model.transformer, "track2"),
        "action_mean": action_mean.detach().cpu(),
        "action_std": action_std.detach().cpu(),
        "metadata": metadata,
    }


def write_deployable_files(output: Path, state: dict, metadata: dict) -> None:
    """Atomically write the files consumed by the Track 2 Wan runtime."""
    output.mkdir(parents=True, exist_ok=True)
    temporary = model_checkpoint_path(output).with_suffix(".pt.tmp")
    torch.save(state, temporary)
    os.replace(temporary, model_checkpoint_path(output))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez(
        output / "action_normalization.npz",
        mean=state["action_mean"].numpy(),
        std=state["action_std"].numpy(),
    )


def save_checkpoint(
    output: Path,
    *,
    model: Track2WanWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    metadata: dict,
) -> None:
    state = deployable_state(
        model=model,
        step=step,
        action_mean=action_mean,
        action_std=action_std,
        metadata=metadata,
    )
    # The root checkpoint carries optimizer state and is the only resumable one.
    state["optimizer"] = optimizer.state_dict()
    state["scheduler"] = scheduler.state_dict()
    write_deployable_files(output, state, metadata)


def save_best_checkpoint(
    output: Path,
    *,
    model: Track2WanWorldModel,
    step: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    metadata: dict,
) -> None:
    """Save the validation-selected runtime checkpoint without optimizer bloat."""
    selected_metadata = {**metadata, "checkpoint_kind": "validation_flow_best", "selected_step": int(step)}
    write_deployable_files(
        output / "best",
        deployable_state(
            model=model,
            step=step,
            action_mean=action_mean,
            action_std=action_std,
            metadata=selected_metadata,
        ),
        selected_metadata,
    )


def restore_checkpoint(
    resume: Path,
    *,
    model: Track2WanWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, float, list[dict]]:
    state = torch.load(resume, map_location="cpu", weights_only=False)
    if state.get("format") not in SUPPORTED_FORMATS:
        raise ValueError("resume checkpoint has the wrong Track 2 Wan format")
    if "optimizer" not in state or "scheduler" not in state:
        raise ValueError("--resume must name the root resumable checkpoint, not output/best/track2_wan_lora.pt")
    checkpoint_trajectory, checkpoint_latent_action = checkpoint_conditioning_modes(state.get("format"))
    if (
        checkpoint_trajectory != model.action_conditioner.trajectory_conditioned
        or checkpoint_latent_action != model.action_conditioner.latent_action_conditioned
    ):
        raise ValueError("resume checkpoint conditioning mode does not match this trainer")
    model.action_conditioner.load_state_dict(state["action_conditioner"], strict=True)
    from wam_pipeline.track2_wan import load_lora_state

    load_lora_state(model.transformer, "track2", state["transformer_lora"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    metadata = state.get("metadata", {})
    best_validation = float(metadata.get("best_validation_future_flow_mse", float("inf")))
    history = metadata.get("validation", [])
    if not isinstance(history, list):
        raise ValueError("resume checkpoint validation history is malformed")
    return int(state["step"]), state["action_mean"].to(device), state["action_std"].to(device), best_validation, history


def initialize_from_checkpoint(
    checkpoint: Path,
    *,
    model: Track2WanWorldModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load v1 visual/action weights into a fresh conditioning-branch optimizer state.

    Added branches use zero-readout tensors, so a v1 checkpoint may omit only
    those tensors. The LoRA and absolute abs14 action path must load completely;
    otherwise this is not a true continuation of the stable v1 model.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("format") not in SUPPORTED_FORMATS:
        raise ValueError("--init-checkpoint has the wrong Track 2 Wan format")
    checkpoint_trajectory, checkpoint_latent_action = checkpoint_conditioning_modes(state.get("format"))
    if checkpoint_trajectory and not model.action_conditioner.trajectory_conditioned:
        raise ValueError("cannot initialize a non-trajectory conditioner from a v2 trajectory checkpoint")
    if checkpoint_latent_action and not model.action_conditioner.latent_action_conditioned:
        raise ValueError("cannot initialize a non-latent-action conditioner from a v3 checkpoint")
    incompatible = model.action_conditioner.load_state_dict(state["action_conditioner"], strict=False)
    expected_missing = set()
    if model.action_conditioner.trajectory_conditioned and not checkpoint_trajectory:
        expected_missing.update(
            name for name in model.action_conditioner.state_dict() if name.startswith("trajectory_")
        )
    if model.action_conditioner.latent_action_conditioned and not checkpoint_latent_action:
        expected_missing.update(
            name for name in model.action_conditioner.state_dict() if name.startswith("latent_action_")
        )
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError("initialization did not load the complete inherited action conditioner")
    from wam_pipeline.track2_wan import load_lora_state

    load_lora_state(model.transformer, "track2", state["transformer_lora"])
    metadata = state.get("metadata", {})
    return state["action_mean"].to(device), state["action_std"].to(device), metadata


def train_rgb_motion_scores(dataset: Track2WindowTorchDataset, cache_path: Path) -> np.ndarray:
    """Measure low-resolution RGB displacement for deterministic weighted sampling."""
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (len(dataset),) and np.isfinite(cached).all() and np.all(cached > 0):
            return cached.astype(np.float32, copy=False)
    scores = np.empty(len(dataset), dtype=np.float32)
    for index in range(len(dataset)):
        item = dataset.dataset[index]
        # Spatial decimation preserves arm/object displacement while keeping
        # startup work small enough to use before a formal run.
        # Weight the unobserved rollout, including the final-context to first-
        # future boundary, instead of already-observed approach motion.
        frames = item.video_frames[CONTEXT_FRAMES - 1 :, ::16, ::16].astype(np.float32, copy=False)
        scores[index] = np.abs(frames[1:] - frames[:-1]).mean(dtype=np.float64) / 255.0
    scores = np.maximum(scores, np.finfo(np.float32).eps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, scores)
    return scores


@torch.no_grad()
def validation_flow_loss(
    loader: DataLoader,
    *,
    vae: AutoencoderKLWan | None,
    model: Track2WanWorldModel,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
    batches: int | None,
    seed: int,
) -> float:
    model.eval()
    values: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, batch in enumerate(loader):
        if batches is not None and batch_index >= batches:
            break
        if len(batch) == 3:
            history, future, clean = batch
            clean = clean.to(device, dtype=next(model.transformer.parameters()).dtype, non_blocking=True)
        else:
            if vae is None or latent_mean is None or latent_std is None:
                raise RuntimeError("VAE is required when validation latents are not cached")
            context, history, future, target = batch
            video = frames_to_wan_video(context, target).to(device, non_blocking=True)
            clean = encode_latents(vae, video, latent_mean, latent_std)
        slots = normalize_action_slots(build_action_slots(history.to(device), future.to(device)), action_mean, action_std)
        noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
        sigma = sample_sigma(clean.shape[0], device, distribution="uniform", generator=generator)
        with torch.autocast(
            device_type=device.type,
            dtype=next(model.transformer.parameters()).dtype,
            enabled=device.type == "cuda",
        ):
            prediction, flow_target, _ = model(clean, noise, sigma, slots)
        values.append(future_flow_loss(prediction, flow_target).detach().float().cpu())
    model.train()
    if not values:
        raise ValueError("validation loader produced no batches")
    return float(torch.stack(values).mean())


def cycling(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Wan LoRA with exact 5+4+8 Track 2 conditioning.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", required=True, help="Local Wan Diffusers directory with transformer/ and vae/.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--inherited-learning-rate",
        type=float,
        help="LR for inherited v1 LoRA/abs14 tensors after motion warmup; defaults to --learning-rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=0,
        help="Held-out batches per validation; 0 evaluates the complete validation split.",
    )
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--sigma-distribution", choices=("uniform", "logit-normal"), default="logit-normal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="Path to a prior track2_wan_lora.pt checkpoint.")
    parser.add_argument(
        "--init-checkpoint",
        help="Deployable v1/v2 checkpoint used to initialize weights but deliberately not optimizer state.",
    )
    parser.add_argument(
        "--trajectory-conditioned",
        action="store_true",
        help="Add a zero-initialized GRU residual over abs14 deltas and displacements from the final history pose.",
    )
    parser.add_argument(
        "--trajectory-hidden-dim",
        type=int,
        help="GRU width for --trajectory-conditioned; defaults to a bounded Wan text width.",
    )
    parser.add_argument(
        "--latent-action-conditioned",
        action="store_true",
        help="Inject zero-initialized 4-slot action residuals aligned to Wan VAE temporal slots.",
    )
    parser.add_argument(
        "--latent-action-hidden-dim",
        type=int,
        help="MLP width for --latent-action-conditioned; defaults to a bounded action width.",
    )
    parser.add_argument(
        "--high-motion-oversample",
        type=float,
        default=0.0,
        help="Nonnegative extra sampling weight for RGB-motion windows above the training median.",
    )
    parser.add_argument(
        "--motion-score-cache",
        help="Optional .npy cache for deterministic per-window RGB-motion scores.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        help="Fixed validation-noise seed; defaults to --seed plus 10,000.",
    )
    parser.add_argument(
        "--trajectory-warmup-steps",
        type=int,
        default=0,
        help="For initialized v2 training, freeze v1 visual/action weights for this many AdamW updates.",
    )
    parser.add_argument(
        "--freeze-trajectory-during-warmup",
        action="store_true",
        help="For v4 initialized from v2, train only the new latent-action branch during warmup.",
    )
    parser.add_argument(
        "--latent-cache",
        help="Validated frozen-VAE cache directory made by cache_track2_wan_latents.py.",
    )
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.gradient_accumulation < 1
        or args.learning_rate <= 0
        or args.inherited_learning_rate is not None and args.inherited_learning_rate <= 0
        or args.validation_interval < 1
        or args.validation_batches < 0
        or args.save_interval < 1
        or args.trajectory_hidden_dim is not None and args.trajectory_hidden_dim < 1
        or args.latent_action_hidden_dim is not None and args.latent_action_hidden_dim < 1
        or args.high_motion_oversample < 0
        or args.trajectory_warmup_steps < 0
    ):
        raise SystemExit("steps, batch size, gradient accumulation, and learning rate must be positive")
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint and not (args.trajectory_conditioned or args.latent_action_conditioned):
        raise SystemExit("--init-checkpoint requires trajectory or latent-action conditioning")
    if args.trajectory_warmup_steps and not (args.trajectory_conditioned or args.latent_action_conditioned):
        raise SystemExit("--trajectory-warmup-steps requires a newly added conditioning branch")
    if args.freeze_trajectory_during_warmup and not (
        args.trajectory_conditioned and args.latent_action_conditioned and args.trajectory_warmup_steps
    ):
        raise SystemExit("--freeze-trajectory-during-warmup requires combined conditioning and positive warmup")
    if args.validation_interval % args.gradient_accumulation or args.save_interval % args.gradient_accumulation:
        raise SystemExit("validation/save intervals must be divisible by --gradient-accumulation for exact resume")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA training requested but CUDA is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    root, base_model, output = Path(args.dataset_root), Path(args.base_model), Path(args.output)
    if not (base_model / "transformer").is_dir() or not (base_model / "vae").is_dir():
        raise SystemExit("--base-model must be a local Wan Diffusers model containing transformer/ and vae/")

    train = Track2WindowTorchDataset(root, "train")
    validation = Track2WindowTorchDataset(root, "validation")
    use_latent_cache = args.latent_cache is not None
    vae, model, lora_names = load_components(
        base_model,
        device,
        dtype,
        args.lora_rank,
        load_vae=not use_latent_cache,
        trajectory_conditioned=args.trajectory_conditioned,
        trajectory_hidden_dim=args.trajectory_hidden_dim,
        latent_action_conditioned=args.latent_action_conditioned,
        latent_action_hidden_dim=args.latent_action_hidden_dim,
    )
    latent_mean = latent_std = None
    if vae is not None:
        latent_mean, latent_std = vae_latent_stats(vae, device=device, dtype=dtype)
    if use_latent_cache:
        # Read these public VAE dimensions from its config without loading its
        # multi-GB weights alongside the DiT.
        vae_config = json.loads((base_model / "vae" / "config.json").read_text())
        spatial = 256 // int(vae_config["scale_factor_spatial"])
        expected_channels = int(vae_config["z_dim"])
        train.latent_cache = open_verified_cache(
            args.latent_cache,
            split="train",
            dataset_root=root,
            base_model=base_model,
            expected_windows=len(train),
            expected_channels=expected_channels,
            expected_latent_frames=TRACK2_LATENT_FRAMES,
            expected_height=spatial,
            expected_width=spatial,
        )
        validation.latent_cache = open_verified_cache(
            args.latent_cache,
            split="validation",
            dataset_root=root,
            base_model=base_model,
            expected_windows=len(validation),
            expected_channels=expected_channels,
            expected_latent_frames=TRACK2_LATENT_FRAMES,
            expected_height=spatial,
            expected_width=spatial,
        )
    sampler = None
    motion_sampling_metadata: dict[str, float | str | None] = {
        "high_motion_oversample": float(args.high_motion_oversample),
        "motion_score_cache": None,
        "motion_score_mean": None,
        "motion_score_median": None,
        "motion_score_p90": None,
    }
    if args.high_motion_oversample > 0:
        score_cache = Path(args.motion_score_cache) if args.motion_score_cache else output / "train_rgb_motion_scores.npy"
        scores = train_rgb_motion_scores(train, score_cache)
        median = float(np.median(scores))
        p90 = float(np.percentile(scores, 90))
        scale = max(p90 - median, float(np.finfo(np.float32).eps))
        weights = 1.0 + args.high_motion_oversample * np.clip((scores - median) / scale, 0.0, 2.0)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights.astype(np.double)), num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        motion_sampling_metadata = {
            "high_motion_oversample": float(args.high_motion_oversample),
            "motion_score_cache": str(score_cache.resolve()),
            "motion_score_mean": float(scores.mean()),
            "motion_score_median": median,
            "motion_score_p90": p90,
        }
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if len(train_loader) == 0:
        raise SystemExit("training data has fewer windows than --batch-size")
    action_mean, action_std = action_statistics(train)
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    inherited_lr = args.inherited_learning_rate or args.learning_rate
    trajectory_params = trajectory_parameters(model)
    latent_action_params = latent_action_parameters(model)
    if args.trajectory_conditioned and not trajectory_params:
        raise RuntimeError("trajectory-conditioned model has no trajectory parameters")
    if args.latent_action_conditioned and not latent_action_params:
        raise RuntimeError("latent-action-conditioned model has no latent action parameters")
    inherited_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("action_conditioner.trajectory_")
        and not name.startswith("action_conditioner.latent_action_")
        and parameter.requires_grad
    ]
    parameter_groups = [{"params": inherited_params, "lr": inherited_lr, "name": "inherited_v1"}]
    if args.trajectory_conditioned:
        parameter_groups.append({"params": trajectory_params, "lr": args.learning_rate, "name": "trajectory_v2"})
    if args.latent_action_conditioned:
        parameter_groups.append({"params": latent_action_params, "lr": args.learning_rate, "name": "latent_action_v3"})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    optimizer_steps = math.ceil(args.steps / args.gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_steps, eta_min=args.learning_rate * 0.1
    )
    start_step = 0
    best_validation = float("inf")
    history: list[dict] = []
    initialization_metadata: dict[str, object] = {"init_checkpoint": None}
    resumed_metadata: dict[str, object] = {}
    if args.resume:
        start_step, action_mean, action_std, best_validation, history = restore_checkpoint(
            Path(args.resume), model=model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        resumed_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        resumed_metadata = resumed_state.get("metadata", {})
    elif args.init_checkpoint:
        initialized_mean, initialized_std, source_metadata = initialize_from_checkpoint(
            Path(args.init_checkpoint), model=model, device=device
        )
        # Normalization defines the coordinate system for absolute and relative
        # actions. It must be preserved when continuing a learned v1 adapter.
        action_mean, action_std = initialized_mean, initialized_std
        initialization_metadata = {
            "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
            "init_checkpoint_format": source_metadata.get("format"),
            "init_checkpoint_step": source_metadata.get("step"),
            "init_optimizer": "fresh",
        }
    if start_step >= args.steps:
        raise SystemExit(f"checkpoint already reached step {start_step}; --steps must exceed it when resuming")
    if args.resume and (args.trajectory_conditioned or args.latent_action_conditioned):
        saved_warmup = resumed_metadata.get("trajectory_warmup_steps")
        if not isinstance(saved_warmup, int) or saved_warmup < 0:
            raise SystemExit("resumed trajectory checkpoint is missing a valid trajectory_warmup_steps value")
        if args.trajectory_warmup_steps not in {0, saved_warmup}:
            raise SystemExit("--trajectory-warmup-steps must match the resumed trajectory checkpoint")
        # Warmup is a property of the trajectory run, not of one invocation.
        # A resumed process must preserve the original global update boundary.
        args.trajectory_warmup_steps = saved_warmup
        saved_freeze_trajectory = resumed_metadata.get("freeze_trajectory_during_warmup", False)
        if not isinstance(saved_freeze_trajectory, bool):
            raise SystemExit("resumed trajectory checkpoint has invalid freeze_trajectory_during_warmup")
        if args.freeze_trajectory_during_warmup not in {False, saved_freeze_trajectory}:
            raise SystemExit("--freeze-trajectory-during-warmup must match the resumed checkpoint")
        args.freeze_trajectory_during_warmup = saved_freeze_trajectory
    completed_optimizer_steps = start_step // args.gradient_accumulation
    warmup_active = (
        (args.trajectory_conditioned or args.latent_action_conditioned)
        and args.trajectory_warmup_steps > completed_optimizer_steps
    )
    if warmup_active:
        set_inherited_trainable(model, False, freeze_trajectory=args.freeze_trajectory_during_warmup)
    assert_frozen_wan_base(model)
    model.train()
    model.transformer.enable_gradient_checkpointing()
    iterator = cycling(train_loader)
    layout = Track2WanLayout()
    checkpoint_format = (
        FORMAT_V4
        if args.trajectory_conditioned and args.latent_action_conditioned
        else FORMAT_V3
        if args.latent_action_conditioned
        else FORMAT_V2
        if args.trajectory_conditioned
        else FORMAT_V1
    )
    validation_seed = args.validation_seed if args.validation_seed is not None else args.seed + 10_000
    metadata_base = {
        "format": checkpoint_format,
        "dataset_root": str(root.resolve()),
        "base_model": str(base_model.resolve()),
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "track2_layout": layout.as_dict(),
        "history_actions": CONTEXT_ACTIONS,
        "future_actions": PREDICTION_FRAMES,
        "action_dim": ACTION_DIM,
        "lora_rank": args.lora_rank,
        "lora_trainable_tensor_count": len(lora_names),
        "trajectory_conditioned": args.trajectory_conditioned,
        "trajectory_hidden_dim": model.action_conditioner.trajectory_hidden_dim,
        "latent_action_conditioned": args.latent_action_conditioned,
        "latent_action_hidden_dim": model.action_conditioner.latent_action_hidden_dim,
        "learning_rate": args.learning_rate,
        "inherited_learning_rate": inherited_lr,
        "trajectory_warmup_steps": args.trajectory_warmup_steps,
        "freeze_trajectory_during_warmup": args.freeze_trajectory_during_warmup,
        "gradient_accumulation": args.gradient_accumulation,
        "optimizer_steps": optimizer_steps,
        "validation_batches": "all" if args.validation_batches == 0 else args.validation_batches,
        "latent_cache": str(Path(args.latent_cache).resolve()) if args.latent_cache else None,
        "motion_sampling": motion_sampling_metadata,
        "validation_noise_seed": validation_seed,
        **initialization_metadata,
        "loss": "future_latent_flow_mse_only",
        "rlinf_enabled": False,
    }
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        if warmup_active and step == args.trajectory_warmup_steps * args.gradient_accumulation + 1:
            set_inherited_trainable(model, True)
            assert_frozen_wan_base(model)
            warmup_active = False
            print(
                json.dumps(
                    {"step": step, "event": "trajectory_warmup_complete", "inherited_v1_trainable": True}
                ),
                flush=True,
            )
        batch = next(iterator)
        if use_latent_cache:
            history_actions, future_actions, clean = batch
            clean = clean.to(device, dtype=dtype, non_blocking=True)
        else:
            context, history_actions, future_actions, target = batch
            if vae is None or latent_mean is None or latent_std is None:
                raise RuntimeError("VAE is required when training latents are not cached")
            video = frames_to_wan_video(context, target).to(device, non_blocking=True)
            clean = encode_latents(vae, video, latent_mean, latent_std)
        slots = build_action_slots(history_actions.to(device, non_blocking=True), future_actions.to(device, non_blocking=True))
        slots = normalize_action_slots(slots, action_mean, action_std)
        noise = torch.randn_like(clean)
        sigma = sample_sigma(clean.shape[0], device, distribution=args.sigma_distribution)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            prediction, target_flow, _ = model(clean, noise, sigma, slots)
            loss = future_flow_loss(prediction, target_flow) / args.gradient_accumulation
        loss.backward()
        if step % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "future_flow_mse": float(loss.detach().cpu()) * args.gradient_accumulation,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "new_conditioning_learning_rate": optimizer.param_groups[-1]["lr"],
                        "trajectory_warmup_active": warmup_active,
                        "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
                    }
                ),
                flush=True,
            )
        needs_validation = step % args.validation_interval == 0 or step == args.steps
        improved = False
        if needs_validation:
            validation_loss = validation_flow_loss(
                val_loader,
                vae=vae,
                model=model,
                latent_mean=latent_mean,
                latent_std=latent_std,
                action_mean=action_mean,
                action_std=action_std,
                device=device,
                batches=None if args.validation_batches == 0 else args.validation_batches,
                seed=validation_seed,
            )
            history.append({"step": step, "future_flow_mse": validation_loss})
            improved = validation_loss < best_validation
            if improved:
                best_validation = validation_loss
            print(json.dumps({"step": step, "validation_future_flow_mse": validation_loss, "best": best_validation}), flush=True)
        if improved:
            best_metadata = {
                **metadata_base,
                "step": step,
                "best_validation_future_flow_mse": best_validation,
                "validation": history,
            }
            save_best_checkpoint(
                output,
                model=model,
                step=step,
                action_mean=action_mean,
                action_std=action_std,
                metadata=best_metadata,
            )
        if step % args.save_interval == 0 or step == args.steps or improved:
            metadata = {
                **metadata_base,
                "step": step,
                "best_validation_future_flow_mse": best_validation,
                "validation": history,
            }
            save_checkpoint(
                output,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                action_mean=action_mean,
                action_std=action_std,
                metadata=metadata,
            )


if __name__ == "__main__":
    main()
