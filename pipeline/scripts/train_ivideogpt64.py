#!/usr/bin/env python3
"""Train the minimal real iVideoGPT-64 Track 2 action-conditioning adapter."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from wam_pipeline.ivideogpt_upstream import import_upstream, load_bair_llm, set_context_length


EPISODE_PATTERN = re.compile(r"^episode(\d+)_\d{5}\.npz$")


def episode_id(path: Path) -> int:
    match = EPISODE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"window name must be episode<id>_<start>.npz: {path.name}")
    return int(match.group(1))


class WindowDataset(Dataset):
    def __init__(self, directory: Path, episodes: list[int] | None = None) -> None:
        self.paths = sorted(directory.glob("episode*_*.npz"))
        if episodes is not None:
            allowed = set(episodes)
            self.paths = [path for path in self.paths if episode_id(path) in allowed]
        if not self.paths:
            raise ValueError("no prepared NPZ windows found")
        with np.load(self.paths[0], allow_pickle=False) as sample:
            self._validate(sample, self.paths[0])

    @staticmethod
    def _validate(sample, path: Path) -> None:
        expected = {
            "context_frames": (5, 256, 256, 3),
            "history_actions": (4, 14),
            "future_actions": (8, 14),
            "target_frames": (8, 256, 256, 3),
        }
        for name, shape in expected.items():
            if name not in sample or sample[name].shape != shape:
                actual = None if name not in sample else sample[name].shape
                raise ValueError(f"{path}:{name} has {actual}; expected {shape}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as sample:
            self._validate(sample, self.paths[index])
            frames = np.concatenate([sample["context_frames"], sample["target_frames"]], axis=0)
            # The extra final action is required by upstream action[:, context-1:-1].
            actions = np.concatenate(
                [sample["history_actions"], sample["future_actions"], sample["future_actions"][-1:]], axis=0
            ).astype(np.float32)
        return torch.from_numpy(frames.copy()), torch.from_numpy(actions.copy())


def frames_to_64(frames: torch.Tensor) -> torch.Tensor:
    """Convert uint8 RGB to the upstream iVideoGPT [0,1] 64-pixel convention."""
    batch, length = frames.shape[:2]
    frames = frames.permute(0, 1, 4, 2, 3).reshape(batch * length, 3, 256, 256).float()
    frames = functional.interpolate(frames, size=(64, 64), mode="bilinear", align_corners=False)
    return frames.div(255.0).reshape(batch, length, 3, 64, 64)


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit action normalization from training episodes only, never validation."""
    action_sum = np.zeros(14, dtype=np.float64)
    square_sum = np.zeros(14, dtype=np.float64)
    count = 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as sample:
            actions = np.concatenate([sample["history_actions"], sample["future_actions"]], axis=0).astype(np.float64)
        action_sum += actions.sum(axis=0)
        square_sum += np.square(actions).sum(axis=0)
        count += len(actions)
    mean = action_sum / count
    variance = np.maximum(square_sum / count - np.square(mean), 1e-8)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32))


def normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (actions - mean) / std


def evenly_sample_validation_indices(dataset: WindowDataset, maximum: int) -> list[int]:
    """Choose a fixed, episode-balanced validation subset without leakage."""
    if maximum < 1:
        raise ValueError("validation sample count must be positive")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, path in enumerate(dataset.paths):
        grouped[episode_id(path)].append(index)
    if maximum >= len(dataset):
        return list(range(len(dataset)))

    episodes = sorted(grouped)
    selected: list[int] = []
    # Allocate the bounded validation budget across all held-out episodes.
    base, remainder = divmod(maximum, len(episodes))
    for rank, episode in enumerate(episodes):
        paths = grouped[episode]
        wanted = min(len(paths), base + (1 if rank < remainder else 0))
        if wanted == 0:
            continue
        positions = np.linspace(0, len(paths) - 1, num=wanted, dtype=np.int64)
        selected.extend(paths[position] for position in positions)
    return sorted(selected)


def token_loss(
    loader: DataLoader,
    tokenizer,
    model,
    device: torch.device,
    dtype: torch.dtype,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    max_batches: int,
) -> float:
    """Evaluate teacher-forced token cross-entropy on unseen full episodes."""
    model.eval()
    losses = []
    with torch.no_grad():
        for index, (raw_frames, actions) in enumerate(loader):
            if index >= max_batches:
                break
            frames = frames_to_64(raw_frames).to(device)
            actions = normalize_actions(actions.to(device=device, dtype=dtype), action_mean, action_std)
            input_ids, labels = tokenizer.tokenize(frames, context_length=5)
            losses.append(float(model(input_ids=input_ids, labels=labels, action=actions).loss.cpu()))
    if not losses:
        raise RuntimeError("validation loader yielded no batches")
    return float(np.mean(losses))


def write_metadata(
    output: Path,
    upstream_checkpoint: Path,
    windows: Path,
    train_window_count: int,
    validation_window_count: int,
    split_manifest: Path | None,
    status: str,
    validation_losses: list[dict[str, float]],
    checkpoint_step: int | None = None,
    best_validation_token_loss: float | None = None,
    validation_sample_count: int | None = None,
    llm_tuning: str = "frozen",
) -> None:
    np.savez(
        output / "track2_ivideogpt_config.npz",
        context_frames=np.asarray(5),
        action_dim=np.asarray(14),
        prediction_frames=np.asarray(8),
        working_resolution=np.asarray(64),
        serving_resolution=np.asarray(256),
    )
    manifest = {
        "backend": "ivideogpt",
        "upstream_checkpoint": str(upstream_checkpoint.resolve()),
        "windows": str(windows.resolve()),
        "train_window_count": train_window_count,
        "validation_window_count": validation_window_count,
        "split_manifest": str(split_manifest.resolve()) if split_manifest else None,
        "context_frames": 5,
        "history_actions": 4,
        "future_actions": 8,
        "target_frames": 8,
        "action_dim": 14,
        "working_resolution": 64,
        "serving_resolution": 256,
        "status": status,
        "validation_losses": validation_losses,
        "checkpoint_step": checkpoint_step,
        "best_validation_token_loss": best_validation_token_loss,
        "validation_sample_count": validation_sample_count,
        "llm_tuning": llm_tuning,
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def save_checkpoint(
    output: Path,
    model,
    upstream: Path,
    windows: Path,
    train_window_count: int,
    validation_window_count: int,
    split_manifest: Path | None,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    source: Path,
    step: int,
    losses: list[float],
    validation_losses: list[dict[str, float]],
    best_validation_token_loss: float,
    validation_sample_count: int,
    status: str,
    llm_tuning: str,
    lora_config: dict[str, Any] | None,
) -> None:
    """Write a standalone runtime checkpoint; every snapshot is loadable."""
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "format": "track2-ivideogpt-v2",
        "action_linear": {name: value.detach().cpu() for name, value in model.action_linear.state_dict().items()},
        "steps": step,
        "losses": losses,
        "upstream_source": str(source),
        "llm_tuning": llm_tuning,
    }
    if llm_tuning == "lora":
        from peft import get_peft_model_state_dict

        state["lora_config"] = lora_config
        state["llm_lora"] = {
            name: value.detach().cpu() for name, value in get_peft_model_state_dict(model.llm).items()
        }
    torch.save(state, output / "track2_ivideogpt_state.pt")
    np.savez(
        output / "action_normalization.npz",
        mean=action_mean.float().cpu().numpy(),
        std=action_std.float().cpu().numpy(),
    )
    write_metadata(
        output,
        upstream,
        windows,
        train_window_count,
        validation_window_count,
        split_manifest,
        status,
        validation_losses,
        checkpoint_step=step,
        best_validation_token_loss=best_validation_token_loss,
        validation_sample_count=validation_sample_count,
        llm_tuning=llm_tuning,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, help="Directory produced by adapt_robotwin.py")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--upstream-checkpoint", required=True, help="Downloaded iVideoGPT 64 checkpoint")
    parser.add_argument("--steps", type=int, default=4, help="Action-projection optimization steps (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-manifest", help="Episode-disjoint split JSON from make_episode_split.py")
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--llm-tuning", choices=("frozen", "lora"), default="frozen")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated Llama projection modules adapted when --llm-tuning lora",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.validation_interval < 1 or args.validation_batches < 1:
        raise SystemExit("steps, batch size, and validation settings must be positive")
    if args.lora_r < 1 or args.lora_alpha < 1 or not 0 <= args.lora_dropout < 1:
        raise SystemExit("LoRA rank/alpha must be positive and dropout must be in [0, 1)")

    windows = Path(args.windows)
    output = Path(args.output)
    upstream = Path(args.upstream_checkpoint)
    split_manifest = Path(args.split_manifest) if args.split_manifest else None
    split = None
    if split_manifest:
        split = json.loads(split_manifest.read_text())
        if split.get("format") != "track2-episode-split-v1":
            raise SystemExit("unsupported split manifest format")
    dataset = WindowDataset(windows, split["train_episodes"] if split else None)
    validation_dataset = WindowDataset(windows, split["validation_episodes"] if split else None)
    if not (upstream / "tokenizer" / "config.json").is_file() or not (upstream / "transformer" / "config.json").is_file():
        raise SystemExit("upstream checkpoint must contain tokenizer/ and transformer/")
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_metadata(
            output,
            upstream,
            windows,
            len(dataset),
            len(validation_dataset),
            split_manifest,
            "prepared-not-trained",
            [],
            llm_tuning=args.llm_tuning,
        )
        print(json.dumps({"status": "prepared-not-trained", "train_windows": len(dataset), "validation_windows": len(validation_dataset)}, indent=2))
        return

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    CompressiveVQModel, HeadModelWithAction, source = import_upstream()
    tokenizer = CompressiveVQModel.from_pretrained(str(upstream / "tokenizer"))
    set_context_length(tokenizer, 5)
    tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    llm = load_bair_llm(upstream / "transformer", dtype, device).eval()
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    lora_config: dict[str, Any] | None = None
    if args.llm_tuning == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = tuple(name.strip() for name in args.lora_target_modules.split(",") if name.strip())
        if not target_modules:
            raise SystemExit("--lora-target-modules cannot be empty")
        llm = get_peft_model(
            llm,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(target_modules),
                bias="none",
            ),
        )
        lora_config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(target_modules),
        }
    model = HeadModelWithAction(
        llm=llm,
        action_dim=14,
        prelude_tokens_num=(256 + 1) * 5 - 1,
        tokens_num_per_dyna=16,
        context=5,
        segment_length=13,
    ).to(device, dtype=dtype)
    model.train()
    if args.llm_tuning == "frozen":
        model.llm.eval()  # Freeze dropout while optimizing only the action adapter.
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(json.dumps({"llm_tuning": args.llm_tuning, "trainable_parameters": sum(p.numel() for p in trainable_parameters)}))
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    action_mean, action_std = action_statistics(dataset)
    action_mean, action_std = action_mean.to(device=device, dtype=dtype), action_std.to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, generator=generator)
    validation_sample_count = min(len(validation_dataset), args.validation_batches * args.batch_size)
    validation_indices = evenly_sample_validation_indices(validation_dataset, validation_sample_count)
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    iterator = iter(loader)
    losses = []
    validation_losses: list[dict[str, float]] = []
    best_validation_token_loss = float("inf")
    best_step = 0
    for step in range(1, args.steps + 1):
        try:
            raw_frames, actions = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_frames, actions = next(iterator)
        frames = frames_to_64(raw_frames).to(device)
        actions = normalize_actions(actions.to(device=device, dtype=dtype), action_mean, action_std)
        # ``inference_mode`` creates special tensors that the causal-LM loss
        # cannot retain for the action adapter's backward pass.  ``no_grad``
        # still avoids tokenizer gradients while producing ordinary tensors.
        with torch.no_grad():
            input_ids, labels = tokenizer.tokenize(frames, context_length=5)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            result = model(input_ids=input_ids, labels=labels, action=actions)
            loss = result.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(json.dumps({"step": step, "loss": losses[-1]}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            validation_loss = token_loss(
                validation_loader, tokenizer, model, device, dtype, action_mean, action_std, args.validation_batches
            )
            validation_losses.append({"step": float(step), "token_loss": validation_loss})
            model.train()
            if args.llm_tuning == "frozen":
                model.llm.eval()
            checkpoint = output / "checkpoints" / f"checkpoint_step_{step:06d}"
            save_checkpoint(
                checkpoint,
                model,
                upstream,
                windows,
                len(dataset),
                len(validation_dataset),
                split_manifest,
                action_mean,
                action_std,
                source,
                step,
                losses,
                validation_losses,
                min(best_validation_token_loss, validation_loss),
                validation_sample_count,
                "validation-checkpoint",
                args.llm_tuning,
                lora_config,
            )
            if validation_loss < best_validation_token_loss:
                best_validation_token_loss = validation_loss
                best_step = step
                save_checkpoint(
                    output / "best",
                    model,
                    upstream,
                    windows,
                    len(dataset),
                    len(validation_dataset),
                    split_manifest,
                    action_mean,
                    action_std,
                    source,
                    step,
                    losses,
                    validation_losses,
                    best_validation_token_loss,
                    validation_sample_count,
                    "best-validation-action-head",
                    args.llm_tuning,
                    lora_config,
                )
                # The requested output itself remains directly runnable and
                # always points to the best validation checkpoint.
                save_checkpoint(
                    output,
                    model,
                    upstream,
                    windows,
                    len(dataset),
                    len(validation_dataset),
                    split_manifest,
                    action_mean,
                    action_std,
                    source,
                    step,
                    losses,
                    validation_losses,
                    best_validation_token_loss,
                    validation_sample_count,
                    "best-validation-action-head",
                    args.llm_tuning,
                    lora_config,
                )
            print(json.dumps({"step": step, "validation_token_loss": validation_loss}), flush=True)

    # Keep the canonical root's history current while retaining its best weights.
    write_metadata(
        output,
        upstream,
        windows,
        len(dataset),
        len(validation_dataset),
        split_manifest,
        "best-validation-action-head",
        validation_losses,
        checkpoint_step=best_step,
        best_validation_token_loss=best_validation_token_loss,
        validation_sample_count=validation_sample_count,
        llm_tuning=args.llm_tuning,
    )
    print(
        json.dumps(
            {
                "status": "trained-action-head",
                "steps": args.steps,
                "final_train_loss": losses[-1],
                "final_validation_token_loss": validation_losses[-1]["token_loss"],
                "best_validation_token_loss": best_validation_token_loss,
                "best_checkpoint_step": best_step,
                "validation_samples_per_check": validation_sample_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
