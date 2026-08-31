#!/usr/bin/env python3
"""Train a parent on the same multi-chunk distribution used by Track 2 RL.

Each item is a contiguous sequence of ordinary 5-context/8-future training
windows.  A chunk is rolled out autoregressively, its predicted last five
frames become the next chunk's context, and the graph is detached only at the
chunk boundary.  Losses are backpropagated per chunk and accumulated before a
single optimizer step, so memory stays bounded while the model is exposed to
its own long-horizon context drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_reward_aligned_autoregressive_unet import (
    OneStepActionUNet,
    official_prompts,
    reward_progress_loss,
    rollout,
    save,
    to_frames,
    visual_loss,
)


CHUNK_FRAMES = 8


class MultiChunkWindows(Dataset):
    """Contiguous 8-frame chunks reconstructed from stride-divisible windows."""

    def __init__(
        self,
        root: Path,
        episode_ids: list[int],
        chunks: int,
        chunk_stride: int,
        arm_filter: str = "all",
        instruction_by_seed: dict[int, str] | None = None,
        instruction_by_episode: dict[int, str] | None = None,
    ) -> None:
        if chunks < 2:
            raise ValueError("multi-chunk training requires at least two chunks")
        if chunk_stride != CHUNK_FRAMES:
            raise ValueError("chunk-stride must equal the official 8-frame chunk")
        if arm_filter not in {"all", "left", "right"}:
            raise ValueError("arm-filter must be all, left, or right")
        allowed = set(episode_ids)
        index: dict[tuple[int, int], Path] = {}
        metadata: dict[tuple[int, int], tuple[bool, bool, int]] = {}
        for path in sorted(root.glob("episode*_*.npz")):
            episode_id = int(path.name.split("_")[0][7:])
            if episode_id not in allowed:
                continue
            with np.load(path, allow_pickle=False) as values:
                start = int(values["start"])
                key = (episode_id, start)
                if key in index:
                    raise ValueError(f"duplicate window {key}")
                index[key] = path
                # Older/synthetic window packs may omit the audit fields.  Do
                # not discard those windows: infer the active arm from the
                # action block with the larger mean motion and use neutral
                # defaults for labels that are only sampling hints.
                if "arm_right" in values.files:
                    arm_right = bool(values["arm_right"])
                else:
                    actions = np.concatenate(
                        (values["history_actions"], values["future_actions"]),
                        axis=0,
                    )
                    delta = np.abs(np.diff(actions.astype(np.float32), axis=0))
                    arm_right = bool(
                        delta[:, 7:].mean() > delta[:, :7].mean()
                    )
                capture_success = (
                    bool(values["capture_success"])
                    if "capture_success" in values.files
                    else True
                )
                synthetic_seed = (
                    int(values["synthetic_seed"])
                    if "synthetic_seed" in values.files
                    else 0
                )
                metadata[key] = (arm_right, capture_success, synthetic_seed)
        sequences: list[tuple[Path, ...]] = []
        sequence_metadata: list[tuple[bool, bool, int, int, int]] = []
        for episode_id, start in sorted(index):
            keys = [
                (episode_id, start + chunk * chunk_stride)
                for chunk in range(chunks)
            ]
            if not all(key in index for key in keys):
                continue
            rows = [metadata[key] for key in keys]
            if len(set(rows)) != 1:
                raise ValueError(f"metadata changes inside sequence {keys}")
            arm_right, capture_success, synthetic_seed = rows[0]
            if arm_filter == "left" and arm_right:
                continue
            if arm_filter == "right" and not arm_right:
                continue
            sequences.append(tuple(index[key] for key in keys))
            sequence_metadata.append(
                (arm_right, capture_success, start, episode_id, synthetic_seed)
            )
        if not sequences:
            raise ValueError("no complete multi-chunk sequences selected")
        self.sequences = sequences
        self.arm_right = np.asarray(
            [row[0] for row in sequence_metadata], dtype=np.bool_
        )
        self.capture_success = np.asarray(
            [row[1] for row in sequence_metadata], dtype=np.bool_
        )
        self.start = np.asarray(
            [row[2] for row in sequence_metadata], dtype=np.int64
        )
        self.episode_id = np.asarray(
            [row[3] for row in sequence_metadata], dtype=np.int64
        )
        self.synthetic_seed = np.asarray(
            [row[4] for row in sequence_metadata], dtype=np.int64
        )
        self.chunks = chunks
        self.instruction_by_seed = instruction_by_seed or {}
        self.instruction_by_episode = instruction_by_episode or {}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        paths = self.sequences[index]
        contexts = []
        histories = []
        futures = []
        targets = []
        instructions = []
        for path in paths:
            with np.load(path, allow_pickle=False) as values:
                contexts.append(values["context_frames"].copy())
                histories.append(values["history_actions"].copy())
                futures.append(values["future_actions"].copy())
                targets.append(values["target_frames"].copy())
                if "task_instruction" in values.files:
                    instructions.append(str(values["task_instruction"]))
        for chunk in range(1, len(paths)):
            if not np.array_equal(contexts[chunk], targets[chunk - 1][-5:]):
                raise ValueError(f"non-contiguous RGB windows: {paths[chunk - 1:chunk + 1]}")
            if not np.array_equal(histories[chunk], futures[chunk - 1][-4:]):
                raise ValueError(f"non-contiguous actions: {paths[chunk - 1:chunk + 1]}")
        if instructions and len(set(instructions)) != 1:
            raise ValueError(f"instruction changes inside {paths[0]}")
        seed = int(self.synthetic_seed[index])
        instruction = (
            instructions[0]
            if instructions
            else self.instruction_by_seed.get(
                seed,
                self.instruction_by_episode.get(int(self.episode_id[index]), ""),
            )
        )
        return (
            torch.from_numpy(contexts[0]),
            torch.from_numpy(histories[0]),
            torch.from_numpy(np.concatenate(futures, axis=0)),
            torch.from_numpy(np.concatenate(targets, axis=0)),
            torch.tensor(bool(self.arm_right[index])),
            torch.tensor(bool(self.capture_success[index])),
            instruction,
        )


def load_instruction_map(path: str | None) -> dict[int, str] | None:
    if path is None:
        return None
    document = json.loads(Path(path).read_text())
    raw = document.get("seed_to_instruction", document)
    return {int(seed): str(value) for seed, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--reset-manifest", required=True)
    parser.add_argument("--instruction-map")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-stride", type=int, default=8)
    parser.add_argument("--arm-filter", choices=("all", "left", "right"), default="left")
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--reward-loss-weight", type=float, default=0.02)
    parser.add_argument(
        "--reward-objective", choices=("logit", "probability"), default="probability"
    )
    parser.add_argument("--reward-probability-scale-floor", type=float, default=1e-5)
    parser.add_argument("--reward-delta-weight", type=float, default=1.0)
    parser.add_argument("--reward-terminal-weight", type=float, default=2.0)
    parser.add_argument("--right-weight", type=float, default=2.0)
    parser.add_argument("--success-weight", type=float, default=3.0)
    parser.add_argument("--late-weight", type=float, default=3.0)
    parser.add_argument("--late-start", type=int, default=64)
    parser.add_argument("--terminal-visual-weight", type=float, default=1.0)
    parser.add_argument("--prompts-per-arm", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1616)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    positive_counts = (
        args.steps,
        args.batch_size,
        args.chunks,
        args.chunk_stride,
        args.prompts_per_arm,
        args.checkpoint_interval,
    )
    if min(positive_counts) < 1:
        raise SystemExit("counts must be positive")
    positive_values = (
        args.learning_rate,
        args.reward_loss_weight,
        args.reward_probability_scale_floor,
        args.reward_delta_weight,
        args.reward_terminal_weight,
        args.right_weight,
        args.success_weight,
        args.late_weight,
        args.terminal_visual_weight,
        args.max_grad_norm,
    )
    if args.reward_loss_weight < 0 or min(
        args.learning_rate, args.reward_probability_scale_floor,
        args.reward_delta_weight, args.reward_terminal_weight,
        args.right_weight, args.success_weight, args.late_weight,
        args.terminal_visual_weight, args.max_grad_norm
    ) <= 0:
        raise SystemExit("learning rate and non-reward weights must be positive")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    instruction_by_episode = {
        int(episode): str(instruction)
        for episode, instruction in split.get("episode_to_instruction", {}).items()
    }
    dataset = MultiChunkWindows(
        Path(args.windows),
        split["train_episodes"],
        chunks=args.chunks,
        chunk_stride=args.chunk_stride,
        arm_filter=args.arm_filter,
        instruction_by_seed=load_instruction_map(args.instruction_map),
        instruction_by_episode=instruction_by_episode,
    )
    weights = np.ones(len(dataset), dtype=np.float64)
    weights[dataset.arm_right] *= args.right_weight
    weights[dataset.capture_success] *= args.success_weight
    weights[dataset.start >= args.late_start] *= args.late_weight
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    iterator = iter(loader)
    device = torch.device(args.device)

    init = Path(args.init_checkpoint)
    state = torch.load(init / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise ValueError("initial checkpoint is not an autoregressive parent")
    model = OneStepActionUNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    with np.load(init / "action_normalization.npz", allow_pickle=False) as values:
        mean = torch.from_numpy(values["mean"].astype(np.float32)).to(device)
        std = torch.from_numpy(values["std"].astype(np.float32)).to(device)

    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        args.reward_checkpoint, config={"t5_model_name": args.t5_model}
    ).to(device).eval().requires_grad_(False)
    left_prompts = official_prompts(
        Path(args.reset_manifest), "left", args.prompts_per_arm
    )
    right_prompts = official_prompts(
        Path(args.reset_manifest), "right", args.prompts_per_arm
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    output = Path(args.output)
    history_log = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        context_raw, history_raw, future_raw, target_raw, arm_right, capture_success, exact = batch
        context = to_frames(context_raw, device)
        target = to_frames(target_raw, device)
        history_actions = ((history_raw.to(device) - mean) / std).float()
        future = ((future_raw.to(device) - mean) / std).float()
        future = future.reshape(future.shape[0], args.chunks, CHUNK_FRAMES, -1)
        target = target.reshape(
            target.shape[0], args.chunks, CHUNK_FRAMES, *target.shape[2:]
        )
        prompts = []
        for index, (is_right, instruction) in enumerate(zip(arm_right.tolist(), exact)):
            if instruction:
                lowered = str(instruction).lower()
                expected = "right arm" if is_right else "left arm"
                opposite = "left arm" if is_right else "right arm"
                if opposite in lowered or expected not in lowered:
                    raise ValueError(
                        f"episode instruction is not explicitly arm-consistent: "
                        f"expected {expected!r}, got {instruction!r}"
                    )
                prompts.append(instruction)
            else:
                choices = right_prompts if is_right else left_prompts
                prompts.append(choices[(step + index) % len(choices)])

        optimizer.zero_grad(set_to_none=True)
        chunk_rows = []
        for chunk_index in range(args.chunks):
            chunk_target = target[:, chunk_index]
            target_context_last = (
                context[:, -1]
                if chunk_index == 0
                else target[:, chunk_index - 1, -1]
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = rollout(
                    model, context, history_actions, future[:, chunk_index]
                )
                image_loss = visual_loss(
                    prediction, chunk_target, target_context_last
                )
            with torch.autocast(device_type=device.type, enabled=False):
                task_loss, reward_parts = reward_progress_loss(
                    reward_model,
                    prediction.float(),
                    chunk_target.float(),
                    target_context_last.float(),
                    prompts,
                    args.reward_objective,
                    args.reward_probability_scale_floor,
                    args.reward_delta_weight,
                    args.reward_terminal_weight,
                )
            chunk_visual_weight = (
                args.terminal_visual_weight
                if chunk_index == args.chunks - 1 else 1.0
            )
            loss = chunk_visual_weight * image_loss.float() + args.reward_loss_weight * task_loss
            (loss / args.chunks).backward()
            chunk_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "visual_loss": float(image_loss.detach().cpu()),
                    "reward_loss": float(task_loss.detach().cpu()),
                    **{
                        key: float(value.cpu())
                        for key, value in reward_parts.items()
                    },
                }
            )
            context = torch.cat((context, prediction.detach()), dim=1)[:, -5:]
            history_actions = future[:, chunk_index, -4:].detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()
        if step == 1 or step % 10 == 0:
            row = {
                "step": step,
                "loss": float(np.mean([item["loss"] for item in chunk_rows])),
                "visual_loss": float(
                    np.mean([item["visual_loss"] for item in chunk_rows])
                ),
                "reward_loss": float(
                    np.mean([item["reward_loss"] for item in chunk_rows])
                ),
                "grad_norm": float(grad_norm.detach().cpu()),
                "right_fraction": float(arm_right.float().mean()),
                "success_fraction": float(capture_success.float().mean()),
                "chunk_metrics": chunk_rows,
            }
            history_log.append(row)
            print(json.dumps(row), flush=True)

        if step % args.checkpoint_interval == 0 or step == args.steps:
            metadata = {
                "format": "track2-multichunk-reward-aligned-autoregressive-parent-v1",
                "checkpoint_step": step,
                "initialization_checkpoint": str(init.resolve()),
                "windows": str(Path(args.windows).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "train_sequence_count": len(dataset),
                "frozen_official_reward_checkpoint": str(
                    Path(args.reward_checkpoint).resolve()
                ),
                "reward_model_modified": False,
                "truncated_bptt": {
                    "chunk_frames": CHUNK_FRAMES,
                    "chunks": args.chunks,
                    "predicted_context_between_chunks": True,
                    "detach_at_chunk_boundary": True,
                    "optimizer_step_after_all_chunks": True,
                },
                "training": vars(args),
                "prompt_protocol": {
                    "left": left_prompts,
                    "right": right_prompts,
                },
                "sampling": {
                    "expected_right_fraction": float(
                        weights[dataset.arm_right].sum() / weights.sum()
                    ),
                    "expected_success_fraction": float(
                        weights[dataset.capture_success].sum() / weights.sum()
                    ),
                    "expected_late_fraction": float(
                        weights[dataset.start >= args.late_start].sum()
                        / weights.sum()
                    ),
                },
                "history": history_log,
            }
            save(
                output / f"checkpoint_step_{step:06d}",
                model,
                mean,
                std,
                metadata,
            )


if __name__ == "__main__":
    main()
