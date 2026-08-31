#!/usr/bin/env python3
"""Fine-tune the parent against frozen official reward-progress targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wam_pipeline.autoregressive_unet import OneStepActionUNet


def window_metadata(values) -> tuple[bool, bool, int, bool]:
    if "arm_right" in values.files:
        arm_right = bool(values["arm_right"])
    else:
        actions = np.concatenate(
            (values["history_actions"], values["future_actions"]), axis=0
        )
        delta = np.abs(np.diff(actions, axis=0))
        arm_right = bool(delta[:, 7:].mean() > delta[:, :7].mean())
    capture_success = (
        bool(values["capture_success"])
        if "capture_success" in values.files
        else True
    )
    return (
        arm_right,
        capture_success,
        int(values["start"]),
        "synthetic_seed" in values.files,
    )


class RewardWindows(Dataset):
    def __init__(
        self,
        root: Path,
        episode_ids: list[int],
        instruction_by_seed: dict[int, str] | None = None,
        min_start: int | None = None,
        max_start: int | None = None,
        arm_filter: str = "both",
        instruction_by_episode: dict[int, str] | None = None,
    ) -> None:
        allowed = set(episode_ids)
        self.paths = []
        metadata = []
        for path in sorted(root.glob("episode*_*.npz")):
            if int(path.name.split("_")[0][7:]) not in allowed:
                continue
            with np.load(path, allow_pickle=False) as values:
                row = window_metadata(values)
            if min_start is not None and row[2] < min_start:
                continue
            if max_start is not None and row[2] > max_start:
                continue
            if arm_filter == "right" and not row[0]:
                continue
            if arm_filter == "left" and row[0]:
                continue
            self.paths.append(path)
            metadata.append(row)
        if not self.paths:
            raise ValueError("no windows selected")
        self.arm_right = np.asarray([row[0] for row in metadata], dtype=np.bool_)
        self.capture_success = np.asarray([row[1] for row in metadata], dtype=np.bool_)
        self.start = np.asarray([row[2] for row in metadata], dtype=np.int64)
        self.is_synthetic = np.asarray([row[3] for row in metadata], dtype=np.bool_)
        self.instruction_by_seed = instruction_by_seed or {}
        self.instruction_by_episode = instruction_by_episode or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index], allow_pickle=False) as values:
            arm_right, capture_success, _, _ = window_metadata(values)
            synthetic_seed = (
                int(values["synthetic_seed"])
                if "synthetic_seed" in values.files
                else -1
            )
            episode_id = int(self.paths[index].name.split("_")[0][7:])
            return (
                torch.from_numpy(values["context_frames"].copy()),
                torch.from_numpy(values["history_actions"].copy()),
                torch.from_numpy(values["future_actions"].copy()),
                torch.from_numpy(values["target_frames"].copy()),
                torch.tensor(arm_right),
                torch.tensor(capture_success),
                (
                    str(values["task_instruction"])
                    if "task_instruction" in values.files
                    else self.instruction_by_seed.get(
                        synthetic_seed, self.instruction_by_episode.get(episode_id, "")
                    )
                ),
            )


def official_prompts(manifest: Path, side: str, limit: int) -> list[str]:
    rows = json.loads(manifest.read_text())["episodes"]
    prompts = sorted({str(row["instruction"]) for row in rows if f"{side} arm" in str(row["instruction"]).lower()})
    if not prompts:
        raise ValueError(f"no {side}-arm prompts in {manifest}")
    if len(prompts) <= limit:
        return prompts
    return [prompts[index] for index in np.linspace(0, len(prompts) - 1, limit, dtype=int)]


def to_frames(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.permute(0, 1, 4, 2, 3).to(device, non_blocking=True).float().div(255)


def rollout(model, context, history, future) -> torch.Tensor:
    predictions = []
    for action in future.unbind(1):
        prediction = model(context, torch.cat((history, action[:, None]), 1)).clamp(0, 1)
        predictions.append(prediction)
        context = torch.cat((context[:, 1:], prediction[:, None]), 1)
        history = torch.cat((history[:, 1:], action[:, None]), 1)
    return torch.stack(predictions, 1)


def visual_loss(prediction: torch.Tensor, target: torch.Tensor, context_last: torch.Tensor) -> torch.Tensor:
    previous_target = torch.cat((context_last[:, None], target[:, :-1]), 1)
    previous_prediction = torch.cat((context_last[:, None], prediction[:, :-1]), 1)
    changed = (target - previous_target).abs().mean(2, keepdim=True) >= 0.03
    motion_weight = 1 + 2 * changed.to(prediction.dtype)
    horizon = torch.arange(1, prediction.shape[1] + 1, device=prediction.device, dtype=prediction.dtype)
    horizon = (horizon.pow(0.5) / horizon.pow(0.5).mean()).view(1, -1, 1, 1, 1)
    difference = prediction - target
    pixel = (horizon * motion_weight * difference.abs()).mean() + 0.05 * (
        horizon * motion_weight * difference.square()
    ).mean()
    temporal = (
        horizon
        * motion_weight
        * ((prediction - previous_prediction) - (target - previous_target)).abs()
    ).mean()
    flat_prediction = prediction.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    pred_high = flat_prediction - F.avg_pool2d(flat_prediction, 3, 1, 1, count_include_pad=False)
    target_high = flat_target - F.avg_pool2d(flat_target, 3, 1, 1, count_include_pad=False)
    texture = (pred_high - target_high).abs().mean()
    return pixel + 0.35 * temporal + 0.15 * texture


def reward_progress_loss(
    reward_model,
    prediction: torch.Tensor,
    target: torch.Tensor,
    context_last: torch.Tensor,
    instructions: list[str],
    objective: str,
    probability_scale_floor: float,
    delta_weight: float,
    terminal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch, time = prediction.shape[:2]
    repeated = [instruction for instruction in instructions for _ in range(time)]
    predicted_output = reward_model(prediction.flatten(0, 1).float(), instructions=repeated)
    predicted_logits = predicted_output["logits"].reshape(batch, time)
    predicted_probabilities = predicted_output["probabilities"].reshape(batch, time)
    with torch.no_grad():
        target_output = reward_model(target.flatten(0, 1).float(), instructions=repeated)
        context_output = reward_model(context_last.float(), instructions=instructions)
        target_logits = target_output["logits"].reshape(batch, time)
        context_logits = context_output["logits"]
        target_probabilities = target_output["probabilities"].reshape(batch, time)
        context_probabilities = context_output["probabilities"]
    if objective == "logit":
        predicted_values, target_values, context_values = (
            predicted_logits, target_logits, context_logits
        )
        scale = torch.ones((batch, 1), device=prediction.device)
    elif objective == "probability":
        predicted_values, target_values, context_values = (
            predicted_probabilities, target_probabilities, context_probabilities
        )
        scale = torch.cat((context_values[:, None], target_values), 1).std(1, keepdim=True)
        scale = scale.clamp_min(probability_scale_floor)
    else:
        raise ValueError(f"unsupported reward objective: {objective}")
    absolute = F.smooth_l1_loss(
        (predicted_values - target_values) / scale, torch.zeros_like(predicted_values), beta=0.1
    )
    predicted_delta = torch.diff(torch.cat((context_values[:, None], predicted_values), 1), dim=1)
    target_delta = torch.diff(torch.cat((context_values[:, None], target_values), 1), dim=1)
    delta = F.smooth_l1_loss(
        (predicted_delta - target_delta) / scale, torch.zeros_like(predicted_delta), beta=0.05
    )
    terminal_difference = (predicted_values[:, -1] - context_values) - (
        target_values[:, -1] - context_values
    )
    terminal = F.smooth_l1_loss(
        terminal_difference / scale[:, 0], torch.zeros_like(terminal_difference), beta=0.05
    )
    loss = absolute + delta_weight * delta + terminal_weight * terminal
    return loss, {
        "reward_absolute_loss": absolute.detach(),
        "reward_delta_loss": delta.detach(),
        "reward_terminal_loss": terminal.detach(),
        "reward_probability_mae": (predicted_probabilities - target_probabilities).abs().mean().detach(),
        "reward_probability_delta_mae": (
            torch.diff(torch.cat((context_probabilities[:, None], predicted_probabilities), 1), dim=1)
            - torch.diff(torch.cat((context_probabilities[:, None], target_probabilities), 1), dim=1)
        ).abs().mean().detach(),
        "reward_probability_terminal_mae": (
            predicted_probabilities[:, -1] - target_probabilities[:, -1]
        ).abs().mean().detach(),
    }


def save(output: Path, model, mean, std, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-autoregressive-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(
        output / "track2_autoregressive_unet_config.npz",
        context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8),
        working_resolution=np.asarray(256), serving_resolution=np.asarray(256),
    )
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--reset-manifest", required=True)
    parser.add_argument("--instruction-map", help="Optional JSON seed-to-exact-instruction mapping")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--reward-loss-weight", type=float, default=0.01)
    parser.add_argument("--reward-objective", choices=("logit", "probability"), default="logit")
    parser.add_argument("--reward-probability-scale-floor", type=float, default=1e-3)
    parser.add_argument("--reward-delta-weight", type=float, default=1.0)
    parser.add_argument("--reward-terminal-weight", type=float, default=2.0)
    parser.add_argument("--arm-filter", choices=("both", "left", "right"), default="both")
    parser.add_argument("--right-weight", type=float, default=2.0)
    parser.add_argument("--success-weight", type=float, default=3.0)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--late-start", type=int, default=80)
    parser.add_argument("--min-start", type=int)
    parser.add_argument("--max-start", type=int)
    parser.add_argument("--official-weight", type=float, default=1.0)
    parser.add_argument("--prompts-per-arm", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.checkpoint_interval, args.prompts_per_arm) < 1:
        raise SystemExit("counts must be positive")
    if min(args.learning_rate, args.reward_loss_weight, args.reward_delta_weight, args.reward_terminal_weight, args.right_weight, args.success_weight, args.late_weight, args.reward_probability_scale_floor, args.official_weight) <= 0:
        raise SystemExit("weights and learning rate must be positive")
    if args.min_start is not None and args.max_start is not None and args.min_start > args.max_start:
        raise SystemExit("min-start cannot exceed max-start")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    torch.manual_seed(args.seed)
    split = json.loads(Path(args.split_manifest).read_text())
    instruction_by_seed = None
    if args.instruction_map:
        instruction_document = json.loads(Path(args.instruction_map).read_text())
        raw_mapping = instruction_document.get("seed_to_instruction", instruction_document)
        instruction_by_seed = {int(seed): str(value) for seed, value in raw_mapping.items()}
    dataset = RewardWindows(
        Path(args.windows),
        split["train_episodes"],
        instruction_by_seed,
        min_start=args.min_start,
        max_start=args.max_start,
        arm_filter=args.arm_filter,
        instruction_by_episode={
            int(episode): str(instruction)
            for episode, instruction in split.get("episode_to_instruction", {}).items()
        },
    )
    weights = np.ones(len(dataset), dtype=np.float64)
    weights[~dataset.is_synthetic] *= args.official_weight
    weights[dataset.arm_right] *= args.right_weight
    weights[dataset.capture_success] *= args.success_weight
    weights[dataset.start >= args.late_start] *= args.late_weight
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
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
    left_prompts = official_prompts(Path(args.reset_manifest), "left", args.prompts_per_arm)
    right_prompts = official_prompts(Path(args.reset_manifest), "right", args.prompts_per_arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output)
    history = []
    for step in range(1, args.steps + 1):
        try:
            context, history_actions, future, target, arm_right, capture_success, exact_instruction = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            context, history_actions, future, target, arm_right, capture_success, exact_instruction = next(iterator)
        context = to_frames(context, device)
        target = to_frames(target, device)
        history_actions = ((history_actions.to(device) - mean) / std).float()
        future = ((future.to(device) - mean) / std).float()
        prompts = []
        for index, (is_right, exact) in enumerate(zip(arm_right.tolist(), exact_instruction)):
            if exact:
                prompts.append(exact)
            else:
                choices = right_prompts if is_right else left_prompts
                prompts.append(choices[(step + index) % len(choices)])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = rollout(model, context, history_actions, future)
            image_loss = visual_loss(prediction, target, context[:, -1])
        # The official Track2 environment evaluates this checkpoint in FP32;
        # keeping its logits in FP32 is essential in the low-probability range.
        with torch.autocast(device_type=device.type, enabled=False):
            task_loss, reward_parts = reward_progress_loss(
                reward_model,
                prediction.float(),
                target.float(),
                context[:, -1].float(),
                prompts,
                args.reward_objective,
                args.reward_probability_scale_floor,
                args.reward_delta_weight,
                args.reward_terminal_weight,
            )
        loss = image_loss.float() + args.reward_loss_weight * task_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "visual_loss": float(image_loss.detach().cpu()),
                "reward_loss": float(task_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "right_fraction": float(arm_right.float().mean()),
                "success_fraction": float(capture_success.float().mean()),
                **{key: float(value.cpu()) for key, value in reward_parts.items()},
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps:
            metadata = {
                "format": "track2-reward-aligned-autoregressive-parent-v1",
                "checkpoint_step": step,
                "initialization_checkpoint": str(init.resolve()),
                "windows": str(Path(args.windows).resolve()),
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "train_windows": len(dataset),
                "frozen_official_reward_checkpoint": str(Path(args.reward_checkpoint).resolve()),
                "reward_model_modified": False,
                "training": vars(args),
                "prompt_protocol": {"left": left_prompts, "right": right_prompts},
                "instruction_map": str(Path(args.instruction_map).resolve()) if args.instruction_map else None,
                "sampling": {
                    "expected_right_fraction": float(weights[dataset.arm_right].sum() / weights.sum()),
                    "expected_success_fraction": float(weights[dataset.capture_success].sum() / weights.sum()),
                    "expected_late_fraction": float(weights[dataset.start >= args.late_start].sum() / weights.sum()),
                    "expected_official_fraction": float(weights[~dataset.is_synthetic].sum() / weights.sum()),
                },
                "history": history,
            }
            save(output / f"checkpoint_step_{step:06d}", model, mean, std, metadata)


if __name__ == "__main__":
    main()
