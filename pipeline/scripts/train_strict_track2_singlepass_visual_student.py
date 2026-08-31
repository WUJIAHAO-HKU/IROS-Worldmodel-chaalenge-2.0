#!/usr/bin/env python3
"""Distill frozen V15.7 policy-state coverage into one multi-source flow U-Net."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wam_pipeline.multisource_flow_unet import MultiSourceActionFlowUNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arm_from_actions(history: np.ndarray, future: np.ndarray) -> str:
    actions = np.concatenate((history, future), axis=0)
    delta = np.abs(np.diff(actions, axis=0))
    return "right" if delta[:, 7:].mean() > delta[:, :7].mean() else "left"


class TeacherStream:
    def __init__(self, manifest: dict, rng: np.random.Generator, cache_files: int = 3):
        self.records = manifest["records"]
        self.rng = rng
        self.cache_files = cache_files
        self.cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.queues = {"left": deque(), "right": deque()}

    def _refill(self, arm: str) -> None:
        order = self.rng.permutation(len(self.records))
        refs = []
        for record_index in order:
            indices = list(self.records[int(record_index)]["sample_indices_by_arm"][arm])
            self.rng.shuffle(indices)
            refs.extend((int(record_index), int(index)) for index in indices)
        if not refs:
            raise RuntimeError(f"teacher cache has no {arm} samples")
        self.queues[arm].extend(refs)

    def _load(self, record_index: int) -> dict[str, np.ndarray]:
        cached = self.cache.get(record_index)
        if cached is not None:
            self.cache.move_to_end(record_index)
            return cached
        record = self.records[record_index]
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"teacher cache hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            loaded = {
                name: values[name].copy()
                for name in (
                    "context_frames",
                    "history_actions",
                    "future_actions",
                    "predicted_frames",
                )
            }
        self.cache[record_index] = loaded
        while len(self.cache) > self.cache_files:
            self.cache.popitem(last=False)
        return loaded

    def next(self, arm: str):
        if not self.queues[arm]:
            self._refill(arm)
        record_index, sample_index = self.queues[arm].popleft()
        values = self._load(record_index)
        return tuple(
            values[name][sample_index]
            for name in (
                "context_frames",
                "history_actions",
                "future_actions",
                "predicted_frames",
            )
        )


class GroundTruthStream:
    def __init__(self, root: Path, split: dict, rng: np.random.Generator, arm_by_name: dict[str, str]):
        allowed = {int(value) for value in split["train_episodes"]}
        self.rng = rng
        self.paths = {"left": [], "right": []}
        for path in sorted(root.glob("episode*_*.npz")):
            episode = int(path.name.split("_")[0][7:])
            if episode not in allowed:
                continue
            arm = arm_by_name.get(path.name)
            if arm not in self.paths:
                raise RuntimeError(f"ground-truth source manifest has no valid arm for {path.name}")
            self.paths[arm].append(path)
        if not all(self.paths.values()):
            raise RuntimeError("ground-truth mixture must contain both arms")

    def next(self, arm: str):
        path = self.paths[arm][int(self.rng.integers(len(self.paths[arm])))]
        with np.load(path, allow_pickle=False) as values:
            return tuple(
                values[name].copy()
                for name in (
                    "context_frames",
                    "history_actions",
                    "future_actions",
                    "target_frames",
                )
            )


def tensors(sample, device, mean, std):
    context, history, future, target = sample
    context = torch.from_numpy(np.ascontiguousarray(context))[None].permute(0, 1, 4, 2, 3).to(device).float().div(255)
    target = torch.from_numpy(np.ascontiguousarray(target))[None].permute(0, 1, 4, 2, 3).to(device).float().div(255)
    actions = np.concatenate((history, future), axis=0)
    actions = torch.from_numpy(np.ascontiguousarray(actions))[None].to(device).float()
    actions = (actions - mean) / std
    return context, actions, target


def balanced_validation_paths(
    paths: list[Path], maximum: int, arm_by_name: dict[str, str]
) -> list[Path]:
    """Deterministically cover both arms and the full episode/horizon range."""
    groups: dict[str, list[Path]] = {"left": [], "right": []}
    for path in paths:
        arm = arm_by_name.get(path.name)
        if arm not in groups:
            raise RuntimeError(f"validation source manifest has no valid arm for {path.name}")
        groups[arm].append(path)
    if not all(groups.values()):
        raise RuntimeError("validation selection must contain both arms")
    quotas = {"left": maximum // 2, "right": maximum - maximum // 2}
    selected: dict[str, list[Path]] = {}
    for arm, values in groups.items():
        count = min(quotas[arm], len(values))
        indices = np.linspace(0, len(values) - 1, count, dtype=int)
        selected[arm] = [values[int(index)] for index in indices]
    interleaved = []
    for index in range(max(map(len, selected.values()))):
        for arm in ("left", "right"):
            if index < len(selected[arm]):
                interleaved.append(selected[arm][index])
    return interleaved


def visual_loss(prediction, target, context_last, source_scale: float, source_weight=None):
    eps = 1e-3
    difference = prediction - target
    charbonnier = torch.sqrt(difference.square() + eps * eps)
    previous = torch.cat((context_last[:, None], target[:, :-1]), dim=1)
    motion = (target - previous).abs().mean(dim=2, keepdim=True)
    motion_weight = 1.0 + 1.5 * (motion >= 0.03).to(target.dtype)
    pixel = (motion_weight * charbonnier).mean()
    flat_prediction, flat_target = prediction.flatten(0, 1), target.flatten(0, 1)
    edge_x = F.l1_loss(flat_prediction[..., 1:] - flat_prediction[..., :-1], flat_target[..., 1:] - flat_target[..., :-1])
    edge_y = F.l1_loss(flat_prediction[..., 1:, :] - flat_prediction[..., :-1, :], flat_target[..., 1:, :] - flat_target[..., :-1, :])
    laplacian = F.l1_loss(
        flat_prediction - F.avg_pool2d(flat_prediction, 5, 1, 2),
        flat_target - F.avg_pool2d(flat_target, 5, 1, 2),
    )
    prediction_sequence = torch.cat((context_last[:, None], prediction), dim=1)
    target_sequence = torch.cat((context_last[:, None], target), dim=1)
    prediction_delta = prediction_sequence[:, 1:] - prediction_sequence[:, :-1]
    target_delta = target_sequence[:, 1:] - target_sequence[:, :-1]
    temporal = F.l1_loss(prediction_delta, target_delta)
    acceleration = F.l1_loss(
        prediction_delta[:, 1:] - prediction_delta[:, :-1],
        target_delta[:, 1:] - target_delta[:, :-1],
    )
    dark = (target.mean(dim=2, keepdim=True) < 0.28).to(target.dtype)
    structure = (dark * difference.abs()).sum() / dark.sum().clamp_min(1)
    # Copy the context only where a pixel remains static for the *entire*
    # horizon.  A per-frame static mask would incorrectly resurrect the old
    # gripper after it moves and then pauses, producing the observed black lag.
    static = (motion.amax(dim=1, keepdim=True) < 0.01).to(target.dtype).expand_as(motion)
    identity = (static * (prediction - context_last[:, None]).abs()).sum() / static.sum().clamp_min(1)
    source_entropy = prediction.new_zeros(())
    if source_weight is not None:
        source_entropy = -(
            source_weight.clamp_min(1e-6) * source_weight.clamp_min(1e-6).log()
        ).sum(dim=2).mean()
    total = source_scale * (
        pixel
        + 0.35 * (edge_x + edge_y + laplacian)
        + 0.4 * temporal
        + 0.15 * acceleration
        + structure
        + 0.25 * identity
        + 0.02 * source_entropy
    )
    return total, {
        "pixel": pixel.detach(),
        "texture": (edge_x + edge_y + laplacian).detach(),
        "temporal": temporal.detach(),
        "acceleration": acceleration.detach(),
        "structure": structure.detach(),
        "identity": identity.detach(),
        "source_entropy": source_entropy.detach(),
    }


@torch.inference_mode()
def validate(model, paths, device, mean, std, maximum: int):
    model.eval()
    selected = paths[:maximum]
    mae = []
    for path in selected:
        with np.load(path, allow_pickle=False) as values:
            sample = tuple(
                values[name].copy()
                for name in (
                    "context_frames",
                    "history_actions",
                    "future_actions",
                    "target_frames",
                )
            )
        context, actions, target = tensors(sample, device, mean, std)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(context, actions).clamp(0, 1)
        mae.append(float(F.l1_loss(prediction.float(), target).cpu()))
    model.train()
    return float(np.mean(mae))


def save(output: Path, model, mean, std, base_channels: int, metadata: dict):
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format": "track2-multisource-flow-unet-v1", "state_dict": model.state_dict()},
        output / "model.pt",
    )
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(
        output / "track2_multisource_flow_unet_config.npz",
        context_frames=np.asarray(5),
        action_dim=np.asarray(14),
        prediction_frames=np.asarray(8),
        base_channels=np.asarray(base_channels),
    )
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-windows", type=Path, required=True)
    parser.add_argument("--ground-truth-split", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--teacher-probability", type=float, default=0.5)
    parser.add_argument("--teacher-loss-scale", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=166)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.steps < 1 or args.gradient_accumulation < 1:
        raise SystemExit("steps and gradient accumulation must be positive")
    if not 0 <= args.teacher_probability <= 1 or not 0 < args.teacher_loss_scale <= 1:
        raise SystemExit("invalid teacher mixture weights")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v166-singlepass-student-pilot-preregistration-v1":
        raise SystemExit("unexpected student pilot preregistration")
    if sha256(Path(__file__).resolve()) != prereg["training"]["script_sha256"]:
        raise SystemExit("training script differs from preregistration")
    registered = prereg["training"]
    actual_training = {
        "steps": args.steps,
        "gradient_accumulation": args.gradient_accumulation,
        "teacher_probability": args.teacher_probability,
        "teacher_loss_scale": args.teacher_loss_scale,
        "learning_rate": args.learning_rate,
        "validation_interval": args.validation_interval,
        "validation_samples": args.validation_samples,
        "seed": args.seed,
        "device": args.device,
    }
    for key, value in actual_training.items():
        if registered[key] != value:
            raise SystemExit(
                f"training argument {key}={value!r} differs from preregistered {registered[key]!r}"
            )
    if sha256(args.teacher_manifest) != prereg["data"]["teacher_manifest_sha256"]:
        raise SystemExit("teacher manifest differs from preregistration")
    if sha256(args.ground_truth_split) != prereg["data"]["ground_truth_split_sha256"]:
        raise SystemExit("ground-truth split differs from preregistration")
    for filename, key in (
        ("model.pt", "model_sha256"),
        ("action_normalization.npz", "action_normalization_sha256"),
        ("track2_multisource_flow_unet_config.npz", "config_sha256"),
    ):
        if sha256(args.initial_checkpoint / filename) != prereg["initialization"][key]:
            raise SystemExit(f"initial checkpoint artifact differs: {filename}")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    teacher_doc = json.loads(args.teacher_manifest.read_text())
    if teacher_doc.get("format") != "strict-track2-v157-local-policy-teacher-cache-v1":
        raise SystemExit("unexpected teacher manifest")
    split = json.loads(args.ground_truth_split.read_text())
    source_manifest_path = args.ground_truth_windows / "window_sources.json"
    if sha256(source_manifest_path) != prereg["data"]["ground_truth_window_sources_sha256"]:
        raise SystemExit("ground-truth window source manifest differs from preregistration")
    source_rows = json.loads(source_manifest_path.read_text())
    arm_by_name = {str(row["path"]): str(row["arm"]) for row in source_rows}
    if len(arm_by_name) != len(source_rows):
        raise SystemExit("ground-truth window source manifest contains duplicate paths")
    teacher = TeacherStream(teacher_doc, rng)
    ground_truth = GroundTruthStream(args.ground_truth_windows, split, rng, arm_by_name)
    validation_allowed = {int(value) for value in split["validation_episodes"]}
    validation_paths = [
        path
        for path in sorted(args.ground_truth_windows.glob("episode*_*.npz"))
        if int(path.name.split("_")[0][7:]) in validation_allowed
    ]
    if not validation_paths:
        raise SystemExit("ground-truth validation split is empty")
    validation_paths = balanced_validation_paths(validation_paths, args.validation_samples, arm_by_name)
    validation_arm_counts = {"left": 0, "right": 0}
    for path in validation_paths:
        validation_arm_counts[arm_by_name[path.name]] += 1
    config = np.load(args.initial_checkpoint / "track2_multisource_flow_unet_config.npz", allow_pickle=False)
    base_channels = int(config["base_channels"])
    normalization = np.load(args.initial_checkpoint / "action_normalization.npz", allow_pickle=False)
    device = torch.device(args.device)
    mean = torch.from_numpy(normalization["mean"].astype(np.float32)).to(device)
    std = torch.from_numpy(normalization["std"].astype(np.float32)).to(device)
    state = torch.load(args.initial_checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-multisource-flow-unet-v1":
        raise SystemExit("initial checkpoint has an incompatible format")
    model = MultiSourceActionFlowUNet(base_channels).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial_mae = validate(
        model, validation_paths, device, mean, std, args.validation_samples
    )
    history = [{"step": 0, "ground_truth_validation_mae": initial_mae}]
    best_mae = initial_mae
    best_step = 0
    print(json.dumps({"step": 0, "validation_mae": initial_mae}), flush=True)
    source_counts = {"teacher": 0, "ground_truth": 0, "left": 0, "right": 0}
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        last_parts = None
        for accumulation in range(args.gradient_accumulation):
            arm = "left" if (step * args.gradient_accumulation + accumulation) % 2 == 0 else "right"
            use_teacher = bool(rng.random() < args.teacher_probability)
            sample = teacher.next(arm) if use_teacher else ground_truth.next(arm)
            source = "teacher" if use_teacher else "ground_truth"
            source_counts[source] += 1
            source_counts[arm] += 1
            context, actions, target = tensors(sample, device, mean, std)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, _, source_weight = model(context, actions, return_flow=True)
                loss, last_parts = visual_loss(
                    prediction,
                    target,
                    context[:, -1],
                    args.teacher_loss_scale if use_teacher else 1.0,
                    source_weight,
                )
            (loss / args.gradient_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "source": source,
                "arm": arm,
                **{key: float(value.cpu()) for key, value in (last_parts or {}).items()},
            }), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            value = validate(
                model, validation_paths, device, mean, std, args.validation_samples
            )
            history.append({"step": step, "ground_truth_validation_mae": value})
            is_best = value < best_mae
            if is_best:
                best_mae = value
                best_step = step
            metadata = {
                "format": "strict-track2-v166-singlepass-visual-student-training-v1",
                "preregistration": str(args.preregistration.resolve()),
                "preregistration_sha256": sha256(args.preregistration),
                "teacher_manifest": str(args.teacher_manifest.resolve()),
                "teacher_manifest_sha256": sha256(args.teacher_manifest),
                "ground_truth_windows": str(args.ground_truth_windows.resolve()),
                "ground_truth_split": str(args.ground_truth_split.resolve()),
                "ground_truth_split_sha256": sha256(args.ground_truth_split),
                "ground_truth_window_sources": str(source_manifest_path.resolve()),
                "ground_truth_window_sources_sha256": sha256(source_manifest_path),
                "initial_checkpoint": str(args.initial_checkpoint.resolve()),
                "steps": args.steps,
                "checkpoint_step": step,
                "gradient_accumulation": args.gradient_accumulation,
                "teacher_probability": args.teacher_probability,
                "teacher_loss_scale": args.teacher_loss_scale,
                "source_counts": source_counts,
                "validation": history,
                "validation_paths": [str(path.resolve()) for path in validation_paths],
                "validation_arm_counts": validation_arm_counts,
                "initial_ground_truth_validation_mae": initial_mae,
                "best_ground_truth_validation_mae": best_mae,
                "best_checkpoint_step": best_step,
                "reward_alignment_pending": True,
            }
            checkpoint = args.output / "checkpoints" / f"checkpoint_step_{step:06d}"
            save(checkpoint, model, mean, std, base_channels, metadata)
            if is_best:
                save(args.output / "best", model, mean, std, base_channels, metadata)
            print(json.dumps({"step": step, "validation_mae": value, "best_mae": best_mae}), flush=True)
    run_metadata = {
        "format": "strict-track2-v166-singlepass-visual-student-run-v1",
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "completed_steps": args.steps,
        "source_counts": source_counts,
        "validation": history,
        "validation_paths": [str(path.resolve()) for path in validation_paths],
        "validation_arm_counts": validation_arm_counts,
        "initial_ground_truth_validation_mae": initial_mae,
        "best_ground_truth_validation_mae": best_mae,
        "best_checkpoint_step": best_step,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_run_manifest.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    if best_step > 0:
        best_manifest_path = args.output / "best" / "training_manifest.json"
        best_metadata = json.loads(best_manifest_path.read_text())
        best_metadata.update({
            "completed_steps": args.steps,
            "source_counts": source_counts,
            "validation": history,
            "best_ground_truth_validation_mae": best_mae,
            "best_checkpoint_step": best_step,
        })
        best_manifest_path.write_text(json.dumps(best_metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
