#!/usr/bin/env python3
"""Train v13.1 with balanced, episode-disjoint validation for both arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_occlusion_head_v131 import ContactOcclusionHeadV131
from wam_pipeline.contact_occlusion_head_v132 import ContactOcclusionHeadV132
from train_contact_occlusion_head_v13 import contact_score, episode, images, loss_function, semantic_mask


def active_arm(history: np.ndarray, future: np.ndarray) -> tuple[int, np.ndarray]:
    actions = np.concatenate((history, future), axis=0)
    delta = np.abs(np.diff(actions, axis=0))
    activity = np.asarray((delta[:, :7].mean(), delta[:, 7:].mean()))
    arm = int(activity.argmax())
    return arm, future[:, arm * 7:(arm + 1) * 7].copy()


class DualArmDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 action_mean: np.ndarray, action_std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.action_mean, self.action_std = action_mean, action_std
        self.arms = []
        for index in indices:
            with np.load(windows / names[index], allow_pickle=False) as window:
                arm, _ = active_arm(window["history_actions"], window["future_actions"])
            self.arms.append(arm)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]; name = self.names[index]
        with np.load(self.windows / name, allow_pickle=False) as window:
            y0, y1, x0, x1 = CONTACT_REGION
            context = window["context_frames"][-1, y0:y1, x0:x1].copy()
            target = window["target_frames"][:, y0:y1, x0:x1].copy()
            arm, active = active_arm(window["history_actions"], window["future_actions"])
        active = (active - self.action_mean[arm]) / self.action_std[arm]
        parent_crop = self.parent[index, :, y0:y1, x0:x1]
        parent_labels = semantic_mask(parent_crop); source_label = semantic_mask(context[None])[0]
        source_semantic = np.stack((source_label == 1, source_label == 2)).astype(np.float32)
        parent_semantic = np.stack((parent_labels == 1, parent_labels == 2), axis=1).astype(np.float32)
        return parent_crop, context, active, arm, semantic_mask(target), source_semantic, parent_semantic, name


def forward_model(model, last, parent, actions, arms, source_semantic, parent_semantic):
    if isinstance(model, ContactOcclusionHeadV132):
        return model(last, parent, actions, arms, source_semantic, parent_semantic)
    return model(last, parent, actions, arms)


@torch.inference_mode()
def evaluate(loader, model, device) -> dict:
    model.eval(); totals = {arm: {"intersection": np.zeros(3), "union": np.zeros(3),
                                  "truth": np.zeros(3), "correct": 0, "pixels": 0,
                                  "frame_i": np.zeros(8), "frame_u": np.zeros(8), "samples": 0}
                              for arm in (0, 1)}
    for parent, last, actions, arms, target, source_semantic, parent_semantic, _ in loader:
        parent = images(parent, device); last = last.permute(0, 3, 1, 2).to(device).float() / 255
        actions, arms, target = actions.to(device).float(), arms.to(device).long(), target.to(device).long()
        source_semantic = source_semantic.to(device).float(); parent_semantic = parent_semantic.to(device).float()
        prediction = forward_model(model, last, parent, actions, arms, source_semantic, parent_semantic).argmax(dim=2)
        for index, arm_value in enumerate(arms.tolist()):
            state = totals[arm_value]; pred, truth = prediction[index], target[index]
            state["correct"] += int((pred == truth).sum()); state["pixels"] += truth.numel(); state["samples"] += 1
            for label in (1, 2):
                pm, tm = pred == label, truth == label
                state["intersection"][label] += int((pm & tm).sum())
                state["union"][label] += int((pm | tm).sum()); state["truth"][label] += int(tm.sum())
            for time in range(8):
                pm, tm = pred[time] == 2, truth[time] == 2
                state["frame_i"][time] += int((pm & tm).sum()); state["frame_u"][time] += int((pm | tm).sum())
    result = {}
    for arm, state in totals.items():
        prefix = f"arm{arm}"
        result[f"{prefix}_samples"] = state["samples"]
        result[f"{prefix}_pixel_accuracy"] = state["correct"] / max(state["pixels"], 1)
        for name, label in (("bottle", 1), ("gripper", 2)):
            result[f"{prefix}_{name}_iou"] = state["intersection"][label] / max(state["union"][label], 1)
            result[f"{prefix}_{name}_recall"] = state["intersection"][label] / max(state["truth"][label], 1)
        frames = state["frame_i"] / np.maximum(state["frame_u"], 1)
        result[f"{prefix}_gripper_frame_iou"] = frames.tolist()
        result[f"{prefix}_minimum_frame_iou"] = float(frames.min())
    result["worst_arm_gripper_iou"] = min(result["arm0_gripper_iou"], result["arm1_gripper_iou"])
    result["worst_arm_minimum_frame_iou"] = min(result["arm0_minimum_frame_iou"], result["arm1_minimum_frame_iou"])
    return result


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--model-version", choices=("v13.1", "v13.2"), default="v13.1")
    parser.add_argument("--prior-strength", type=float, default=1.5)
    parser.add_argument("--action-dropout", type=float, default=0.0)
    parser.add_argument("--semantic-prior-dropout", type=float, default=0.0)
    parser.add_argument("--arm0-dev-episode", default="episode36"); parser.add_argument("--arm1-dev-episode", default="episode47")
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent, names = cache["prediction"], cache["windows"].astype(str).tolist()
    selected, metadata = [], {}
    y0, y1, x0, x1 = CONTACT_REGION
    for index, name in enumerate(names):
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            score = contact_score(window["target_frames"][:, y0:y1, x0:x1])
            arm, active = active_arm(window["history_actions"], window["future_actions"])
        if score[0] >= 2 and score[1] >= 1000 and score[2] >= 80:
            selected.append(index); metadata[index] = (arm, active)
    dev_episodes = {args.arm0_dev_episode, args.arm1_dev_episode}
    train_indices = [i for i in selected if episode(names[i]) not in dev_episodes]
    dev_indices = [i for i in selected if episode(names[i]) in dev_episodes]
    for expected_arm, dev_episode in enumerate((args.arm0_dev_episode, args.arm1_dev_episode)):
        observed = {metadata[i][0] for i in dev_indices if episode(names[i]) == dev_episode}
        if observed != {expected_arm}: raise ValueError(f"{dev_episode} expected arm{expected_arm}, got {observed}")
    action_mean = np.zeros((2, 7), np.float32); action_std = np.ones((2, 7), np.float32)
    for arm in (0, 1):
        values = np.concatenate([metadata[i][1] for i in train_indices if metadata[i][0] == arm])
        action_mean[arm] = values.mean(0); action_std[arm] = np.maximum(values.std(0), 1e-4)
    train = DualArmDataset(Path(args.windows), parent, names, train_indices, action_mean, action_std)
    dev = DualArmDataset(Path(args.windows), parent, names, dev_indices, action_mean, action_std)
    counts = np.bincount(train.arms, minlength=2); weights = [1.0 / counts[arm] for arm in train.arms]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                        persistent_workers=True)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    device = torch.device(args.device)
    model = (ContactOcclusionHeadV132(args.base_channels, args.prior_strength)
             if args.model_version == "v13.2" else ContactOcclusionHeadV131(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); history = []
    initial = evaluate(dev_loader, model, device); history.append({"step": 0, **initial}); print(json.dumps(history[-1]), flush=True)
    best = initial["worst_arm_gripper_iou"]; iterator = iter(loader)
    for step in range(1, args.steps + 1):
        model.train()
        try: parent_batch, last, actions, arms, target, source_semantic, parent_semantic, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            parent_batch, last, actions, arms, target, source_semantic, parent_semantic, _ = next(iterator)
        parent_batch = images(parent_batch, device); last = last.permute(0, 3, 1, 2).to(device).float() / 255
        actions, arms, target = actions.to(device).float(), arms.to(device).long(), target.to(device).long()
        source_semantic = source_semantic.to(device).float(); parent_semantic = parent_semantic.to(device).float()
        if isinstance(model, ContactOcclusionHeadV132):
            if args.action_dropout:
                drop = torch.rand(len(actions), device=device) < args.action_dropout
                actions = torch.where(drop[:, None, None], torch.zeros_like(actions), actions)
            if args.semantic_prior_dropout:
                parent_semantic = parent_semantic.clone()
                drop = torch.rand(len(actions), device=device) < args.semantic_prior_dropout
                starts = torch.randint(1, 6, (len(actions),), device=device)
                for sample in torch.nonzero(drop, as_tuple=False).flatten().tolist():
                    start = int(starts[sample])
                    decay = torch.linspace(0.5, 0.0, 8 - start, device=device)
                    parent_semantic[sample, start:, 1] *= decay[:, None, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = forward_model(model, last, parent_batch, actions, arms, source_semantic, parent_semantic)
            loss, parts = loss_function(logits, target)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss),
                              **{key: float(value) for key, value in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            metrics = evaluate(dev_loader, model, device); history.append({"step": step, **metrics}); print(json.dumps(history[-1]), flush=True)
            value = {"format": f"track2-contact-occlusion-head-{args.model_version}", "state_dict": model.state_dict(),
                     "step": step, "metrics": metrics, "base_channels": args.base_channels,
                     "model_version": args.model_version, "prior_strength": args.prior_strength,
                     "action_mean": torch.from_numpy(action_mean), "action_std": torch.from_numpy(action_std),
                     "dev_episodes": sorted(dev_episodes)}
            atomic_save(value, output / "latest.pt")
            if metrics["worst_arm_gripper_iou"] > best:
                best = metrics["worst_arm_gripper_iou"]; atomic_save(value, output / "best.pt")
    manifest = {"format": f"track2-contact-occlusion-{args.model_version}-training", "steps": args.steps,
                "model_version": args.model_version, "prior_strength": args.prior_strength,
                "action_dropout": args.action_dropout,
                "semantic_prior_dropout": args.semantic_prior_dropout,
                "selected_windows": len(selected), "train_windows": len(train), "dev_windows": len(dev),
                "train_arm_counts": counts.tolist(), "dev_arm_counts": np.bincount(dev.arms, minlength=2).tolist(),
                "train_episodes": sorted({episode(names[i]) for i in train_indices}),
                "dev_episodes": sorted(dev_episodes), "selection_metric": "worst_arm_gripper_iou", "history": history}
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
