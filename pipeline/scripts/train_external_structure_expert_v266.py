#!/usr/bin/env python3
"""Pretrain black/gray geometry externally, then fit official RGB residuals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_contact_occlusion_head_v13 import contact_score, episode
from train_contact_occlusion_head_v131 import active_arm
from train_dual_tiny_experts_v150 import (
    SIZE, _mask_tensor, _resize_mask, _resize_rgb, _rgb_tensor,
    evaluate_gripper, gripper_loss,
)
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.dual_tiny_experts_v150 import TinyBlackGripperExpert


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary); os.replace(temporary, path)


class ExternalGeometryDataset(Dataset):
    def __init__(self, path: str, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        self.path, self.indices, self.mean, self.std = path, indices.astype(np.int64), mean, std
        self.handle = None

    def __len__(self): return len(self.indices)

    def _handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def __getitem__(self, item):
        data = self._handle(); index = int(self.indices[item]); arm = int(data["arm"][index])
        source = np.asarray(data["source_rgb"][index]).transpose(2, 0, 1).astype(np.float32) / 255
        source_structure = np.asarray(data["source_structure"][index], np.float32)[None]
        source_bottle = np.asarray(data["source_bottle"][index], np.float32)[None]
        target = np.asarray(data["target_structure"][index], np.float32)[:, None]
        action = (np.asarray(data["action"][index], np.float32) - self.mean[arm]) / self.std[arm]
        return source, action.astype(np.float32), arm, source_structure, source_bottle, target


class OfficialCombinedStructureDataset(Dataset):
    def __init__(self, windows: Path, parent: np.ndarray, names: list[str], indices: list[int],
                 mean: np.ndarray, std: np.ndarray) -> None:
        self.windows, self.parent, self.names, self.indices = windows, parent, names, indices
        self.mean, self.std = mean, std
        self.arms = []
        for index in indices:
            with np.load(windows / names[index], allow_pickle=False) as value:
                arm, _ = active_arm(value["history_actions"], value["future_actions"])
            self.arms.append(arm)

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]; name = self.names[index]
        y0, y1, x0, x1 = CONTACT_REGION
        with np.load(self.windows / name, allow_pickle=False) as value:
            source = value["context_frames"][-1, y0:y1, x0:x1]
            target = value["target_frames"][:, y0:y1, x0:x1]
            arm, action = active_arm(value["history_actions"], value["future_actions"])
        parent = self.parent[index, :, y0:y1, x0:x1]
        source_label = structure_semantic_mask(source[None])[0]
        parent_label = structure_semantic_mask(parent)
        target_label = structure_semantic_mask(target)
        action = (action - self.mean[arm]) / self.std[arm]
        return (
            _rgb_tensor(_resize_rgb(source)), _rgb_tensor(_resize_rgb(parent)),
            _rgb_tensor(_resize_rgb(target)), action.astype(np.float32), arm,
            _mask_tensor(_resize_mask(np.isin(source_label, (2, 3)))),
            _mask_tensor(_resize_mask(np.isin(parent_label, (2, 3)))),
            _mask_tensor(_resize_mask(parent_label == 1)),
            _mask_tensor(_resize_mask(np.isin(target_label, (2, 3)))), name,
        )


def external_loss(model, batch, device):
    source, actions, arms, source_structure, source_bottle, target = batch
    source, actions, source_structure, source_bottle, target = [
        value.to(device, non_blocking=True).float()
        for value in (source, actions, source_structure, source_bottle, target)
    ]
    arms = arms.to(device, non_blocking=True).long()
    parent = source[:, None].expand(-1, 8, -1, -1, -1)
    parent_structure = source_structure[:, None].expand(-1, 8, -1, -1, -1)
    parent_bottle = source_bottle[:, None].expand(-1, 8, -1, -1, -1)
    logits, _ = model(source, parent, actions, arms, source_structure, parent_structure, parent_bottle)
    probability = logits.sigmoid()
    changed = torch.zeros_like(target)
    previous = torch.cat((source_structure[:, None], target[:, :-1]), dim=1)
    changed = (target != previous).float()
    weight = 1 + 5 * target + 5 * changed
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum()
    intersection = (probability * target).sum((-1, -2, -3))
    false_positive = (probability * (1 - target)).sum((-1, -2, -3))
    false_negative = ((1 - probability) * target).sum((-1, -2, -3))
    dice = 1 - ((2 * intersection + 1) / (probability.sum((-1, -2, -3))
                                             + target.sum((-1, -2, -3)) + 1)).mean()
    tversky = 1 - ((intersection + 1) / (intersection + .7 * false_positive
                                         + .3 * false_negative + 1)).mean()
    temporal = ((probability[:, 1:] - probability[:, :-1])
                - (target[:, 1:] - target[:, :-1])).abs().mean()
    return bce + dice + tversky + .25 * temporal, {
        "bce": bce, "dice": dice, "tversky": tversky, "temporal": temporal,
    }


@torch.inference_mode()
def evaluate_external(loader, model, device):
    model.eval(); totals = {arm: {"i": 0, "u": 0, "truth": 0, "pred": 0} for arm in (0, 1)}
    for batch in loader:
        source, actions, arms, source_structure, source_bottle, target = batch
        source, actions, source_structure, source_bottle, target = [
            value.to(device).float() for value in (source, actions, source_structure, source_bottle, target)
        ]
        parent = source[:, None].expand(-1, 8, -1, -1, -1)
        parent_structure = source_structure[:, None].expand(-1, 8, -1, -1, -1)
        parent_bottle = source_bottle[:, None].expand(-1, 8, -1, -1, -1)
        logits, _ = model(source, parent, actions, arms.to(device).long(), source_structure,
                          parent_structure, parent_bottle)
        prediction = logits.sigmoid() >= .5
        for sample, arm in enumerate(arms.tolist()):
            truth = target[sample] > .5; pred = prediction[sample]; state = totals[arm]
            state["i"] += int((truth & pred).sum()); state["u"] += int((truth | pred).sum())
            state["truth"] += int(truth.sum()); state["pred"] += int(pred.sum())
    result = {}
    for arm, state in totals.items():
        result[f"arm{arm}_iou"] = state["i"] / max(state["u"], 1)
        result[f"arm{arm}_precision"] = state["i"] / max(state["pred"], 1)
        result[f"arm{arm}_recall"] = state["i"] / max(state["truth"], 1)
    result["selection_score"] = 1 - min(result["arm0_iou"], result["arm1_iou"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-cache", required=True)
    parser.add_argument("--official-windows", required=True)
    parser.add_argument("--official-parent-cache", required=True)
    parser.add_argument("--normalization-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external-steps", type=int, default=700)
    parser.add_argument("--official-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--external-learning-rate", type=float, default=3e-4)
    parser.add_argument("--official-learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=266)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    checkpoint = torch.load(args.normalization_checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["action_mean"], np.float32)
    std = np.asarray(checkpoint["action_std"], np.float32)
    device = torch.device(args.device); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model = TinyBlackGripperExpert(args.base_channels).to(device)

    with h5py.File(args.external_cache, "r") as data:
        episodes = np.asarray(data["episode"]); arms = np.asarray(data["arm"])
    train_indices = np.flatnonzero(episodes < 450); dev_indices = np.flatnonzero(episodes >= 450)
    train_external = ExternalGeometryDataset(args.external_cache, train_indices, mean, std)
    dev_external = ExternalGeometryDataset(args.external_cache, dev_indices, mean, std)
    counts = np.bincount(arms[train_indices], minlength=2)
    weights = [1 / max(counts[arm], 1) for arm in arms[train_indices]]
    sampler = WeightedRandomSampler(weights, max(len(train_indices), args.external_steps * args.batch_size),
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_external, batch_size=args.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_external, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.external_learning_rate, weight_decay=1e-4)
    external_history = [{"step": 0, **evaluate_external(dev_loader, model, device)}]
    iterator = iter(train_loader); best_external = external_history[0]["selection_score"]
    atomic_save({"format": "track2-v26.6-external-structure-pretrain",
                 "state_dict": model.state_dict(), "step": 0, "metrics": external_history[0],
                 "base_channels": args.base_channels, "action_mean": torch.from_numpy(mean),
                 "action_std": torch.from_numpy(std)}, output / "external_best.pt")
    for step in range(1, args.external_steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, parts = external_loss(model, batch, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"phase": "external", "step": step, "loss": float(loss),
                              **{k: float(v) for k, v in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.external_steps:
            metrics = evaluate_external(dev_loader, model, device)
            external_history.append({"step": step, **metrics}); print(json.dumps(external_history[-1]), flush=True)
            if metrics["selection_score"] < best_external:
                best_external = metrics["selection_score"]
                atomic_save({"format": "track2-v26.6-external-structure-pretrain",
                             "state_dict": model.state_dict(), "step": step, "metrics": metrics,
                             "base_channels": args.base_channels, "action_mean": torch.from_numpy(mean),
                             "action_std": torch.from_numpy(std)}, output / "external_best.pt")
    pretrained = torch.load(output / "external_best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(pretrained["state_dict"], strict=True)

    with np.load(args.official_parent_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; names = cache["windows"].astype(str).tolist()
    windows = Path(args.official_windows); selected = []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as value:
            target = value["target_frames"]; y0, y1, x0, x1 = CONTACT_REGION
            score = contact_score(target[:, y0:y1, x0:x1])
        if score[0] >= 2 and score[1] >= 1000 and score[2] >= 80:
            selected.append(index)
    dev_episodes = {"episode36", "episode47"}
    official_train_indices = [i for i in selected if episode(names[i]) not in dev_episodes]
    official_dev_indices = [i for i in selected if episode(names[i]) in dev_episodes]
    official_train = OfficialCombinedStructureDataset(windows, parent, names, official_train_indices, mean, std)
    official_dev = OfficialCombinedStructureDataset(windows, parent, names, official_dev_indices, mean, std)
    counts = np.bincount(official_train.arms, minlength=2)
    weights = [1 / max(counts[arm], 1) for arm in official_train.arms]
    sampler = WeightedRandomSampler(weights, max(len(official_train), args.official_steps * args.batch_size),
                                    replacement=True, generator=torch.Generator().manual_seed(args.seed + 1))
    train_loader = DataLoader(official_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(official_dev, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.official_learning_rate, weight_decay=1e-4)
    official_history = [{"step": 0, **evaluate_gripper(dev_loader, model, device)}]
    best_official = official_history[0]["selection_score"]; iterator = iter(train_loader)
    atomic_save({"format": "track2-external-structure-expert-v26.6", "expert": "structure",
                 "state_dict": model.state_dict(), "step": 0, "metrics": official_history[0],
                 "base_channels": args.base_channels, "action_mean": torch.from_numpy(mean),
                 "action_std": torch.from_numpy(std), "input_size": SIZE,
                 "external_pretrain_step": pretrained["step"]}, output / "best.pt")
    for step in range(1, args.official_steps + 1):
        model.train()
        try: batch = next(iterator)
        except StopIteration: iterator = iter(train_loader); batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, parts = gripper_loss(model, batch, device)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"phase": "official", "step": step, "loss": float(loss),
                              **{k: float(v) for k, v in parts.items()}}), flush=True)
        if step % args.validation_interval == 0 or step == args.official_steps:
            metrics = evaluate_gripper(dev_loader, model, device)
            official_history.append({"step": step, **metrics}); print(json.dumps(official_history[-1]), flush=True)
            checkpoint = {"format": "track2-external-structure-expert-v26.6", "expert": "structure",
                          "state_dict": model.state_dict(), "step": step, "metrics": metrics,
                          "base_channels": args.base_channels, "action_mean": torch.from_numpy(mean),
                          "action_std": torch.from_numpy(std), "input_size": SIZE,
                          "external_pretrain_step": pretrained["step"]}
            if metrics["selection_score"] < best_official:
                best_official = metrics["selection_score"]; atomic_save(checkpoint, output / "best.pt")
    manifest = {
        "format": "track2-v26.6-external-structure-expert-training", "config": vars(args),
        "external_train_windows": len(train_external), "external_dev_windows": len(dev_external),
        "official_train_windows": len(official_train), "official_dev_windows": len(official_dev),
        "external_history": external_history, "official_history": official_history,
        "best_external_score": best_external, "best_official_score": best_official,
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "best_official_score": best_official}), flush=True)


if __name__ == "__main__":
    main()
