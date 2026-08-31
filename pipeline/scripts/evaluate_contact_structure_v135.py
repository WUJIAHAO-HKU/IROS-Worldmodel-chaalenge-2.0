#!/usr/bin/env python3
"""External per-arm and per-frame evaluation of the v13.5 structure head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_head_v135 import ContactStructureHeadV135
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from train_contact_occlusion_head_v131 import active_arm


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for name, label in (("bottle", 1), ("black", 2), ("grey", 3)):
        pred, truth = prediction == label, target == label
        intersection = np.logical_and(pred, truth).sum()
        result[name + "_iou"] = float(intersection / max(np.logical_or(pred, truth).sum(), 1))
        result[name + "_recall"] = float(intersection / max(truth.sum(), 1))
        result[name + "_precision"] = float(intersection / max(pred.sum(), 1))
        if label >= 2:
            result[name + "_frame_iou"] = [
                float(np.logical_and(pred[:, time], truth[:, time]).sum() /
                      max(np.logical_or(pred[:, time], truth[:, time]).sum(), 1))
                for time in range(8)
            ]
            result[name + "_frame_recall"] = [
                float(np.logical_and(pred[:, time], truth[:, time]).sum() /
                      max(truth[:, time].sum(), 1)) for time in range(8)
            ]
            result[name + "_frame_precision"] = [
                float(np.logical_and(pred[:, time], truth[:, time]).sum() /
                      max(pred[:, time].sum(), 1)) for time in range(8)
            ]
    return result


@torch.inference_mode()
def infer(model, checkpoint, parent, context, names, windows, device, action_clips):
    y0, y1, x0, x1 = CONTACT_REGION; outputs, arms = [], []
    mean, std = checkpoint["action_mean"].numpy(), checkpoint["action_std"].numpy()
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        action = np.clip((action - mean[arm]) / std[arm], -action_clips[arm], action_clips[arm])
        source_label = structure_semantic_mask(context[index, -1:, y0:y1, x0:x1])[0]
        parent_label = structure_semantic_mask(parent[index, :, y0:y1, x0:x1])
        source_semantic = np.stack([source_label == label for label in (1, 2, 3)]).astype(np.float32)
        parent_semantic = np.stack(
            [parent_label == label for label in (1, 2, 3)], axis=1
        ).astype(np.float32)
        logits = model(
            torch.from_numpy(context[index, -1, y0:y1, x0:x1]).permute(2, 0, 1)[None].to(device).float() / 255,
            torch.from_numpy(parent[index, :, y0:y1, x0:x1]).permute(0, 3, 1, 2)[None].to(device).float() / 255,
            torch.from_numpy(action)[None].to(device).float(),
            torch.tensor([arm], device=device),
            torch.from_numpy(source_semantic)[None].to(device),
            torch.from_numpy(parent_semantic)[None].to(device),
        )
        outputs.append(logits.softmax(2)[0].cpu().numpy()); arms.append(arm)
    return np.stack(outputs), np.asarray(arms, np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True); parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache"); parser.add_argument("--arm0-action-clip", type=float, default=4.0)
    parser.add_argument("--arm1-action-clip", type=float, default=5.0); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; target_frames = cache["target"]; context = cache["context"]
        names = cache["windows"].astype(str).tolist()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device); model = ContactStructureHeadV135(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    probability, arms = infer(
        model, checkpoint, parent, context, names, Path(args.windows), device,
        np.asarray((args.arm0_action_clip, args.arm1_action_clip), np.float32)
    )
    y0, y1, x0, x1 = CONTACT_REGION
    truth = np.stack([
        structure_semantic_mask(frames[:, y0:y1, x0:x1]) for frames in target_frames
    ])
    predicted = probability.argmax(axis=2)
    report = {"format": "track2-contact-structure-v13.5-external-eval",
              "checkpoint_step": int(checkpoint["step"]), "sample_count": len(parent),
              "parameters": {"arm_action_clips": [args.arm0_action_clip, args.arm1_action_clip]},
              "arms": {}}
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm)
        report["arms"][f"arm{arm}"] = {"sample_count": len(indices),
                                          **metrics(predicted[indices], truth[indices])}
    report["overall"] = metrics(predicted, truth)
    report["windows"] = [
        {"window": name, "arm": int(arm), "metrics": metrics(predicted[i:i + 1], truth[i:i + 1])}
        for i, (name, arm) in enumerate(zip(names, arms))
    ]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_cache:
        Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_cache, probability=probability.astype(np.float16),
                            prediction=parent, target=target_frames, context=context,
                            windows=np.asarray(names), arm_id=arms)
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__": main()
