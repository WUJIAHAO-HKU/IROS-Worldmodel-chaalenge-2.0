#!/usr/bin/env python3
"""Calibrate the external geometry expert on v15 dev, then freeze on Validation64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_dual_tiny_experts_v150 import MetricEvaluator, arm_metrics, load_cache
from train_contact_occlusion_head_v131 import active_arm
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.dual_tiny_experts_v150 import TinyBlackGripperExpert, render_black_gripper


SIZE = 96


def resize_rgb(value):
    if value.ndim == 3:
        return cv2.resize(value, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.stack([resize_rgb(frame) for frame in value])


def resize_mask(value):
    if value.ndim == 2:
        return cv2.resize(value.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([resize_mask(frame) for frame in value])


def rgb(value, device):
    axes = (2, 0, 1) if value.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(value.transpose(axes).copy()).to(device).float() / 255


def mask(value, device):
    value = value[None] if value.ndim == 2 else value[:, None]
    return torch.from_numpy(value.astype(np.float32)).to(device)


@torch.inference_mode()
def deltas(parent, context, names, windows, checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TinyBlackGripperExpert(int(state["base_channels"])).to(device)
    model.load_state_dict(state["state_dict"], strict=True); model.eval()
    mean, std = state["action_mean"].numpy(), state["action_std"].numpy()
    output = np.zeros_like(parent, dtype=np.float32); arms = []
    y0, y1, x0, x1 = CONTACT_REGION
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as value:
            arm, action = active_arm(value["history_actions"], value["future_actions"])
        arms.append(arm); action = (action - mean[arm]) / std[arm]
        source = context[index, -1, y0:y1, x0:x1]
        current = parent[index, :, y0:y1, x0:x1]
        source_label = structure_semantic_mask(source[None])[0]
        parent_label = structure_semantic_mask(current)
        source_structure = np.isin(source_label, (2, 3))
        parent_structure = np.isin(parent_label, (2, 3))
        parent_bottle = parent_label == 1
        logits, residual = model(
            rgb(resize_rgb(source), device)[None], rgb(resize_rgb(current), device)[None],
            torch.from_numpy(action[None]).to(device).float(), torch.tensor([arm], device=device),
            mask(resize_mask(source_structure), device)[None],
            mask(resize_mask(parent_structure), device)[None],
            mask(resize_mask(parent_bottle), device)[None],
        )
        height, width = current.shape[1:3]
        logits = F.interpolate(logits.flatten(0, 1), (height, width), mode="bilinear",
                               align_corners=False).unflatten(0, (1, 8))
        residual = F.interpolate(residual.flatten(0, 1), (height, width), mode="bilinear",
                                 align_corners=False).unflatten(0, (1, 8))
        rendered, _, _ = render_black_gripper(
            rgb(current, device)[None], mask(source_structure, device)[None],
            mask(parent_structure, device)[None], logits, residual, 1.0,
        )
        rendered = rendered[0].permute(0, 2, 3, 1).cpu().numpy() * 255
        output[index, :, y0:y1, x0:x1] = rendered - current.astype(np.float32)
    return output, np.asarray(arms)


def compose(parent, delta, arms, selections):
    value = parent.astype(np.float32).copy()
    for arm in (0, 1):
        selected = arms == arm; setting = selections[f"arm{arm}"]
        gate = (np.arange(8) >= setting["start_frame_zero_based"]).astype(np.float32)
        value[selected] += setting["strength"] * delta[selected] * gate[None, :, None, None, None]
    return np.clip(np.round(value), 0, 255).astype(np.uint8)


def safe(parent_metrics, metrics, tolerance=1e-8):
    scalars = ("rgb_mae", "contact_rgb_mae", "structure_rgb_mae")
    if any(metrics[key] > parent_metrics[key] + tolerance for key in scalars): return False
    if any(np.any(np.asarray(metrics[key]) > np.asarray(parent_metrics[key]) + tolerance)
           for key in ("frame_rgb_mae", "frame_contact_rgb_mae")): return False
    return True


def calibrate(parent, target, delta, arms):
    evaluator = MetricEvaluator(target, arms); selections = {}
    trials = {}
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm); baseline = evaluator(parent, indices)
        candidates = []
        for strength in (0.0, .1, .25, .5, .75, 1.0):
            for start in (0, 2, 4, 6):
                setting = {f"arm{side}": {"strength": 0.0, "start_frame_zero_based": 0}
                           for side in (0, 1)}
                setting[f"arm{arm}"] = {"strength": strength, "start_frame_zero_based": start}
                value = compose(parent, delta, arms, setting)
                metrics = evaluator(value, indices)
                record = {"strength": strength, "start_frame_zero_based": start,
                          "safe": safe(baseline, metrics), "metrics": metrics}
                candidates.append(record)
        accepted = [value for value in candidates if value["safe"]]
        best = min(accepted, key=lambda value: (
            value["metrics"]["structure_rgb_mae"], value["metrics"]["contact_rgb_mae"],
            value["metrics"]["rgb_mae"], value["strength"],
        ))
        selections[f"arm{arm}"] = {"strength": best["strength"],
                                    "start_frame_zero_based": best["start_frame_zero_based"]}
        trials[f"arm{arm}"] = candidates
    selected = compose(parent, delta, arms, selections)
    return selections, evaluator(parent), evaluator(selected), trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); windows = Path(args.windows)
    dev_parent, dev_target, dev_context, dev_names = load_cache(args.dev_cache, windows, None)
    dev_delta, dev_arms = deltas(dev_parent, dev_context, dev_names, windows, args.checkpoint, device)
    selections, dev_base, dev_selected, trials = calibrate(dev_parent, dev_target, dev_delta, dev_arms)

    parent, target, context, names = load_cache(args.validation_cache, windows, None)
    delta, arms = deltas(parent, context, names, windows, args.checkpoint, device)
    prediction = compose(parent, delta, arms, selections); evaluator = MetricEvaluator(target, arms)
    parent_metrics, metrics = evaluator(parent), evaluator(prediction)
    report = {
        "format": "track2-v26.6-external-structure-expert-evaluation",
        "checkpoint": str(Path(args.checkpoint).resolve()), "selections": selections,
        "dev": {"sample_count": len(dev_parent), "parent": dev_base, "selected": dev_selected,
                "trials": trials},
        "validation64": {"sample_count": len(parent), "parent": parent_metrics,
                         "selected": metrics, "parent_arms": arm_metrics(parent, evaluator, arms),
                         "selected_arms": arm_metrics(prediction, evaluator, arms)},
    }
    for scope in ("dev", "validation64"):
        base, value = report[scope]["parent"], report[scope]["selected"]
        report[scope]["improvement_percent"] = {
            key: (base[key] - value[key]) / base[key] * 100
            for key in ("rgb_mae", "contact_rgb_mae", "structure_rgb_mae",
                        "target_texture_rgb_mae", "active_arm_texture_rgb_mae")
        }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(output / "validation64_predictions.npz", prediction=prediction,
                        target=target, context=context, windows=names, arm_id=arms)
    print(json.dumps({"selections": selections, "dev": report["dev"]["improvement_percent"],
                      "validation64": report["validation64"]["improvement_percent"]}, indent=2))


if __name__ == "__main__": main()
