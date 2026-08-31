#!/usr/bin/env python3
"""Stratified external evaluation for the balanced dual-arm v13.1 head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_occlusion_head_v131 import ContactOcclusionHeadV131
from wam_pipeline.contact_occlusion_head_v132 import ContactOcclusionHeadV132
from train_contact_occlusion_head_v13 import contact_score, semantic_mask
from train_contact_occlusion_head_v131 import active_arm
from evaluate_contact_occlusion_head_v13 import learned_metrics, render_sequence, rgb_metrics


@torch.inference_mode()
def infer(model, checkpoint, parent, context, names, windows, device, action_clips):
    y0, y1, x0, x1 = CONTACT_REGION; probabilities, arms = [], []
    mean, std = checkpoint["action_mean"].numpy(), checkpoint["action_std"].numpy()
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            arm, actions = active_arm(window["history_actions"], window["future_actions"])
        clip = float(action_clips[arm])
        actions = np.clip((actions - mean[arm]) / std[arm], -clip, clip); arms.append(arm)
        last = torch.from_numpy(context[index, -1, y0:y1, x0:x1]).permute(2, 0, 1)[None].to(device).float() / 255
        prediction = torch.from_numpy(parent[index, :, y0:y1, x0:x1]).permute(0, 3, 1, 2)[None].to(device).float() / 255
        action = torch.from_numpy(actions)[None].to(device).float(); arm_tensor = torch.tensor([arm], device=device)
        if isinstance(model, ContactOcclusionHeadV132):
            source_label = semantic_mask(context[index, -1:, y0:y1, x0:x1])[0]
            parent_label = semantic_mask(parent[index, :, y0:y1, x0:x1])
            source_semantic = torch.from_numpy(
                np.stack((source_label == 1, source_label == 2)).astype(np.float32))[None].to(device)
            parent_semantic = torch.from_numpy(
                np.stack((parent_label == 1, parent_label == 2), axis=1).astype(np.float32))[None].to(device)
            logits = model(last, prediction, action, arm_tensor, source_semantic, parent_semantic)
        else:
            logits = model(last, prediction, action, arm_tensor)
        probabilities.append(logits.softmax(2)[0].cpu().numpy())
    return np.stack(probabilities), np.asarray(arms, np.uint8)


def delta(before: dict, after: dict) -> dict:
    return {key: after[key] - before[key] for key in before}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-cache", required=True); parser.add_argument("--windows", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--output-cache"); parser.add_argument("--gripper-threshold", type=float, default=0.55)
    parser.add_argument("--arm0-gripper-threshold", type=float)
    parser.add_argument("--arm1-gripper-threshold", type=float)
    parser.add_argument("--bottle-threshold", type=float, default=0.50); parser.add_argument("--green-ratio", type=float, default=0.58)
    parser.add_argument("--decay-ratio", type=float, default=0.65); parser.add_argument("--recovery-ratio", type=float, default=0.65)
    parser.add_argument("--arm0-recovery-ratio", type=float); parser.add_argument("--arm1-recovery-ratio", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-clip", type=float, default=5.0)
    parser.add_argument("--arm0-action-clip", type=float); parser.add_argument("--arm1-action-clip", type=float)
    parser.add_argument("--episodes", help="Optional comma-separated episode subset")
    parser.add_argument("--material-ceiling", type=float, default=92.0)
    parser.add_argument("--stale-confidence-margin", type=float, default=0.08)
    parser.add_argument("--stale-guard-radius", type=int, default=1)
    parser.add_argument("--stale-inpaint-radius", type=float, default=3.0)
    parser.add_argument("--paint-erosion", type=int, default=1)
    parser.add_argument("--paint-kernel", choices=("square", "cross"), default="square")
    parser.add_argument("--prior-strength-override", type=float); args = parser.parse_args()
    with np.load(args.prediction_cache, allow_pickle=False) as cache:
        parent = cache["prediction"]; names_array = cache["windows"].astype(str)
        selected = np.arange(len(names_array))
        if args.episodes:
            episodes = set(args.episodes.split(","))
            selected = np.asarray([i for i, name in enumerate(names_array)
                                   if name.split("_")[0] in episodes], dtype=np.int64)
        parent = parent[selected]; names = names_array[selected].tolist()
        if "target" in cache.files and "context" in cache.files:
            target, context = cache["target"][selected], cache["context"][selected]
        else:
            target_parts, context_parts = [], []
            for name in names:
                with np.load(Path(args.windows) / name, allow_pickle=False) as window:
                    target_parts.append(window["target_frames"].copy())
                    context_parts.append(window["context_frames"].copy())
            target, context = np.stack(target_parts), np.stack(context_parts)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if checkpoint.get("model_version") == "v13.2":
        prior_strength = (args.prior_strength_override if args.prior_strength_override is not None
                          else float(checkpoint["prior_strength"]))
        model = ContactOcclusionHeadV132(int(checkpoint["base_channels"]), prior_strength)
    else:
        model = ContactOcclusionHeadV131(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    action_clips = np.asarray((args.arm0_action_clip if args.arm0_action_clip is not None else args.action_clip,
                               args.arm1_action_clip if args.arm1_action_clip is not None else args.action_clip))
    probability, arms = infer(model, checkpoint, parent, context, names, Path(args.windows), device, action_clips)
    thresholds = np.asarray((args.arm0_gripper_threshold if args.arm0_gripper_threshold is not None else args.gripper_threshold,
                             args.arm1_gripper_threshold if args.arm1_gripper_threshold is not None else args.gripper_threshold))
    recovery_ratios = np.asarray((args.arm0_recovery_ratio if args.arm0_recovery_ratio is not None else args.recovery_ratio,
                                  args.arm1_recovery_ratio if args.arm1_recovery_ratio is not None else args.recovery_ratio))
    rendered, diagnostics = [], []
    for prediction, history, prob, arm in zip(parent, context, probability, arms):
        value, detail = render_sequence(prediction, history, prob, float(thresholds[arm]),
                                        args.bottle_threshold, args.green_ratio, args.decay_ratio,
                                        float(recovery_ratios[arm]), args.material_ceiling,
                                        args.stale_confidence_margin, args.stale_guard_radius,
                                        args.stale_inpaint_radius, args.paint_erosion, args.paint_kernel)
        rendered.append(value); diagnostics.append(detail)
    rendered = np.stack(rendered); y0, y1, x0, x1 = CONTACT_REGION
    contact = []
    for value in target:
        score = contact_score(value[:, y0:y1, x0:x1])
        contact.append(score[0] >= 2 and score[1] >= 1000 and score[2] >= 80)
    contact = np.asarray(contact)
    report = {"format": "track2-contact-occlusion-head-v13.1-stratified-eval",
              "sample_count": len(parent), "checkpoint_step": int(checkpoint["step"]),
              "parameters": {"gripper_threshold": args.gripper_threshold,
                             "arm_gripper_thresholds": thresholds.tolist(), "bottle_threshold": args.bottle_threshold,
                             "green_ratio": args.green_ratio, "decay_ratio": args.decay_ratio,
                             "recovery_ratio": args.recovery_ratio,
                             "arm_recovery_ratios": recovery_ratios.tolist(), "action_clip": args.action_clip,
                             "arm_action_clips": action_clips.tolist(),
                             "material_ceiling": args.material_ceiling,
                             "stale_confidence_margin": args.stale_confidence_margin,
                             "stale_guard_radius": args.stale_guard_radius,
                             "stale_inpaint_radius": args.stale_inpaint_radius,
                             "paint_erosion": args.paint_erosion,
              "paint_kernel": args.paint_kernel}, "arms": {}}
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm); contact_indices = np.flatnonzero((arms == arm) & contact)
        accepted_indices = np.asarray([i for i in indices if diagnostics[i]["accepted"]], dtype=np.int64)
        if not len(indices):
            report["arms"][f"arm{arm}"] = {
                "sample_count": 0,
                "contact_sample_count": 0,
                "accepted_sample_count": 0,
            }
            continue
        baseline = rgb_metrics(parent[indices], target[indices]); result = rgb_metrics(rendered[indices], target[indices])
        arm_report = {"sample_count": len(indices), "contact_sample_count": len(contact_indices),
                      "accepted_sample_count": len(accepted_indices), "baseline": baseline,
                      "rendered": result, "delta": delta(baseline, result)}
        if len(contact_indices):
            arm_report["contact_segmentation"] = learned_metrics(
                probability[contact_indices], target[contact_indices], float(thresholds[arm]))
        if len(accepted_indices):
            accepted_before = rgb_metrics(parent[accepted_indices], target[accepted_indices])
            accepted_after = rgb_metrics(rendered[accepted_indices], target[accepted_indices])
            arm_report["accepted_baseline"] = accepted_before; arm_report["accepted_rendered"] = accepted_after
            arm_report["accepted_delta"] = delta(accepted_before, accepted_after)
        report["arms"][f"arm{arm}"] = arm_report
    baseline = rgb_metrics(parent, target); result = rgb_metrics(rendered, target)
    report["overall"] = {"baseline": baseline, "rendered": result, "delta": delta(baseline, result),
                         "accepted_sample_count": sum(x["accepted"] for x in diagnostics)}
    report["windows"] = [{"window": name, "arm": int(arm), **detail}
                         for name, arm, detail in zip(names, arms, diagnostics)]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.output_cache:
        Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_cache, prediction=rendered, target=target, context=context,
                            windows=np.asarray(names), arm_id=arms, layer_probability=probability.astype(np.float16))
    print(json.dumps({key: value for key, value in report.items() if key != "windows"}, indent=2))


if __name__ == "__main__": main()
