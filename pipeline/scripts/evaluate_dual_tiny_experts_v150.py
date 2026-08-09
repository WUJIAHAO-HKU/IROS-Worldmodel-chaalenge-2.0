#!/usr/bin/env python3
"""Apply/tune the independent glyph and black-gripper specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_beam_mask, _observed_logo_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.dual_tiny_experts_v150 import (
    TinyBlackGripperExpert, TinyGlyphMotionExpert, render_black_gripper, render_glyph,
)


SIZE = 96


def episode(name: str) -> str:
    match = re.search(r"episode\d+", name)
    if not match:
        raise ValueError(f"no episode in {name}")
    return match.group(0)


def active_arm(history: np.ndarray, future: np.ndarray) -> tuple[int, np.ndarray]:
    actions = np.concatenate((history, future), axis=0)
    delta = np.abs(np.diff(actions, axis=0))
    arm = int(np.asarray((delta[:, :7].mean(), delta[:, 7:].mean())).argmax())
    return arm, future[:, arm * 7:(arm + 1) * 7].copy()


def right_texture_mask(frame: np.ndarray) -> np.ndarray:
    output = np.zeros(frame.shape[:2], dtype=bool)
    y0, y1, x0, x1 = REGIONS["right"]
    local = _observed_logo_mask(frame[y0:y1, x0:x1])
    if local.any():
        output[y0:y1, x0:x1] = cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0
    return output


def resize_rgb(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3:
        return cv2.resize(value, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.stack([resize_rgb(frame) for frame in value])


def resize_mask(value: np.ndarray) -> np.ndarray:
    if value.ndim == 2:
        return cv2.resize(value.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return np.stack([resize_mask(frame) for frame in value])


def rgb(value: np.ndarray, device: torch.device) -> torch.Tensor:
    axes = (2, 0, 1) if value.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(value.transpose(axes).copy()).to(device).float() / 255.0


def mask(value: np.ndarray, device: torch.device) -> torch.Tensor:
    value = value[None] if value.ndim == 2 else value[:, None]
    return torch.from_numpy(value.astype(np.float32)).to(device)


def clean_glyph(frame: np.ndarray) -> np.ndarray:
    existing = _observed_logo_mask(frame)
    if not existing.any(): return frame
    clear = cv2.dilate(existing, np.ones((3, 3), np.uint8))
    return cv2.inpaint(frame, clear * 255, 2.0, cv2.INPAINT_TELEA)


def load_model(path: str, expert: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = TinyGlyphMotionExpert if expert == "glyph" else TinyBlackGripperExpert
    model = model_type(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.to(device).eval()
    mean = checkpoint["action_mean"].numpy(); std = checkpoint["action_std"].numpy()
    return model, mean, std, checkpoint


def load_cache(path: str, windows: Path, episodes: set[str] | None):
    with np.load(path, allow_pickle=False) as cache:
        prediction = cache["prediction"]
        names = cache["windows"].astype(str)
        target = cache["target"] if "target" in cache.files else None
        context = cache["context"] if "context" in cache.files else None
    selected = np.arange(len(names))
    if episodes:
        selected = np.asarray([i for i, name in enumerate(names) if episode(name) in episodes])
    prediction = prediction[selected]; names = names[selected]
    if target is not None: target = target[selected]
    if context is not None: context = context[selected]
    if target is None or context is None:
        targets, contexts = [], []
        for name in names:
            with np.load(windows / name, allow_pickle=False) as value:
                targets.append(value["target_frames"]); contexts.append(value["context_frames"])
        target, context = np.stack(targets), np.stack(contexts)
    return prediction, target, context, names


@torch.inference_mode()
def expert_deltas(parent: np.ndarray, context: np.ndarray, names: np.ndarray, windows: Path,
                  glyph_model, glyph_stats, gripper_model, gripper_stats,
                  device: torch.device):
    glyph_delta = np.zeros_like(parent, dtype=np.float32)
    glyph_alpha = np.zeros(parent.shape[:-1] + (1,), dtype=np.float32)
    black_delta = np.zeros_like(parent, dtype=np.float32)
    glyph_confidences = np.zeros(parent.shape[:2], dtype=np.float32)
    arms, details = [], []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        arms.append(arm); detail = {"window": str(name), "arm": int(arm)}
        # Glyph geometry is predicted at the training scale, then used to warp
        # the original 128px crop so the observed letter strokes are never
        # downsampled in the final render.
        side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
        source_crop = context[index, -1, y0:y1, x0:x1]
        parent_crop = parent[index, :, y0:y1, x0:x1]
        source_logo = _observed_logo_mask(source_crop)
        glyph_confidence = np.zeros(parent.shape[1], np.float32)
        if source_logo.sum() >= 8:
            source_beam = _observed_beam_mask(source_crop)
            parent_beam = np.stack([_observed_beam_mask(frame) for frame in parent_crop])
            gmean, gstd = glyph_stats
            normalized = (action - gmean[arm]) / gstd[arm]
            source_small, parent_small = resize_rgb(source_crop), resize_rgb(parent_crop)
            logo_small = resize_mask(source_logo); source_beam_small = resize_mask(source_beam)
            parent_beam_small = resize_mask(parent_beam)
            matrix, confidence = glyph_model(
                rgb(source_small, device)[None], rgb(parent_small, device)[None],
                torch.from_numpy(normalized[None]).to(device).float(),
                torch.tensor([arm], device=device), mask(logo_small, device)[None],
                mask(source_beam_small, device)[None], mask(parent_beam_small, device)[None],
            )
            glyph_confidence = confidence[0].float().cpu().numpy()
            glyph_confidences[index] = glyph_confidence
            clean = np.stack([clean_glyph(frame) for frame in parent_crop])
            rendered, alpha = render_glyph(
                rgb(clean, device)[None], rgb(source_crop, device)[None],
                mask(source_logo, device)[None], mask(parent_beam, device)[None],
                matrix, confidence, 1.0,
            )
            rendered_np = rendered[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
            glyph_delta[index, :, y0:y1, x0:x1] = rendered_np - parent_crop.astype(np.float32)
            glyph_alpha[index, :, y0:y1, x0:x1] = alpha[0].permute(0, 2, 3, 1).float().cpu().numpy()
        # Black gripper expert stays at its trained scale; its probability and
        # residual are smoothly lifted to the native contact crop.
        cy0, cy1, cx0, cx1 = CONTACT_REGION
        source_contact = context[index, -1, cy0:cy1, cx0:cx1]
        parent_contact = parent[index, :, cy0:cy1, cx0:cx1]
        source_label = structure_semantic_mask(source_contact[None])[0]
        parent_label = structure_semantic_mask(parent_contact)
        source_black = source_label == 2; parent_black = parent_label == 2; parent_bottle = parent_label == 1
        bmean, bstd = gripper_stats; normalized = (action - bmean[arm]) / bstd[arm]
        logits, residual = gripper_model(
            rgb(resize_rgb(source_contact), device)[None], rgb(resize_rgb(parent_contact), device)[None],
            torch.from_numpy(normalized[None]).to(device).float(), torch.tensor([arm], device=device),
            mask(resize_mask(source_black), device)[None], mask(resize_mask(parent_black), device)[None],
            mask(resize_mask(parent_bottle), device)[None],
        )
        height, width = parent_contact.shape[1:3]
        logits = F.interpolate(logits.flatten(0, 1), (height, width), mode="bilinear",
                               align_corners=False).unflatten(0, (1, parent.shape[1]))
        residual = F.interpolate(residual.flatten(0, 1), (height, width), mode="bilinear",
                                 align_corners=False).unflatten(0, (1, parent.shape[1]))
        rendered, probability, support = render_black_gripper(
            rgb(parent_contact, device)[None], mask(source_black, device)[None],
            mask(parent_black, device)[None], logits, residual, 1.0,
        )
        rendered_np = rendered[0].permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
        black_delta[index, :, cy0:cy1, cx0:cx1] = rendered_np - parent_contact.astype(np.float32)
        detail.update({"source_glyph_pixels": int(source_logo.sum()),
                       "glyph_confidence": glyph_confidence.tolist(),
                       "black_probability_mean": float(probability.mean()),
                       "black_support_fraction": float((support > 0.1).float().mean())})
        details.append(detail)
    return glyph_delta, glyph_alpha, glyph_confidences, black_delta, np.asarray(arms), details


class MetricEvaluator:
    """Cache target-only semantic masks across the strength grid."""

    def __init__(self, target: np.ndarray, arms: np.ndarray) -> None:
        self.target = target
        y0, y1, x0, x1 = CONTACT_REGION
        self.structure = np.stack([
            np.isin(structure_semantic_mask(sequence[:, y0:y1, x0:x1]), (2, 3))
            for sequence in target
        ])
        self.texture = np.stack([
            [right_texture_mask(frame) for frame in sequence] for sequence in target
        ])
        active_texture = np.zeros(target.shape[:-1], dtype=bool)
        for index, (sequence, arm) in enumerate(zip(target, arms)):
            side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
            for time, frame in enumerate(sequence):
                local = _observed_logo_mask(frame[y0:y1, x0:x1])
                if local.any():
                    active_texture[index, time, y0:y1, x0:x1] = cv2.dilate(
                        local, np.ones((5, 5), np.uint8)
                    ) > 0
        self.active_texture = active_texture

    def __call__(self, prediction: np.ndarray, indices: np.ndarray | None = None) -> dict:
        if indices is None:
            target = self.target; structure = self.structure; texture = self.texture
            active_texture = self.active_texture
        else:
            target = self.target[indices]; structure = self.structure[indices]; texture = self.texture[indices]
            active_texture = self.active_texture[indices]
            prediction = prediction[indices]
        y0, y1, x0, x1 = CONTACT_REGION
        error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
        crop = error[:, :, y0:y1, x0:x1]
        structure_rgb = np.repeat(structure[..., None], 3, axis=4)
        texture_rgb = np.repeat(texture[..., None], 3, axis=4)
        active_texture_rgb = np.repeat(active_texture[..., None], 3, axis=4)
        frame_texture = []
        for time in range(target.shape[1]):
            time_mask = texture_rgb[:, time]
            frame_texture.append(float(error[:, time][time_mask].mean()) if time_mask.any() else float("nan"))
        return {"rgb_mae": float(error.mean()), "contact_rgb_mae": float(crop.mean()),
                "structure_rgb_mae": float(crop[structure_rgb].mean()) if structure_rgb.any() else float("nan"),
                "target_texture_rgb_mae": float(error[texture_rgb].mean()) if texture_rgb.any() else float("nan"),
                "active_arm_texture_rgb_mae": float(error[active_texture_rgb].mean()) if active_texture_rgb.any() else float("nan"),
                "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
                "frame_contact_rgb_mae": crop.mean(axis=(0, 2, 3, 4)).tolist(),
                "frame_target_texture_rgb_mae": frame_texture}

    def compact(self, prediction: np.ndarray, indices: np.ndarray) -> dict:
        """Scalar-only view used by the internal routing grid."""
        target = self.target[indices]; structure = self.structure[indices]
        active_texture = self.active_texture[indices]
        error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
        y0, y1, x0, x1 = CONTACT_REGION; crop = error[:, :, y0:y1, x0:x1]
        structure_rgb = np.repeat(structure[..., None], 3, axis=4)
        texture_rgb = np.repeat(active_texture[..., None], 3, axis=4)
        return {"rgb_mae": float(error.mean()), "contact_rgb_mae": float(crop.mean()),
                "structure_rgb_mae": float(crop[structure_rgb].mean()) if structure_rgb.any() else float("nan"),
                "active_arm_texture_rgb_mae": float(error[texture_rgb].mean()) if texture_rgb.any() else float("nan"),
                "frame_rgb_mae": error.mean(axis=(0, 2, 3, 4)).tolist(),
                "frame_contact_rgb_mae": crop.mean(axis=(0, 2, 3, 4)).tolist()}


def arm_metrics(prediction: np.ndarray, evaluator: MetricEvaluator, arms: np.ndarray) -> dict:
    result = {}
    for arm in (0, 1):
        selected = np.flatnonzero(arms == arm)
        result[f"arm{arm}"] = {"sample_count": len(selected)}
        if len(selected): result[f"arm{arm}"].update(evaluator(prediction, selected))
    return result


def compose(parent, glyph_delta, glyph_alpha, glyph_confidence, black_delta,
            glyph_strength, gripper_strength, glyph_threshold, glyph_prefix_frames,
            gripper_start_frame=0):
    # Glyph owns overlap pixels. This prevents the dark-structure residual from
    # washing out the sharp strokes transported by the other specialist.
    time = np.arange(parent.shape[1])[None]
    frame_gate = ((glyph_confidence >= glyph_threshold)
                  & (time < glyph_prefix_frames)).astype(np.float32)
    gated_alpha = glyph_alpha * frame_gate[:, :, None, None, None]
    black_gate = (gated_alpha < 0.05).astype(np.float32)
    value = parent.astype(np.float32) + glyph_strength * glyph_delta * frame_gate[:, :, None, None, None]
    gripper_gate = (time >= gripper_start_frame).astype(np.float32)
    value += (gripper_strength * black_delta * black_gate
              * gripper_gate[:, :, None, None, None])
    return np.clip(np.round(value), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--glyph-checkpoint", required=True); parser.add_argument("--gripper-checkpoint", required=True)
    parser.add_argument("--output-cache", required=True); parser.add_argument("--output-report", required=True)
    parser.add_argument("--episodes", nargs="*"); parser.add_argument("--device", default="cuda")
    parser.add_argument("--glyph-threshold", type=float, default=0.5)
    parser.add_argument("--glyph-prefix-frames", type=int, default=8)
    parser.add_argument("--glyph-strength", type=float); parser.add_argument("--gripper-strength", type=float)
    parser.add_argument("--tune-dev", action="store_true")
    parser.add_argument("--routing-report",
                        help="Internal-dev report whose per-arm routing is frozen for evaluation")
    args = parser.parse_args(); device = torch.device(args.device); windows = Path(args.windows)
    parent, target, context, names = load_cache(
        args.parent_cache, windows, set(args.episodes) if args.episodes else None
    )
    glyph_model, glyph_mean, glyph_std, glyph_checkpoint = load_model(args.glyph_checkpoint, "glyph", device)
    gripper_model, gripper_mean, gripper_std, gripper_checkpoint = load_model(args.gripper_checkpoint, "gripper", device)
    glyph_delta, glyph_alpha, glyph_confidence, black_delta, arms, details = expert_deltas(
        parent, context, names, windows, glyph_model, (glyph_mean, glyph_std),
        gripper_model, (gripper_mean, gripper_std), device,
    )
    evaluator = MetricEvaluator(target, arms)
    parent_metrics = evaluator(parent); trials = []
    parent_arms = arm_metrics(parent, evaluator, arms)
    if args.tune_dev:
        # Coarse, predeclared grid: 162 routes per arm. Validation never tunes
        # these values, and native-resolution statistics stay inexpensive.
        glyph_strengths = (0.0, 0.5, 1.0)
        gripper_strengths = (0.0, 0.5, 1.0)
        gripper_starts = (0, 2, 4)
        glyph_thresholds = (0.5, 0.75, 0.9)
        glyph_prefixes = (2, 8)
        arm_selections = {}
        for arm in (0, 1):
            indices = np.flatnonzero(arms == arm); arm_trials = []
            if not len(indices): continue
            arm_parent = parent_arms[f"arm{arm}"]
            for glyph_threshold in glyph_thresholds:
                for glyph_prefix in glyph_prefixes:
                    for glyph_strength in glyph_strengths:
                        for gripper_strength in gripper_strengths:
                            for gripper_start in gripper_starts:
                                prediction = compose(
                                    parent[indices], glyph_delta[indices], glyph_alpha[indices],
                                    glyph_confidence[indices], black_delta[indices],
                                    glyph_strength, gripper_strength,
                                    glyph_threshold, glyph_prefix, gripper_start,
                                )
                                score = evaluator.compact(prediction, indices)
                                arm_trials.append({"arm": arm, "glyph_threshold": glyph_threshold,
                                                   "glyph_prefix_frames": glyph_prefix,
                                                   "glyph_strength": glyph_strength,
                                                   "gripper_strength": gripper_strength,
                                                   "gripper_start_frame": gripper_start,
                                                   "metrics": score})
            safe = [trial for trial in arm_trials
                    if trial["metrics"]["rgb_mae"] <= arm_parent["rgb_mae"]
                    and trial["metrics"]["contact_rgb_mae"] <= arm_parent["contact_rgb_mae"]
                    and trial["metrics"]["structure_rgb_mae"] <= arm_parent["structure_rgb_mae"]
                    and trial["metrics"]["active_arm_texture_rgb_mae"]
                    <= arm_parent["active_arm_texture_rgb_mae"]
                    and np.all(np.asarray(trial["metrics"]["frame_rgb_mae"])
                               <= np.asarray(arm_parent["frame_rgb_mae"]))
                    and np.all(np.asarray(trial["metrics"]["frame_contact_rgb_mae"])
                               <= np.asarray(arm_parent["frame_contact_rgb_mae"]))]
            candidates = safe or arm_trials
            def arm_key(trial):
                value = trial["metrics"]
                texture_ratio = (value["active_arm_texture_rgb_mae"]
                                 / max(arm_parent["active_arm_texture_rgb_mae"], 1e-9))
                structure_ratio = (value["structure_rgb_mae"]
                                   / max(arm_parent["structure_rgb_mae"], 1e-9))
                return (texture_ratio + structure_ratio, value["rgb_mae"])
            arm_selections[f"arm{arm}"] = min(candidates, key=arm_key)
            trials.extend(arm_trials)
    elif args.routing_report:
        arm_selections = json.loads(Path(args.routing_report).read_text())["arm_selections"]
    else:
        if args.glyph_strength is None or args.gripper_strength is None:
            raise ValueError("locked evaluation requires a routing report or both expert strengths")
        arm_selections = {f"arm{arm}": {"arm": arm, "glyph_threshold": args.glyph_threshold,
                          "glyph_prefix_frames": args.glyph_prefix_frames,
                          "glyph_strength": args.glyph_strength,
                          "gripper_strength": args.gripper_strength,
                          "gripper_start_frame": 0} for arm in (0, 1)}
    output = parent.copy()
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm)
        if not len(indices): continue
        route = arm_selections[f"arm{arm}"]
        output[indices] = compose(
            parent[indices], glyph_delta[indices], glyph_alpha[indices], glyph_confidence[indices],
            black_delta[indices], route["glyph_strength"], route["gripper_strength"],
            route["glyph_threshold"], route["glyph_prefix_frames"],
            route.get("gripper_start_frame", 0),
        )
    selected = {"metrics": evaluator(output), "arms": arm_metrics(output, evaluator, arms)}
    report = {"format": "track2-dual-tiny-experts-v15.0-evaluation",
              "parent": "frozen cache", "sample_count": len(names),
              "episodes": sorted({episode(name) for name in names}),
              "checkpoints": {"glyph": {"path": args.glyph_checkpoint, "step": glyph_checkpoint["step"]},
                              "gripper": {"path": args.gripper_checkpoint, "step": gripper_checkpoint["step"]}},
              "glyph_threshold": args.glyph_threshold, "parent_metrics": parent_metrics,
              "parent_arms": parent_arms,
              "arm_selections": arm_selections,
              "selected": selected, "trials": trials, "windows": details}
    output_cache = Path(args.output_cache); output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, prediction=output, target=target, context=context,
                        windows=names, arm_id=arms)
    output_report = Path(args.output_report); output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("trials", "windows")}, indent=2))


if __name__ == "__main__":
    main()
