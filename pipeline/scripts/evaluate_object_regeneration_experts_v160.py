#!/usr/bin/env python3
"""Tune on episode-disjoint dev or evaluate the two v16 object experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_beam_mask, _observed_logo_mask
from wam_pipeline.contact_occlusion_head_v13 import CONTACT_REGION
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.object_regeneration_experts_v160 import (
    GlyphAtlasProjectionExpert, GripperInstanceFlowExpert,
    render_atlas_glyph, render_gripper_instances,
)
from evaluate_dual_tiny_experts_v150 import MetricEvaluator, arm_metrics, load_cache
from train_contact_occlusion_head_v13 import episode
from train_contact_occlusion_head_v131 import active_arm
from train_object_regeneration_experts_v160 import AtlasAligner, clean_glyph


def rgb(value: np.ndarray, device: torch.device) -> torch.Tensor:
    axes = (2, 0, 1) if value.ndim == 3 else (0, 3, 1, 2)
    return torch.from_numpy(value.transpose(axes).copy()).to(device).float().div_(255)


def mask(value: np.ndarray, device: torch.device) -> torch.Tensor:
    value = value[None] if value.ndim == 2 else value[:, None]
    return torch.from_numpy(value.astype(np.float32)).to(device)


def load_model(path: str, expert: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_type = GlyphAtlasProjectionExpert if expert == "glyph" else GripperInstanceFlowExpert
    model = model_type(int(checkpoint["base_channels"])); model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["action_mean"].numpy(), checkpoint["action_std"].numpy(), checkpoint


@torch.inference_mode()
def expert_candidates(parent: np.ndarray, context: np.ndarray, names: np.ndarray, windows: Path,
                      atlas_path: str, glyph_model, glyph_stats, gripper_model, gripper_stats,
                      device: torch.device):
    glyph = parent.copy(); gripper = parent.copy(); glyph_alpha = np.zeros(parent.shape[:-1] + (1,), np.float32)
    aligner = AtlasAligner(atlas_path); arms, details = [], []
    for index, name in enumerate(names):
        with np.load(windows / name, allow_pickle=False) as window:
            arm, action = active_arm(window["history_actions"], window["future_actions"])
        arms.append(arm); side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
        atlas_rgb, atlas_mask, atlas_ok = aligner(context[index, -1], side)
        crop = parent[index, :, y0:y1, x0:x1]; clean = np.stack([clean_glyph(frame) for frame in crop])
        beam = np.stack([_observed_beam_mask(frame) for frame in crop])
        old_glyph = np.stack([_observed_logo_mask(frame) for frame in crop])
        mean, std = glyph_stats; normalized = (action - mean[arm]) / std[arm]
        flow, visibility, edit = glyph_model(
            rgb(atlas_rgb, device)[None], rgb(crop, device)[None],
            torch.from_numpy(normalized[None]).to(device).float(), torch.tensor([arm], device=device),
            mask(atlas_mask, device)[None], mask(beam, device)[None], mask(old_glyph, device)[None])
        rendered, alpha, _ = render_atlas_glyph(rgb(crop, device)[None], rgb(atlas_rgb, device)[None],
                                                mask(atlas_mask, device)[None], mask(beam, device)[None],
                                                flow, visibility, edit,
                                                parent_clean=rgb(clean, device)[None])
        glyph[index, :, y0:y1, x0:x1] = np.round(rendered[0].permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        glyph_alpha[index, :, y0:y1, x0:x1] = alpha[0].permute(0, 2, 3, 1).cpu().numpy()

        cy0, cy1, cx0, cx1 = CONTACT_REGION
        source = context[index, -1, cy0:cy1, cx0:cx1]; crop = parent[index, :, cy0:cy1, cx0:cx1]
        source_label = structure_semantic_mask(source[None])[0]; parent_label = structure_semantic_mask(crop)
        mean, std = gripper_stats; normalized = (action - mean[arm]) / std[arm]
        semantic, flow, visibility, replacement, edit = gripper_model(
            rgb(source, device)[None], rgb(crop, device)[None],
            torch.from_numpy(normalized[None]).to(device).float(), torch.tensor([arm], device=device),
            mask(source_label == 2, device)[None], mask(source_label == 3, device)[None],
            mask(parent_label == 2, device)[None], mask(parent_label == 3, device)[None],
            mask(parent_label == 1, device)[None])
        rendered, probability, alpha, _ = render_gripper_instances(
            rgb(crop, device)[None], rgb(source, device)[None],
            mask(source_label == 2, device)[None], mask(source_label == 3, device)[None],
            mask(parent_label == 2, device)[None], mask(parent_label == 3, device)[None],
            semantic, flow, visibility, replacement, edit)
        gripper[index, :, cy0:cy1, cx0:cx1] = np.round(rendered[0].permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        details.append({"window": str(name), "arm": int(arm), "atlas_aligned": bool(atlas_ok),
                        "glyph_alpha_fraction": float((glyph_alpha[index] > .05).mean()),
                        "gripper_edit_fraction": float((alpha > .05).float().mean()),
                        "black_probability_mean": float(probability[:, :, 1].mean()),
                        "grey_probability_mean": float(probability[:, :, 2].mean())})
        if (index + 1) % 16 == 0: print(json.dumps({"inference": index + 1, "total": len(names)}), flush=True)
    return glyph, gripper, glyph_alpha, np.asarray(arms), details


def compose(parent: np.ndarray, glyph: np.ndarray, gripper: np.ndarray, glyph_alpha: np.ndarray,
            arms: np.ndarray, routes: dict) -> np.ndarray:
    output = parent.astype(np.float32).copy()
    glyph_delta = glyph.astype(np.float32) - parent.astype(np.float32)
    gripper_delta = gripper.astype(np.float32) - parent.astype(np.float32)
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm)
        for time in range(parent.shape[1]):
            route = routes[f"arm{arm}"][time]; g, s = route["glyph_strength"], route["gripper_strength"]
            ownership = (glyph_alpha[indices, time] < .05).astype(np.float32)
            output[indices, time] += g * glyph_delta[indices, time] + s * gripper_delta[indices, time] * ownership
    return np.round(np.clip(output, 0, 255)).astype(np.uint8)


def frame_metrics(prediction: np.ndarray, target: np.ndarray, arm_indices: np.ndarray,
                  time: int, text_mask: np.ndarray, structure_mask: np.ndarray) -> dict:
    return frame_metrics_arrays(prediction[arm_indices, time], target[arm_indices, time],
                                text_mask[arm_indices, time], structure_mask[arm_indices, time])


def frame_metrics_arrays(prediction: np.ndarray, target: np.ndarray,
                         text_mask: np.ndarray, structure_mask: np.ndarray) -> dict:
    error = np.abs(prediction.astype(np.float32) - target.astype(np.float32))
    y0, y1, x0, x1 = CONTACT_REGION; contact = error[:, y0:y1, x0:x1]
    sm = np.repeat(structure_mask[..., None], 3, -1)
    tm = np.repeat(text_mask[..., None], 3, -1)
    return {"rgb": float(error.mean()), "contact": float(contact.mean()),
            "structure": float(contact[sm].mean()) if sm.any() else 0.0,
            "text": float(error[tm].mean()) if tm.any() else 0.0}


def tune_routes(parent, target, glyph, gripper, glyph_alpha, arms):
    y0, y1, x0, x1 = CONTACT_REGION
    structure = np.stack([np.isin(structure_semantic_mask(value[:, y0:y1, x0:x1]), (2, 3)) for value in target])
    text = np.zeros(target.shape[:-1], bool)
    for index, arm in enumerate(arms):
        side = "left" if arm == 0 else "right"; ay0, ay1, ax0, ax1 = REGIONS[side]
        for time, frame in enumerate(target[index]):
            local = _observed_logo_mask(frame[ay0:ay1, ax0:ax1])
            if local.any(): text[index, time, ay0:ay1, ax0:ax1] = cv2.dilate(local, np.ones((5, 5), np.uint8)) > 0
    routes = {}; trials = []
    for arm in (0, 1):
        indices = np.flatnonzero(arms == arm); routes[f"arm{arm}"] = []
        for time in range(8):
            baseline = frame_metrics(parent, target, indices, time, text, structure)
            candidates = []
            for gs in (0., .1, .25, .5, 1.):
                for ss in (0., .1, .25, .5, 1.):
                    candidate = parent[indices, time].astype(np.float32).copy()
                    ownership = (glyph_alpha[indices, time] < .05).astype(np.float32)
                    candidate += gs * (glyph[indices, time].astype(np.float32) - parent[indices, time])
                    candidate += ss * (gripper[indices, time].astype(np.float32) - parent[indices, time]) * ownership
                    candidate = np.round(np.clip(candidate, 0, 255)).astype(np.uint8)
                    value = frame_metrics_arrays(candidate, target[indices, time], text[indices, time],
                                                 structure[indices, time])
                    safe = all(value[key] <= baseline[key] + 1e-8 for key in ("rgb", "contact", "structure", "text"))
                    score = sum(value[key] / max(baseline[key], 1e-6) for key in ("structure", "text")) + .25 * value["rgb"] / max(baseline["rgb"], 1e-6)
                    candidates.append({"arm": arm, "horizon": time + 1, "glyph_strength": gs,
                                       "gripper_strength": ss, "safe": safe, "score": score,
                                       "metrics": value, "parent_metrics": baseline})
            accepted = [value for value in candidates if value["safe"]]
            choice = min(accepted or candidates, key=lambda value: (value["score"], value["metrics"]["rgb"]))
            routes[f"arm{arm}"].append({key: choice[key] for key in ("glyph_strength", "gripper_strength")})
            trials.extend(candidates)
    return routes, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--atlas", required=True); parser.add_argument("--glyph-checkpoint", required=True)
    parser.add_argument("--gripper-checkpoint", required=True); parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True); parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--tune-dev", action="store_true"); parser.add_argument("--routing-report")
    parser.add_argument("--device", default="cuda"); args = parser.parse_args(); device = torch.device(args.device)
    windows = Path(args.windows); parent, target, context, names = load_cache(
        args.parent_cache, windows, set(args.episodes) if args.episodes else None)
    glyph_model, gm, gs, glyph_checkpoint = load_model(args.glyph_checkpoint, "glyph", device)
    gripper_model, sm, ss, gripper_checkpoint = load_model(args.gripper_checkpoint, "gripper", device)
    glyph, gripper, glyph_alpha, arms, details = expert_candidates(
        parent, context, names, windows, args.atlas, glyph_model, (gm, gs),
        gripper_model, (sm, ss), device)
    if args.tune_dev:
        routes, trials = tune_routes(parent, target, glyph, gripper, glyph_alpha, arms)
    elif args.routing_report:
        routes = json.loads(Path(args.routing_report).read_text())["routes"]; trials = []
    else:
        raise ValueError("use --tune-dev or provide --routing-report")
    output = compose(parent, glyph, gripper, glyph_alpha, arms, routes)
    evaluator = MetricEvaluator(target, arms)
    report = {"format": "track2-object-regeneration-experts-v16.0-evaluation", "sample_count": len(names),
              "episodes": sorted({episode(name) for name in names}),
              "checkpoints": {"glyph": {"path": args.glyph_checkpoint, "step": glyph_checkpoint["step"]},
                              "gripper": {"path": args.gripper_checkpoint, "step": gripper_checkpoint["step"]}},
              "routes": routes, "parent_metrics": evaluator(parent), "parent_arms": arm_metrics(parent, evaluator, arms),
              "selected": {"metrics": evaluator(output), "arms": arm_metrics(output, evaluator, arms)},
              "trials": trials, "windows": details}
    output_cache = Path(args.output_cache); output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, prediction=output, target=target, context=context, windows=names, arm_id=arms)
    output_report = Path(args.output_report); output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("trials", "windows")}, indent=2))


if __name__ == "__main__": main()
