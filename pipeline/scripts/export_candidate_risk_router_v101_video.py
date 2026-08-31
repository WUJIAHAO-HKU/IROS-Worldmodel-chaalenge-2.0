#!/usr/bin/env python3
"""Export labeled v8 versus v10.1 candidate-router evaluation videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.candidate_risk_router_v101 import CandidateRiskRouterV101
from train_candidate_risk_router_v10 import frames, load_flow


PALETTE = np.asarray(
    [
        [32, 32, 32],
        [230, 70, 70],
        [70, 180, 90],
        [70, 120, 230],
        [235, 180, 55],
        [190, 80, 215],
    ],
    dtype=np.uint8,
)


def highpass(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(0, 1)
    smooth = F.avg_pool2d(flat, 5, stride=1, padding=2, count_include_pad=False)
    return (flat - smooth).unflatten(0, value.shape[:2])


def tile(image: np.ndarray, title: str) -> np.ndarray:
    canvas = Image.new("RGB", (256, 288), "white")
    canvas.paste(Image.fromarray(image, mode="RGB"), (0, 32))
    ImageDraw.Draw(canvas).text((6, 10), title, fill="black")
    return np.asarray(canvas)


def uint8_frames(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(0, 1, 3, 4, 2)
        .cpu()
        .numpy()
    )


def edge_mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = denominator = 0.0
    for dimension in (-1, -2):
        error = (torch.diff(prediction, dim=dimension) - torch.diff(target, dim=dimension)).abs()
        numerator += float(error.sum())
        denominator += error.numel()
    return 255.0 * numerator / denominator


def metrics(
    parent: torch.Tensor,
    routed: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    choice: torch.Tensor,
) -> dict:
    previous = torch.cat((context[:, -1:], target[:, :-1]), dim=1)
    moving = (target - previous).abs().mean(dim=2) >= 0.03
    dark = target.mean(dim=2) < 0.30
    parent_error = (parent - target).abs()
    routed_error = (routed - target).abs()

    def masked_mae(error: torch.Tensor, mask: torch.Tensor) -> float:
        count = int(mask.sum()) * 3
        return 255.0 * float((error * mask[:, :, None]).sum()) / max(count, 1)

    parent_mae = 255.0 * float(parent_error.mean())
    routed_mae = 255.0 * float(routed_error.mean())
    result = {
        "parent_rgb_mae": parent_mae,
        "router_rgb_mae": routed_mae,
        "rgb_improvement_percent": 100.0 * (parent_mae - routed_mae) / parent_mae,
        "parent_moving_rgb_mae": masked_mae(parent_error, moving),
        "router_moving_rgb_mae": masked_mae(routed_error, moving),
        "parent_highpass_mae": 255.0 * float((highpass(parent) - highpass(target)).abs().mean()),
        "router_highpass_mae": 255.0 * float((highpass(routed) - highpass(target)).abs().mean()),
        "parent_edge_mae": edge_mae(parent, target),
        "router_edge_mae": edge_mae(routed, target),
        "parent_dark_rgb_mae": masked_mae(parent_error, dark),
        "router_dark_rgb_mae": masked_mae(routed_error, dark),
        "selected_transport_block_fraction": float((choice != 0).float().mean()),
    }
    result["moving_rgb_improvement_percent"] = 100.0 * (
        result["parent_moving_rgb_mae"] - result["router_moving_rgb_mae"]
    ) / max(result["parent_moving_rgb_mae"], 1e-12)
    result["highpass_improvement_percent"] = 100.0 * (
        result["parent_highpass_mae"] - result["router_highpass_mae"]
    ) / result["parent_highpass_mae"]
    result["edge_improvement_percent"] = 100.0 * (
        result["parent_edge_mae"] - result["router_edge_mae"]
    ) / result["parent_edge_mae"]
    result["dark_improvement_percent"] = 100.0 * (
        result["parent_dark_rgb_mae"] - result["router_dark_rgb_mae"]
    ) / max(result["parent_dark_rgb_mae"], 1e-12)
    result["per_frame"] = []
    for index in range(target.shape[1]):
        parent_frame = 255.0 * float(parent_error[:, index].mean())
        routed_frame = 255.0 * float(routed_error[:, index].mean())
        result["per_frame"].append(
            {
                "frame": index,
                "parent_rgb_mae": parent_frame,
                "router_rgb_mae": routed_frame,
                "rgb_improvement_percent": 100.0 * (parent_frame - routed_frame) / parent_frame,
                "selected_transport_block_fraction": float((choice[:, index] != 0).float().mean()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--flow-head", required=True)
    parser.add_argument("--risk-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--highpass-protection", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0.0 <= args.highpass_protection <= 1.0:
        raise ValueError("highpass-protection must be in [0,1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent_predictions = cache["prediction"]
        cache_names = cache["windows"].astype(str).tolist()
    parent_by_name = {name: index for index, name in enumerate(cache_names)}

    base_checkpoint = Path(args.base_checkpoint)
    flow_model, active_mean, active_std = load_flow(base_checkpoint, Path(args.flow_head), device)
    with np.load(base_checkpoint / "action_normalization.npz", allow_pickle=False) as normalization:
        action_mean = torch.from_numpy(normalization["mean"]).to(device)
        action_std = torch.from_numpy(normalization["std"]).to(device)
    checkpoint = torch.load(args.risk_checkpoint, map_location="cpu", weights_only=False)
    if str(checkpoint.get("router_version")) != "v10.1":
        raise ValueError("risk checkpoint is not v10.1")
    router = CandidateRiskRouterV101(int(checkpoint["base_channels"]))
    router.load_state_dict(checkpoint["state_dict"], strict=True)
    router = router.requires_grad_(False).to(device).eval()

    for window_arg in args.windows:
        window_path = Path(window_arg)
        name = window_path.name
        if name not in parent_by_name:
            raise KeyError(f"{name} is absent from {args.parent_cache}")
        with np.load(window_path, allow_pickle=False) as window:
            context_np = window["context_frames"].copy()
            history_np = window["history_actions"].copy()
            future_np = window["future_actions"].copy()
            target_np = window["target_frames"].copy()
        parent_np = parent_predictions[parent_by_name[name]][None]
        context = frames(torch.from_numpy(context_np[None]), device)
        parent = frames(torch.from_numpy(parent_np), device)
        target = frames(torch.from_numpy(target_np[None]), device)
        actions = torch.from_numpy(np.concatenate((history_np, future_np), axis=0)[None]).to(device).float()

        with torch.inference_mode():
            active, arm = flow_model.active_arm_actions(actions)
            flow = flow_model(
                context,
                (actions - action_mean) / action_std,
                (active - active_mean) / active_std,
                arm,
                parent,
            )
            candidates = flow["candidates"]
            risk = router(
                context,
                parent,
                candidates[:, :, 1:],
                flow["refined_flow"],
                flow["visibility_logits"],
                active,
                arm,
            )
            choice, score = router.safe_choice(
                risk["advantage_mean"],
                risk["advantage_uncertainty"],
                args.risk_weight,
                args.margin,
            )
            full_choice = choice.repeat_interleave(4, 2).repeat_interleave(4, 3)
            routed_raw = candidates.gather(
                2, full_choice[:, :, None, None].expand(-1, -1, 1, 3, -1, -1)
            ).squeeze(2)
            routed = (
                routed_raw
                + args.highpass_protection * (highpass(parent) - highpass(routed_raw))
            ).clamp(0, 1)

        result = metrics(parent, routed, target, context, choice)
        result.update(
            {
                "format": "track2-candidate-risk-router-v10.1-video",
                "window": str(window_path.resolve()),
                "parent_cache": str(Path(args.parent_cache).resolve()),
                "risk_checkpoint": str(Path(args.risk_checkpoint).resolve()),
                "risk_step": int(checkpoint["step"]),
                "risk_weight": args.risk_weight,
                "margin": args.margin,
                "highpass_protection": args.highpass_protection,
                "mean_safe_score": float(score.mean()),
            }
        )
        parent_u8 = uint8_frames(parent)[0]
        routed_u8 = uint8_frames(routed)[0]
        full_choice_np = full_choice[0].byte().cpu().numpy()
        visuals = []
        last_context = context_np[-1]
        for index in range(target_np.shape[0]):
            per_frame = result["per_frame"][index]
            source_map = PALETTE[full_choice_np[index]]
            visuals.append(
                np.concatenate(
                    [
                        tile(last_context, "last observed frame"),
                        tile(target_np[index], f"ground truth t+{index + 1}"),
                        tile(parent_u8[index], f"v8 parent MAE {per_frame['parent_rgb_mae']:.3f}"),
                        tile(routed_u8[index], f"v10.1 HP MAE {per_frame['router_rgb_mae']:.3f}"),
                        tile(source_map, f"transport blocks {100 * per_frame['selected_transport_block_fraction']:.2f}%"),
                    ],
                    axis=1,
                )
            )

        stem = f"best_v101_hp05_{window_path.stem}"
        gif_path = output_dir / f"{stem}.gif"
        mp4_path = output_dir / f"{stem}.mp4"
        imageio.mimsave(gif_path, visuals, fps=args.fps, loop=0)
        imageio.mimsave(mp4_path, visuals, fps=args.fps, codec="libx264", quality=8)
        Image.fromarray(np.concatenate(visuals, axis=0), mode="RGB").save(
            output_dir / f"{stem}_contact_sheet.png"
        )
        np.savez_compressed(
            output_dir / f"{stem}_predictions.npz",
            parent=parent_u8,
            prediction=routed_u8,
            target=target_np,
            choice=choice[0].byte().cpu().numpy(),
        )
        (output_dir / f"{stem}.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"wrote {stem}: RGB {result['parent_rgb_mae']:.6f} -> {result['router_rgb_mae']:.6f} "
            f"({result['rgb_improvement_percent']:+.4f}%), selected "
            f"{100 * result['selected_transport_block_fraction']:.3f}%"
        )


if __name__ == "__main__":
    main()
