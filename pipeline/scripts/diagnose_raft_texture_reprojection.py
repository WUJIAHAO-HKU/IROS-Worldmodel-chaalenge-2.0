#!/usr/bin/env python3
"""Diagnose texture restoration by warping the last observation to predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def warp(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).unsqueeze(0)
    scale = torch.tensor((2.0 / (width - 1), 2.0 / (height - 1)), device=flow.device, dtype=flow.dtype)
    return functional.grid_sample(
        source, base + flow.permute(0, 2, 3, 1) * scale, mode="bilinear", padding_mode="border", align_corners=True
    )


def blur(value: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    return functional.avg_pool2d(value, kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    high_prediction = prediction - blur(prediction)
    high_target = target - blur(target)
    edge_prediction_x = prediction[..., 1:] - prediction[..., :-1]
    edge_target_x = target[..., 1:] - target[..., :-1]
    edge_prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    edge_target_y = target[..., 1:, :] - target[..., :-1, :]
    dark = target.mean(dim=1, keepdim=True) < 0.30
    return {
        "rgb_mae_0_255": float((prediction - target).abs().mean() * 255.0),
        "highpass_mae_0_255": float((high_prediction - high_target).abs().mean() * 255.0),
        "edge_mae_0_255": float(0.5 * ((edge_prediction_x - edge_target_x).abs().mean() + (edge_prediction_y - edge_target_y).abs().mean()) * 255.0),
        "dark_region_mae_0_255": float((prediction - target).abs()[dark.expand_as(target)].mean() * 255.0),
    }


def annotated_row(frames: list[np.ndarray], labels: list[str]) -> Image.Image:
    width, height = frames[0].shape[1], frames[0].shape[0]
    row = Image.new("RGB", (width * len(frames), height + 22), "white")
    draw = ImageDraw.Draw(row)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        row.paste(Image.fromarray(frame), (index * width, 22))
        draw.text((index * width + 4, 4), label, fill="black")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle-flow", action="store_true", help="Use target-to-context flow only as a hard upper-bound diagnostic.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.window, allow_pickle=False) as window:
        context = window["context_frames"][-1]
        target_numpy = window["target_frames"]
    prediction_numpy = np.load(args.prediction, allow_pickle=False)
    device = torch.device(args.device)
    prediction = torch.from_numpy(prediction_numpy.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
    target = torch.from_numpy(target_numpy.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
    source = torch.from_numpy(context.copy()).permute(2, 0, 1).unsqueeze(0).expand(8, -1, -1, -1).to(device).float().div(255.0)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    first, second = weights.transforms()(target if args.oracle_flow else prediction, source)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        flow = model(first, second)[-1].float()
    reprojected = warp(source, flow).clamp(0, 1)
    prediction_low, reprojected_low = blur(prediction), blur(reprojected)
    photometric = (prediction_low - reprojected_low).abs().mean(dim=1, keepdim=True)
    candidates: dict[str, torch.Tensor] = {"baseline": prediction}
    for strength in (0.25, 0.5, 0.75, 1.0):
        for tau in (0.03, 0.05, 0.08):
            confidence = torch.exp(-photometric / tau)
            restored = prediction + strength * confidence * ((reprojected - reprojected_low) - (prediction - prediction_low))
            candidates[f"s{strength:g}_t{tau:g}"] = restored.clamp(0, 1)
    for strength in (0.1, 0.2, 0.3, 0.5):
        for tau in (0.02, 0.03, 0.05):
            confidence = torch.exp(-photometric / tau)
            restored = prediction + strength * confidence * (reprojected - prediction)
            candidates[f"full_s{strength:g}_t{tau:g}"] = restored.clamp(0, 1)
    report = {name: metrics(value, target) for name, value in candidates.items()}
    best = min((name for name in report if name != "baseline"), key=lambda name: report[name]["highpass_mae_0_255"] + 0.25 * report[name]["rgb_mae_0_255"])
    report["selection"] = {"best": best, "criterion": "highpass_mae + 0.25 * rgb_mae"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    values = [prediction, reprojected, candidates[best], target]
    labels = ["baseline", "reprojected context", best, "ground truth"]
    frames = [[value[index].mul(255).round().byte().permute(1, 2, 0).cpu().numpy() for value in values] for index in range(8)]
    sheet = Image.new("RGB", (256 * 4, (256 + 22) * 8), "white")
    for index, row_frames in enumerate(frames):
        sheet.paste(annotated_row(row_frames, labels), (0, index * (256 + 22)))
    sheet.save(output / "contact_sheet.png")
    np.save(output / "best_prediction.npy", candidates[best].mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
