#!/usr/bin/env python3
"""Measure a target-aware multi-context texture transport upper bound.

This is deliberately an oracle diagnostic, not a deployable predictor: ground-truth
future frames are used to estimate backward flow and to choose among transported
sources.  The report answers whether better motion/occlusion routing could plausibly
beat the current parent, and how much error remains in pixels not recoverable from
the five observed frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item < 1 or 256 % item for item in result):
        raise argparse.ArgumentTypeError("block sizes must be positive divisors of 256")
    return result


def load_prediction_cache(path: Path) -> tuple[np.ndarray, list[str]]:
    cache = np.load(path, allow_pickle=False, mmap_mode="r")
    if set(cache.files) != {"prediction", "windows"}:
        raise ValueError(f"invalid prediction cache: {path}")
    prediction = cache["prediction"]
    names = [str(name) for name in cache["windows"]]
    if prediction.shape[1:] != (8, 256, 256, 3) or prediction.dtype != np.uint8:
        raise ValueError(f"unexpected prediction array: {prediction.shape} {prediction.dtype}")
    return prediction, names


def warp(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).unsqueeze(0)
    scale = flow.new_tensor((2.0 / (width - 1), 2.0 / (height - 1)))
    return F.grid_sample(
        source,
        base + flow.permute(0, 2, 3, 1) * scale,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


@torch.inference_mode()
def transport_all_sources(
    model: torch.nn.Module,
    transforms,
    context: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Return [8,5,3,256,256] target-to-context transported RGB."""
    steps, sources = target.shape[0], context.shape[0]
    first = target[:, None].expand(-1, sources, -1, -1, -1).flatten(0, 1)
    second = context[None].expand(steps, -1, -1, -1, -1).flatten(0, 1)
    warped: list[torch.Tensor] = []
    for start in range(0, len(first), batch_size):
        stop = min(start + batch_size, len(first))
        normalized_first, normalized_second = transforms(first[start:stop], second[start:stop])
        with torch.autocast(device_type=first.device.type, dtype=torch.float16, enabled=first.device.type == "cuda"):
            flow = model(normalized_first, normalized_second)[-1]
        warped.append(warp(second[start:stop], flow.float()).clamp(0.0, 1.0))
    return torch.cat(warped).unflatten(0, (steps, sources))


def choose_pixel(candidates: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = (candidates - target[:, None]).abs().mean(dim=2)
    choice = error.argmin(dim=1)
    gather = choice[:, None, None].expand(-1, 1, 3, -1, -1)
    return candidates.gather(1, gather).squeeze(1), error.min(dim=1).values


def choose_blocks(candidates: torch.Tensor, target: torch.Tensor, block: int) -> torch.Tensor:
    steps, count, _, height, width = candidates.shape
    error = (candidates - target[:, None]).abs().mean(dim=2)
    pooled = error.reshape(steps, count, height // block, block, width // block, block).mean(dim=(3, 5))
    choice = pooled.argmin(dim=1)
    choice = choice.repeat_interleave(block, 1).repeat_interleave(block, 2)
    gather = choice[:, None, None].expand(-1, 1, 3, -1, -1)
    return candidates.gather(1, gather).squeeze(1)


def highpass(value: torch.Tensor) -> torch.Tensor:
    low = F.avg_pool2d(value, 5, stride=1, padding=2, count_include_pad=False)
    return value - low


class Accumulator:
    def __init__(self, methods: list[str]) -> None:
        self.methods = methods
        self.total = {name: 0.0 for name in methods}
        self.moving = {name: 0.0 for name in methods}
        self.dark = {name: 0.0 for name in methods}
        self.highpass = {name: 0.0 for name in methods}
        self.total_count = self.moving_count = self.dark_count = self.highpass_count = 0
        self.high_motion_total = {name: 0.0 for name in methods}
        self.high_motion_count = 0

    def add(self, predictions: dict[str, torch.Tensor], target: torch.Tensor, previous: torch.Tensor, high_motion: bool) -> None:
        moving = (target - previous).abs().mean(dim=1) >= 0.03
        dark = target.mean(dim=1) < 0.30
        target_highpass = highpass(target)
        for name, prediction in predictions.items():
            error = (prediction - target).abs()
            self.total[name] += float(error.sum())
            self.moving[name] += float((error * moving[:, None]).sum())
            self.dark[name] += float((error * dark[:, None]).sum())
            self.highpass[name] += float((highpass(prediction) - target_highpass).abs().sum())
            if high_motion:
                self.high_motion_total[name] += float(error.sum())
        count = target.numel()
        self.total_count += count
        self.moving_count += int(moving.sum()) * 3
        self.dark_count += int(dark.sum()) * 3
        self.highpass_count += count
        if high_motion:
            self.high_motion_count += count

    def metrics(self) -> dict[str, dict[str, float | None]]:
        result = {}
        for name in self.methods:
            result[name] = {
                "rgb_mae_0_255": 255.0 * self.total[name] / self.total_count,
                "moving_rgb_mae_0_255": 255.0 * self.moving[name] / max(1, self.moving_count),
                "dark_rgb_mae_0_255": 255.0 * self.dark[name] / max(1, self.dark_count),
                "highpass_mae_0_255": 255.0 * self.highpass[name] / self.highpass_count,
                "high_motion_rgb_mae_0_255": (
                    255.0 * self.high_motion_total[name] / self.high_motion_count if self.high_motion_count else None
                ),
            }
        return result


def to_uint8(value: torch.Tensor) -> np.ndarray:
    return value.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_visualization(path: Path, values: dict[str, torch.Tensor]) -> None:
    arrays = {name: to_uint8(value) for name, value in values.items()}
    labels = list(arrays)
    width = 256 * len(labels)
    canvas = Image.new("RGB", (width, 8 * 278), "white")
    draw = ImageDraw.Draw(canvas)
    for frame in range(8):
        for column, label in enumerate(labels):
            x, y = column * 256, frame * 278
            draw.text((x + 4, y + 4), f"t+{frame + 1} {label}", fill="black")
            canvas.paste(Image.fromarray(arrays[label][frame]), (x, y + 22))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--prediction-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--include-window", action="append", default=[])
    parser.add_argument("--block-sizes", type=parse_ints, default=parse_ints("4,8,16,32"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parent, names = load_prediction_cache(Path(args.prediction_cache))
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = list(np.linspace(0, len(names) - 1, min(args.samples, len(names)), dtype=np.int64))
    for name in args.include_window:
        if name not in name_to_index:
            raise ValueError(f"included window is absent from prediction cache: {name}")
        indices.append(name_to_index[name])
    indices = list(dict.fromkeys(indices))

    methods = ["v8_parent", "last_context_oracle_flow", "multisource_pixel_oracle", "hybrid_pixel_oracle"]
    methods += [f"multisource_block{block}_oracle" for block in args.block_sizes]
    methods += [f"hybrid_block{block}_oracle" for block in args.block_sizes]
    accumulator = Accumulator(methods)
    device = torch.device(args.device)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    transforms = weights.transforms()
    coverage_thresholds = (2.0, 5.0, 10.0, 20.0)
    coverage = {str(value): 0 for value in coverage_thresholds}
    moving_coverage = {str(value): 0 for value in coverage_thresholds}
    coverage_count = moving_coverage_count = 0
    sample_records = []
    visualizations: dict[str, dict[str, torch.Tensor]] = {}

    for completed, index in enumerate(indices, start=1):
        name = names[index]
        with np.load(Path(args.windows) / name, allow_pickle=False) as window:
            context_np = window["context_frames"]
            target_np = window["target_frames"]
        context = torch.from_numpy(context_np.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        target = torch.from_numpy(target_np.copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        parent_prediction = torch.from_numpy(parent[index].copy()).permute(0, 3, 1, 2).to(device).float().div(255.0)
        transported = transport_all_sources(model, transforms, context, target, args.batch_size)
        multisource_pixel, source_error = choose_pixel(transported, target)
        hybrid_candidates = torch.cat((parent_prediction[:, None], transported), dim=1)
        hybrid_pixel, _ = choose_pixel(hybrid_candidates, target)
        predictions = {
            "v8_parent": parent_prediction,
            "last_context_oracle_flow": transported[:, -1],
            "multisource_pixel_oracle": multisource_pixel,
            "hybrid_pixel_oracle": hybrid_pixel,
        }
        for block in args.block_sizes:
            predictions[f"multisource_block{block}_oracle"] = choose_blocks(transported, target, block)
            predictions[f"hybrid_block{block}_oracle"] = choose_blocks(hybrid_candidates, target, block)
        previous = torch.cat((context[-1:], target[:-1]), dim=0)
        motion_score = float((target - previous).abs().mean())
        moving = (target - previous).abs().mean(dim=1) >= 0.03
        accumulator.add(predictions, target, previous, motion_score >= 0.04)
        coverage_count += source_error.numel()
        moving_coverage_count += int(moving.sum())
        for threshold in coverage_thresholds:
            recoverable = source_error * 255.0 <= threshold
            coverage[str(threshold)] += int(recoverable.sum())
            moving_coverage[str(threshold)] += int((recoverable & moving).sum())
        sample_records.append({"window": name, "motion_score": motion_score})
        if name in args.include_window:
            visualizations[name] = {
                "v8": parent_prediction,
                "last-warp": transported[:, -1],
                "multi-block8": predictions.get("multisource_block8_oracle", multisource_pixel),
                "hybrid-block8": predictions.get("hybrid_block8_oracle", hybrid_pixel),
                "hybrid-pixel": hybrid_pixel,
                "target": target,
            }
        print(json.dumps({"completed": completed, "total": len(indices), "window": name, "motion_score": motion_score}), flush=True)

    metrics = accumulator.metrics()
    baseline = metrics["v8_parent"]["rgb_mae_0_255"]
    for value in metrics.values():
        value["relative_rgb_improvement_over_v8_percent"] = 100.0 * (baseline - value["rgb_mae_0_255"]) / baseline
    result = {
        "format": "track2-multisource-transport-oracle-v1",
        "warning": "Diagnostic ceiling only: future targets are used for RAFT flow and candidate routing.",
        "prediction_cache": str(Path(args.prediction_cache).resolve()),
        "sample_count": len(indices),
        "high_motion_sample_count": sum(record["motion_score"] >= 0.04 for record in sample_records),
        "block_sizes": args.block_sizes,
        "metrics": metrics,
        "source_recoverability": {
            threshold: {
                "all_pixel_fraction": coverage[threshold] / coverage_count,
                "moving_pixel_fraction": moving_coverage[threshold] / max(1, moving_coverage_count),
            }
            for threshold in coverage
        },
        "targets": {"full_rgb_mae_30pct": 4.163416, "high_motion_rgb_mae_30pct": 7.301133},
        "samples": sample_records,
    }
    report = output / "report.json"
    temporary = report.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, report)
    for name, values in visualizations.items():
        save_visualization(output / f"{Path(name).stem}_contact_sheet.png", values)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
