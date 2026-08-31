#!/usr/bin/env python3
"""Create train-only RAFT backward-flow targets for action-conditioned warping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def selected_paths(windows: Path, split_manifest: Path) -> list[Path]:
    split = json.loads(split_manifest.read_text())
    allowed = set(split["train_episodes"])
    paths = [path for path in sorted(windows.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed]
    if not paths:
        raise ValueError("no train windows found")
    return paths


def warp(source: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Apply target-to-source pixel flow to a batch of source frames."""
    batch, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).unsqueeze(0)
    scale = torch.tensor((2.0 / (width - 1), 2.0 / (height - 1)), device=flow.device, dtype=flow.dtype)
    grid = base + flow.permute(0, 2, 3, 1) * scale
    return functional.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--source-mode",
        choices=("last", "all-context"),
        default="last",
        help="Use only the newest frame or all five context frames as warp sources.",
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=16,
        help="Maximum target/source image pairs passed to RAFT at once.",
    )
    parser.add_argument("--flow-resolution", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.pair_batch_size < 1 or args.flow_resolution < 8 or args.flow_resolution > 256:
        raise SystemExit("batch size and pair batch size must be positive and flow resolution must be between 8 and 256")

    paths = selected_paths(Path(args.windows), Path(args.split_manifest))
    if args.limit is not None:
        paths = paths[: args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_count = 5 if args.source_mode == "all-context" else 1
    expected_shape = (
        (8, source_count, 2, args.flow_resolution, args.flow_resolution)
        if args.source_mode == "all-context"
        else (8, 2, args.flow_resolution, args.flow_resolution)
    )
    pending = []
    for path in paths:
        candidate = output / f"{path.stem}.npy"
        if not candidate.is_file():
            pending.append(path)
            continue
        try:
            if np.load(candidate, mmap_mode="r", allow_pickle=False).shape != expected_shape:
                pending.append(path)
        except Exception:
            pending.append(path)
    if not pending:
        print(json.dumps({"status": "complete", "train_window_count": len(paths), "written": 0, "output": str(output.resolve())}))
        return

    device = torch.device(args.device)
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    preprocess = weights.transforms()
    diagnostics: list[tuple[float, float]] = []
    for start in range(0, len(pending), args.batch_size):
        group = pending[start : start + args.batch_size]
        contexts, targets = [], []
        for path in group:
            with np.load(path, allow_pickle=False) as data:
                contexts.append(data["context_frames"].copy())
                targets.append(data["target_frames"].copy())
        source = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3).float().div(255.0).to(device)
        if args.source_mode == "last":
            source = source[:, -1:]
        source = source.to(device)
        target = torch.from_numpy(np.stack(targets)).permute(0, 1, 4, 2, 3).float().div(255.0).to(device)
        steps = target.shape[1]
        # RAFT(first, second) predicts displacement from first-image pixels to
        # second-image pixels. Supplying target then source gives the backward
        # flow required by grid_sample(source, target_grid + flow).
        source_count = source.shape[1]
        first = target[:, :, None].expand(-1, -1, source_count, -1, -1, -1).flatten(0, 2)
        second = source[:, None].expand(-1, steps, -1, -1, -1, -1).flatten(0, 2)
        predicted_pairs = []
        for pair_start in range(0, len(first), args.pair_batch_size):
            first_batch, second_batch = preprocess(
                first[pair_start : pair_start + args.pair_batch_size], second[pair_start : pair_start + args.pair_batch_size]
            )
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted_pairs.append(model(first_batch, second_batch)[-1].float())
        flow = torch.cat(predicted_pairs).reshape(
            len(group), steps, source_count, 2, source.shape[-2], source.shape[-1]
        )
        if len(diagnostics) < 8:
            # The newest context source is the meaningful copy/warp diagnostic.
            full_source = source[:, -1:,].expand(-1, steps, -1, -1, -1).flatten(0, 1)
            warped = warp(full_source, flow[:, :, -1].flatten(0, 1)).reshape_as(target)
            diagnostics.extend(
                zip(
                    (target - source[:, -1:]).abs().mean(dim=(1, 2, 3, 4)).cpu().tolist(),
                    (target - warped).abs().mean(dim=(1, 2, 3, 4)).cpu().tolist(),
                )
            )
        downsampled = functional.interpolate(
            flow.flatten(0, 2), size=(args.flow_resolution, args.flow_resolution), mode="bilinear", align_corners=True
        ).reshape(len(group), steps, source_count, 2, args.flow_resolution, args.flow_resolution)
        downsampled.mul_(args.flow_resolution / source.shape[-1])
        values = downsampled.cpu().numpy().astype(np.float16, copy=False)
        if args.source_mode == "last":
            values = values[:, :, 0]
        for path, value in zip(group, values):
            temporary = output / f".{path.stem}.tmp.npy"
            np.save(temporary, value)
            temporary.replace(output / f"{path.stem}.npy")
        print(
            json.dumps(
                {
                    "completed": min(start + len(group), len(pending)),
                    "pending": len(pending),
                    "total_train_windows": len(paths),
                }
            ),
            flush=True,
        )
    copied = np.asarray(diagnostics, dtype=np.float32)
    manifest = {
        "format": (
            "track2-raft-backward-flow-targets-v2"
            if args.source_mode == "all-context"
            else "track2-raft-backward-flow-targets-v1"
        ),
        "source_windows": str(Path(args.windows).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "split": "train_episodes_only",
        "window_count": len(paths),
        "flow_resolution": args.flow_resolution,
        "raft_weights": str(weights.url),
        "diagnostic_copy_last_mae": float(copied[:, 0].mean()) if len(copied) else None,
        "diagnostic_raft_warp_mae": float(copied[:, 1].mean()) if len(copied) else None,
    }
    if args.source_mode == "all-context":
        manifest["source_mode"] = args.source_mode
        manifest["source_count"] = source_count
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
