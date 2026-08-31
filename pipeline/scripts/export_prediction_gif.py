#!/usr/bin/env python3
"""Export a labeled Track 2 input/prediction/ground-truth comparison GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz


def labeled(image: np.ndarray, text: str) -> np.ndarray:
    """Add a small header without altering the 256-pixel RGB model input."""
    canvas = Image.new("RGB", (256, 280), "white")
    canvas.paste(Image.fromarray(image, mode="RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), text, fill="black")
    return np.asarray(canvas)


def motion_score(context: np.ndarray, target: np.ndarray) -> float:
    """Mean true rollout displacement from the final observation onward."""
    sequence = np.concatenate((context[-1:], target), axis=0).astype(np.float32)
    return float(np.abs(sequence[1:] - sequence[:-1]).mean() / 255.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument("--backend", choices=("synthetic", "ivideogpt", "track2-wan", "official-rlinf-wan", "residual-unet", "flow-residual-unet", "temporal-unet", "direct-video-unet", "direct-flow-unet", "wan-flow-ensemble", "autoregressive-flow-ensemble", "local-motion-texture-fusion", "structure-gated-local-fusion", "multisource-flow-unet", "autoregressive-unet", "autoregressive-structure-unet", "hybrid-unet"), default="ivideogpt")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-base-model", help="Local Wan Diffusers base required by --backend track2-wan.")
    parser.add_argument("--wan-inference-steps", type=int, default=30)
    parser.add_argument("--wan-inference-solver", choices=("euler", "heun"), default="euler")
    parser.add_argument(
        "--official-diffsynth-root",
        help="Pinned DiffSynth checkout required by --backend official-rlinf-wan.",
    )
    parser.add_argument(
        "--official-wan-inference-steps",
        type=int,
        default=5,
        help="Official RLinf baseline is 5 steps.",
    )
    args = parser.parse_args()

    window = load_window_npz(args.window)
    backend = build_backend(
        args.backend,
        args.checkpoint_dir,
        args.device,
        wan_base_model=args.wan_base_model,
        wan_inference_steps=args.wan_inference_steps,
        wan_inference_solver=args.wan_inference_solver,
        official_diffsynth_root=args.official_diffsynth_root,
        official_wan_inference_steps=args.official_wan_inference_steps,
    )
    prediction = backend.predict(
        window.context_frames, window.history_actions, window.future_actions, args.seed, instruction=None
    )
    mae_by_frame = np.abs(prediction.astype(np.float32) - window.target_frames.astype(np.float32)).mean(axis=(1, 2, 3))
    metrics = {
        "window": str(Path(args.window).resolve()),
        "backend": args.backend,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "seed": args.seed,
        "wan_inference_steps": args.wan_inference_steps if args.backend == "track2-wan" else None,
        "wan_inference_solver": args.wan_inference_solver if args.backend == "track2-wan" else None,
        "official_wan_inference_steps": (
            args.official_wan_inference_steps if args.backend == "official-rlinf-wan" else None
        ),
        "motion_score": motion_score(window.context_frames, window.target_frames),
        "mae_by_prediction_frame": [float(value) for value in mae_by_frame],
        "mae_mean": float(mae_by_frame.mean()),
        "mae_max": float(mae_by_frame.max()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output.with_suffix(".prediction.npy"), prediction)

    frames = []
    for index in range(5):
        blank = np.full_like(window.context_frames[index], 255)
        frames.append(np.concatenate([labeled(window.context_frames[index], f"context o{index}"), labeled(blank, "prediction"), labeled(blank, "ground truth")], axis=1))
    for index in range(8):
        blank = np.full_like(prediction[index], 255)
        frames.append(
            np.concatenate(
                [
                    labeled(blank, f"future action u{index}"),
                    labeled(prediction[index], f"prediction p{index}"),
                    labeled(window.target_frames[index], f"ground truth o{index + 5}"),
                ],
                axis=1,
            )
        )
    imageio.mimsave(output, frames, fps=args.fps, loop=0)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {output} with {len(frames)} frames")
    print(f"prediction min={int(prediction.min())} max={int(prediction.max())} mean={float(prediction.mean()):.3f}")
    print(f"mae_mean={metrics['mae_mean']:.4f} mae_max={metrics['mae_max']:.4f} motion={metrics['motion_score']:.5f}")


if __name__ == "__main__":
    main()
