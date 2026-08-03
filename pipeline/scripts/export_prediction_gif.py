#!/usr/bin/env python3
"""Export a labeled Track 2 input/prediction/ground-truth comparison GIF."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", required=True)
    parser.add_argument("--backend", choices=("synthetic", "ivideogpt", "residual-unet", "flow-residual-unet", "temporal-unet", "autoregressive-unet", "hybrid-unet"), default="ivideogpt")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    window = load_window_npz(args.window)
    backend = build_backend(args.backend, args.checkpoint_dir, args.device)
    prediction = backend.predict(
        window.context_frames, window.history_actions, window.future_actions, args.seed, instruction=None
    )
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
    print(f"wrote {output} with {len(frames)} frames")
    print(f"prediction min={int(prediction.min())} max={int(prediction.max())} mean={float(prediction.mean()):.3f}")


if __name__ == "__main__":
    main()
