#!/usr/bin/env python3
"""Measure deterministic open-loop Track 2 predictions on episode-held-out windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from wam_pipeline.backends import build_backend
from wam_pipeline.data import load_window_npz


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("--samples must be positive")
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    return [paths[index] for index in indices]


def file_sha256(path: Path) -> str | None:
    """Return a stable digest for a deployable checkpoint file when present."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_artifact_sha256(checkpoint_dir: Path, backend: str) -> str | None:
    """Identify the exact deployable artifact, including a packaged ensemble."""
    if backend == "wan-flow-ensemble":
        from wam_pipeline.wan_flow_ensemble_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    if backend == "autoregressive-flow-ensemble":
        from wam_pipeline.autoregressive_flow_ensemble_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    if backend == "local-motion-texture-fusion":
        from wam_pipeline.local_fusion_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    if backend == "structure-gated-local-fusion":
        from wam_pipeline.structure_refiner_runtime import ensemble_artifact_sha256

        return ensemble_artifact_sha256(checkpoint_dir)
    filename = (
        "track2_wan_lora.pt"
        if backend == "track2-wan"
        else "dit_model.safetensors"
        if backend == "official-rlinf-wan"
        else "model.pt"
    )
    return file_sha256(checkpoint_dir / filename)


class ResumablePredictionCache:
    """Durably persist each rollout before proceeding to the next window."""

    FORMAT = "track2-resumable-prediction-cache-v1"

    def __init__(self, root: Path, metadata: dict[str, object], count: int) -> None:
        self.root = root.resolve()
        self.metadata = {"format": self.FORMAT, **metadata}
        self.prediction_path = self.root / "predictions.npy"
        self.completed_path = self.root / "completed.npy"
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"resumable prediction cache manifest is invalid: {manifest_path}") from exc
            if existing != self.metadata:
                raise RuntimeError("resumable prediction cache belongs to a different evaluation")
            if not self.prediction_path.is_file() or not self.completed_path.is_file():
                raise RuntimeError("resumable prediction cache is incomplete")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise RuntimeError(f"refusing to reuse cache directory without a manifest: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
            prediction_temp = self.prediction_path.with_suffix(".npy.tmp")
            completed_temp = self.completed_path.with_suffix(".npy.tmp")
            predictions = np.lib.format.open_memmap(
                prediction_temp, mode="w+", dtype=np.uint8, shape=(count, 8, 256, 256, 3)
            )
            predictions.flush()
            del predictions
            completed = np.lib.format.open_memmap(completed_temp, mode="w+", dtype=np.bool_, shape=(count,))
            completed[:] = False
            completed.flush()
            del completed
            os.replace(prediction_temp, self.prediction_path)
            os.replace(completed_temp, self.completed_path)
            manifest_path.write_text(json.dumps(self.metadata, indent=2) + "\n")
        self.predictions = np.load(self.prediction_path, mmap_mode="r+", allow_pickle=False)
        self.completed = np.load(self.completed_path, mmap_mode="r+", allow_pickle=False)
        expected_predictions = (count, 8, 256, 256, 3)
        if self.predictions.shape != expected_predictions or self.predictions.dtype != np.uint8:
            raise RuntimeError("resumable prediction cache prediction array has an invalid shape or dtype")
        if self.completed.shape != (count,) or self.completed.dtype != np.bool_:
            raise RuntimeError("resumable prediction cache completion array has an invalid shape or dtype")

    def get(self, index: int) -> np.ndarray | None:
        if not bool(self.completed[index]):
            return None
        return np.array(self.predictions[index], copy=True)

    def put(self, index: int, prediction: np.ndarray) -> None:
        if prediction.shape != (8, 256, 256, 3) or prediction.dtype != np.uint8:
            raise RuntimeError("cannot cache an invalid Track 2 prediction")
        self.predictions[index] = prediction
        self.predictions.flush()
        # A completed marker is only durable after all prediction bytes are durable.
        self.completed[index] = True
        self.completed.flush()

    @property
    def completed_count(self) -> int:
        return int(np.count_nonzero(self.completed))


def labeled(image: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.new("RGB", (256, 280), "white")
    canvas.paste(Image.fromarray(image, mode="RGB"), (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), text, fill="black")
    return np.asarray(canvas)


def write_comparison_gif(path: Path, context: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    """Persist the already-computed failure rollout without another model call."""
    frames = []
    for index in range(5):
        blank = np.full_like(context[index], 255)
        frames.append(
            np.concatenate(
                [
                    labeled(context[index], f"context o{index}"),
                    labeled(blank, "prediction"),
                    labeled(blank, "ground truth"),
                ],
                axis=1,
            )
        )
    for index in range(8):
        blank = np.full_like(prediction[index], 255)
        frames.append(
            np.concatenate(
                [
                    labeled(blank, f"future action u{index}"),
                    labeled(prediction[index], f"prediction p{index}"),
                    labeled(target[index], f"ground truth o{index + 5}"),
                ],
                axis=1,
            )
        )
    imageio.mimsave(path, frames, fps=3, loop=0)


def window_motion_score(context: np.ndarray, target: np.ndarray) -> float:
    """Measure observed-to-future RGB displacement for motion-subset reporting."""
    sequence = np.concatenate((context[-1:], target), axis=0).astype(np.float32)
    return float(np.abs(sequence[1:] - sequence[:-1]).mean() / 255.0)


def write_progress_report(
    path: Path,
    *,
    backend: str,
    checkpoint_dir: Path,
    artifact_sha256: str | None,
    completed: int,
    total: int,
    model_maes: list[np.ndarray],
    cache_hits: int,
    accept_mae: float,
) -> None:
    """Atomically expose resumable evaluation status without claiming a final pass."""
    values = np.asarray(model_maes, dtype=np.float32)
    peaks = values.max(axis=1)
    result = {
        "format": "track2-open-loop-eval-progress-v1",
        "backend": backend,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_model_sha256": artifact_sha256,
        "completed_windows": completed,
        "total_windows": total,
        "cache_hits": cache_hits,
        "model_mae_mean": float(values.mean()),
        "worst_window_prediction_frame_mae": float(peaks.max()),
        "windows_passing_all_frames": int((peaks < accept_mae).sum()),
        "accept_mae_0_255": float(accept_mae),
        "final": completed == total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backend", choices=("ivideogpt", "track2-wan", "official-rlinf-wan", "residual-unet", "flow-residual-unet", "temporal-unet", "direct-video-unet", "direct-flow-unet", "recursive-flow-unet", "wan-flow-ensemble", "autoregressive-flow-ensemble", "local-motion-texture-fusion", "structure-gated-local-fusion", "multisource-flow-unet", "autoregressive-unet", "autoregressive-structure-unet", "hybrid-unet"), default="ivideogpt")
    parser.add_argument("--split", choices=("train", "validation", "local-test"), default="validation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-base-model", help="Local Wan Diffusers base required by --backend track2-wan.")
    parser.add_argument("--wan-inference-steps", type=int, default=30)
    parser.add_argument(
        "--wan-inference-solver",
        choices=("euler", "heun"),
        default="euler",
        help="ODE integrator for the native Track 2 Wan flow model.",
    )
    parser.add_argument(
        "--official-diffsynth-root",
        help="Pinned DiffSynth checkout required by --backend official-rlinf-wan.",
    )
    parser.add_argument("--official-wan-inference-steps", type=int, default=5)
    parser.add_argument(
        "--high-motion-threshold",
        type=float,
        default=0.04,
        help="Mean true inter-frame RGB change used for the difficult-motion subset.",
    )
    parser.add_argument(
        "--accept-mae",
        type=float,
        default=2.5,
        help="Hard acceptance threshold in 0--255 RGB MAE for every window and prediction frame.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit unsuccessfully unless every selected window and future frame meets --accept-mae.",
    )
    parser.add_argument(
        "--measure-action-sensitivity",
        action="store_true",
        help="Also run a perturbed-action prediction for each window; not required for strict MAE acceptance.",
    )
    parser.add_argument(
        "--worst-windows",
        type=int,
        default=20,
        help="Number of worst held-out windows retained in the compact failure summary.",
    )
    parser.add_argument(
        "--failure-gif-count",
        type=int,
        default=3,
        help="Export this many worst comparison GIFs on a strict-evaluation failure; 0 disables exports.",
    )
    parser.add_argument(
        "--failure-gif-dir",
        help="Directory for failure GIFs; defaults beside --output when --require-pass is used.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Emit cumulative evaluation status every N completed windows; 0 disables progress messages.",
    )
    parser.add_argument(
        "--progress-output",
        help="Optional atomic JSON status written at each --progress-interval; it is not a final evaluation report.",
    )
    parser.add_argument(
        "--prediction-cache",
        help="Optional atomic .npz cache of selected predictions and their ordered window names.",
    )
    parser.add_argument(
        "--resume-cache-dir",
        help="Optional directory that durably caches each prediction so an interrupted evaluation resumes exactly.",
    )
    parser.add_argument(
        "--export-worst-gifs",
        action="store_true",
        help="Export --failure-gif-count worst rollouts even when --require-pass is not set.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split = json.loads(Path(args.split_manifest).read_text())
    key = args.split.replace("-", "_") + "_episodes"
    episodes = split[key]
    paths = [
        path
        for episode in episodes
        for path in sorted(Path(args.windows).glob(f"episode{episode}_*.npz"))
    ]
    if not paths:
        raise SystemExit(f"no windows for {args.split} split")
    if (
        args.accept_mae <= 0
        or args.worst_windows < 0
        or args.failure_gif_count < 0
        or args.progress_interval < 0
    ):
        raise SystemExit("--accept-mae must be positive and summary/GIF counts must be non-negative")

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
    selected = evenly_spaced(paths, args.samples)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    artifact_sha256 = checkpoint_artifact_sha256(checkpoint_dir, args.backend)
    ensemble_metadata = None
    if args.backend == "wan-flow-ensemble":
        from wam_pipeline.wan_flow_ensemble_runtime import load_ensemble_config, wan_base_identity

        config = load_ensemble_config(checkpoint_dir)
        actual_base = wan_base_identity(args.wan_base_model)
        if config["wan"]["base_model"] != actual_base:
            raise RuntimeError("active Wan base identity does not match the packaged ensemble")
        ensemble_metadata = {
            "format": config["format"],
            "wan_weight": config["wan_weight"],
            "direct_flow_weight": config["direct_flow_weight"],
            "wan_inference": {
                "steps": config["wan"]["inference_steps"],
                "solver": config["wan"]["inference_solver"],
            },
            "wan_base_model": actual_base,
        }
    elif args.backend == "autoregressive-flow-ensemble":
        from wam_pipeline.autoregressive_flow_ensemble_runtime import load_ensemble_config

        config = load_ensemble_config(checkpoint_dir)
        ensemble_metadata = {
            "format": config["format"],
            "context_motion_statistic": config["context_motion_statistic"],
            "context_motion_threshold": config["context_motion_threshold"],
            "high_motion_policy": config["high_motion_policy"],
            "low_motion_autoregressive_weight": config["low_motion_autoregressive_weight"],
            "low_motion_direct_flow_weight": config["low_motion_direct_flow_weight"],
        }
    elif args.backend == "local-motion-texture-fusion":
        from wam_pipeline.local_fusion_runtime import load_ensemble_config

        config = load_ensemble_config(checkpoint_dir)
        ensemble_metadata = {
            "format": config["format"],
            "selection_reports": config.get("selection_reports"),
        }
    elif args.backend == "structure-gated-local-fusion":
        from wam_pipeline.structure_refiner_runtime import load_ensemble_config

        config = load_ensemble_config(checkpoint_dir)
        ensemble_metadata = {
            "format": config["format"],
            "baseline_artifact_sha256": config["baseline"]["artifact_sha256"],
            "selection_reports": config.get("selection_reports"),
        }
    resume_cache = None
    if args.resume_cache_dir:
        resume_cache = ResumablePredictionCache(
            Path(args.resume_cache_dir),
            {
                "backend": args.backend,
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_model_sha256": artifact_sha256,
                "seed": args.seed,
                "selected_windows": [path.name for path in selected],
                "wan_base_model": str(Path(args.wan_base_model).resolve()) if args.wan_base_model else None,
                "wan_inference_steps": args.wan_inference_steps,
                "wan_inference_solver": args.wan_inference_solver,
            },
            len(selected),
        )
        print(
            json.dumps(
                {
                    "event": "resumable_cache_ready",
                    "completed_windows": resume_cache.completed_count,
                    "total_windows": len(selected),
                    "path": str(resume_cache.root),
                }
            ),
            flush=True,
        )
    model_maes: list[np.ndarray] = []
    copy_last_maes: list[np.ndarray] = []
    predictions: list[np.ndarray] | None = [] if args.prediction_cache and resume_cache is None else None
    motion_scores: list[float] = []
    action_sensitivity: list[float] = []
    gif_candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray]] = []
    cache_hits = 0
    for index, path in enumerate(selected, start=1):
        window = load_window_npz(path)
        prediction = resume_cache.get(index - 1) if resume_cache else None
        cache_hit = prediction is not None
        if prediction is None:
            prediction = backend.predict(
                window.context_frames,
                window.history_actions,
                window.future_actions,
                args.seed,
                instruction=None,
            )
            if resume_cache:
                resume_cache.put(index - 1, prediction)
            elif predictions is not None:
                predictions.append(prediction)
        if cache_hit:
            cache_hits += 1
        target = window.target_frames.astype(np.float32)
        previous = np.concatenate([window.context_frames[-1:].astype(np.float32), target[:-1]], axis=0)
        model_maes.append(np.abs(prediction.astype(np.float32) - target).mean(axis=(1, 2, 3)))
        copy_last_maes.append(np.abs(window.context_frames[-1:].astype(np.float32) - target).mean(axis=(1, 2, 3)))
        motion_scores.append(window_motion_score(window.context_frames, window.target_frames))
        if args.measure_action_sensitivity:
            altered_actions = window.future_actions.copy()
            altered_actions[:, 0] += 0.05
            altered = backend.predict(
                window.context_frames,
                window.history_actions,
                altered_actions,
                args.seed,
                instruction=None,
            )
            action_sensitivity.append(float(np.abs(prediction.astype(np.float32) - altered.astype(np.float32)).mean()))
        if args.failure_gif_count:
            peak = float(model_maes[-1].max())
            gif_candidates.append(
                (peak, len(model_maes) - 1, window.context_frames.copy(), prediction.copy(), window.target_frames.copy())
            )
            gif_candidates.sort(key=lambda value: (-value[0], value[1]))
            del gif_candidates[args.failure_gif_count :]
        if args.progress_interval and (index % args.progress_interval == 0 or index == len(selected)):
            completed = np.asarray(model_maes, dtype=np.float32)
            peaks = completed.max(axis=1)
            print(
                json.dumps(
                    {
                        "event": "evaluation_progress",
                        "completed_windows": index,
                        "total_windows": len(selected),
                        "rgb_mae_mean": float(completed.mean()),
                        "worst_window_prediction_frame_mae": float(peaks.max()),
                        "windows_passing_all_frames": int((peaks < args.accept_mae).sum()),
                    }
                ),
                flush=True,
            )
            if args.progress_output:
                write_progress_report(
                    Path(args.progress_output),
                    backend=args.backend,
                    checkpoint_dir=checkpoint_dir,
                    artifact_sha256=artifact_sha256,
                    completed=index,
                    total=len(selected),
                    model_maes=model_maes,
                    cache_hits=cache_hits,
                    accept_mae=args.accept_mae,
                )

    model_by_frame = np.mean(model_maes, axis=0)
    copy_by_frame = np.mean(copy_last_maes, axis=0)
    high_motion = np.asarray(motion_scores) >= args.high_motion_threshold
    high_model_by_frame = np.mean(np.asarray(model_maes)[high_motion], axis=0) if high_motion.any() else None
    high_copy_by_frame = np.mean(np.asarray(copy_last_maes)[high_motion], axis=0) if high_motion.any() else None
    per_window = np.asarray(model_maes, dtype=np.float32)
    per_window_mean = per_window.mean(axis=1)
    per_window_peak = per_window.max(axis=1)
    # A candidate passes only when every held-out window and every future frame
    # meets the pixel-MAE target. Means alone can hide catastrophic rollouts.
    acceptance = {
        "threshold_mae_0_255": float(args.accept_mae),
        "window_count": int(len(per_window)),
        "windows_passing_mean_over_8_frames": int((per_window_mean < args.accept_mae).sum()),
        "windows_passing_all_8_prediction_frames": int((per_window_peak < args.accept_mae).sum()),
        "max_window_mean_over_8_frames": float(per_window_mean.max()),
        "max_window_prediction_frame_mae": float(per_window_peak.max()),
        "p50_window_mean_over_8_frames": float(np.quantile(per_window_mean, 0.50)),
        "p90_window_mean_over_8_frames": float(np.quantile(per_window_mean, 0.90)),
        "p99_window_mean_over_8_frames": float(np.quantile(per_window_mean, 0.99)),
        "all_windows_and_frames_pass": bool((per_window_peak < args.accept_mae).all()),
    }
    window_acceptance = [
        {
            "window": path.name,
            "mean_mae_over_8_frames": float(per_window[index].mean()),
            "max_prediction_frame_mae": float(per_window[index].max()),
            "mae_by_prediction_frame": [float(value) for value in per_window[index]],
            "passes_all_prediction_frames": bool(per_window[index].max() < args.accept_mae),
        }
        for index, path in enumerate(selected)
    ]
    worst_indices = np.argsort(-per_window_peak)[: args.worst_windows]
    worst_windows = [
        {
            "window": selected[index].name,
            "motion_score": motion_scores[index],
            "mean_mae_over_8_frames": float(per_window_mean[index]),
            "max_prediction_frame_mae": float(per_window_peak[index]),
            "mae_by_prediction_frame": [float(value) for value in per_window[index]],
        }
        for index in worst_indices
    ]
    worst_by_prediction_frame = [
        {
            "prediction_frame": frame,
            "window": selected[int(np.argmax(per_window[:, frame]))].name,
            "mae": float(per_window[:, frame].max()),
            "motion_score": motion_scores[int(np.argmax(per_window[:, frame]))],
        }
        for frame in range(per_window.shape[1])
    ]
    passed = acceptance["all_windows_and_frames_pass"]
    output = Path(args.output)
    failure_gifs: list[str] = []
    if (args.export_worst_gifs or (args.require_pass and not passed)) and args.failure_gif_count:
        gif_dir = Path(args.failure_gif_dir) if args.failure_gif_dir else output.parent / f"{output.stem}_worst_gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        for rank, (peak, index, context, prediction, target) in enumerate(gif_candidates, start=1):
            path = gif_dir / f"rank{rank:02d}_{selected[index].stem}_peak_{peak:.3f}.gif"
            write_comparison_gif(path, context, prediction, target)
            failure_gifs.append(str(path.resolve()))
    result = {
        "format": "track2-open-loop-eval-v1",
        "split": args.split,
        "episodes": episodes,
        "sample_count": len(selected),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_model_sha256": artifact_sha256,
        "backend": args.backend,
        "wan_inference": (
            {"steps": args.wan_inference_steps, "solver": args.wan_inference_solver}
            if args.backend == "track2-wan"
            else None
        ),
        "ensemble": ensemble_metadata,
        "model_mae_by_prediction_frame": [float(value) for value in model_by_frame],
        "model_mae_mean": float(model_by_frame.mean()),
        "copy_last_mae_by_prediction_frame": [float(value) for value in copy_by_frame],
        "copy_last_mae_mean": float(copy_by_frame.mean()),
        "high_motion_threshold": args.high_motion_threshold,
        "high_motion_sample_count": int(high_motion.sum()),
        "high_motion_model_mae_by_prediction_frame": (
            [float(value) for value in high_model_by_frame] if high_model_by_frame is not None else None
        ),
        "high_motion_model_mae_mean": float(high_model_by_frame.mean()) if high_model_by_frame is not None else None,
        "high_motion_copy_last_mae_by_prediction_frame": (
            [float(value) for value in high_copy_by_frame] if high_copy_by_frame is not None else None
        ),
        "high_motion_copy_last_mae_mean": float(high_copy_by_frame.mean()) if high_copy_by_frame is not None else None,
        "action_perturbation_mean_absolute_pixel_delta": (
            float(np.mean(action_sensitivity)) if args.measure_action_sensitivity else None
        ),
        "hard_acceptance": acceptance,
        "worst_validation_windows": worst_windows,
        "worst_window_by_prediction_frame": worst_by_prediction_frame,
        "failure_gifs": failure_gifs,
        "window_acceptance": window_acceptance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.prediction_cache:
        cache = Path(args.prediction_cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                prediction=(resume_cache.predictions if resume_cache else np.stack(predictions)).astype(np.uint8, copy=False),
                windows=np.asarray([path.name for path in selected]),
            )
        os.replace(temporary, cache)
    print(json.dumps(result, indent=2))
    if args.require_pass and not passed:
        raise SystemExit("strict Track 2 acceptance failed; see the evaluation JSON for every failing window")


if __name__ == "__main__":
    main()
