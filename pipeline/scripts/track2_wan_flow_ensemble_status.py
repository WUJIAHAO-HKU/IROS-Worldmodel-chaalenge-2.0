#!/usr/bin/env python3
"""Report the durable state of a resumable Wan/direct-flow evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def motion_score(context: np.ndarray, target: np.ndarray) -> float:
    sequence = np.concatenate((context[-1:], target), axis=0).astype(np.float32)
    return float(np.abs(sequence[1:] - sequence[:-1]).mean() / 255.0)


def partial_metrics(cache_dir: Path, windows_dir: Path, windows: list[str], count: int, high_motion_threshold: float) -> dict:
    """Read only the durable prefix; never present it as a full validation score."""
    from wam_pipeline.data import load_window_npz

    predictions_path = cache_dir / "predictions.npy"
    if not predictions_path.is_file():
        raise ValueError(f"resumable evaluation cache is missing predictions: {predictions_path}")
    predictions = np.load(predictions_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(windows), 8, 256, 256, 3)
    if predictions.shape != expected_shape or predictions.dtype != np.uint8:
        raise ValueError("resumable evaluation predictions have an invalid shape or dtype")
    if count == 0:
        return {
            "prefix_window_count": 0,
            "partial_only": True,
            "model_mae_mean": None,
            "high_motion_model_mae_mean": None,
        }
    maes: list[np.ndarray] = []
    high_motion: list[bool] = []
    for index, name in enumerate(windows[:count]):
        window = load_window_npz(windows_dir / name)
        target = window.target_frames.astype(np.float32)
        maes.append(np.abs(np.asarray(predictions[index], dtype=np.float32) - target).mean(axis=(1, 2, 3)))
        high_motion.append(motion_score(window.context_frames, window.target_frames) >= high_motion_threshold)
    values = np.asarray(maes, dtype=np.float32)
    high_values = values[np.asarray(high_motion, dtype=np.bool_)]
    return {
        "prefix_window_count": count,
        "partial_only": True,
        "model_mae_by_prediction_frame": [float(value) for value in values.mean(axis=0)],
        "model_mae_mean": float(values.mean()),
        "worst_window_prediction_frame_mae": float(values.max()),
        "high_motion_threshold": high_motion_threshold,
        "high_motion_prefix_window_count": int(len(high_values)),
        "high_motion_model_mae_mean": float(high_values.mean()) if len(high_values) else None,
    }


def status(
    cache_dir: str | Path,
    progress_path: str | Path | None,
    evaluation_path: str | Path | None,
    windows_dir: str | Path | None = None,
    high_motion_threshold: float = 0.04,
) -> dict:
    root = Path(cache_dir).resolve()
    manifest_path = root / "manifest.json"
    completed_path = root / "completed.npy"
    if not manifest_path.is_file() or not completed_path.is_file():
        raise ValueError(f"resumable evaluation cache is missing manifest or completion array: {root}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid resumable evaluation manifest: {manifest_path}") from exc
    windows = manifest.get("selected_windows")
    if manifest.get("format") != "track2-resumable-prediction-cache-v1" or not isinstance(windows, list):
        raise ValueError("unsupported resumable evaluation cache manifest")
    completed = np.load(completed_path, mmap_mode="r", allow_pickle=False)
    if completed.dtype != np.bool_ or completed.shape != (len(windows),):
        raise ValueError("resumable evaluation completion array has an invalid shape or dtype")
    count = int(np.count_nonzero(completed))
    # This evaluator writes sequentially. A gap would indicate a damaged cache
    # and must not be presented as a safely resumable prefix.
    if not bool(np.all(completed[:count])) or bool(np.any(completed[count:])):
        raise ValueError("resumable evaluation cache completion markers are not a contiguous prefix")
    result = {
        "format": "track2-wan-flow-ensemble-status-v1",
        "backend": manifest.get("backend"),
        "checkpoint_dir": manifest.get("checkpoint_dir"),
        "checkpoint_model_sha256": manifest.get("checkpoint_model_sha256"),
        "completed_windows": count,
        "total_windows": len(windows),
        "remaining_windows": len(windows) - count,
        "last_completed_window": windows[count - 1] if count else None,
        "next_window": windows[count] if count < len(windows) else None,
        "complete": count == len(windows),
    }
    if progress_path and Path(progress_path).is_file():
        try:
            progress = json.loads(Path(progress_path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evaluation progress JSON: {progress_path}") from exc
        if progress.get("checkpoint_model_sha256") != manifest.get("checkpoint_model_sha256"):
            raise ValueError("evaluation progress JSON belongs to a different checkpoint")
        result["latest_progress"] = progress
    if evaluation_path and Path(evaluation_path).is_file():
        try:
            evaluation = json.loads(Path(evaluation_path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid final evaluation JSON: {evaluation_path}") from exc
        if evaluation.get("checkpoint_model_sha256") != manifest.get("checkpoint_model_sha256"):
            raise ValueError("final evaluation JSON belongs to a different checkpoint")
        result["final_evaluation_written"] = True
        result["strict_all_windows_and_frames_pass"] = evaluation.get("hard_acceptance", {}).get(
            "all_windows_and_frames_pass"
        )
    else:
        result["final_evaluation_written"] = False
    if windows_dir:
        result["durable_prefix_metrics"] = partial_metrics(
            root, Path(windows_dir), windows, count, high_motion_threshold
        )
    return result


def format_markdown(result: dict) -> str:
    """Render a compact, copyable status for operators without GPU access."""
    lines = [
        "# Wan + Direct Flow Full Validation",
        "",
        f"- Progress: `{result['completed_windows']}/{result['total_windows']}`",
        f"- Remaining: `{result['remaining_windows']}`",
        f"- Next window: `{result['next_window'] or 'none'}`",
        f"- Final evaluation written: `{result['final_evaluation_written']}`",
    ]
    metrics = result.get("durable_prefix_metrics")
    if isinstance(metrics, dict):
        lines.extend(
            [
                "",
                "## Durable Prefix (Not Final)",
                "",
                f"- RGB MAE: `{metrics['model_mae_mean']}`",
                f"- High-motion RGB MAE: `{metrics['high_motion_model_mae_mean']}`",
                f"- Worst prediction-frame MAE: `{metrics['worst_window_prediction_frame_mae']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a resumable Wan/direct-flow evaluation status without using GPU.")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_cache",
    )
    parser.add_argument(
        "--progress",
        default="artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_progress.json",
    )
    parser.add_argument(
        "--evaluation",
        default="artifacts/evaluations/track2_wan45_direct_flow_ensemble_v2_validation_all.json",
    )
    parser.add_argument("--windows", default="artifacts/adjust_bottle_windows_full")
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--markdown-output", help="Optional compact Markdown status written atomically.")
    args = parser.parse_args()
    try:
        if args.high_motion_threshold < 0:
            raise ValueError("--high-motion-threshold must be non-negative")
        result = status(
            args.cache_dir,
            args.progress,
            args.evaluation,
            args.windows,
            args.high_motion_threshold,
        )
        if args.markdown_output:
            output = Path(args.markdown_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(format_markdown(result))
            temporary.replace(output)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read Wan/direct-flow evaluation status: {exc}") from exc


if __name__ == "__main__":
    main()
