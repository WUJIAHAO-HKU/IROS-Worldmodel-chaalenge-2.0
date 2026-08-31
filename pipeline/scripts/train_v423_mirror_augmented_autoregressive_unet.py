#!/usr/bin/env python3
"""Train a right-arm world model with on-the-fly physical mirror augmentation.

Only declared public training episodes are augmented.  Public validation stays
real, episode-disjoint, and right-arm-only.  No outcome or reward field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from wam_pipeline.autoregressive_unet import OneStepActionUNet


MIRROR_SIGN = np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)


def mirror_joint14(values: np.ndarray) -> np.ndarray:
    """Apply the audited Aloha left/right reflection to the final dimension."""
    if values.shape[-1] != 14:
        raise ValueError(f"expected joint14 values, got {values.shape}")
    output = np.empty_like(values, dtype=np.float32)
    output[..., :7] = values[..., 7:14] * MIRROR_SIGN
    output[..., 7:14] = values[..., :7] * MIRROR_SIGN
    return output


class WindowDataset(Dataset):
    """Trajectory windows with an optional longer, contiguous rollout target.

    Source files contain eight future actions and frames.  For a horizon above
    eight, this view concatenates subsequent windows from *the same public
    episode*.  This keeps the original data immutable and, importantly, does
    not inspect any outcome field embedded in a source window.
    """

    def __init__(self, directory: Path, episodes: list[int], rollout_horizon: int = 8) -> None:
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be positive")
        allowed = set(episodes)
        all_paths = [path for path in sorted(directory.glob("episode*_*.npz")) if int(path.name.split("_")[0][7:]) in allowed]
        if not all_paths:
            raise ValueError("no windows selected")
        self.rollout_horizon = rollout_horizon
        chunks = (rollout_horizon + 7) // 8

        def key(path: Path) -> tuple[int, int]:
            episode_text, start_text = path.stem.split("_")
            return int(episode_text[7:]), int(start_text)

        by_key = {key(path): path for path in all_paths}
        groups = []
        for path in all_paths:
            episode, start = key(path)
            group = tuple(by_key.get((episode, start + 8 * offset)) for offset in range(chunks))
            if all(item is not None for item in group):
                groups.append(group)
        self.source_groups: list[tuple[Path, ...]] = [tuple(item for item in group if item is not None) for group in groups]
        self.paths = [group[0] for group in self.source_groups]
        if not self.paths:
            raise ValueError(f"no contiguous windows selected for horizon {rollout_horizon}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        arrays = self.load_arrays(index)
        return (
            torch.from_numpy(arrays["context_frames"]),
            torch.from_numpy(arrays["history_actions"]),
            torch.from_numpy(arrays["future_actions"]),
            torch.from_numpy(arrays["target_frames"]),
        )

    def load_arrays(self, index: int) -> dict[str, np.ndarray]:
        """Load one sample without reading labels or outcome metadata."""
        sources = self.source_groups[index]
        with np.load(sources[0], allow_pickle=False) as data:
            result = {
                "context_frames": data["context_frames"].copy(),
                "history_actions": data["history_actions"].copy(),
            }
        future, targets = [], []
        for source in sources:
            with np.load(source, allow_pickle=False) as data:
                future.append(data["future_actions"].copy())
                targets.append(data["target_frames"].copy())
        result["future_actions"] = np.concatenate(future, axis=0)[: self.rollout_horizon]
        result["target_frames"] = np.concatenate(targets, axis=0)[: self.rollout_horizon]
        return result

    def load_actions(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load only the action arrays required for arm-motion sampling.

        Keeping this separate from :meth:`load_arrays` avoids decompressing
        RGB targets while computing a statistic that depends exclusively on
        public joint14 action data.
        """
        sources = self.source_groups[index]
        future = []
        history = None
        for source_index, source in enumerate(sources):
            with np.load(source, allow_pickle=False) as data:
                if source_index == 0:
                    history = data["history_actions"].copy()
                future.append(data["future_actions"].copy())
        if history is None:
            raise RuntimeError("window has no action source")
        return history, np.concatenate(future, axis=0)[: self.rollout_horizon]


class MirrorAugmentedRightDataset(Dataset):
    """Real-right windows plus left windows reflected into the right domain."""

    def __init__(self, base: WindowDataset, arm_by_episode: dict[int, str]) -> None:
        self.base = base
        self.rollout_horizon = base.rollout_horizon
        samples: list[tuple[int, bool]] = []
        for index, path in enumerate(base.paths):
            episode = int(path.name.split("_")[0][7:])
            arm = arm_by_episode.get(episode)
            if arm not in {"left", "right"}:
                raise ValueError(f"episode {episode} has no valid public arm label")
            samples.append((index, arm == "left"))
        self.samples = samples
        self.paths = [base.paths[index] for index, _ in samples]
        self.source_groups = [base.source_groups[index] for index, _ in samples]
        self.mirror_flags = np.asarray([mirror for _, mirror in samples], dtype=np.bool_)
        self.real_right_window_count = int((~self.mirror_flags).sum())
        self.mirrored_left_window_count = int(self.mirror_flags.sum())
        if not self.real_right_window_count or not self.mirrored_left_window_count:
            raise ValueError("mirror-augmented training requires both real-right and public-left windows")

    def __len__(self) -> int:
        return len(self.samples)

    def load_arrays(self, index: int) -> dict[str, np.ndarray]:
        source_index, mirrored = self.samples[index]
        arrays = self.base.load_arrays(source_index)
        if not mirrored:
            return arrays
        arrays["context_frames"] = np.flip(arrays["context_frames"], axis=2).copy()
        arrays["target_frames"] = np.flip(arrays["target_frames"], axis=2).copy()
        arrays["history_actions"] = mirror_joint14(arrays["history_actions"])
        arrays["future_actions"] = mirror_joint14(arrays["future_actions"])
        return arrays

    def load_actions(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        source_index, mirrored = self.samples[index]
        history, future = self.base.load_actions(source_index)
        if mirrored:
            history, future = mirror_joint14(history), mirror_joint14(future)
        return history, future

    def __getitem__(self, index: int):
        arrays = self.load_arrays(index)
        return (
            torch.from_numpy(arrays["context_frames"]),
            torch.from_numpy(arrays["history_actions"]),
            torch.from_numpy(arrays["future_actions"]),
            torch.from_numpy(arrays["target_frames"]),
        )


def frames_for_model(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 1, 4, 2, 3).float().div(255.0)


def action_statistics(dataset: WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    total = np.zeros(14, dtype=np.float64)
    squared = np.zeros(14, dtype=np.float64)
    count = 0
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            actions = np.concatenate([data["history_actions"], data["future_actions"]]).astype(np.float64)
        total += actions.sum(0)
        squared += np.square(actions).sum(0)
        count += len(actions)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - np.square(mean), 1e-8))
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def window_motion_scores(dataset: WindowDataset) -> np.ndarray:
    """Return mean true inter-frame motion over each selected rollout target."""
    scores = []
    for index in range(len(dataset)):
        data = dataset.load_arrays(index)
        target = data["target_frames"].astype(np.float32) / 255.0
        previous = np.concatenate(
            [data["context_frames"][-1:].astype(np.float32) / 255.0, target[:-1]], axis=0
        )
        scores.append(float(np.abs(target - previous).mean()))
    return np.asarray(scores, dtype=np.float64)


def cached_training_statistics(
    dataset: WindowDataset,
    cache_path: Path | None,
    windows: Path,
    split_manifest: Path,
    require_arm_metadata: bool,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray | None]:
    augmentation = getattr(dataset, "mirror_flags", np.empty(0, dtype=np.bool_))
    identity = hashlib.sha256(
        f"horizon={dataset.rollout_horizon}\n".encode()
        + "\n".join("|".join(path.name for path in group) for group in dataset.source_groups).encode()
        + np.asarray(augmentation, dtype=np.bool_).tobytes()
    ).hexdigest()
    arm_by_path: dict[str, bool] = {}
    arm_source_sha256 = "embedded"
    source_manifest_path = windows / "window_sources.json"
    # The action-only mode must be a hard data boundary: do not even read
    # simulator arm labels or their sidecar manifest.  They are unnecessary
    # because selection and sampling both derive the active arm from joint14.
    if require_arm_metadata and source_manifest_path.is_file():
        source_bytes = source_manifest_path.read_bytes()
        records = json.loads(source_bytes)
        arm_by_path = {
            str(record["path"]): str(record["arm"]) == "right" for record in records
        }
        arm_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            if (
                str(data["format"]) == "track2-autoregressive-training-statistics-v2"
                and str(data["windows"]) == str(windows.resolve())
                and str(data["split_manifest"]) == str(split_manifest.resolve())
                and str(data["window_identity_sha256"]) == identity
                and int(data["window_count"]) == len(dataset)
                and (
                    not require_arm_metadata
                    or str(data["arm_source_sha256"]) == arm_source_sha256
                )
            ):
                mean = np.asarray(data["mean"], dtype=np.float32)
                std = np.asarray(data["std"], dtype=np.float32)
                scores = np.asarray(data["motion_scores"], dtype=np.float64)
                arm_right = (
                    np.asarray(data["arm_right"], dtype=np.bool_)
                    if require_arm_metadata
                    else None
                )
                if (
                    mean.shape == (14,)
                    and std.shape == (14,)
                    and scores.shape == (len(dataset),)
                    and (arm_right is None or arm_right.shape == (len(dataset),))
                ):
                    return torch.from_numpy(mean), torch.from_numpy(std), scores, arm_right
        raise ValueError(f"training statistics cache does not match this dataset: {cache_path}")
    total = np.zeros(14, dtype=np.float64)
    squared = np.zeros(14, dtype=np.float64)
    count = 0
    scores = []
    arm_flags = []
    # All sampling statistics come from the same compressed-NPZ pass.  The
    # previous implementation opened every multi-megabyte window three times,
    # which made large joint-domain experiments spend minutes on redundant I/O.
    for index, path in enumerate(dataset.paths):
        data = dataset.load_arrays(index)
        actions = np.concatenate([data["history_actions"], data["future_actions"]]).astype(np.float64)
        target = data["target_frames"].astype(np.float32) / 255.0
        previous = np.concatenate(
            [data["context_frames"][-1:].astype(np.float32) / 255.0, target[:-1]],
            axis=0,
        )
        total += actions.sum(0)
        squared += np.square(actions).sum(0)
        count += len(actions)
        scores.append(float(np.abs(target - previous).mean()))
        if require_arm_metadata:
            with np.load(path, allow_pickle=False) as metadata:
                if "arm_right" in metadata.files:
                    arm_value = np.asarray(metadata["arm_right"])
                elif path.name in arm_by_path:
                    arm_value = np.asarray(arm_by_path[path.name])
                else:
                    raise ValueError(
                        f"{path}: arm_right is absent and window_sources.json has no arm label"
                    )
            if arm_value.shape != ():
                raise ValueError(f"{path}: arm_right must be scalar")
            arm_flags.append(bool(arm_value))
    mean_array = total / count
    std_array = np.sqrt(np.maximum(squared / count - np.square(mean_array), 1e-8))
    mean = torch.from_numpy(mean_array.astype(np.float32))
    std = torch.from_numpy(std_array.astype(np.float32))
    scores = np.asarray(scores, dtype=np.float64)
    arm_right = np.asarray(arm_flags, dtype=np.bool_) if require_arm_metadata else None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                format=np.asarray("track2-autoregressive-training-statistics-v2"),
                windows=np.asarray(str(windows.resolve())),
                split_manifest=np.asarray(str(split_manifest.resolve())),
                window_identity_sha256=np.asarray(identity),
                window_count=np.asarray(len(dataset)),
                arm_source_sha256=np.asarray(arm_source_sha256),
                mean=mean.numpy(),
                std=std.numpy(),
                motion_scores=scores,
                arm_right=arm_right if arm_right is not None else np.empty(0, dtype=np.bool_),
            )
        temporary.replace(cache_path)
    return mean, std, scores, arm_right


def high_motion_sampling(
    scores: np.ndarray, threshold: float, oversample_factor: float
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Oversample windows with sustained future motion without touching held-out data."""
    high_motion = scores >= threshold
    weights = np.ones(len(scores), dtype=np.float64)
    weights[high_motion] = oversample_factor
    expected_fraction = float(weights[high_motion].sum() / weights.sum()) if high_motion.any() else 0.0
    return torch.from_numpy(weights), {
        "threshold": float(threshold),
        "oversample_factor": float(oversample_factor),
        "window_count": int(len(scores)),
        "high_motion_window_count": int(high_motion.sum()),
        "high_motion_fraction": float(high_motion.mean()),
        "expected_high_motion_fraction": expected_fraction,
        "motion_mean": float(scores.mean()),
        "motion_p50": float(np.quantile(scores, 0.50)),
        "motion_p90": float(np.quantile(scores, 0.90)),
    }


def window_arm_flags(dataset: WindowDataset) -> np.ndarray:
    """Read the active arm recorded by the simulator for every train window."""
    flags = []
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as data:
            if "arm_right" not in data.files:
                raise ValueError(f"{path}: arm_right is required for arm-aware sampling")
            value = np.asarray(data["arm_right"])
            if value.shape != ():
                raise ValueError(f"{path}: arm_right must be scalar")
            flags.append(bool(value))
    return np.asarray(flags, dtype=np.bool_)


def window_action_arm_motion(
    dataset: WindowDataset, action_std: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Measure normalized joint motion in the official left/right joint14 blocks.

    Grippers are deliberately excluded: their binary range can dominate a
    short window even when the corresponding six-joint arm is stationary.
    The scale comes from training actions only and no task outcome is read.
    """
    scale = np.asarray(action_std.cpu(), dtype=np.float64)
    if scale.shape != (14,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("action std must contain 14 finite positive values")
    left_motion, right_motion = [], []
    for index, path in enumerate(dataset.paths):
        data = dataset.load_arrays(index)
        actions = np.concatenate(
            [data["history_actions"], data["future_actions"]]
        ).astype(np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"{path}: expected [time, 14] joint actions, got {actions.shape}")
        normalized_delta = np.abs(np.diff(actions, axis=0)) / scale
        left_motion.append(float(normalized_delta[:, :6].mean()))
        right_motion.append(float(normalized_delta[:, 7:13].mean()))
    return np.asarray(left_motion), np.asarray(right_motion)


def apply_arm_sampling(
    weights: torch.Tensor,
    arm_right: np.ndarray | None,
    right_oversample_factor: float,
    action_left_motion: np.ndarray | None = None,
    action_right_motion: np.ndarray | None = None,
    source: str = "metadata",
    dominance_margin: float = 1.10,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Multiply verified right-arm weights without using task outcomes.

    ``metadata`` preserves the historical behavior. ``action`` trusts only
    normalized joint motion. ``consensus`` is the safe v171 mode: a window is
    boosted only when the simulator label and joint14 motion both say right.
    Mismatched and ambiguous windows remain available at their original weight.
    """
    if arm_right is not None and arm_right.shape != (len(weights),):
        raise ValueError("arm flags and sampling weights must have equal length")
    if source not in {"metadata", "action", "consensus"}:
        raise ValueError("arm sampling source must be metadata, action, or consensus")
    if dominance_margin < 1.0:
        raise ValueError("arm dominance margin must be at least 1")
    if action_left_motion is None or action_right_motion is None:
        if source != "metadata":
            raise ValueError("action motion is required for action/consensus sampling")
        action_left = np.zeros(len(weights), dtype=np.bool_)
        action_right = np.zeros(len(weights), dtype=np.bool_)
    else:
        if action_left_motion.shape != (len(weights),) or action_right_motion.shape != (len(weights),):
            raise ValueError("action motion and sampling weights must have equal length")
        action_left = action_left_motion > action_right_motion * dominance_margin
        action_right = action_right_motion > action_left_motion * dominance_margin

    if source != "action" and arm_right is None:
        raise ValueError("arm metadata is required for metadata/consensus sampling")
    if source == "metadata":
        assert arm_right is not None
        selected_right = arm_right
    elif source == "action":
        selected_right = action_right
    else:
        assert arm_right is not None
        selected_right = arm_right & action_right
    adjusted = weights.clone().to(torch.float64)
    right = torch.from_numpy(selected_right)
    adjusted[right] *= right_oversample_factor
    total = float(adjusted.sum())
    report: dict[str, float | int | str] = {
        "source": source,
        "dominance_margin": float(dominance_margin),
        "right_oversample_factor": float(right_oversample_factor),
        "boosted_right_window_count": int(selected_right.sum()),
        "expected_right_fraction": float(adjusted[right].sum() / total) if total else 0.0,
        "metadata_right_available": arm_right is not None,
    }
    if arm_right is not None:
        report.update({
            "left_window_count": int((~arm_right).sum()),
            "right_window_count": int(arm_right.sum()),
        })
    if action_left_motion is not None and action_right_motion is not None:
        decisive = action_left | action_right
        report.update(
            {
                "action_dominant_left_count": int(action_left.sum()),
                "action_dominant_right_count": int(action_right.sum()),
                "action_ambiguous_count": int((~decisive).sum()),
            }
        )
        if arm_right is not None:
            agreement = ((~arm_right & action_left) | (arm_right & action_right))
            report.update(
                {
                    "metadata_action_agreement_all": float(agreement.mean()),
                    "metadata_action_agreement_decisive": float(agreement[decisive].mean()) if decisive.any() else 0.0,
                    "metadata_right_action_right_fraction": float(action_right[arm_right].mean()) if arm_right.any() else 0.0,
                }
            )
    return adjusted, report


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor | None = None,
    motion_weight: float = 1.0,
    motion_threshold: float = 0.03,
    horizon_loss_power: float = 0.0,
    temporal_delta_weight: float = 0.0,
    texture_laplacian_weight: float = 0.0,
    temporal_delta_pool: int = 1,
    lowfreq_anchor_weight: float = 0.0,
) -> torch.Tensor:
    """Weight motion, later horizons, temporal deltas, and local texture."""
    if min(horizon_loss_power, temporal_delta_weight, texture_laplacian_weight, lowfreq_anchor_weight) < 0.0:
        raise ValueError("horizon, temporal-delta, and texture weights must be non-negative")
    if temporal_delta_pool < 1:
        raise ValueError("temporal-delta pooling must be positive")
    horizon_scale: torch.Tensor | float = 1.0
    flat_horizon_scale: torch.Tensor | float = 1.0
    if prediction.ndim == 5 and horizon_loss_power > 0.0:
        horizon_scale = torch.arange(
            1, prediction.shape[1] + 1, device=prediction.device, dtype=prediction.dtype
        ).pow(horizon_loss_power)
        horizon_scale = (horizon_scale / horizon_scale.mean()).view(1, -1, 1, 1, 1)
        flat_horizon_scale = horizon_scale.expand(prediction.shape[0], -1, -1, -1, -1).reshape(
            -1, 1, 1, 1
        )
    weights: torch.Tensor | float = 1.0
    if motion_weight > 1.0:
        if previous is None:
            raise ValueError("previous frames are required for motion-weighted loss")
        changed = (target - previous).abs().mean(dim=-3, keepdim=True) >= motion_threshold
        weights = 1.0 + (motion_weight - 1.0) * changed.to(target.dtype)
        pixel = (horizon_scale * weights * (prediction - target).abs()).mean() + 0.05 * (
            horizon_scale * weights * (prediction - target).square()
        ).mean()
    else:
        if horizon_loss_power > 0.0 and prediction.ndim == 5:
            pixel = (horizon_scale * (prediction - target).abs()).mean() + 0.05 * (
                horizon_scale * (prediction - target).square()
            ).mean()
        else:
            pixel = functional.l1_loss(prediction, target) + 0.05 * functional.mse_loss(prediction, target)

    temporal_delta = prediction.new_zeros(())
    lowfreq_anchor = prediction.new_zeros(())
    if temporal_delta_weight > 0.0:
        if previous is None:
            raise ValueError("previous frames are required for temporal-delta loss")
        if prediction.ndim == 5:
            predicted_previous = torch.cat([previous[:, :1], prediction[:, :-1]], dim=1)
        else:
            predicted_previous = previous
        delta_difference = (prediction - predicted_previous) - (target - previous)
        delta_weights = weights
        if temporal_delta_pool > 1:
            original_shape = delta_difference.shape
            if delta_difference.ndim == 5:
                delta_difference = delta_difference.flatten(0, 1)
                if isinstance(delta_weights, torch.Tensor):
                    delta_weights = delta_weights.flatten(0, 1)
            delta_difference = functional.avg_pool2d(
                delta_difference, kernel_size=temporal_delta_pool, stride=temporal_delta_pool
            )
            if isinstance(delta_weights, torch.Tensor):
                delta_weights = functional.avg_pool2d(
                    delta_weights, kernel_size=temporal_delta_pool, stride=temporal_delta_pool
                )
            if len(original_shape) == 5:
                delta_difference = delta_difference.unflatten(0, original_shape[:2])
                if isinstance(delta_weights, torch.Tensor):
                    delta_weights = delta_weights.unflatten(0, original_shape[:2])
        temporal_delta = (horizon_scale * delta_weights * delta_difference.abs()).mean()

    if lowfreq_anchor_weight > 0.0:
        if previous is None:
            raise ValueError("previous frames are required for low-frequency anchor loss")
        if prediction.ndim == 5:
            predicted_previous = torch.cat([previous[:, :1], prediction[:, :-1]], dim=1)
        else:
            predicted_previous = previous
        pred_flat = prediction.flatten(0, 1) if prediction.ndim == 5 else prediction
        pred_prev_flat = predicted_previous.flatten(0, 1) if predicted_previous.ndim == 5 else predicted_previous
        target_flat = target.flatten(0, 1) if target.ndim == 5 else target
        target_prev_flat = previous.flatten(0, 1) if previous.ndim == 5 else previous
        pred_low = functional.avg_pool2d(pred_flat, 5, 1, 2)
        pred_prev_low = functional.avg_pool2d(pred_prev_flat, 5, 1, 2)
        target_low = functional.avg_pool2d(target_flat, 5, 1, 2)
        target_prev_low = functional.avg_pool2d(target_prev_flat, 5, 1, 2)
        lowfreq_scale = flat_horizon_scale if prediction.ndim == 5 else 1.0
        lowfreq_anchor = (
            lowfreq_scale
            * ((pred_low - pred_prev_low) - (target_low - target_prev_low)).abs()
        ).mean()

    flat_weights: torch.Tensor | float = weights
    if prediction.ndim == 5:
        prediction = prediction.flatten(0, 1)
        target = target.flatten(0, 1)
        if isinstance(flat_weights, torch.Tensor):
            flat_weights = flat_weights.flatten(0, 1)
    coarse = (
        flat_horizon_scale
        * (functional.avg_pool2d(prediction, kernel_size=2) - functional.avg_pool2d(target, kernel_size=2)).abs()
    ).mean()
    edge_x = (
        flat_horizon_scale
        * ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    ).mean()
    edge_y = (
        flat_horizon_scale
        * ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    ).mean()
    texture_laplacian = prediction.new_zeros(())
    if texture_laplacian_weight > 0.0:
        prediction_highpass = prediction - functional.avg_pool2d(
            prediction, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        target_highpass = target - functional.avg_pool2d(
            target, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        texture_laplacian = (
            flat_horizon_scale * flat_weights * (prediction_highpass - target_highpass).abs()
        ).mean()
    return (
        pixel
        + 0.2 * coarse
        + 0.2 * (edge_x + edge_y) / 2.0
        + temporal_delta_weight * temporal_delta
        + lowfreq_anchor_weight * lowfreq_anchor
        + texture_laplacian_weight * texture_laplacian
    )


def rollout(model, context, history, future) -> torch.Tensor:
    """Generate eight frames using only previous predictions after the first step."""
    predictions = []
    for action in future.unbind(dim=1):
        next_frame = model(context, torch.cat([history, action[:, None]], dim=1)).clamp(0.0, 1.0)
        predictions.append(next_frame)
        context = torch.cat([context[:, 1:], next_frame[:, None]], dim=1)
        history = torch.cat([history[:, 1:], action[:, None]], dim=1)
    return torch.stack(predictions, dim=1)


def evaluate(loader, model, device, mean, std, high_motion_threshold: float) -> dict[str, float | int | list[float] | None]:
    model.eval()
    errors = []
    copy_errors = []
    motion_scores = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for context, history, future, target in loader:
            context = frames_for_model(context).to(device, non_blocking=True)
            target = frames_for_model(target).to(device, non_blocking=True)
            history = ((history.to(device, non_blocking=True) - mean) / std).float()
            future = ((future.to(device, non_blocking=True) - mean) / std).float()
            prediction = rollout(model, context, history, future)
            errors.append((prediction.float() - target.float()).abs().mean(dim=(2, 3, 4)).cpu())
            copy_errors.append((context[:, -1:].float() - target.float()).abs().mean(dim=(2, 3, 4)).cpu())
            previous = torch.cat([context[:, -1:], target[:, :-1]], dim=1)
            motion_scores.append((target.float() - previous.float()).abs().mean(dim=(1, 2, 3, 4)).cpu())
    error = torch.cat(errors)
    copy_error = torch.cat(copy_errors)
    motion = torch.cat(motion_scores)
    high_motion = motion >= high_motion_threshold

    def aggregate(mask: torch.Tensor) -> dict[str, float | list[float] | None]:
        if not bool(mask.any()):
            return {"mae": None, "first_mae": None, "copy_mae": None, "mae_by_prediction_frame": None}
        selected_error = error[mask]
        selected_copy = copy_error[mask]
        return {
            "mae": float(selected_error.mean()),
            "first_mae": float(selected_error[:, 0].mean()),
            "copy_mae": float(selected_copy.mean()),
            "mae_by_prediction_frame": [float(value) for value in selected_error.mean(dim=0)],
        }

    overall = aggregate(torch.ones(len(error), dtype=torch.bool))
    high = aggregate(high_motion)
    return {
        "rollout_mae": overall["mae"],
        "first_frame_mae": overall["first_mae"],
        "copy_last_mae": overall["copy_mae"],
        "mae_by_prediction_frame": overall["mae_by_prediction_frame"],
        "high_motion_threshold": float(high_motion_threshold),
        "high_motion_sample_count": int(high_motion.sum()),
        "high_motion_rollout_mae": high["mae"],
        "high_motion_first_frame_mae": high["first_mae"],
        "high_motion_copy_last_mae": high["copy_mae"],
        "high_motion_mae_by_prediction_frame": high["mae_by_prediction_frame"],
    }


def selection_metric(
    result: dict,
    high_motion_weight: float,
    right_result: dict | None = None,
    right_weight: float = 0.0,
    right_high_motion_weight: float = 0.0,
) -> float:
    value = float(result["rollout_mae"])
    high_mae = result["high_motion_rollout_mae"]
    if high_mae is not None:
        value = (1.0 - high_motion_weight) * value + high_motion_weight * float(high_mae)
    if right_result is not None:
        value = (1.0 - right_weight) * value + right_weight * float(right_result["rollout_mae"])
        right_high_mae = right_result["high_motion_rollout_mae"]
        if right_high_mae is not None:
            value = (
                (1.0 - right_high_motion_weight) * value
                + right_high_motion_weight * float(right_high_mae)
            )
    return value


def load_visual_initialization(model: OneStepActionUNet, checkpoint: Path) -> list[str]:
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-residual-unet-v1":
        raise SystemExit("--init-checkpoint must use track2-residual-unet-v1")
    source = state["state_dict"]
    target = model.state_dict()
    copied = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    model.load_state_dict(copied, strict=False)
    return sorted(copied)


def load_autoregressive_initialization(model: OneStepActionUNet, checkpoint: Path) -> list[str]:
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    if state.get("format") != "track2-autoregressive-unet-v1":
        raise SystemExit("--init-autoregressive-checkpoint must use track2-autoregressive-unet-v1")
    model.load_state_dict(state["state_dict"], strict=True)
    return sorted(state["state_dict"])


def save_checkpoint(output: Path, model, mean, std, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "track2-autoregressive-unet-v1", "state_dict": model.state_dict()}, output / "model.pt")
    np.savez(output / "action_normalization.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    np.savez(output / "track2_autoregressive_unet_config.npz", context_frames=np.asarray(5), action_dim=np.asarray(14), prediction_frames=np.asarray(8), working_resolution=np.asarray(256), serving_resolution=np.asarray(256))
    (output / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def save_training_state(
    output: Path,
    model,
    optimizer,
    step: int,
    target_steps: int,
    training_config: dict,
    history: list[dict],
    best_selection_metric: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "training_state.pt.tmp"
    torch.save(
        {
            "format": "track2-autoregressive-training-state-v1",
            "step": step,
            "target_steps": target_steps,
            "training_config": training_config,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_selection_metric": best_selection_metric,
        },
        temporary,
    )
    temporary.replace(output / "training_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--init-autoregressive-checkpoint")
    parser.add_argument("--normalization-checkpoint")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--train-rollout-steps", type=int, default=1)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--horizon-loss-power", type=float, default=0.0)
    parser.add_argument("--temporal-delta-weight", type=float, default=0.0)
    parser.add_argument("--texture-laplacian-weight", type=float, default=0.0)
    parser.add_argument("--temporal-delta-pool", type=int, default=1)
    parser.add_argument("--lowfreq-anchor-weight", type=float, default=0.0)
    parser.add_argument("--high-motion-threshold", type=float, default=0.04)
    parser.add_argument("--high-motion-oversample-factor", type=float, default=1.0)
    parser.add_argument("--right-arm-oversample-factor", type=float, default=1.0)
    parser.add_argument(
        "--arm-sampling-source",
        choices=("metadata", "action", "consensus"),
        default="metadata",
        help="v171 should use consensus; metadata preserves historical runs",
    )
    parser.add_argument("--arm-dominance-margin", type=float, default=1.10)
    parser.add_argument("--high-motion-selection-weight", type=float, default=0.0)
    parser.add_argument("--right-action-selection-weight", type=float, default=0.0)
    parser.add_argument("--right-high-motion-selection-weight", type=float, default=0.0)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--early-stop-validations", type=int, default=0)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.0)
    parser.add_argument("--statistics-cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if args.motion_weight < 1.0 or args.motion_threshold < 0.0 or min(
        args.horizon_loss_power, args.temporal_delta_weight, args.texture_laplacian_weight,
        args.lowfreq_anchor_weight
    ) < 0.0:
        raise SystemExit(
            "motion weight must be >= 1; motion threshold, horizon loss power, temporal-delta weight, "
            "and texture-laplacian weight must be >= 0"
        )
    if args.temporal_delta_pool < 1:
        raise SystemExit("temporal-delta pooling must be positive")
    if args.high_motion_threshold < 0.0 or args.high_motion_oversample_factor < 1.0:
        raise SystemExit("high-motion threshold must be >= 0 and oversample factor must be >= 1")
    if args.right_arm_oversample_factor < 1.0:
        raise SystemExit("right-arm oversample factor must be >= 1")
    if args.arm_dominance_margin < 1.0:
        raise SystemExit("arm dominance margin must be >= 1")
    if not all(
        0.0 <= weight <= 1.0
        for weight in (
            args.high_motion_selection_weight,
            args.right_action_selection_weight,
            args.right_high_motion_selection_weight,
        )
    ):
        raise SystemExit("selection weights must be in [0, 1]")
    if args.rollout_horizon < 1:
        raise SystemExit("rollout-horizon must be positive")
    if not 1 <= args.train_rollout_steps <= args.rollout_horizon:
        raise SystemExit("train-rollout-steps must be within [1, rollout-horizon]")
    if min(args.steps, args.batch_size, args.validation_interval, args.validation_batches, args.checkpoint_interval) < 1:
        raise SystemExit("steps, batch size, validation counts, and checkpoint interval must be positive")
    if args.num_workers < 0:
        raise SystemExit("num-workers must be non-negative")
    if args.arm_sampling_source != "action":
        raise SystemExit("mirror-augmented training requires --arm-sampling-source action")
    if args.early_stop_validations < 0 or not 0.0 <= args.minimum_relative_improvement < 1.0:
        raise SystemExit("early-stop-validations must be non-negative and relative improvement in [0, 1)")
    torch.manual_seed(args.seed)
    windows, split_path = Path(args.windows), Path(args.split_manifest)
    split = json.loads(split_path.read_text())
    arm_by_episode = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    train_episodes = [int(value) for value in split["train_episodes"]]
    validation_episodes = [int(value) for value in split["validation_episodes"]]
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("public training and validation episodes must be disjoint")
    validation_right_episodes = [episode for episode in validation_episodes if arm_by_episode[episode] == "right"]
    if not validation_right_episodes:
        raise ValueError("public validation has no real-right episodes")
    train_base = WindowDataset(windows, train_episodes, args.rollout_horizon)
    train = MirrorAugmentedRightDataset(train_base, arm_by_episode)
    validation = WindowDataset(windows, validation_right_episodes, args.rollout_horizon)
    device = torch.device(args.device)
    data_mean, data_std, train_motion_scores, train_arm_right = cached_training_statistics(
        train,
        Path(args.statistics_cache) if args.statistics_cache else None,
        windows,
        split_path,
        require_arm_metadata=args.arm_sampling_source != "action",
    )
    normalization_source = None
    if args.normalization_checkpoint:
        normalization_root = Path(args.normalization_checkpoint).resolve()
        with np.load(normalization_root / "action_normalization.npz", allow_pickle=False) as values:
            mean = torch.from_numpy(np.asarray(values["mean"], dtype=np.float32))
            std = torch.from_numpy(np.asarray(values["std"], dtype=np.float32))
        if mean.shape != (14,) or std.shape != (14,):
            raise ValueError("locked action normalization must have shape [14]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or bool((std <= 0).any()):
            raise ValueError("locked action normalization must be finite with positive std")
        normalization_source = str(normalization_root)
    else:
        mean, std = data_mean, data_std
    mean, std = mean.to(device), std.to(device)
    train_weights, motion_sampling = high_motion_sampling(
        train_motion_scores, args.high_motion_threshold, args.high_motion_oversample_factor
    )
    action_left_motion, action_right_motion = window_action_arm_motion(train, data_std)
    train_weights, arm_sampling = apply_arm_sampling(
        train_weights,
        train_arm_right,
        args.right_arm_oversample_factor,
        action_left_motion,
        action_right_motion,
        args.arm_sampling_source,
        args.arm_dominance_margin,
    )
    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=len(train), replacement=True, generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    sample_count = min(len(validation), args.validation_batches * args.batch_size)
    validation_indices = np.linspace(0, len(validation) - 1, sample_count, dtype=np.int64).tolist()
    validation_loader = DataLoader(Subset(validation, validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    validation_left_motion, validation_right_motion = window_action_arm_motion(validation, data_std)
    validation_action_left = validation_left_motion > validation_right_motion * args.arm_dominance_margin
    validation_action_right = validation_right_motion > validation_left_motion * args.arm_dominance_margin
    left_validation_indices = [index for index in validation_indices if validation_action_left[index]]
    right_validation_indices = [index for index in validation_indices if validation_action_right[index]]
    left_validation_loader = DataLoader(Subset(validation, left_validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    right_validation_loader = DataLoader(Subset(validation, right_validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.init_checkpoint and args.init_autoregressive_checkpoint:
        raise SystemExit("use only one initialization checkpoint")
    model = OneStepActionUNet().to(device)
    if args.init_autoregressive_checkpoint:
        initialized_keys = load_autoregressive_initialization(model, Path(args.init_autoregressive_checkpoint))
    elif args.init_checkpoint:
        initialized_keys = load_visual_initialization(model, Path(args.init_checkpoint))
    else:
        initialized_keys = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output, best_selection_metric, history, start_step = Path(args.output), float("inf"), [], 0
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rollout_horizon": args.rollout_horizon,
        "train_rollout_steps": args.train_rollout_steps,
        "motion_weight": args.motion_weight,
        "motion_threshold": args.motion_threshold,
        "horizon_loss_power": args.horizon_loss_power,
        "temporal_delta_weight": args.temporal_delta_weight,
        "texture_laplacian_weight": args.texture_laplacian_weight,
        "temporal_delta_pool": args.temporal_delta_pool,
        "lowfreq_anchor_weight": args.lowfreq_anchor_weight,
        "high_motion_threshold": args.high_motion_threshold,
        "high_motion_oversample_factor": args.high_motion_oversample_factor,
        "right_arm_oversample_factor": args.right_arm_oversample_factor,
        "arm_sampling_source": args.arm_sampling_source,
        "arm_dominance_margin": args.arm_dominance_margin,
        "high_motion_selection_weight": args.high_motion_selection_weight,
        "right_action_selection_weight": args.right_action_selection_weight,
        "right_high_motion_selection_weight": args.right_high_motion_selection_weight,
        "normalization_checkpoint": normalization_source,
        "augmentation": {
            "format": "track2-public-left-to-right-physical-mirror-v1",
            "mirror_sign": MIRROR_SIGN.tolist(),
            "image_transform": "horizontal_flip",
            "real_right_window_count": train.real_right_window_count,
            "mirrored_left_window_count": train.mirrored_left_window_count,
            "validation_real_right_only": True,
            "validation_right_episodes": validation_right_episodes,
        },
        "early_stop_validations": args.early_stop_validations,
        "minimum_relative_improvement": args.minimum_relative_improvement,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }
    state_path = output / "training_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != "track2-autoregressive-training-state-v1":
            raise ValueError("unsupported autoregressive training state")
        if int(state.get("target_steps", -1)) != args.steps or state.get("training_config") != training_config:
            raise ValueError("autoregressive resume state does not match the requested training run")
        start_step = int(state["step"])
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        history = list(state.get("history", []))
        best_selection_metric = float(state.get("best_selection_metric", float("inf")))
        print(json.dumps({"resumed_from_step": start_step, "target_step": args.steps}), flush=True)

    if start_step == 0:
        baseline = evaluate(validation_loader, model, device, mean, std, args.high_motion_threshold)
        baseline_right = dict(baseline) if len(right_validation_indices) == sample_count else None
        baseline["step"] = 0.0
        baseline["action_left_sample_count"] = len(left_validation_indices)
        baseline["action_right_sample_count"] = len(right_validation_indices)
        baseline["action_ambiguous_sample_count"] = sample_count - len(left_validation_indices) - len(right_validation_indices)
        baseline["action_left"] = None
        baseline["action_right"] = baseline_right
        baseline_value = selection_metric(
            baseline,
            args.high_motion_selection_weight,
            baseline["action_right"],
            args.right_action_selection_weight,
            args.right_high_motion_selection_weight,
        )
        baseline["selection_metric"] = baseline_value
        baseline["role"] = "frozen_initialization_reference"
        history = [baseline]
        best_selection_metric = baseline_value
        print(json.dumps({"step": 0, "validation": baseline, "best_selection_metric": best_selection_metric}), flush=True)
        model.train()
    else:
        if not history or float(history[0].get("step", -1)) != 0.0:
            raise ValueError("resumed v423 state lacks the frozen step-0 reference")
        baseline_value = float(history[0]["selection_metric"])

    stop_requested = False
    early_gate_failed = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"signal": signum, "status": "checkpoint_requested"}), flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    iterator = iter(train_loader)
    for step in range(start_step + 1, args.steps + 1):
        try:
            context, action_history, future, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            context, action_history, future, target = next(iterator)
        context = frames_for_model(context).to(device, non_blocking=True)
        target = frames_for_model(target).to(device, non_blocking=True)
        action_history = ((action_history.to(device, non_blocking=True) - mean) / std).float()
        future = ((future.to(device, non_blocking=True) - mean) / std).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if args.train_rollout_steps == 1:
                prediction = model(context, torch.cat([action_history, future[:, :1]], dim=1))
                loss = reconstruction_loss(
                    prediction,
                    target[:, 0],
                    context[:, -1],
                    args.motion_weight,
                    args.motion_threshold,
                    args.horizon_loss_power,
                    args.temporal_delta_weight,
                    args.texture_laplacian_weight,
                    args.temporal_delta_pool,
                    args.lowfreq_anchor_weight,
                )
            else:
                prediction = rollout(model, context, action_history, future[:, : args.train_rollout_steps])
                target_rollout = target[:, : args.train_rollout_steps]
                previous = torch.cat([context[:, -1:], target_rollout[:, :-1]], dim=1)
                loss = reconstruction_loss(
                    prediction,
                    target_rollout,
                    previous,
                    args.motion_weight,
                    args.motion_threshold,
                    args.horizon_loss_power,
                    args.temporal_delta_weight,
                    args.texture_laplacian_weight,
                    args.temporal_delta_pool,
                    args.lowfreq_anchor_weight,
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            first_prediction = prediction if args.train_rollout_steps == 1 else prediction[:, 0]
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "first_frame_mae": float(functional.l1_loss(first_prediction.float(), target[:, 0]).detach().cpu())}), flush=True)
        if step % args.validation_interval == 0 or step == args.steps:
            result = evaluate(validation_loader, model, device, mean, std, args.high_motion_threshold)
            left_action_result = evaluate(left_validation_loader, model, device, mean, std, args.high_motion_threshold) if left_validation_indices else None
            right_action_result = (
                dict(result)
                if len(right_validation_indices) == sample_count
                else evaluate(right_validation_loader, model, device, mean, std, args.high_motion_threshold)
                if right_validation_indices
                else None
            )
            model.train()
            result["step"] = float(step)
            result["action_left_sample_count"] = len(left_validation_indices)
            result["action_right_sample_count"] = len(right_validation_indices)
            result["action_ambiguous_sample_count"] = sample_count - len(left_validation_indices) - len(right_validation_indices)
            result["action_left"] = left_action_result
            result["action_right"] = right_action_result
            current_selection_metric = selection_metric(
                result,
                args.high_motion_selection_weight,
                right_action_result,
                args.right_action_selection_weight,
                args.right_high_motion_selection_weight,
            )
            result["selection_metric"] = current_selection_metric
            history.append(result)
            initialization = args.init_autoregressive_checkpoint or args.init_checkpoint
            metadata = {"backend": "autoregressive_unet", "format": "track2-autoregressive-unet-v1", "windows": str(Path(args.windows).resolve()), "split_manifest": str(Path(args.split_manifest).resolve()), "train_window_count": len(train), "validation_window_count": len(validation), "validation_sample_count": sample_count, "context_frames": 5, "history_actions": 4, "future_actions": args.rollout_horizon, "target_frames": args.rollout_horizon, "serving_prediction_frames": 8, "action_dim": 14, "working_resolution": 256, "serving_resolution": 256, "initialization_checkpoint": str(Path(initialization).resolve()) if initialization else None, "normalization_checkpoint": normalization_source, "data_action_statistics_ignored_for_normalization": normalization_source is not None, "train_rollout_steps": args.train_rollout_steps, "motion_weight": args.motion_weight, "motion_threshold": args.motion_threshold, "horizon_loss_power": args.horizon_loss_power, "temporal_delta_weight": args.temporal_delta_weight, "texture_laplacian_weight": args.texture_laplacian_weight, "temporal_delta_pool": args.temporal_delta_pool, "high_motion_sampling": motion_sampling, "arm_sampling": arm_sampling, "high_motion_selection_weight": args.high_motion_selection_weight, "right_action_selection_weight": args.right_action_selection_weight, "right_high_motion_selection_weight": args.right_high_motion_selection_weight, "augmentation": training_config["augmentation"], "frozen_initialization_selection_metric": baseline_value, "minimum_relative_improvement": args.minimum_relative_improvement, "initialized_parameter_count": len(initialized_keys), "checkpoint_step": step, "validation": history, "best_selection_metric": min(best_selection_metric, current_selection_metric)}
            save_checkpoint(output / "checkpoints" / f"checkpoint_step_{step:06d}", model, mean, std, metadata)
            if current_selection_metric < best_selection_metric:
                best_selection_metric = current_selection_metric
                metadata["best_checkpoint_step"] = step
                metadata["best_rollout_mae"] = result["rollout_mae"]
                metadata["best_high_motion_rollout_mae"] = result["high_motion_rollout_mae"]
                save_checkpoint(output, model, mean, std, metadata)
                save_checkpoint(output / "best", model, mean, std, metadata)
            print(json.dumps({"step": step, "validation": result, "best_selection_metric": best_selection_metric}), flush=True)
            trained_validation_count = len(history) - 1
            required_metric = baseline_value * (1.0 - args.minimum_relative_improvement)
            if (
                args.early_stop_validations > 0
                and trained_validation_count >= args.early_stop_validations
                and best_selection_metric > required_metric
            ):
                early_gate_failed = True
                stop_requested = True
                output.mkdir(parents=True, exist_ok=True)
                (output / "early_gate.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "step": step,
                            "baseline_selection_metric": baseline_value,
                            "best_selection_metric": best_selection_metric,
                            "required_selection_metric": required_metric,
                            "trained_validation_count": trained_validation_count,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps({"step": step, "status": "early_gate_failed", "required_selection_metric": required_metric}), flush=True)
        if step % args.checkpoint_interval == 0 or step == args.steps or stop_requested:
            save_training_state(
                output, model, optimizer, step, args.steps, training_config, history, best_selection_metric
            )
        if stop_requested:
            status = "checkpointed_after_early_gate_failure" if early_gate_failed else "checkpointed_for_gpu_yield"
            print(json.dumps({"step": step, "status": status}), flush=True)
            return


if __name__ == "__main__":
    main()
