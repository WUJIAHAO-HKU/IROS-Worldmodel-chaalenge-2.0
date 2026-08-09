"""Model backends behind the exact same 5-frame/8-action inference contract."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class ModelBackend(ABC):
    """A backend only predicts RGB frames; it never computes reward or actions."""

    @abstractmethod
    def predict(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seed: int,
        instruction: str | None,
    ) -> np.ndarray:
        """Return [8, 256, 256, 3] uint8 future RGB frames."""


class SyntheticActionBackend(ModelBackend):
    """Deterministic protocol-test backend, deliberately not a contest model."""

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3):
            raise ValueError("unexpected context frame shape")
        if history_actions.shape != (CONTEXT_FRAMES - 1, ACTION_DIM):
            raise ValueError("unexpected history action shape")
        if future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("unexpected future action shape")
        # The seed-only dither makes different seeds valid stochastic samples while preserving repeatability.
        rng = np.random.default_rng(seed)
        output = np.empty((PREDICTION_FRAMES, 256, 256, 3), dtype=np.uint8)
        frame = context_frames[-1].astype(np.float32)
        action_scale = np.array([8.0, 5.0, 3.0], dtype=np.float32)
        for index, action in enumerate(future_actions):
            shift = np.rint(np.tanh(action[:2]) * 5).astype(int)
            frame = np.roll(frame, shift=(int(shift[1]), int(shift[0])), axis=(0, 1))
            tint = np.tanh(action[2:5]) * action_scale
            frame = np.clip(frame + tint.reshape(1, 1, 3), 0, 255)
            noise = rng.integers(-1, 2, size=(256, 256, 1), dtype=np.int16)
            output[index] = np.clip(frame + noise, 0, 255).astype(np.uint8)
        return output


class IVideoGPTBackend(ModelBackend):
    """Runtime adapter for a fine-tuned 64x64 iVideoGPT checkpoint.

    This class intentionally refuses to pretend that a BAIR 4-D checkpoint is
    compatible with Track 2. Supply a checkpoint fine-tuned with the provided
    5-context/14-action configuration before using this backend.
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime
        metadata_path = self.checkpoint_dir / "track2_ivideogpt_config.npz"
        if not metadata_path.exists():
            raise RuntimeError(
                "iVideoGPT checkpoint is not Track-2-adapted; run scripts/train_ivideogpt64.py first"
            )
        # The concrete loader is deferred so protocol tests do not require torch/transformers.
        try:
            from .ivideogpt_runtime import load_track2_ivideogpt
        except ImportError as exc:
            raise RuntimeError("iVideoGPT runtime dependencies are not installed") from exc
        self._runtime = load_track2_ivideogpt(self.checkpoint_dir, self.device)
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        runtime = self._load_runtime()
        return runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class ResidualUNetBackend(ModelBackend):
    """Track-2-specific residual baseline with the same external contract."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .residual_unet_runtime import Track2ResidualUNet

            self._runtime = Track2ResidualUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class FlowResidualUNetBackend(ModelBackend):
    """Appearance-preserving action-conditioned Track 2 predictor."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .flow_residual_unet_runtime import Track2FlowResidualUNet

            self._runtime = Track2FlowResidualUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class TemporalUNetBackend(ModelBackend):
    """Per-step action-conditioned Track 2 predictor."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .temporal_unet_runtime import Track2TemporalUNet

            self._runtime = Track2TemporalUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class AutoregressiveUNetBackend(ModelBackend):
    """One-step model recursively rolled forward for the exact 8-frame contract."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .autoregressive_unet_runtime import Track2AutoregressiveUNet

            self._runtime = Track2AutoregressiveUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class HybridUNetBackend(ModelBackend):
    """Horizon-specialized Track 2 predictor packaged as one checkpoint directory."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .hybrid_unet_runtime import Track2HorizonHybridUNet

            self._runtime = Track2HorizonHybridUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class V15CompositeBackend(ModelBackend):
    """Validated v15 composite exposed through the online Track-2 contract."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .v15_runtime import Track2V15Runtime

            self._runtime = Track2V15Runtime(self.checkpoint_dir, self.library_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


def build_backend(
    name: str,
    checkpoint_dir: str | None = None,
    device: str = "cuda",
    *,
    v15_library_dir: str | Path | None = None,
) -> ModelBackend:
    if name == "synthetic":
        return SyntheticActionBackend()
    if name == "ivideogpt":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend ivideogpt")
        return IVideoGPTBackend(checkpoint_dir, device)
    if name == "residual-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend residual-unet")
        return ResidualUNetBackend(checkpoint_dir, device)
    if name == "flow-residual-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend flow-residual-unet")
        return FlowResidualUNetBackend(checkpoint_dir, device)
    if name == "temporal-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend temporal-unet")
        return TemporalUNetBackend(checkpoint_dir, device)
    if name == "autoregressive-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend autoregressive-unet")
        return AutoregressiveUNetBackend(checkpoint_dir, device)
    if name == "hybrid-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend hybrid-unet")
        return HybridUNetBackend(checkpoint_dir, device)
    if name == "v15-composite":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v15-composite")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v15-composite")
        return V15CompositeBackend(checkpoint_dir, v15_library_dir, device)
    raise ValueError(f"unknown backend {name!r}")
