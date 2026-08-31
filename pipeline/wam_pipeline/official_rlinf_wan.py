"""Exact inference adapter for an official RLinf Wan RobotWin checkpoint.

The organizer's RLinf environment owns the reference calling convention.  This
module deliberately mirrors that convention while keeping the heavyweight
DiffSynth import isolated from the Track 2 HTTP service and its unit tests.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


OFFICIAL_RLINF_WAN_RUNTIME_REVISION = "2a2e05fa1f724828b243f272540989b19a6e54f8"
REQUIRED_CHECKPOINT_FILES = ("dit_model.safetensors", "Wan2.2_VAE.pth")


@dataclass(frozen=True)
class OfficialRLinfWanCheckpoint:
    """Verified local layout expected by RLinf's RobotWin Wan environment."""

    root: Path
    dit_model: Path
    vae: Path
    dataset: Path


def sha256_file(path: Path) -> str:
    """Hash one file without loading a multi-gigabyte checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_official_rlinf_wan_checkpoint(checkpoint_dir: str | Path) -> OfficialRLinfWanCheckpoint:
    """Reject incomplete or lookalike checkpoint directories before inference."""
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"RLinf Wan checkpoint directory does not exist: {root}")
    files: dict[str, Path] = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"RLinf Wan checkpoint is missing required file: {path}")
        files[name] = path
    dataset = root / "dataset"
    if not dataset.is_dir() or not any(dataset.glob("*.npy")):
        raise ValueError(
            "RLinf Wan checkpoint needs dataset/ with official reset trajectory .npy files: "
            f"{dataset}"
        )
    return OfficialRLinfWanCheckpoint(root, files["dit_model.safetensors"], files["Wan2.2_VAE.pth"], dataset)


def build_official_action_slots(history_actions: np.ndarray, future_actions: np.ndarray) -> np.ndarray:
    """Build the exact 13 action slots consumed by RLinf's ``WanEnv``.

    Slot 0 is the reference-image anchor.  Slots 1--4 are actions aligned with
    the four remaining observed images; slots 5--12 drive the eight futures.
    """
    history = np.asarray(history_actions, dtype=np.float32)
    future = np.asarray(future_actions, dtype=np.float32)
    expected_history = (CONTEXT_FRAMES - 1, ACTION_DIM)
    expected_future = (PREDICTION_FRAMES, ACTION_DIM)
    if history.shape != expected_history:
        raise ValueError(f"history_actions must have shape {expected_history}, got {history.shape}")
    if future.shape != expected_future:
        raise ValueError(f"future_actions must have shape {expected_future}, got {future.shape}")
    if not np.isfinite(history).all() or not np.isfinite(future).all():
        raise ValueError("RLinf Wan actions contain NaN or infinity")
    return np.concatenate((np.zeros((1, ACTION_DIM), dtype=np.float32), history, future), axis=0)


def _runtime_commit(runtime_root: Path) -> str:
    """Read a pinned checkout revision without trusting the caller's label."""
    git_dir = runtime_root / ".git"
    if not git_dir.exists():
        raise ValueError(f"DiffSynth runtime is not a Git checkout: {runtime_root}")
    completed = subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise ValueError(f"cannot read DiffSynth runtime revision: {runtime_root}")
    return completed.stdout.strip()


def validate_official_rlinf_wan_runtime(diffsynth_root: str | Path) -> Path:
    """Verify the isolated, pinned DiffSynth source before importing it."""
    root = Path(diffsynth_root).expanduser().resolve()
    if not (root / "diffsynth" / "pipelines" / "wan_video_new.py").is_file():
        raise ValueError(f"DiffSynth Wan runtime source is incomplete: {root}")
    revision = _runtime_commit(root)
    if revision != OFFICIAL_RLINF_WAN_RUNTIME_REVISION:
        raise ValueError(
            "DiffSynth runtime revision does not match the pinned RLinf source: "
            f"expected {OFFICIAL_RLINF_WAN_RUNTIME_REVISION}, got {revision}"
        )
    return root


class OfficialRLinfWanRuntime:
    """Load and call the same DiffSynth Wan pipeline used by official RLinf."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        diffsynth_root: str | Path,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        self.checkpoint = validate_official_rlinf_wan_checkpoint(checkpoint_dir)
        self.diffsynth_root = validate_official_rlinf_wan_runtime(diffsynth_root)
        self.device = device
        self.inference_steps = int(inference_steps)
        if self.inference_steps < 1:
            raise ValueError("official RLinf Wan inference_steps must be positive")
        self._pipe: Any | None = None
        self._torch: Any | None = None

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        # The checkout is isolated under artifacts/; do not import the known
        # dirty vendor working tree in third_party/.
        root_text = str(self.diffsynth_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            torch = importlib.import_module("torch")
            module = importlib.import_module("diffsynth.pipelines.wan_video_new")
        except Exception as exc:
            raise RuntimeError(
                "official RLinf Wan runtime dependencies are unavailable; install them in an isolated environment"
            ) from exc
        try:
            Path(module.__file__).resolve().relative_to(self.diffsynth_root)
        except (AttributeError, ValueError) as exc:
            raise RuntimeError(
                "a different DiffSynth package is already imported; restart with only the pinned runtime"
            ) from exc
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        try:
            pipe = module.WanVideoPipeline.from_pretrained(
                torch_dtype=dtype,
                device="cpu",
                model_configs=[
                    module.ModelConfig(path=str(self.checkpoint.dit_model), offload_device="cpu"),
                    module.ModelConfig(path=str(self.checkpoint.vae), offload_device="cpu"),
                ],
            )
            pipe.dit.to(self.device)
            pipe.vae.to(self.device)
            pipe.dit.eval()
            pipe.vae.eval()
        except Exception as exc:
            raise RuntimeError("failed to load the supplied RLinf Wan checkpoint with pinned DiffSynth") from exc
        self._torch = torch
        self._pipe = pipe
        return pipe

    def predict(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seed: int,
        instruction: str | None = None,
    ) -> np.ndarray:
        del instruction  # Official RobotWin WanEnv uses action conditioning, not text conditioning.
        context = np.asarray(context_frames, dtype=np.uint8)
        if context.shape != (CONTEXT_FRAMES, 256, 256, 3):
            raise ValueError(f"context_frames must have shape {(CONTEXT_FRAMES, 256, 256, 3)}, got {context.shape}")
        slots = build_official_action_slots(history_actions, future_actions)
        pipe = self._load_pipe()
        torch = self._torch
        assert torch is not None
        torch.manual_seed(int(seed))
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
        images = [Image.fromarray(frame, mode="RGB") for frame in context]
        action_tensor = torch.from_numpy(slots).unsqueeze(0).to(self.device)
        # This call is intentionally line-for-line equivalent to WanEnv's
        # B=1 path, except it returns only the Track 2 future chunk.
        try:
            output = pipe(
                seed=int(seed),
                tiled=False,
                input_image=[images[0]],
                input_image4=[images[-4:]],
                action=action_tensor,
                height=256,
                width=256,
                num_frames=CONTEXT_FRAMES + PREDICTION_FRAMES,
                num_inference_steps=self.inference_steps,
                cfg_scale=1.0,
                progress_bar_cmd=lambda value: value,
                batch_size=1,
            )
        except Exception as exc:
            raise RuntimeError("official RLinf Wan inference failed") from exc
        if not isinstance(output, (list, tuple)) or len(output) != 1:
            raise RuntimeError("official RLinf Wan returned an invalid batch")
        video = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in output[0]], axis=0)
        expected = (CONTEXT_FRAMES + PREDICTION_FRAMES, 256, 256, 3)
        if video.shape != expected:
            raise RuntimeError(f"official RLinf Wan returned {video.shape}, expected {expected}")
        return video[CONTEXT_FRAMES:].copy()
