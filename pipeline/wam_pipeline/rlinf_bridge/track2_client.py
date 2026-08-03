"""Strict client for the published WorldArena Track 2 prediction API."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import requests

from ..images import decode_png_base64, encode_png_base64
from ..profile import ACTION_DIM, API_VERSION, CONTEXT_ACTIONS, CONTEXT_FRAMES, OFFICIAL_PROFILE_ID, PREDICTION_FRAMES


@dataclass(frozen=True)
class Track2ServiceClient:
    """Call a submitted service without relying on RLinf's private pickle protocol."""

    base_url: str
    bearer_token: str
    model_version: str
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.bearer_token:
            raise ValueError("bearer_token must be non-empty")
        if not self.model_version:
            raise ValueError("model_version must be non-empty")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}", "Content-Type": "application/json"}

    def assert_ready(self) -> None:
        health = requests.get(f"{self.base_url}/v1/health", timeout=self.timeout_seconds)
        health.raise_for_status()
        body = health.json()
        if body.get("status") != "ready" or body.get("api_version") != API_VERSION:
            raise RuntimeError("Track 2 world-model service is not ready")
        if body.get("model_version") != self.model_version:
            raise RuntimeError("Track 2 world-model service model_version does not match RLinf config")

        capabilities = requests.get(
            f"{self.base_url}/v1/capabilities", headers=self._headers, timeout=self.timeout_seconds
        )
        capabilities.raise_for_status()
        body = capabilities.json()
        if body.get("model_version") != self.model_version:
            raise RuntimeError("capabilities model_version does not match RLinf config")
        profiles = body.get("profiles", [])
        if not any(profile.get("profile_id") == OFFICIAL_PROFILE_ID for profile in profiles):
            raise RuntimeError("service does not expose the official Track 2 Adjust Bottle profile")

    def predict_batch(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seeds: np.ndarray | list[int],
        instructions: list[str | None] | None = None,
    ) -> np.ndarray:
        """Return [B,8,256,256,3] uint8 predictions with exact API alignment."""
        context_frames = np.asarray(context_frames)
        history_actions = np.asarray(history_actions, dtype=np.float32)
        future_actions = np.asarray(future_actions, dtype=np.float32)
        seeds = np.asarray(seeds)
        if context_frames.ndim != 5 or context_frames.shape[1:] != (CONTEXT_FRAMES, 256, 256, 3):
            raise ValueError("context_frames must be [B,5,256,256,3]")
        batch = context_frames.shape[0]
        if context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be uint8")
        if history_actions.shape != (batch, CONTEXT_ACTIONS, ACTION_DIM):
            raise ValueError("history_actions must be [B,4,14]")
        if future_actions.shape != (batch, PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("future_actions must be [B,8,14]")
        if seeds.shape != (batch,) or not np.issubdtype(seeds.dtype, np.integer):
            raise ValueError("seeds must be integer [B]")
        if not np.isfinite(history_actions).all() or not np.isfinite(future_actions).all():
            raise ValueError("actions must be finite")
        if instructions is None:
            instructions = [None] * batch
        if len(instructions) != batch:
            raise ValueError("instructions must have B items")

        request_id = str(uuid.uuid4())
        samples = []
        for index in range(batch):
            samples.append(
                {
                    "sample_id": f"rlinf-{index}",
                    "seed": int(seeds[index]),
                    "context": {
                        "frames": [encode_png_base64(frame) for frame in context_frames[index]],
                        "actions": history_actions[index].tolist(),
                        "states": None,
                        "instruction": instructions[index],
                    },
                    "actions": future_actions[index].tolist(),
                }
            )
        payload = {
            "api_version": API_VERSION,
            "request_id": request_id,
            "model_version": self.model_version,
            "profile_id": OFFICIAL_PROFILE_ID,
            "samples": samples,
        }
        response = requests.post(
            f"{self.base_url}/v1/predict", headers=self._headers, json=payload, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        body = response.json()
        if (
            body.get("api_version") != API_VERSION
            or body.get("request_id") != request_id
            or body.get("model_version") != self.model_version
        ):
            raise RuntimeError("Track 2 service response does not echo immutable identifiers")
        predictions = body.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != batch:
            raise RuntimeError("Track 2 service returned an invalid batch size")
        frames: list[np.ndarray] = []
        for index, prediction in enumerate(predictions):
            if prediction.get("sample_id") != samples[index]["sample_id"]:
                raise RuntimeError("Track 2 service changed response sample ordering")
            raw_frames = prediction.get("frames")
            if not isinstance(raw_frames, list) or len(raw_frames) != PREDICTION_FRAMES:
                raise RuntimeError("Track 2 service returned an invalid prediction horizon")
            frames.append(np.stack([decode_png_base64(frame) for frame in raw_frames]))
        return np.stack(frames)
