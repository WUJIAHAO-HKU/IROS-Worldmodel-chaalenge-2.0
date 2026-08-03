"""Request validation for the official stateless Track 2 API."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .images import decode_png_base64
from .profile import (
    ACTION_DIM,
    API_VERSION,
    CONTEXT_ACTIONS,
    CONTEXT_FRAMES,
    MAX_BATCH_SIZE,
    OFFICIAL_PROFILE_ID,
    PREDICTION_FRAMES,
)


class RequestValidationError(ValueError):
    """A caller supplied an invalid API payload."""


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    seed: int
    frames: np.ndarray
    history_actions: np.ndarray
    future_actions: np.ndarray
    instruction: str | None


@dataclass(frozen=True)
class PredictionRequest:
    request_id: str
    model_version: str
    samples: tuple[PredictionSample, ...]


def _error(message: str) -> RequestValidationError:
    return RequestValidationError(message)


def _finite_actions(value: Any, expected_length: int, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != expected_length:
        raise _error(f"{field} must have length {expected_length}")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise _error(f"{field} must be numeric") from exc
    if result.shape != (expected_length, ACTION_DIM):
        raise _error(f"{field} must have shape [{expected_length}, {ACTION_DIM}]")
    if not np.isfinite(result).all():
        raise _error(f"{field} must contain only finite values")
    return result


def validate_predict_payload(payload: Any, expected_model_version: str) -> PredictionRequest:
    """Validate everything before inference so batches never partially succeed."""
    if not isinstance(payload, dict):
        raise _error("request body must be a JSON object")
    if payload.get("api_version") != API_VERSION:
        raise _error("unsupported api_version")
    if payload.get("model_version") != expected_model_version:
        raise _error("unsupported model_version")
    if payload.get("profile_id") != OFFICIAL_PROFILE_ID:
        raise _error("unsupported profile_id")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        raise _error("request_id must be a UUID string")
    try:
        uuid.UUID(request_id)
    except (ValueError, AttributeError) as exc:
        raise _error("request_id must be a UUID string") from exc
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not 1 <= len(raw_samples) <= MAX_BATCH_SIZE:
        raise _error(f"samples must contain 1 through {MAX_BATCH_SIZE} items")

    seen_ids: set[str] = set()
    samples: list[PredictionSample] = []
    for index, raw_sample in enumerate(raw_samples):
        field = f"samples[{index}]"
        if not isinstance(raw_sample, dict):
            raise _error(f"{field} must be an object")
        sample_id = raw_sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise _error(f"{field}.sample_id must be a non-empty string")
        if sample_id in seen_ids:
            raise _error(f"{field}.sample_id must be unique")
        seen_ids.add(sample_id)
        seed = raw_sample.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= (2**63 - 1):
            raise _error(f"{field}.seed must be an integer in [0, 2^63-1]")
        context = raw_sample.get("context")
        if not isinstance(context, dict):
            raise _error(f"{field}.context must be an object")
        frames = context.get("frames")
        if not isinstance(frames, list) or len(frames) != CONTEXT_FRAMES:
            raise _error(f"{field}.context.frames must have length {CONTEXT_FRAMES}")
        decoded_frames = np.stack([decode_png_base64(image) for image in frames])
        history_actions = _finite_actions(
            context.get("actions"), CONTEXT_ACTIONS, f"{field}.context.actions"
        )
        states = context.get("states")
        if states is not None:
            raise _error(f"{field}.context.states must be null for the official profile")
        instruction = context.get("instruction")
        if instruction is not None and not isinstance(instruction, str):
            raise _error(f"{field}.context.instruction must be a string or null")
        future_actions = _finite_actions(raw_sample.get("actions"), PREDICTION_FRAMES, f"{field}.actions")
        samples.append(
            PredictionSample(sample_id, seed, decoded_frames, history_actions, future_actions, instruction)
        )
    return PredictionRequest(request_id, expected_model_version, tuple(samples))


def payload_digest(payload: Any) -> str:
    """Stable payload fingerprint used to enforce request-id idempotency."""
    import hashlib
    import json

    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
