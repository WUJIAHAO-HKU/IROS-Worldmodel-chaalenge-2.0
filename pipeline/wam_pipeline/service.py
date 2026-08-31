"""Reference implementation of the official Track 2 HTTP service contract."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .backends import ModelBackend, build_backend
from .contracts import RequestValidationError, payload_digest, validate_predict_payload
from .images import ImageValidationError, encode_png_base64_batch
from .profile import (
    API_VERSION,
    MAX_BATCH_SIZE,
    MAX_CONCURRENCY,
    MAX_REQUEST_BYTES,
    capabilities,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceSettings:
    model_version: str
    bearer_token: str
    backend_name: str = "synthetic"
    checkpoint_dir: str | None = None
    device: str = "cuda"
    batch_workers: int = 1
    native_batch_enabled: bool = False
    native_batch_micro_size: int = MAX_BATCH_SIZE
    release_cuda_cache: bool = False
    image_codec_workers: int = 1

    def __post_init__(self) -> None:
        if self.batch_workers < 1:
            raise ValueError("batch_workers must be at least one")
        if not 1 <= self.native_batch_micro_size <= MAX_BATCH_SIZE:
            raise ValueError(
                f"native_batch_micro_size must be between one and {MAX_BATCH_SIZE}"
            )
        if self.image_codec_workers < 1:
            raise ValueError("image_codec_workers must be at least one")


class IdempotencyCache:
    """Small in-memory request cache; do not persist official request payloads."""

    def __init__(self, max_items: int = 128) -> None:
        self._items: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
        self._max_items = max_items

    def get(self, request_id: str, digest: str) -> tuple[str, dict[str, Any] | None]:
        item = self._items.get(request_id)
        if item is None:
            return "miss", None
        stored_digest, response = item
        self._items.move_to_end(request_id)
        if stored_digest != digest:
            return "conflict", None
        return "hit", response

    def put(self, request_id: str, digest: str, response: dict[str, Any]) -> None:
        self._items[request_id] = (digest, response)
        self._items.move_to_end(request_id)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)


def _error(status: int, code: str, message: str, request_id: str | None, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id, "retryable": retryable}},
    )


def _request_id_from_untrusted(payload: object) -> str | None:
    return payload.get("request_id") if isinstance(payload, dict) and isinstance(payload.get("request_id"), str) else None


def create_app(settings: ServiceSettings, backend: ModelBackend | None = None) -> FastAPI:
    """Create an app whose only prediction output is future RGB frames."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.backend = backend or build_backend(
        settings.backend_name,
        settings.checkpoint_dir,
        settings.device,
        v15_library_dir=os.environ.get("WAM_V15_LIBRARY_DIR"),
        v216_library_index=os.environ.get("WAM_V216_LIBRARY_INDEX"),
    )
    app.state.cache = IdempotencyCache()
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    app.state.batch_workers = settings.batch_workers
    app.state.native_batch_enabled = settings.native_batch_enabled
    app.state.native_batch_micro_size = settings.native_batch_micro_size
    app.state.release_cuda_cache = settings.release_cuda_cache
    app.state.image_codec_workers = settings.image_codec_workers
    app.state.backend_warmed = False
    app.state.ready = True

    async def predict_one(sample):
        frames = await asyncio.to_thread(
            app.state.backend.predict,
            sample.frames,
            sample.history_actions,
            sample.future_actions,
            sample.seed,
            sample.instruction,
        )
        if frames.shape != (8, 256, 256, 3) or frames.dtype.name != "uint8":
            raise RuntimeError("backend returned an invalid Track 2 frame tensor")
        if app.state.release_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return frames

    async def predict_native_batch(samples):
        frames_by_sample = []
        micro_size = app.state.native_batch_micro_size
        for begin in range(0, len(samples), micro_size):
            micro = samples[begin : begin + micro_size]
            frames = await asyncio.to_thread(
                app.state.backend.predict_batch,
                np.stack([sample.frames for sample in micro]),
                np.stack([sample.history_actions for sample in micro]),
                np.stack([sample.future_actions for sample in micro]),
                np.asarray([sample.seed for sample in micro], dtype=np.int64),
                [sample.instruction for sample in micro],
            )
            expected = (len(micro), 8, 256, 256, 3)
            if frames.shape != expected or frames.dtype.name != "uint8":
                raise RuntimeError("backend returned an invalid Track 2 batch tensor")
            frames_by_sample.extend(frames)
            if app.state.release_cuda_cache:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return frames_by_sample

    def authorized(authorization: str | None) -> bool:
        return authorization == f"Bearer {settings.bearer_token}"

    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ready" if app.state.ready else "starting",
            "api_version": API_VERSION,
            "model_version": settings.model_version,
        }

    @app.get("/v1/capabilities")
    async def get_capabilities(authorization: str | None = Header(default=None)):
        if not authorized(authorization):
            return _error(401, "UNAUTHENTICATED", "missing or invalid bearer token", None, False)
        return capabilities(settings.model_version)

    @app.post("/v1/predict")
    async def predict(request: Request, authorization: str | None = Header(default=None)):
        if not authorized(authorization):
            return _error(401, "UNAUTHENTICATED", "missing or invalid bearer token", None, False)
        raw = await request.body()
        if len(raw) > MAX_REQUEST_BYTES:
            return _error(413, "PAYLOAD_TOO_LARGE", "request exceeds the official byte limit", None, False)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return _error(400, "INVALID_ARGUMENT", "request body is not valid JSON", None, False)
        request_id = _request_id_from_untrusted(payload)
        try:
            digest = payload_digest(payload)
        except (TypeError, ValueError):
            return _error(400, "INVALID_ARGUMENT", "request contains non-JSON values", request_id, False)
        cache_state, cached = app.state.cache.get(request_id or "", digest)
        if cache_state == "conflict":
            return _error(409, "REQUEST_ID_CONFLICT", "request_id was used with another payload", request_id, False)
        if cache_state == "hit":
            return JSONResponse(status_code=200, content=cached)
        try:
            parsed = validate_predict_payload(
                payload, settings.model_version, settings.image_codec_workers
            )
        except (RequestValidationError, ImageValidationError) as exc:
            message = str(exc)
            if "unsupported" in message:
                return _error(422, "UNSUPPORTED_VERSION", message, request_id, False)
            return _error(400, "INVALID_ARGUMENT", message, request_id, False)
        if not app.state.ready:
            return _error(503, "NOT_READY", "model is not ready", parsed.request_id, True)
        if app.state.semaphore.locked():
            return _error(429, "OVERLOADED", "maximum concurrency reached", parsed.request_id, True)
        async with app.state.semaphore:
            try:
                frames_by_sample = []
                remaining_samples = list(parsed.samples)
                native_batch = getattr(app.state.backend, "predict_batch", None)
                if (
                    app.state.native_batch_enabled
                    and native_batch is not None
                    and len(remaining_samples) > 1
                ):
                    frames_by_sample = await predict_native_batch(remaining_samples)
                    remaining_samples = []
                    app.state.backend_warmed = True
                # Composite backends load their immutable runtime lazily.  Warm
                # exactly one sample before enabling intra-request parallelism,
                # preventing concurrent duplicate checkpoint loads.
                if not app.state.backend_warmed and remaining_samples:
                    frames_by_sample.append(await predict_one(remaining_samples.pop(0)))
                    app.state.backend_warmed = True
                if app.state.batch_workers == 1:
                    for sample in remaining_samples:
                        frames_by_sample.append(await predict_one(sample))
                else:
                    worker_slots = asyncio.Semaphore(app.state.batch_workers)

                    async def bounded_predict(sample):
                        async with worker_slots:
                            return await predict_one(sample)

                    frames_by_sample.extend(
                        await asyncio.gather(
                            *(bounded_predict(sample) for sample in remaining_samples)
                        )
                    )
                encoded_frames = encode_png_base64_batch(
                    [frame for frames in frames_by_sample for frame in frames],
                    app.state.image_codec_workers,
                )
                predictions = []
                for index, sample in enumerate(parsed.samples):
                    predictions.append(
                        {
                            "sample_id": sample.sample_id,
                            "frames": encoded_frames[index * 8 : (index + 1) * 8],
                        }
                    )
                response = {
                    "api_version": API_VERSION,
                    "request_id": parsed.request_id,
                    "model_version": settings.model_version,
                    "predictions": predictions,
                }
            except Exception:
                LOGGER.exception("Track 2 world-model inference failed request_id=%s", parsed.request_id)
                return _error(500, "INTERNAL", "world-model inference failed", parsed.request_id, True)
        app.state.cache.put(parsed.request_id, digest, response)
        return JSONResponse(status_code=200, content=response)

    return app


def settings_from_env() -> ServiceSettings:
    return ServiceSettings(
        model_version=os.environ.get("WAM_MODEL_VERSION", "ivideogpt64-track2-dev"),
        bearer_token=os.environ.get("WAM_BEARER_TOKEN", "local-dev-token"),
        backend_name=os.environ.get("WAM_BACKEND", "synthetic"),
        checkpoint_dir=os.environ.get("WAM_CHECKPOINT_DIR"),
        device=os.environ.get("WAM_DEVICE", "cuda"),
        batch_workers=int(os.environ.get("WAM_BATCH_WORKERS", "1")),
        native_batch_enabled=os.environ.get("WAM_NATIVE_BATCH_ENABLED", "0") == "1",
        native_batch_micro_size=int(
            os.environ.get("WAM_NATIVE_BATCH_MICRO_SIZE", str(MAX_BATCH_SIZE))
        ),
        release_cuda_cache=os.environ.get("WAM_RELEASE_CUDA_CACHE", "0") == "1",
        image_codec_workers=int(os.environ.get("WAM_IMAGE_CODEC_WORKERS", "1")),
    )
