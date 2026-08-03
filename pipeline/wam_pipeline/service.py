"""Reference implementation of the official Track 2 HTTP service contract."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from .backends import ModelBackend, build_backend
from .contracts import RequestValidationError, payload_digest, validate_predict_payload
from .images import ImageValidationError, encode_png_base64
from .profile import API_VERSION, MAX_CONCURRENCY, MAX_REQUEST_BYTES, capabilities


@dataclass
class ServiceSettings:
    model_version: str
    bearer_token: str
    backend_name: str = "synthetic"
    checkpoint_dir: str | None = None
    device: str = "cuda"


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
    app.state.backend = backend or build_backend(settings.backend_name, settings.checkpoint_dir, settings.device)
    app.state.cache = IdempotencyCache()
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    app.state.ready = True

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
            parsed = validate_predict_payload(payload, settings.model_version)
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
                predictions = []
                for sample in parsed.samples:
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
                    predictions.append(
                        {"sample_id": sample.sample_id, "frames": [encode_png_base64(frame) for frame in frames]}
                    )
                response = {
                    "api_version": API_VERSION,
                    "request_id": parsed.request_id,
                    "model_version": settings.model_version,
                    "predictions": predictions,
                }
            except Exception:
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
    )
