"""RLinf's local `/reset` and `/chunk_step` transport backed by `/v1/predict`.

The bridge is intentionally separate from the submission service.  RLinf's Wan
HTTP prototype serializes PyTorch tensors with pickle, while the contest API is
stateless JSON+PNG.  This server consumes the former locally and calls the
latter exactly once per world-model rollout chunk; rewards stay in RLinf.
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from .track2_client import Track2ServiceClient


def _decode_payload(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _encode_payload(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


@dataclass
class BridgeState:
    current_obs: torch.Tensor | None = None  # [B,3,1,T,256,256], values in [-1,1]
    condition_action: torch.Tensor | None = None  # [B,5,14]
    task_descriptions: list[str] = field(default_factory=list)
    elapsed_steps: int = 0
    chunk_index: int = 0


class Track2RLinfBridge:
    """State adapter with the exact temporal update used by the RLinf Wan env."""

    def __init__(self, client: Track2ServiceClient) -> None:
        self.client = client
        self.state = BridgeState()
        self.lock = threading.Lock()

    @staticmethod
    def _to_uint8_frames(current_obs: torch.Tensor) -> np.ndarray:
        if current_obs.ndim != 6 or current_obs.shape[1:3] != (3, 1) or current_obs.shape[-2:] != (256, 256):
            raise ValueError("current_obs must be [B,3,1,T,256,256]")
        frames = current_obs[:, :, 0, -5:].permute(0, 2, 3, 4, 1)
        return ((frames.float().cpu().numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        current_obs = payload.get("current_obs")
        condition_action = payload.get("condition_action")
        if not isinstance(current_obs, torch.Tensor) or not isinstance(condition_action, torch.Tensor):
            raise ValueError("RLinf reset payload is missing tensors")
        if current_obs.shape[3] != 5 or condition_action.shape != (current_obs.shape[0], 5, 14):
            raise ValueError("RLinf reset payload does not match the Track 2 5-frame/14D profile")
        with self.lock:
            self.state = BridgeState(
                current_obs=current_obs.detach().cpu().float().contiguous(),
                condition_action=condition_action.detach().cpu().float().contiguous(),
                task_descriptions=[str(value) for value in payload.get("task_descriptions", [])],
                elapsed_steps=int(payload.get("elapsed_steps", 0)),
            )
        return {"status": "reset"}

    def chunk_step(self, actions: Any) -> dict[str, Any]:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions.detach().cpu().float().contiguous()
        with self.lock:
            if self.state.current_obs is None or self.state.condition_action is None:
                raise RuntimeError("reset must be called before chunk_step")
            batch = self.state.current_obs.shape[0]
            if actions.shape != (batch, 8, 14):
                raise ValueError("RLinf action chunk must be [B,8,14]")
            context = self._to_uint8_frames(self.state.current_obs)
            history = self.state.condition_action[:, -4:].numpy()
            instructions = self.state.task_descriptions or [None] * batch
            if len(instructions) != batch:
                instructions = [None] * batch
            seeds = np.arange(batch, dtype=np.int64) + self.state.chunk_index * batch
            prediction = self.client.predict_batch(context, history, actions.numpy(), seeds, instructions)
            generated = torch.from_numpy(prediction).permute(0, 4, 1, 2, 3).float().div(127.5).sub(1.0).unsqueeze(2)
            current = torch.cat([self.state.current_obs, generated], dim=3)
            current = current[:, :, :, -13:].contiguous()
            self.state.current_obs = current
            # Wan reserves the first condition slot for the reference frame.  The
            # remaining four slots are the actions that produced the five-frame
            # context used by the next request: u4 through u7 after one chunk.
            self.state.condition_action[:, 1:, :] = actions[:, -4:, :]
            self.state.elapsed_steps += 8
            self.state.chunk_index += 1
            return {"current_obs": current, "elapsed_steps": self.state.elapsed_steps}


def create_app(bridge: Track2RLinfBridge) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/reset")
    def reset(body: dict[str, str]):
        try:
            return {"payload": _encode_payload(bridge.reset(_decode_payload(body["payload"]))) }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/chunk_step")
    def chunk_step(body: dict[str, str]):
        try:
            request = _decode_payload(body["payload"])
            return {"payload": _encode_payload(bridge.chunk_step(request["actions"]))}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Bridge RLinf Wan HTTP transport to a Track 2 API service")
    parser.add_argument("--world-model-url", default=os.environ.get("WAM_API_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--token", default=os.environ.get("WAM_BEARER_TOKEN", "local-dev-token"))
    parser.add_argument("--model-version", default=os.environ.get("WAM_MODEL_VERSION"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if not args.model_version:
        raise SystemExit("--model-version or WAM_MODEL_VERSION is required")
    client = Track2ServiceClient(args.world_model_url, args.token, args.model_version, args.timeout)
    client.assert_ready()
    uvicorn.run(create_app(Track2RLinfBridge(client)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
