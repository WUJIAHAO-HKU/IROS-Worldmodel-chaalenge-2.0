from __future__ import annotations

import asyncio
import uuid

import httpx
import numpy as np
import torch

from wam_pipeline.backends import SyntheticActionBackend
from wam_pipeline.contracts import validate_predict_payload
from wam_pipeline.data import Trajectory, make_windows
from wam_pipeline.images import encode_png_base64
from wam_pipeline.mb_rl import run_smoke
from wam_pipeline.profile import ACTION_DIM, API_VERSION, OFFICIAL_PROFILE_ID
from wam_pipeline.service import ServiceSettings, create_app
from wam_pipeline.rlinf_bridge.server import Track2RLinfBridge


def _image(value: int) -> dict:
    return encode_png_base64(np.full((256, 256, 3), value, dtype=np.uint8))


def _payload() -> dict:
    return {
        "api_version": API_VERSION,
        "request_id": str(uuid.uuid4()),
        "model_version": "test-model",
        "profile_id": OFFICIAL_PROFILE_ID,
        "samples": [
            {
                "sample_id": "sample",
                "seed": 7,
                "context": {
                    "frames": [_image(index) for index in range(5)],
                    "actions": np.zeros((4, ACTION_DIM), dtype=np.float32).tolist(),
                    "states": None,
                },
                "actions": np.zeros((8, ACTION_DIM), dtype=np.float32).tolist(),
            }
        ],
    }


def test_window_alignment():
    frames = np.zeros((15, 256, 256, 3), dtype=np.uint8)
    actions = np.arange(15 * ACTION_DIM, dtype=np.float32).reshape(15, ACTION_DIM)
    window = make_windows(Trajectory(frames, actions, "test"))[0]
    assert np.array_equal(window.history_actions, actions[:4])
    assert np.array_equal(window.future_actions, actions[4:12])
    assert len(window.target_frames) == 8


def test_synthetic_backend_and_smoke():
    frames = np.zeros((13, 256, 256, 3), dtype=np.uint8)
    actions = np.ones((13, ACTION_DIM), dtype=np.float32) * 0.1
    window = make_windows(Trajectory(frames, actions, "test"))[0]
    backend = SyntheticActionBackend()
    first = backend.predict(window.context_frames, window.history_actions, window.future_actions, 3, None)
    second = backend.predict(window.context_frames, window.history_actions, window.future_actions, 3, None)
    assert np.array_equal(first, second)
    result = run_smoke(window, backend, rounds=2)
    assert result.generated_frames == 16
    assert np.isfinite(result.reward)


def test_service_idempotency_and_contract():
    app = create_app(ServiceSettings("test-model", "token"), SyntheticActionBackend())

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer token"}
            payload = _payload()
            response = await client.post("/v1/predict", json=payload, headers=headers)
            assert response.status_code == 200
            assert len(response.json()["predictions"][0]["frames"]) == 8
            retry = await client.post("/v1/predict", json=payload, headers=headers)
            assert retry.json() == response.json()
            payload["samples"][0]["seed"] = 8
            conflict = await client.post("/v1/predict", json=payload, headers=headers)
            assert conflict.status_code == 409

    asyncio.run(run())


def test_rlinf_bridge_temporal_update():
    class Client:
        def predict_batch(self, context, history, future, seeds, instructions):
            assert context.shape == (1, 5, 256, 256, 3)
            assert history.shape == (1, 4, ACTION_DIM)
            assert future.shape == (1, 8, ACTION_DIM)
            # This makes the expected temporal result inspectable without HTTP.
            return np.full((1, 8, 256, 256, 3), 128, dtype=np.uint8)

    bridge = Track2RLinfBridge(Client())
    current = torch.zeros((1, 3, 1, 5, 256, 256), dtype=torch.float32)
    condition = torch.full((1, 5, ACTION_DIM), -2.0, dtype=torch.float32)
    bridge.reset({"current_obs": current, "condition_action": condition, "task_descriptions": ["adjust bottle"]})
    output = bridge.chunk_step(torch.ones((1, 8, ACTION_DIM), dtype=torch.float32))
    assert output["elapsed_steps"] == 8
    assert output["current_obs"].shape == (1, 3, 1, 13, 256, 256)
    assert bridge.state.condition_action.shape == (1, 5, ACTION_DIM)
    assert torch.equal(bridge.state.condition_action[:, :1], condition[:, :1])
    assert torch.equal(bridge.state.condition_action[:, 1:], torch.ones((1, 4, ACTION_DIM)))
