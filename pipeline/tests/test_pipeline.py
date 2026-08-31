from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

import httpx
import numpy as np
import torch

from wam_pipeline.backends import SyntheticActionBackend
from wam_pipeline.contracts import validate_predict_payload
from wam_pipeline.data import Trajectory, make_windows
from wam_pipeline.images import (
    decode_png_base64_batch,
    encode_png_base64,
    encode_png_base64_batch,
)
from wam_pipeline.mb_rl import run_smoke
from wam_pipeline.profile import ACTION_DIM, API_VERSION, OFFICIAL_PROFILE_ID
from wam_pipeline.service import ServiceSettings, create_app
from wam_pipeline.rlinf_bridge.server import Track2RLinfBridge
from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


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


def test_parallel_png_codec_is_byte_and_pixel_exact():
    images = [
        np.random.default_rng(index).integers(
            0, 256, size=(256, 256, 3), dtype=np.uint8
        )
        for index in range(8)
    ]
    sequential = encode_png_base64_batch(images, workers=1)
    parallel = encode_png_base64_batch(images, workers=4)
    assert parallel == sequential
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(
            decode_png_base64_batch(parallel, workers=4), images
        )
    )


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


def test_service_parallel_batch_preserves_sample_order_and_bounds_workers():
    class TrackingBackend:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def predict(self, context, history, future, seed, instruction):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            return np.full((8, 256, 256, 3), seed, dtype=np.uint8)

    backend = TrackingBackend()
    app = create_app(
        ServiceSettings("test-model", "token", batch_workers=3), backend
    )

    async def run():
        payload = _payload()
        template = payload["samples"][0]
        payload["samples"] = []
        for index in range(7):
            sample = json.loads(json.dumps(template))
            sample["sample_id"] = f"sample-{index}"
            sample["seed"] = index
            payload["samples"].append(sample)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/predict",
                json=payload,
                headers={"Authorization": "Bearer token"},
            )
        assert response.status_code == 200
        assert [item["sample_id"] for item in response.json()["predictions"]] == [
            f"sample-{index}" for index in range(7)
        ]

    asyncio.run(run())
    assert 2 <= backend.max_active <= 3


def test_service_uses_native_batch_and_preserves_sample_order():
    class NativeBatchBackend:
        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0

        def predict(self, context, history, future, seed, instruction):
            self.single_calls += 1
            return np.full((8, 256, 256, 3), seed, dtype=np.uint8)

        def predict_batch(self, context, history, future, seeds, instructions):
            self.batch_calls += 1
            assert context.shape == (5, 5, 256, 256, 3)
            assert history.shape == (5, 4, ACTION_DIM)
            assert future.shape == (5, 8, ACTION_DIM)
            assert instructions == [None] * 5
            return np.stack(
                [np.full((8, 256, 256, 3), seed, dtype=np.uint8) for seed in seeds]
            )

    backend = NativeBatchBackend()
    app = create_app(
        ServiceSettings("test-model", "token", native_batch_enabled=True), backend
    )

    async def run():
        payload = _payload()
        template = payload["samples"][0]
        payload["samples"] = []
        for index in range(5):
            sample = json.loads(json.dumps(template))
            sample["sample_id"] = f"sample-{index}"
            sample["seed"] = index
            payload["samples"].append(sample)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/predict",
                json=payload,
                headers={"Authorization": "Bearer token"},
            )
        assert response.status_code == 200
        decoded = response.json()["predictions"]
        assert [item["sample_id"] for item in decoded] == [
            f"sample-{index}" for index in range(5)
        ]

    asyncio.run(run())
    assert backend.batch_calls == 1
    assert backend.single_calls == 0


def test_service_bounds_native_batch_micro_size_without_changing_request_limit():
    class NativeBatchBackend:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, context, history, future, seed, instruction):
            raise AssertionError("single inference should not be used")

        def predict_batch(self, context, history, future, seeds, instructions):
            self.batch_sizes.append(len(context))
            return np.stack(
                [np.full((8, 256, 256, 3), seed, dtype=np.uint8) for seed in seeds]
            )

    backend = NativeBatchBackend()
    app = create_app(
        ServiceSettings(
            "test-model",
            "token",
            native_batch_enabled=True,
            native_batch_micro_size=2,
        ),
        backend,
    )

    async def run():
        payload = _payload()
        template = payload["samples"][0]
        payload["samples"] = []
        for index in range(5):
            sample = json.loads(json.dumps(template))
            sample["sample_id"] = f"sample-{index}"
            sample["seed"] = index
            payload["samples"].append(sample)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/predict",
                json=payload,
                headers={"Authorization": "Bearer token"},
            )
        assert response.status_code == 200

    asyncio.run(run())
    assert backend.batch_sizes == [2, 2, 1]


def test_track2_client_shards_environment_batch_to_official_batch_limit(monkeypatch):
    calls = []

    class Response:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    def fake_post(url, *, headers, json, timeout):
        calls.append(json)
        predictions = []
        for sample in json["samples"]:
            index = int(sample["sample_id"].split("-")[-1])
            frame = _image(index)
            predictions.append(
                {"sample_id": sample["sample_id"], "frames": [frame] * 8}
            )
        return Response(
            {
                "api_version": API_VERSION,
                "request_id": json["request_id"],
                "model_version": json["model_version"],
                "predictions": predictions,
            }
        )

    monkeypatch.setattr("wam_pipeline.rlinf_bridge.track2_client.requests.post", fake_post)
    batch = 17
    client = Track2ServiceClient("http://test", "token", "test-model")
    prediction = client.predict_batch(
        np.zeros((batch, 5, 256, 256, 3), dtype=np.uint8),
        np.zeros((batch, 4, ACTION_DIM), dtype=np.float32),
        np.zeros((batch, 8, ACTION_DIM), dtype=np.float32),
        np.arange(batch, dtype=np.int64),
        [None] * batch,
    )
    assert [len(call["samples"]) for call in calls] == [8, 8, 1]
    assert prediction.shape == (batch, 8, 256, 256, 3)
    assert [int(prediction[index, 0, 0, 0, 0]) for index in range(batch)] == list(
        range(batch)
    )


def test_rlinf_bridge_temporal_update(tmp_path):
    class Client:
        def __init__(self):
            self.histories = []

        def predict_batch(self, context, history, future, seeds, instructions):
            assert context.shape == (1, 5, 256, 256, 3)
            assert history.shape == (1, 4, ACTION_DIM)
            assert future.shape == (1, 8, ACTION_DIM)
            self.histories.append(history.copy())
            # This makes the expected temporal result inspectable without HTTP.
            return np.full((1, 8, 256, 256, 3), 128, dtype=np.uint8)

    audit_dir = tmp_path / "bridge_audit"
    client = Client()
    bridge = Track2RLinfBridge(client, audit_dir=audit_dir)
    current = torch.zeros((1, 3, 1, 5, 256, 256), dtype=torch.float32)
    condition = torch.arange(5 * ACTION_DIM, dtype=torch.float32).reshape(1, 5, ACTION_DIM)
    bridge.reset({"current_obs": current, "condition_action": condition, "task_descriptions": ["adjust bottle"]})
    first_actions = torch.ones((1, 8, ACTION_DIM), dtype=torch.float32)
    output = bridge.chunk_step(first_actions)
    assert output["elapsed_steps"] == 8
    assert output["current_obs"].shape == (1, 3, 1, 13, 256, 256)
    assert bridge.state.condition_action.shape == (1, 5, ACTION_DIM)
    assert torch.equal(bridge.state.condition_action[:, :1], condition[:, :1])
    assert torch.equal(bridge.state.condition_action[:, 1:], torch.ones((1, 4, ACTION_DIM)))
    assert np.array_equal(client.histories[0], condition[:, :4].numpy())
    second_actions = torch.full((1, 8, ACTION_DIM), 2.0, dtype=torch.float32)
    bridge.chunk_step(second_actions)
    assert np.array_equal(client.histories[1], first_actions[:, -4:].numpy())
    audit_path = audit_dir / "rollout_000000.npz"
    assert audit_path.is_file()
    with np.load(audit_path, allow_pickle=False) as audit:
        assert audit["context_frames"].shape == (1, 5, 256, 256, 3)
        assert audit["history_actions"].shape == (1, 4, ACTION_DIM)
        assert audit["future_actions"].shape == (1, 8, ACTION_DIM)
        assert audit["predicted_frames"].shape == (1, 8, 256, 256, 3)
        assert audit["seeds"].tolist() == [0]
        assert json.loads(str(audit["instructions_json"])) == ["adjust bottle"]


def test_rlinf_bridge_can_bound_diagnostic_audit_without_changing_rollout(tmp_path):
    class Client:
        def predict_batch(self, context, history, future, seeds, instructions):
            return np.full((1, 8, 256, 256, 3), 128, dtype=np.uint8)

    audit_dir = tmp_path / "bounded_bridge_audit"
    bridge = Track2RLinfBridge(Client(), audit_dir=audit_dir, audit_max_items=1)
    current = torch.zeros((1, 3, 1, 5, 256, 256), dtype=torch.float32)
    condition = torch.zeros((1, 5, ACTION_DIM), dtype=torch.float32)
    bridge.reset(
        {
            "current_obs": current,
            "condition_action": condition,
            "task_descriptions": ["adjust bottle"],
        }
    )
    actions = torch.zeros((1, 8, ACTION_DIM), dtype=torch.float32)
    first = bridge.chunk_step(actions)
    second = bridge.chunk_step(actions)
    assert first["elapsed_steps"] == 8
    assert second["elapsed_steps"] == 16
    assert sorted(path.name for path in audit_dir.glob("rollout_*.npz")) == [
        "rollout_000000.npz"
    ]
