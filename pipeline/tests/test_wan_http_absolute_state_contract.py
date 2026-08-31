from __future__ import annotations

import numpy as np
import torch

from rlinf.envs.world_model.world_model_wan_http_env import WanHttpProxyEnv


class _FakeHttpClient:
    def __init__(self, current_obs: torch.Tensor) -> None:
        self.current_obs = current_obs
        self.received: torch.Tensor | None = None

    def chunk_step(self, actions: torch.Tensor):
        self.received = actions.clone()
        return {"current_obs": self.current_obs.clone()}


def bare_env() -> WanHttpProxyEnv:
    env = WanHttpProxyEnv.__new__(WanHttpProxyEnv)
    env.num_envs = 2
    env.action_dim = 14
    env.chunk = 8
    env.device = torch.device("cpu")
    env.image_size = (256, 256)
    env.current_obs = torch.zeros(2, 3, 1, 5, 256, 256)
    env.task_descriptions = ["left", "right"]
    env.condition_action = torch.arange(2 * 5 * 14, dtype=torch.float32).reshape(2, 5, 14)
    return env


def test_wrap_obs_uses_latest_absolute_joint_state():
    env = bare_env()
    obs = env._wrap_obs()
    assert torch.equal(obs["states"], env.condition_action[:, -1])
    assert obs["states"].shape == (2, 14)


def test_chunk_update_preserves_absolute_actions_for_next_policy_call():
    env = bare_env()
    expected_frames = torch.ones_like(env.current_obs)
    env._http_client = _FakeHttpClient(expected_frames)
    actions = torch.linspace(-0.7, 1.0, 2 * 8 * 14).reshape(2, 8, 14)
    env._infer_next_chunk_frames(actions)
    assert torch.equal(env._http_client.received, actions)
    assert torch.equal(env.current_obs, expected_frames)
    assert torch.equal(env.condition_action[:, 1:], actions[:, -4:])
    assert torch.equal(env._wrap_obs()["states"], actions[:, -1])


def test_reference_action_injection_aligns_initial_history_without_moving_state():
    env = bare_env()
    original_latest = env.condition_action[:, -1].clone()
    action0 = torch.linspace(-0.4, 0.9, 14)
    action1 = torch.linspace(0.2, 1.0, 14)
    env.dataset = [
        {"start_items": [{"action": action0}]},
        {"start_items": [{"action": action1}]},
    ]
    env._inject_reference_actions(np.asarray([1, 0], dtype=np.int64))
    assert torch.equal(env.condition_action[0, 0], action1)
    assert torch.equal(env.condition_action[1, 0], action0)
    assert torch.equal(env.condition_action[:, -1], original_latest)
