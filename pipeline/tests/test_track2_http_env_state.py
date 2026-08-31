"""Regression tests for Track 2 policy proprioception."""

from __future__ import annotations

import torch

from wam_pipeline.rlinf_bridge.track2_http_env import Track2HttpEnv


def test_wrap_obs_uses_latest_absolute_action_as_policy_state() -> None:
    env = object.__new__(Track2HttpEnv)
    env.device = torch.device("cpu")
    env.current_obs = torch.zeros((2, 3, 1, 5, 4, 4), dtype=torch.float32)
    env.condition_action = torch.zeros((2, 5, 14), dtype=torch.float32)
    env.policy_state = torch.arange(2 * 14, dtype=torch.float32).reshape(2, 14)
    env.task_descriptions = ["left", "right"]
    env.use_active_arm_loss_mask = False

    obs = env._wrap_obs()

    expected = env.policy_state
    assert torch.equal(obs["states"], expected)
    assert obs["states"].data_ptr() != expected.data_ptr()
    assert obs["main_images"].shape == (2, 4, 4, 3)
