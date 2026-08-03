"""Local MBRL smoke test driven through the official prediction API contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import ModelBackend
from .data import Window
from .profile import CONTEXT_FRAMES, PREDICTION_FRAMES


@dataclass(frozen=True)
class SmokeResult:
    rounds: int
    reward: float
    policy_mean_action: np.ndarray
    generated_frames: int


class LinearPolicy:
    """Minimal trainable policy used only to prove a model-based update reaches the backend."""

    def __init__(self, action_dim: int = 14) -> None:
        self.mean_action = np.zeros(action_dim, dtype=np.float32)

    def act(self, frame: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(self.mean_action[None, :], horizon, axis=0)

    def update(self, actions: np.ndarray, reward: float, learning_rate: float) -> None:
        # REINFORCE-style directional update is sufficient for a deterministic integration smoke test.
        self.mean_action = self.mean_action + learning_rate * reward * actions.mean(axis=0)


def image_reward(frames: np.ndarray, target: np.ndarray) -> float:
    """A local proxy reward; it is not the official non-public reward model."""
    error = np.abs(frames[-1].astype(np.float32) - target.astype(np.float32)).mean() / 255.0
    return float(1.0 - error)


def run_smoke(window: Window, backend: ModelBackend, rounds: int = 2, learning_rate: float = 0.05) -> SmokeResult:
    """Run policy -> 8 actions -> WM -> proxy reward -> policy update for a few rounds."""
    policy = LinearPolicy(window.future_actions.shape[1])
    frames = window.context_frames.copy()
    history_actions = window.history_actions.copy()
    total_reward = 0.0
    generated_frames = 0
    for round_index in range(rounds):
        # Use the dataset actions as a deterministic exploration direction in this smoke test.
        exploration = window.future_actions * (1.0 + 0.05 * round_index)
        actions = policy.act(frames[-1], PREDICTION_FRAMES) + exploration
        generated = backend.predict(frames, history_actions, actions, seed=round_index, instruction=None)
        target = window.target_frames[min(PREDICTION_FRAMES - 1, round_index)]
        reward = image_reward(generated, target)
        if not np.isfinite(reward):
            raise RuntimeError("local reward is non-finite")
        policy.update(actions, reward, learning_rate)
        total_reward += reward
        generated_frames += len(generated)
        # Advance exactly 8 transitions: last 5 predicted frames and actions u4..u7 align.
        frames = generated[-CONTEXT_FRAMES:]
        history_actions = actions[-(CONTEXT_FRAMES - 1) :]
    return SmokeResult(rounds, total_reward, policy.mean_action, generated_frames)
