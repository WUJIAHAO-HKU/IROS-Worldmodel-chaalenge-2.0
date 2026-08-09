"""Reward shaping helpers for long-horizon Track 2 policy training."""

from __future__ import annotations

import torch


def shape_progress_rewards(
    chunk_scores: torch.Tensor,
    previous_score: torch.Tensor,
    *,
    delta_weight: float = 1.0,
    terminal_weight: float = 0.0,
    peak_weight: float = 0.0,
    positive_delta_weight: float = 0.0,
    reward_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert official per-frame scores into stable progress rewards.

    The returned tensor keeps the original ``[batch, chunk]`` credit-assignment
    shape.  ``previous_score`` is the score of the observation immediately
    before the chunk, so consecutive chunks telescope without an artificial
    zero-reward baseline.
    """

    if chunk_scores.ndim != 2:
        raise ValueError("chunk_scores must be [batch, chunk]")
    if previous_score.shape != chunk_scores.shape[:1]:
        raise ValueError("previous_score must be [batch]")
    if not torch.isfinite(chunk_scores).all() or not torch.isfinite(previous_score).all():
        raise ValueError("reward inputs must be finite")
    if reward_clip is not None and reward_clip <= 0:
        raise ValueError("reward_clip must be positive")

    deltas = torch.empty_like(chunk_scores)
    deltas[:, 0] = chunk_scores[:, 0] - previous_score
    deltas[:, 1:] = chunk_scores[:, 1:] - chunk_scores[:, :-1]
    shaped = float(delta_weight) * deltas
    if positive_delta_weight:
        shaped = shaped + float(positive_delta_weight) * deltas.clamp_min(0)

    start_to_terminal = chunk_scores[:, -1] - previous_score
    start_to_peak = chunk_scores.max(dim=1).values - previous_score
    shaped[:, -1] += float(terminal_weight) * start_to_terminal
    shaped[:, -1] += float(peak_weight) * start_to_peak.clamp_min(0)
    if reward_clip is not None:
        shaped = shaped.clamp(-float(reward_clip), float(reward_clip))
    return shaped, chunk_scores[:, -1].detach()
