"""Policy-probability diagnostics for Track 2 PPO/GRPO updates."""

from __future__ import annotations

import torch


BEHAVIOR_LOGPROB_SOURCES = ("rollout", "actor_recomputed")


def select_behavior_logprobs(
    current_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    source: str,
) -> torch.Tensor:
    """Select the frozen PPO denominator without detaching the numerator."""

    if source not in BEHAVIOR_LOGPROB_SOURCES:
        raise ValueError(
            f"behavior logprob source must be one of {BEHAVIOR_LOGPROB_SOURCES}, got {source!r}"
        )
    if current_logprobs.shape != rollout_logprobs.shape:
        raise ValueError("current and rollout logprobs must have identical shapes")
    if source == "actor_recomputed":
        return current_logprobs.detach()
    return rollout_logprobs.detach()


def probability_consistency_metrics(
    current_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    *,
    action_dim: int,
    loss_mask: torch.Tensor | None = None,
    clip_ratio_low: float = 0.1,
    clip_ratio_high: float = 0.1,
) -> dict[str, float]:
    """Measure rollout/actor density disagreement per action dimension.

    RLinf's standard token-level KL and clip metrics aggregate the action
    dimension against an unexpanded mask. These diagnostics instead use the
    fully expanded valid-element count and report the two 7-DoF arms
    separately when ``action_dim == 14``.
    """

    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if current_logprobs.shape != rollout_logprobs.shape:
        raise ValueError("current and rollout logprobs must have identical shapes")
    if current_logprobs.numel() == 0 or current_logprobs.numel() % action_dim:
        raise ValueError("logprob element count must be divisible by action_dim")
    if not torch.isfinite(current_logprobs).all() or not torch.isfinite(
        rollout_logprobs
    ).all():
        raise ValueError("policy logprobs must be finite")
    if clip_ratio_low < 0 or clip_ratio_high < 0:
        raise ValueError("clip ratios must be non-negative")

    current = current_logprobs.reshape(current_logprobs.shape[0], -1, action_dim)
    rollout = rollout_logprobs.reshape_as(current)
    if loss_mask is None:
        valid = torch.ones_like(current, dtype=torch.bool)
    else:
        valid = loss_mask.to(device=current.device, dtype=torch.bool)
        while valid.ndim < current.ndim:
            valid = valid.unsqueeze(-1)
        try:
            valid = valid.expand_as(current)
        except RuntimeError as error:
            raise ValueError("loss_mask cannot be broadcast to logprobs") from error

    valid_count = valid.count_nonzero()
    if not bool(valid_count):
        raise ValueError("loss_mask selects no policy logprob elements")
    log_ratio = current.detach().float() - rollout.detach().float()
    selected = log_ratio[valid]
    ratio = selected.clamp(-20.0, 20.0).exp()
    clipped = (ratio < 1.0 - float(clip_ratio_low)) | (
        ratio > 1.0 + float(clip_ratio_high)
    )

    metrics = {
        "actor/prob_audit_log_ratio_mean": selected.mean(),
        "actor/prob_audit_log_ratio_abs_mean": selected.abs().mean(),
        "actor/prob_audit_log_ratio_abs_max": selected.abs().max(),
        "actor/prob_audit_ratio_mean": ratio.mean(),
        "actor/prob_audit_ratio_abs_mean": (ratio - 1.0).abs().mean(),
        "actor/prob_audit_clip_fraction": clipped.float().sum() / valid_count,
    }

    if action_dim == 14:
        for side, arm_slice in (("left", slice(0, 7)), ("right", slice(7, 14))):
            arm_valid = valid[..., arm_slice]
            arm_values = log_ratio[..., arm_slice][arm_valid]
            metrics[f"actor/prob_audit_{side}_log_ratio_abs_mean"] = (
                arm_values.abs().mean()
            )
            metrics[f"actor/prob_audit_{side}_log_ratio_abs_max"] = (
                arm_values.abs().max()
            )
    # EmbodiedFSDPActor aggregates metric lists with NumPy, which cannot
    # consume CUDA tensors. Convert at the worker boundary instead of forcing
    # every caller to remember a device transfer.
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}
