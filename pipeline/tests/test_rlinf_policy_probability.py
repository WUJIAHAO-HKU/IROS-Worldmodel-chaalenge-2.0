import pytest
import torch

from wam_pipeline.rlinf_bridge.policy_probability import (
    probability_consistency_metrics,
    select_behavior_logprobs,
)


def test_probability_metrics_expand_mask_and_split_arms():
    rollout = torch.zeros((2, 2, 14), dtype=torch.float32)
    current = rollout.clone()
    current[0, 0, :7] = 0.2
    current[0, 0, 7:] = -0.1
    mask = torch.tensor([[True, False], [False, False]])

    metrics = probability_consistency_metrics(
        current,
        rollout,
        action_dim=14,
        loss_mask=mask,
        clip_ratio_low=0.05,
        clip_ratio_high=0.05,
    )

    assert metrics["actor/prob_audit_clip_fraction"] == pytest.approx(1.0)
    assert metrics["actor/prob_audit_left_log_ratio_abs_mean"] == pytest.approx(
        0.2
    )
    assert metrics["actor/prob_audit_right_log_ratio_abs_mean"] == pytest.approx(
        0.1
    )


def test_actor_recomputed_behavior_is_detached_snapshot():
    current = torch.randn(1, 8, 14, requires_grad=True)
    rollout = torch.randn_like(current)

    selected = select_behavior_logprobs(current, rollout, "actor_recomputed")

    assert torch.equal(selected, current)
    assert not selected.requires_grad


def test_probability_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="source"):
        select_behavior_logprobs(torch.zeros(1), torch.zeros(1), "invalid")
    with pytest.raises(ValueError, match="finite"):
        probability_consistency_metrics(
            torch.tensor([[[float("nan")]]]),
            torch.zeros(1, 1, 1),
            action_dim=1,
        )
    with pytest.raises(ValueError, match="selects no"):
        probability_consistency_metrics(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            action_dim=1,
            loss_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
