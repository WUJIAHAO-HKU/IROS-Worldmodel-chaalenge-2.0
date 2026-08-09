import pytest
import torch

from wam_pipeline.rlinf_bridge.reward_shaping import shape_progress_rewards


def test_default_progress_rewards_telescope_to_terminal_gain():
    scores = torch.tensor([[0.20, 0.25, 0.23, 0.30]])
    rewards, next_score = shape_progress_rewards(scores, torch.tensor([0.10]))
    assert rewards.sum().item() == pytest.approx(0.20)
    assert next_score.item() == pytest.approx(0.30)


def test_long_horizon_shaping_rewards_peak_and_terminal_progress():
    scores = torch.tensor([[0.20, 0.25, 0.23, 0.30]])
    rewards, _ = shape_progress_rewards(
        scores,
        torch.tensor([0.10]),
        terminal_weight=1.0,
        peak_weight=0.5,
        positive_delta_weight=0.25,
    )
    expected = 0.20 + 0.20 + 0.10 + 0.25 * (0.10 + 0.05 + 0.07)
    assert rewards.sum().item() == pytest.approx(expected)


def test_reward_shaping_clips_and_validates_inputs():
    rewards, _ = shape_progress_rewards(
        torch.tensor([[0.0, 10.0]]), torch.tensor([0.0]), reward_clip=0.5
    )
    assert rewards.abs().max().item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        shape_progress_rewards(torch.zeros(2, 3), torch.zeros(1))
