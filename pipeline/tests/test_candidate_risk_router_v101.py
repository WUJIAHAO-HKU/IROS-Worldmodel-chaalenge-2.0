import torch

from wam_pipeline.candidate_risk_router_v10 import CandidateRiskRouterV10
from wam_pipeline.candidate_risk_router_v101 import CandidateRiskRouterV101


def inputs():
    return (
        torch.rand(1, 5, 3, 256, 256),
        torch.rand(1, 8, 3, 256, 256),
        torch.rand(1, 8, 5, 3, 256, 256),
        torch.randn(1, 8, 5, 2, 256, 256),
        torch.randn(1, 8, 5, 256, 256),
        torch.randn(1, 8, 7),
        torch.tensor([0]),
    )


def test_v101_adds_four_finite_observable_channels():
    model = CandidateRiskRouterV101(base_channels=16).eval()
    values = inputs()
    with torch.inference_mode():
        features = model.observable_features(*values[:5])
        result = model(*values)
    assert features.shape == (1, 8, 5, 27, 64, 64)
    assert torch.isfinite(features).all()
    assert result["advantage_mean"].shape == (1, 8, 5, 64, 64)


def test_zero_initialized_v10_upgrade_is_exact():
    torch.manual_seed(11)
    old = CandidateRiskRouterV10(base_channels=16).eval()
    upgraded = CandidateRiskRouterV101(base_channels=16).eval()
    upgraded.load_v10_state_dict(old.state_dict())
    values = inputs()
    with torch.inference_mode():
        old_result = old(*values)
        new_result = upgraded(*values)
    assert torch.equal(old_result["advantage_mean"], new_result["advantage_mean"])
    assert torch.equal(old_result["advantage_uncertainty"], new_result["advantage_uncertainty"])
