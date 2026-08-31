import torch

from wam_pipeline.candidate_risk_router_v10 import CandidateRiskRouterV10


def test_candidate_risk_router_shapes_and_finite_outputs():
    model = CandidateRiskRouterV10(base_channels=16).eval()
    context = torch.rand(1, 5, 3, 256, 256)
    parent = torch.rand(1, 8, 3, 256, 256)
    transported = torch.rand(1, 8, 5, 3, 256, 256)
    flow = torch.randn(1, 8, 5, 2, 256, 256)
    visibility = torch.randn(1, 8, 5, 256, 256)
    actions = torch.randn(1, 8, 7)
    arm = torch.tensor([1])
    with torch.inference_mode():
        result = model(context, parent, transported, flow, visibility, actions, arm)
    assert result["advantage_mean"].shape == (1, 8, 5, 64, 64)
    assert result["advantage_uncertainty"].shape == (1, 8, 5, 64, 64)
    assert torch.isfinite(result["advantage_mean"]).all()
    assert (result["advantage_uncertainty"] > 0).all()


def test_safe_choice_uses_uncertainty_and_parent_fallback():
    mean = torch.tensor([[[[[2.0]], [[1.0]]]]])
    uncertainty = torch.tensor([[[[[0.7]], [[0.1]]]]])
    choice, score = CandidateRiskRouterV10.safe_choice(mean, uncertainty, risk_weight=2.0, margin=0.7)
    assert choice.item() == 2
    assert torch.allclose(score, torch.tensor([[[[0.8]]]]))
    choice, _ = CandidateRiskRouterV10.safe_choice(mean, uncertainty, risk_weight=2.0, margin=0.9)
    assert choice.item() == 0
