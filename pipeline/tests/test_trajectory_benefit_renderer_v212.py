import torch
from wam_pipeline.trajectory_benefit_renderer_v212 import TrajectoryBenefitRenderer, render, parameter_count


def test_shapes_identity_and_budget():
    model = TrajectoryBenefitRenderer()
    base = torch.rand(1, 8, 3, 32, 32)
    selected = torch.rand_like(base)
    probabilities = torch.rand(1, 8, 5, 4, 4).softmax(2)
    logits = model(base, selected, probabilities, torch.rand(1, 8, 14))
    assert logits.shape == (1, 8, 6, 4, 4)
    output, alpha = render(base, selected, logits)
    assert output.shape == base.shape and alpha.shape == (1, 8, 1, 32, 32)
    assert torch.equal(output[:, :3], base[:, :3])
    logits.mean().backward()
    assert any(value.grad is not None for value in model.parameters())
    assert parameter_count() < 1_000_000
