import torch

from wam_pipeline.direct_overfit_diagnostic_v250 import DirectOverfitDiagnosticV250, parameter_count


def test_shape_identity_and_gradient():
    model = DirectOverfitDiagnosticV250(8)
    base = torch.rand(1, 8, 3, 32, 32)
    output = model(base, torch.rand(1, 5, 3, 32, 32), torch.rand(1, 8, 7), torch.zeros(1).long())
    assert output.shape == base.shape
    assert torch.equal(output, base)
    output.mean().backward()
    assert model.head.weight.grad is not None


def test_parameter_budget():
    assert 1_000_000 < parameter_count(32) < 10_000_000
