import numpy as np
import torch

try:
    from wam_pipeline.geometry_visibility_router_v172 import (
        CANDIDATE_COUNT, GeometryVisibilityRouterV172, hard_render,
        parameter_count, routing_targets,
    )
except ModuleNotFoundError:
    from geometry_visibility_router_v172 import (
        CANDIDATE_COUNT, GeometryVisibilityRouterV172, hard_render,
        parameter_count, routing_targets,
    )


def inputs(size=64):
    parent = torch.zeros(2, 3, size, size)
    candidates = torch.zeros(2, CANDIDATE_COUNT, 3, size, size)
    candidates[:, 0, :, 20:30, 25:35] = 1
    support = torch.zeros(2, CANDIDATE_COUNT, 5, size, size)
    support[:, 0, 0, 20:30, 25:35] = 1
    condition = torch.zeros(2, CANDIDATE_COUNT, 5)
    horizon = torch.tensor([[.25], [.75]])
    arm = torch.tensor([[0.], [1.]])
    return parent, candidates, support, condition, horizon, arm


def test_router_shapes_and_gradients():
    values = inputs(); model = GeometryVisibilityRouterV172(8)
    score = model(*values)
    assert score.shape == (2, CANDIDATE_COUNT, 1, 64, 64)
    score.mean().backward()
    assert any(value.grad is not None for value in model.parameters())


def test_targets_and_hard_parent_fallback():
    parent, candidates, support, condition, horizon, arm = inputs()
    target = parent.clone(); target[:, :, 20:30, 25:35] = 1
    labels, focus, gain = routing_targets(parent, candidates, support, target)
    assert torch.all(labels[:, 20:30, 25:35] == 1)
    assert focus[:, 20:30, 25:35].all() and gain.shape[1] == CANDIDATE_COUNT
    scores = torch.full((2, CANDIDATE_COUNT, 1, 64, 64), -1.0)
    output, choice = hard_render(parent, candidates, support, scores)
    assert torch.equal(output, parent) and not choice.any()
    scores[:, 0, :, 20:30, 25:35] = 2
    output, choice = hard_render(parent, candidates, support, scores)
    assert torch.all(output[:, :, 20:30, 25:35] == 1)
    assert torch.all(choice[:, 20:30, 25:35] == 1)


def test_parameter_budget_is_small():
    assert 100_000 < parameter_count(24) < 2_000_000
