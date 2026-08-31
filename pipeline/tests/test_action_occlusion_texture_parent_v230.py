import torch

from wam_pipeline.action_occlusion_texture_parent_v230 import (
    ActionOcclusionTextureParentV230,
    parameter_count,
)


def inputs(batch=2, size=64):
    base = torch.rand(batch, 3, size, size)
    previous = torch.rand_like(base)
    memory = torch.rand(batch, 5, 3, size, size)
    action = torch.randn(batch, 14)
    pose = torch.zeros(batch, 6)
    horizon = torch.full((batch, 1), 0.5)
    arm = torch.tensor([0, 1])[:batch]
    return base, previous, memory, action, pose, horizon, arm


def test_identity_shape_gradient_and_parameter_budget():
    model = ActionOcclusionTextureParentV230()
    values = inputs()
    output, detail = model(*values)
    assert output.shape == values[0].shape
    assert torch.equal(output, values[0])
    assert detail["flow"].shape == (2, 2, 64, 64)
    assert detail["structure_residual"].shape == values[0].shape
    output.mean().backward()
    assert model.motion_head.weight.grad is not None
    assert model.texture_head.weight.grad is not None
    assert 250_000 < parameter_count() < 2_000_000


def test_pose_heatmaps_are_spatial_not_broadcast():
    pose = torch.tensor([[0.0, 0.0, -0.5, 0.5, 0.75, -0.75]])
    maps = ActionOcclusionTextureParentV230._pose_heatmaps(pose, size=64)
    assert maps.shape == (1, 3, 64, 64)
    maxima = maps.flatten(2).argmax(2)
    ys, xs = maxima // 64, maxima % 64
    expected = ((pose.reshape(1, 3, 2) + 1) * 0.5 * 63).round().long()
    assert torch.all((xs - expected[..., 0]).abs() <= 1)
    assert torch.all((ys - expected[..., 1]).abs() <= 1)


def test_recurrent_state_changes_the_next_prediction_input():
    model = ActionOcclusionTextureParentV230()
    with torch.no_grad():
        model.motion_head.bias[2] = 0.25
        model.motion_head.bias[6] = 2.0
    values = list(inputs(batch=1))
    first, _ = model(*values)
    values[1] = first
    second, _ = model(*values)
    assert not torch.equal(first, values[0])
    assert not torch.equal(second, values[0])
