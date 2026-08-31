import torch

from wam_pipeline.layered_object_state_parent_v240 import (
    LayeredObjectStateParentV240, parameter_count,
)


def inputs(size=32):
    base = torch.rand(1, 8, 3, size, size)
    transported = torch.rand_like(base)
    support = torch.zeros(1, 8, 4, size, size)
    support[:, :, :, 8:24, 8:24] = 1
    cleanup = torch.rand_like(base)
    cleanup_support = torch.zeros(1, 8, 1, size, size)
    actions = torch.rand(1, 8, 7)
    poses = torch.rand(1, 8, 6) * 2 - 1
    arms = torch.zeros(1, dtype=torch.long)
    return base, transported, support, cleanup, cleanup_support, actions, poses, arms


def test_identity_initialization_and_shapes():
    model = LayeredObjectStateParentV240(channels=16)
    values = inputs()
    output, detail = model(*values)
    assert output.shape == values[0].shape
    assert detail["masks"].shape == (1, 8, 4, 32, 32)
    assert detail["flows"].shape == (1, 8, 4, 2, 32, 32)
    assert torch.equal(output, values[0])


def test_teacher_forced_full_rollout_is_differentiable():
    model = LayeredObjectStateParentV240(channels=16)
    values = inputs()
    labels = values[2].clone()
    output, detail = model(*values, target_masks=labels, teacher_forcing=.5)
    (output.mean() + detail["masks"].mean()).backward()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


def test_parameter_budget():
    assert 100_000 < parameter_count(48) < 2_000_000
