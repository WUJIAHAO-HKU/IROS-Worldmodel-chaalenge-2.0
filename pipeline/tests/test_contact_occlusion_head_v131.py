import torch

from wam_pipeline.contact_occlusion_head_v131 import ContactOcclusionHeadV131


def test_dual_arm_shape_gradient_and_expert_selection():
    model = ContactOcclusionHeadV131(base_channels=16)
    last = torch.rand(2, 3, 64, 72); parent = torch.rand(2, 8, 3, 64, 72)
    actions = torch.rand(2, 8, 7); arms = torch.tensor([0, 1])
    output = model(last, parent, actions, arms)
    assert output.shape == (2, 8, 3, 64, 72)
    output[:, -1].mean().backward()
    assert model.arm_residual[0][-1].weight.grad is not None
    assert model.arm_residual[1][-1].weight.grad is not None


def test_arm_identity_changes_output():
    torch.manual_seed(1); model = ContactOcclusionHeadV131(base_channels=16).eval()
    last = torch.rand(1, 3, 32, 36); parent = torch.rand(1, 8, 3, 32, 36); actions = torch.rand(1, 8, 7)
    with torch.no_grad():
        left = model(last, parent, actions, torch.tensor([0]))
        right = model(last, parent, actions, torch.tensor([1]))
    assert not torch.equal(left, right)
