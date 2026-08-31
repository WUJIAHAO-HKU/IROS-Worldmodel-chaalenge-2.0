import torch

from wam_pipeline.contact_occlusion_head_v132 import ContactOcclusionHeadV132


def test_parent_prior_preserves_observed_gripper_and_gradients():
    model = ContactOcclusionHeadV132(base_channels=16, prior_strength=2.0)
    last = torch.rand(2, 3, 32, 36); parent = torch.rand(2, 8, 3, 32, 36)
    actions = torch.rand(2, 8, 7); arms = torch.tensor([0, 1])
    source = torch.zeros(2, 2, 32, 36); prior = torch.zeros(2, 8, 2, 32, 36)
    prior[:, :, 1, 10:20, 12:24] = 1
    output = model(last, parent, actions, arms, source, prior)
    assert output.shape == (2, 8, 3, 32, 36)
    learned = torch.zeros(2, 3, 32, 36)
    added = model.add_parent_prior(learned, prior[:, 0])
    assert torch.equal(added[:, 2, 10:20, 12:24], torch.full((2, 10, 12), 2.0))
    output[:, -1].mean().backward(); assert model.recurrent.gates.weight.grad is not None
