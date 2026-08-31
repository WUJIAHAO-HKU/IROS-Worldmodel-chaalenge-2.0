import torch

from wam_pipeline.contact_occlusion_head_v13 import ContactOcclusionHeadV13


def test_shape_and_temporal_gradient():
    model = ContactOcclusionHeadV13(base_channels=16)
    last = torch.rand(2, 3, 64, 72)
    parent = torch.rand(2, 8, 3, 64, 72)
    actions = torch.rand(2, 8, 14)
    result = model(last, parent, actions)
    assert result.shape == (2, 8, 3, 64, 72)
    result[:, -1].mean().backward()
    assert model.recurrent.gates.weight.grad is not None
