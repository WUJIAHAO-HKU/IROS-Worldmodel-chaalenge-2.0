import torch

from wam_pipeline.contact_rgb_refiner_v138 import ContactRGBRefinerV138


def test_identity_initialization_and_gradients():
    model = ContactRGBRefinerV138(base_channels=16)
    last = torch.rand(2, 3, 32, 36)
    parent = torch.rand(2, 8, 3, 32, 36)
    actions = torch.rand(2, 8, 7)
    arms = torch.tensor((0, 1))
    source_semantic = torch.zeros(2, 3, 32, 36)
    parent_semantic = torch.zeros(2, 8, 3, 32, 36)
    result = model(last, parent, actions, arms, source_semantic,
                   parent_semantic, return_details=True)
    assert result["prediction"].shape == parent.shape
    assert torch.allclose(result["prediction"], parent, atol=0.06)
    result["prediction"].mean().backward()
    assert model.output.weight.grad is not None
