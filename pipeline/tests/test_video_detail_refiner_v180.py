import torch

try:
    from wam_pipeline.video_detail_refiner_v180 import VideoDetailRefinerV180, parameter_count
except ModuleNotFoundError:
    from video_detail_refiner_v180 import VideoDetailRefinerV180, parameter_count


def test_identity_initialization_and_shapes():
    model = VideoDetailRefinerV180(8)
    visual = torch.rand(1, 22, 8, 64, 64); actions = torch.randn(1, 8, 7)
    output, alpha = model(visual, actions, torch.tensor([1]))
    assert output.shape == (1, 3, 8, 64, 64) and alpha.shape == (1, 1, 8, 64, 64)
    assert torch.allclose(output, visual[:, :3])
    output.mean().backward(); assert any(value.grad is not None for value in model.parameters())


def test_parameter_budget_uses_available_gpu_without_becoming_parent_sized():
    assert 2_000_000 < parameter_count(32) < 30_000_000
