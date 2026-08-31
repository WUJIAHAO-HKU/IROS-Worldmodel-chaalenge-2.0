import torch

from wam_pipeline.autoregressive_texture_memory_v190 import OneStepActionTextureMemoryUNet, parameter_count


def test_shapes_persistent_memory_and_gradients():
    model = OneStepActionTextureMemoryUNet()
    context = torch.rand(1, 5, 3, 64, 64); memory = torch.rand_like(context)
    output, detail = model(context, torch.randn(1, 5, 14), memory, True)
    assert output.shape == (1, 3, 64, 64)
    assert detail["flow"].shape == (1, 5, 2, 64, 64)
    assert detail["source_weight"].shape == (1, 5, 64, 64)
    output.mean().backward(); assert model.memory_head.weight.grad is not None


def test_parameter_budget():
    assert 8_000_000 < parameter_count() < 15_000_000


def test_source_expert_logits_and_gradients():
    model=OneStepActionTextureMemoryUNet(use_source_expert=True)
    _,detail=model(torch.rand(1,5,3,32,32),torch.randn(1,5,14),return_diagnostics=True)
    assert detail["source_logits"].shape==(1,5,32,32)
    detail["source_logits"].square().mean().backward()
    assert model.source_expert[-1].weight.grad is not None
