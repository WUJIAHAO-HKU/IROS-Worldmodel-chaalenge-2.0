import torch
from torch import nn

from wam_pipeline.protected_layered_flow_v91 import ProtectedLayeredFlowV91


class DummyFlow(nn.Module):
    def forward(self, context, actions, return_flow=False):
        batch, _, _, height, width = context.shape
        flow = context.new_zeros(batch, 8, 5, 2, height, width)
        weights = context.new_full((batch, 8, 5, 1, height, width), 0.2)
        prediction = context[:, -1:, None].expand(-1, 8, 5, -1, -1, -1).mean(dim=2)
        return (prediction, flow, weights) if return_flow else prediction

    @staticmethod
    def _warp(images, flow):
        batch, steps, sources = flow.shape[:3]
        return images[:, None].expand(-1, steps, -1, -1, -1, -1)


def test_v91_initialization_is_exactly_parent_protected():
    torch.manual_seed(0)
    model = ProtectedLayeredFlowV91(DummyFlow(), base_channels=8).eval()
    context = torch.rand(1, 5, 3, 256, 256)
    parent = torch.rand(1, 8, 3, 256, 256)
    actions = torch.rand(1, 12, 14)
    active, arm = model.active_arm_actions(actions)
    with torch.no_grad():
        result = model(context, actions, active, arm, parent)
    assert result["candidates"].shape == (1, 8, 6, 3, 256, 256)
    assert result["refined_flow"].shape == (1, 8, 5, 2, 256, 256)
    assert result["route_logits"].shape == (1, 8, 6, 64, 64)
    assert torch.equal(result["prediction"], parent)
    assert torch.count_nonzero(result["residual_flow"]) == 0
