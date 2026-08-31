import torch

from wam_pipeline.hybrid_transport_router import HybridTransportRouter


def test_active_arm_factorization_selects_changing_side():
    actions = torch.zeros(2, 12, 14)
    actions[0, :, 2] = torch.arange(12)
    actions[1, :, 10] = torch.arange(12)
    active, arm = HybridTransportRouter.active_arm_actions(actions)
    assert arm.tolist() == [0, 1]
    assert torch.equal(active[0, :, 2], torch.arange(4, 12))
    assert torch.equal(active[1, :, 3], torch.arange(4, 12))


def test_router_initialization_copies_parent_exactly():
    torch.manual_seed(0)
    model = HybridTransportRouter(base_channels=8).eval()
    candidates = torch.rand(1, 8, 7, 3, 256, 256)
    context = torch.rand(1, 5, 3, 256, 256)
    actions = torch.rand(1, 12, 14)
    active, arm = model.active_arm_actions(actions)
    with torch.no_grad():
        prediction, logits, weights = model(candidates, context, active, arm)
    assert prediction.shape == (1, 8, 3, 256, 256)
    assert logits.shape == weights.shape == (1, 8, 7, 64, 64)
    assert torch.equal(prediction, candidates[:, :, 0])
