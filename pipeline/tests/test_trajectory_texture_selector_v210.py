import torch
from wam_pipeline.trajectory_texture_selector_v210 import TrajectoryPatchSelector,temporally_smoothed_labels,parameter_count


def test_shapes_gradients_and_budget():
    model=TrajectoryPatchSelector();logits=model(torch.rand(2,8,3,64,64),torch.rand(2,8,5,3,64,64),torch.rand(2,8,14))
    assert logits.shape==(2,8,5,8,8);logits.square().mean().backward();assert model.network[-1].weight.grad is not None
    assert 100_000<parameter_count()<1_000_000


def test_viterbi_prefers_temporal_consistency():
    cost=torch.ones(1,8,5,2,2);cost[:,:,0]=0.2;cost[:,::2,1]=0.19
    labels=temporally_smoothed_labels(cost,switch_penalty=.05)
    assert torch.equal(labels,torch.zeros_like(labels))
