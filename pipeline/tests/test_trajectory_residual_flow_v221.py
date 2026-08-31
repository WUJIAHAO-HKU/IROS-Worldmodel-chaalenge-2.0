import torch
from wam_pipeline.trajectory_residual_flow_v221 import TrajectoryResidualFlow,parameter_count


def test_identity_shape_gradient_and_budget():
    model=TrajectoryResidualFlow();base=torch.rand(1,8,3,32,32);output,detail=model(base,torch.rand(1,3,32,32),torch.rand(1,8,14),torch.rand(1,8,6));assert output.shape==base.shape;assert torch.allclose(output,base,atol=2e-6);assert detail["flow"].shape==(1,8,2,32,32);output.mean().backward();assert any(p.grad is not None for p in model.parameters());assert parameter_count()<1_000_000


def test_temporal_difference_axis_has_eight_transitions():
    sequence=torch.rand(2,9,3,7,11)
    assert sequence.diff(dim=1).shape==(2,8,3,7,11)
