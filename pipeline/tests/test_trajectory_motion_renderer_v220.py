import torch
from wam_pipeline.trajectory_motion_renderer_v220 import TrajectoryMotionRenderer,render_motion,parameter_count


def test_shape_identity_gradient_and_budget():
    model=TrajectoryMotionRenderer();base=torch.rand(1,8,3,32,32);warped=torch.rand(1,8,5,3,32,32);logits=model(base,warped,torch.rand(1,8,14),torch.rand(1,8,6));assert logits.shape==(1,8,22,4,4);output,beta=render_motion(base,warped,logits);assert torch.equal(output,base);assert not beta.any();logits.mean().backward();assert any(p.grad is not None for p in model.parameters());assert parameter_count()<1_000_000
