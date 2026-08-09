import torch

try:
    from wam_pipeline.dual_tiny_experts_v150 import (
        TinyBlackGripperExpert, TinyGlyphMotionExpert, parameter_counts,
        render_black_gripper, render_glyph,
    )
except ModuleNotFoundError:
    from dual_tiny_experts_v150 import (
        TinyBlackGripperExpert, TinyGlyphMotionExpert, parameter_counts,
        render_black_gripper, render_glyph,
    )


def inputs(batch=2, time=8, size=96):
    source = torch.rand(batch, 3, size, size)
    parent = torch.rand(batch, time, 3, size, size)
    actions = torch.rand(batch, time, 7)
    arm = torch.tensor([0, 1][:batch])
    mask = torch.zeros(batch, 1, size, size); mask[:, :, 40:45, 30:60] = 1
    sequence_mask = mask[:, None].expand(-1, time, -1, -1, -1).clone()
    return source, parent, actions, arm, mask, sequence_mask


def test_glyph_shapes_and_identity():
    source, parent, actions, arm, glyph, beam = inputs()
    model = TinyGlyphMotionExpert(8)
    matrix, confidence = model(source, parent, actions, arm, glyph, beam[:, 0], beam)
    output, alpha = render_glyph(parent, source, glyph, beam, matrix, confidence, strength=0)
    assert matrix.shape == (2, 8, 2, 3)
    assert confidence.shape == (2, 8)
    assert alpha.shape == (2, 8, 1, 96, 96)
    assert torch.equal(output, parent)


def test_gripper_shapes_bounded_and_identity():
    source, parent, actions, arm, black, parent_black = inputs()
    bottle = torch.zeros_like(parent_black)
    model = TinyBlackGripperExpert(8)
    logits, residual = model(source, parent, actions, arm, black, parent_black, bottle)
    output, probability, support = render_black_gripper(
        parent, black, parent_black, logits, residual, strength=0
    )
    assert logits.shape == (2, 8, 1, 96, 96)
    assert residual.shape == (2, 8, 3, 96, 96)
    assert probability.shape == support.shape == logits.shape
    assert torch.equal(output, parent)


def test_parameter_budget_is_small():
    counts = parameter_counts()
    assert 10_000 < counts["glyph"] < 1_000_000
    assert 10_000 < counts["black_gripper"] < 1_000_000
