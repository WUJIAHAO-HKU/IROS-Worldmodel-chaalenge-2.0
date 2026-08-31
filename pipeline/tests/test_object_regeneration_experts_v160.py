import torch

from wam_pipeline.object_regeneration_experts_v160 import (
    GlyphAtlasProjectionExpert, GripperInstanceFlowExpert, parameter_counts,
    render_atlas_glyph, render_gripper_instances,
)


def test_glyph_shapes_and_zero_strength_identity():
    batch, time, height, width = 2, 8, 64, 64
    model = GlyphAtlasProjectionExpert(base_channels=8)
    atlas = torch.rand(batch, 3, height, width)
    parent = torch.rand(batch, time, 3, height, width)
    mask = torch.rand(batch, 1, height, width)
    sequence_mask = torch.rand(batch, time, 1, height, width)
    actions = torch.rand(batch, time, 7)
    arms = torch.tensor((0, 1))
    flow, visibility, edit = model(atlas, parent, actions, arms, mask, sequence_mask, sequence_mask)
    output, alpha, warped = render_atlas_glyph(parent, atlas, mask, sequence_mask,
                                               flow, visibility, edit, strength=0,
                                               parent_clean=torch.zeros_like(parent))
    assert output.shape == parent.shape
    assert alpha.shape == warped.shape == (batch, time, 1, height, width)
    assert torch.equal(output, parent)


def test_gripper_shapes_probabilities_and_zero_strength_identity():
    batch, time, height, width = 2, 8, 80, 90
    model = GripperInstanceFlowExpert(base_channels=8)
    source = torch.rand(batch, 3, height, width)
    parent = torch.rand(batch, time, 3, height, width)
    source_mask = torch.rand(batch, 1, height, width)
    sequence_mask = torch.rand(batch, time, 1, height, width)
    actions = torch.rand(batch, time, 7)
    arms = torch.tensor((0, 1))
    semantic, flow, visibility, replacement, edit = model(
        source, parent, actions, arms, source_mask, source_mask,
        sequence_mask, sequence_mask, sequence_mask)
    output, probability, alpha, warped = render_gripper_instances(
        parent, source, source_mask, source_mask, sequence_mask, sequence_mask,
        semantic, flow, visibility, replacement, edit, strength=0)
    assert output.shape == parent.shape
    assert probability.shape == (batch, time, 3, height, width)
    assert alpha.shape == warped.shape == (batch, time, 1, height, width)
    assert torch.equal(output, parent)
    assert torch.allclose(probability.sum(2), torch.ones_like(probability[:, :, 0]), atol=1e-6)


def test_default_parameter_budget_is_below_two_million():
    counts = parameter_counts()
    assert 0 < counts["glyph_atlas"] < 1_000_000
    assert 0 < counts["gripper_instance_flow"] < 1_000_000
    assert sum(counts.values()) < 2_000_000
