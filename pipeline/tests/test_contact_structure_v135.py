import numpy as np
import torch

from wam_pipeline.contact_structure_head_v135 import ContactStructureHeadV135
from wam_pipeline.contact_structure_v135 import structure_semantic_mask


def test_four_class_shape_and_temporal_gradient():
    model = ContactStructureHeadV135(base_channels=16)
    last = torch.rand(2, 3, 32, 36); parent = torch.rand(2, 8, 3, 32, 36)
    actions = torch.rand(2, 8, 7); arms = torch.tensor([0, 1])
    source = torch.rand(2, 3, 32, 36); prior = torch.rand(2, 8, 3, 32, 36)
    output = model(last, parent, actions, arms, source, prior)
    assert output.shape == (2, 8, 4, 32, 36)
    output[:, -1].mean().backward()
    assert model.recurrent.gates.weight.grad is not None


def test_labels_keep_black_and_grey_as_separate_layers():
    frame = np.full((1, 96, 96, 3), 240, np.uint8)
    frame[0, 30:78, 38:68] = (35, 170, 38)  # green bottle
    frame[0, 42:65, 27:42] = 24             # black contact pad
    frame[0, 25:48, 22:34] = 135            # grey jaw touching black
    label = structure_semantic_mask(frame)[0]
    assert (label == 1).sum() > 200
    assert (label == 2).sum() > 100
    assert (label == 3).sum() > 50
