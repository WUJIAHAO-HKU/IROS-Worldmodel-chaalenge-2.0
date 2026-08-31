import numpy as np
import torch

try:
    from wam_pipeline.object_geometry_v170 import (
        ActionPoseProjector,
        ActionLayerProjector,
        RelativeLayerProjector,
        affine_from_landmarks,
        project_endpose,
        quaternion_matrix_wxyz,
        oracle_update,
        mask_landmarks,
    )
except ModuleNotFoundError:
    from object_geometry_v170 import (
        ActionPoseProjector,
        ActionLayerProjector,
        RelativeLayerProjector,
        affine_from_landmarks,
        project_endpose,
        quaternion_matrix_wxyz,
        oracle_update,
        mask_landmarks,
    )


def test_quaternion_and_projection_identity() -> None:
    np.testing.assert_allclose(quaternion_matrix_wxyz(np.asarray([1, 0, 0, 0])), np.eye(3))
    intrinsic = np.asarray([[100, 0, 160], [0, 100, 120], [0, 0, 1]], np.float32)
    extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1).astype(np.float32)
    value = project_endpose(np.asarray([0, 0, 1, 1, 0, 0, 0], np.float32),
                            intrinsic, extrinsic)
    np.testing.assert_allclose(value[:2], [128, 128], atol=1e-5)
    assert value[6] == 1.0


def test_affine_from_landmarks_translation() -> None:
    source = np.asarray([10, 20, 20, 20, 10, 30, 1], np.float32)
    target = source.copy(); target[0:6:2] += 7; target[1:6:2] -= 3
    matrix = affine_from_landmarks(source, target)
    np.testing.assert_allclose(matrix, [[1, 0, 7], [0, 1, -3]], atol=1e-5)


def test_two_arm_model_shape_and_budget() -> None:
    model = ActionPoseProjector(hidden=64)
    value = model(torch.zeros(6, 7), torch.tensor([0, 1, 0, 1, 0, 1]))
    assert value.shape == (6, 7)
    assert sum(parameter.numel() for parameter in model.parameters()) < 200_000


def test_oracle_update_never_increases_selected_pixel_error() -> None:
    target = np.full((2, 2, 3), 100, np.uint8)
    parent = np.asarray([[[90] * 3, [110] * 3], [[100] * 3, [80] * 3]], np.uint8)
    candidate = np.asarray([[[99] * 3, [130] * 3], [[0] * 3, [101] * 3]], np.uint8)
    output, selected = oracle_update(parent, target, candidate, np.ones((2, 2), bool))
    assert selected.tolist() == [[True, False], [False, True]]
    assert np.abs(output.astype(int) - target.astype(int)).sum() < np.abs(
        parent.astype(int) - target.astype(int)
    ).sum()


def test_mask_landmarks_and_layer_projector() -> None:
    mask = np.zeros((64, 64), np.uint8); mask[20:30, 10:50] = 1
    value = mask_landmarks(mask)
    assert value is not None and value.shape == (7,)
    np.testing.assert_allclose(value[:2], [29.5, 24.5], atol=1e-5)
    model = ActionLayerProjector(hidden=32)
    result = model(torch.zeros(6, 7), torch.tensor([0, 0, 0, 1, 1, 1]),
                   torch.tensor([0, 1, 2, 0, 1, 2]))
    assert result.shape == (6, 7)
    relative = RelativeLayerProjector(hidden=32)
    moved = relative(torch.zeros(6, 22), torch.tensor([0, 0, 0, 1, 1, 1]),
                     torch.tensor([0, 1, 2, 0, 1, 2]))
    assert moved.shape == (6, 7)
