import cv2
import numpy as np

from wam_pipeline.contact_layer_v12 import ContactLayerV12, ContactParameters, bottle_mask, gripper_mask


def scene(offset_x=0):
    image = np.full((256, 256, 3), 242, dtype=np.uint8)
    cv2.ellipse(image, (112 + offset_x, 174), (43, 15), -12, 0, 360, (55, 165, 58), -1)
    cv2.rectangle(image, (135 + offset_x, 154), (166 + offset_x, 184), (22, 22, 22), -1)
    return image


def test_masks_detect_contact_layers():
    image = scene()
    bottle = bottle_mask(image[72:232, 30:210])
    gripper = gripper_mask(image[72:232, 30:210], bottle)
    assert bottle.sum() > 500
    assert gripper.sum() > 300


def test_rigid_contact_reconstruction_restores_gripper():
    source = scene()
    query = scene(5)
    query[150:190, 135:175] = (70, 135, 75)
    cv2.rectangle(query, (153, 162), (169, 181), (45, 45, 45), -1)
    history = np.repeat(source[None], 5, axis=0)
    output, diagnostics = ContactLayerV12().render(
        query,
        history,
        ContactParameters(
            minimum_gripper_area=1,
            minimum_query_gripper_area=1,
            minimum_ecc=0.0,
            minimum_bottle_overlap=0.1,
            maximum_gripper_area_ratio=2.0,
        ),
    )
    assert diagnostics["accepted"]
    assert output[160:180, 142:170].mean() < query[160:180, 142:170].mean()


def test_hard_trimap_produces_opaque_contact_classes():
    source = scene()
    query = scene(5)
    query[150:190, 135:175] = (70, 135, 75)
    cv2.rectangle(query, (153, 162), (169, 181), (45, 45, 45), -1)
    output, diagnostics = ContactLayerV12().render(
        query,
        np.repeat(source[None], 5, axis=0),
        ContactParameters(
            minimum_gripper_area=1,
            minimum_query_gripper_area=1,
            maximum_gripper_area_ratio=2.0,
            reconstruction_mode="hard_trimap",
        ),
    )
    assert diagnostics["accepted"]
    crop = output[153:182, 153:170]
    neutral = crop.max(axis=2) - crop.min(axis=2) < 8
    assert neutral.mean() > 0.6
