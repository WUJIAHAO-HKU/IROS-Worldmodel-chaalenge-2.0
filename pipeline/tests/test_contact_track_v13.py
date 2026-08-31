import cv2
import numpy as np

from wam_pipeline.contact_track_v13 import ContactTrackParameters, ContactTrackV13


def scene(gripper_width=31):
    image = np.full((256, 256, 3), 242, dtype=np.uint8)
    cv2.ellipse(image, (112, 174), (43, 15), -12, 0, 360, (55, 165, 58), -1)
    cv2.rectangle(image, (166 - gripper_width, 154), (166, 184), (22, 22, 22), -1)
    return image


def test_sequence_latch_restores_all_frames():
    source = scene(31)
    predictions = np.stack([scene(width) for width in (28, 24, 12, 9, 11, 14, 25, 28)])
    output, diagnostics = ContactTrackV13().render_sequence(
        predictions,
        np.repeat(source[None], 5, axis=0),
        ContactTrackParameters(
            minimum_source_area=1,
            maximum_source_area=10000,
            minimum_query_area=1,
            latch_area_ratio=0.8,
        ),
    )
    assert diagnostics["accepted"]
    assert diagnostics["accepted_frame_fraction"] == 1.0
    assert all(frame["output_area"] >= frame["input_area"] for frame in diagnostics["frames"])
    assert np.abs(output.astype(np.int16) - predictions.astype(np.int16)).sum() > 0

