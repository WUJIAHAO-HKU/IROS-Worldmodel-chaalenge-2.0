import numpy as np

from evaluate_contact_motion_retrieval_v140 import (
    feather,
    photometric_calibration,
    reproject_observed_glyph,
    temporal_ramp,
)


def test_feather_has_zero_border_and_full_interior():
    value = feather(160, 180, 16)
    assert value.shape == (160, 180)
    assert np.all(value[[0, -1]] == 0)
    assert np.all(value[:, [0, -1]] == 0)
    assert value[80, 90] == 1


def test_input_only_photometric_calibration_recovers_brightness_shift():
    candidate = np.full((256, 256, 3), 190, np.uint8)
    query = np.full((256, 256, 3), 202, np.uint8)
    gain, bias = photometric_calibration(candidate, query, np.eye(2, 3, dtype=np.float32))
    calibrated = candidate.astype(np.float32) * gain + bias
    assert np.abs(calibrated.mean(axis=(0, 1)) - 202).max() < 0.6


def test_temporal_ramp_preserves_prefix_and_crossfades_branch():
    assert np.allclose(temporal_ramp(8, 3, 2), [0, 0, 0, 0.5, 1, 1, 1, 1])
    assert np.allclose(temporal_ramp(8, 3, 1), [0, 0, 0, 1, 1, 1, 1, 1])


def test_observed_glyph_reprojection_restores_local_texture():
    observed = np.full((256, 256, 3), 238, np.uint8)
    crop = observed[0:128, 128:256]
    crop[42:75, 38:126] = 35
    # Three separated bright components form a conservative logo run.
    crop[52:65, 56:61] = 235
    crop[52:65, 68:73] = 235
    crop[52:65, 80:85] = (45, 70, 235)
    query = observed.copy()
    query_crop = query[0:128, 128:256]
    query_crop[50:68, 52:90] = 35
    result, detail = reproject_observed_glyph(query, observed, texture_alpha=1.0,
                                               highpass_strength=0.0)
    assert detail["accepted"]
    assert detail["edited_fraction"] > 0
    assert result[52:65, 128 + 56:128 + 85].mean() > query[52:65, 128 + 56:128 + 85].mean()
