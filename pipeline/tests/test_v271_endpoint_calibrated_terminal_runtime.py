from wam_pipeline.v271_endpoint_calibrated_terminal_runtime import (
    Track2V271EndpointCalibratedTerminal as V271,
)


def test_endpoint_quality_rewards_closer_endpoint():
    assert V271._endpoint_quality(0.05, 0.2) > V271._endpoint_quality(0.5, 0.2)


def test_endpoint_quality_rewards_positive_progress():
    assert V271._endpoint_quality(0.2, 0.3) > V271._endpoint_quality(0.2, -0.2)


def test_strong_regression_has_no_terminal_quality():
    assert V271._endpoint_quality(0.2, -0.25) == 0.0

