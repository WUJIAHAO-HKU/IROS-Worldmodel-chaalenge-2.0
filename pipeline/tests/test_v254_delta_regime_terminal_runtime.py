from wam_pipeline.v254_delta_regime_terminal_runtime import Track2V254DeltaRegimeTerminal as V254


def test_clean_and_ood_regimes_accept_good_direction():
    assert V254._regime_alpha(.005, .99, .1) > 0
    assert V254._regime_alpha(.9, .8, .1) > 0


def test_ambiguous_regime_and_bad_direction_are_rejected():
    assert V254._regime_alpha(.2, .99, .1) == 0
    assert V254._regime_alpha(.005, .90, .1) == 0
    assert V254._regime_alpha(.9, .4, .1) == 0
