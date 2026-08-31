#!/usr/bin/env python3
"""Contracts for the compliant v290 world-model request transformation."""

import numpy as np

from wam_pipeline.v290_right_closed_mirror_runtime import (
    MIRROR_SIGN,
    mirror_actions,
    mirror_prompt,
    route_right,
    use_right_closed_mirror,
)


def main() -> None:
    rng = np.random.default_rng(290)
    actions = rng.normal(size=(8, 14)).astype(np.float32)
    assert np.allclose(mirror_actions(mirror_actions(actions)), actions)
    assert np.allclose(mirror_actions(actions)[:, :7], actions[:, 7:14] * MIRROR_SIGN)
    prompt = "Use the left arm and keep the bottle upright; right arm idle."
    assert mirror_prompt(mirror_prompt(prompt)).lower() == prompt.lower()

    history = np.zeros((4, 14), dtype=np.float32)
    right = np.zeros((8, 14), dtype=np.float32)
    right[:, 7:13] = np.linspace(0, 1, 8)[:, None]
    right[:, 6] = 1.0
    right[:, 13] = 0.0
    assert route_right(history, right)
    assert use_right_closed_mirror(history, right)

    right_open = right.copy()
    right_open[:, 13] = 1.0
    assert route_right(history, right_open)
    assert not use_right_closed_mirror(history, right_open)

    left = mirror_actions(right)
    assert not route_right(mirror_actions(history), left)
    assert not use_right_closed_mirror(mirror_actions(history), left)
    print("V290_RIGHT_CLOSED_MIRROR_CONTRACT_OK")


if __name__ == "__main__":
    main()
