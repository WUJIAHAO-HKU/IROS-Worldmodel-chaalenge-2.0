#!/usr/bin/env python3
"""Static contracts for the v292 terminal-preserving mirror runtime."""

import numpy as np

from wam_pipeline.v292_terminal_preserving_mirror_runtime import (
    MIRROR_SIGN,
    mirror_actions,
    mirror_prompt,
)


def main() -> None:
    rng = np.random.default_rng(292)
    actions = rng.normal(size=(12, 14)).astype(np.float32)
    assert np.allclose(mirror_actions(mirror_actions(actions)), actions)
    assert np.allclose(
        mirror_actions(actions)[:, :7], actions[:, 7:14] * MIRROR_SIGN
    )
    prompt = "Use the left arm; keep the right arm idle."
    assert mirror_prompt(mirror_prompt(prompt)).lower() == prompt.lower()
    print("V292_TERMINAL_PRESERVING_MIRROR_CONTRACT_OK")


if __name__ == "__main__":
    main()
