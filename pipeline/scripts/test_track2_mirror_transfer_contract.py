#!/usr/bin/env python3
"""Pure contract checks for Track2 evaluation-only mirror transfer."""

import torch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    track2_mirror_bimanual_tensor,
    track2_mirror_env_obs,
    track2_mirror_prompt,
    track2_right_route_mask,
)


def main() -> None:
    values = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.8]]
    )
    mirrored = track2_mirror_bimanual_tensor(values)
    assert torch.allclose(track2_mirror_bimanual_tensor(mirrored), values)
    assert torch.allclose(mirrored[0, :7], values[0, 7:14] * values.new_tensor([-1, 1, 1, 1, -1, -1, 1]))

    prompt = "Use the left arm; keep the bottle upright with the RIGHT arm unavailable."
    swapped = track2_mirror_prompt(prompt)
    assert "right arm" in swapped and "left arm unavailable" in swapped
    assert "upright" in swapped
    assert track2_mirror_prompt(swapped).lower() == prompt.lower()

    image = torch.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3)
    obs = {
        "states": values,
        "main_images": image,
        "wrist_images": None,
        "extra_view_images": None,
        "task_descriptions": ["use the right arm"],
    }
    mirrored_obs = track2_mirror_env_obs(obs)
    assert torch.equal(mirrored_obs["main_images"], image.flip(2))
    assert mirrored_obs["task_descriptions"] == ["use the left arm"]
    assert torch.allclose(
        track2_mirror_env_obs(mirrored_obs)["states"], values
    )

    states = torch.zeros((2, 14))
    actions = torch.zeros((2, 8, 14))
    actions[0, :, :6] = 0.2
    actions[1, :, 7:13] = 0.2
    assert track2_right_route_mask(actions, states).tolist() == [False, True]
    assert track2_right_route_mask(actions, states, margin=0.3).tolist() == [False, False]
    print("TRACK2_MIRROR_TRANSFER_CONTRACT_OK")


if __name__ == "__main__":
    main()
