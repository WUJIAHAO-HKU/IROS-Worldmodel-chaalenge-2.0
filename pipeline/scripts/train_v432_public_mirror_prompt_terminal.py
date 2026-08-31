#!/usr/bin/env python3
"""Thin v432 adapter around the audited multi-chunk reward-aligned trainer.

The base trainer remains the implementation of recursive 4x8 rollout and the
frozen official reward loss.  This adapter changes only the public-train40
sampling view: 70% real-right non-terminal sequences, 20% mirrored public-left
sequences, and 10% real-right terminal sequences.  It also doubles the first
8-frame visual loss, matching the exact chunk consumed by the official API.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import torch

import train_multichunk_reward_aligned_autoregressive_unet as base


MIRROR_SIGN = torch.tensor([-1, 1, 1, 1, -1, -1, 1], dtype=torch.float32)
MIX_UNITS = {"real_right": 7, "mirrored_left": 2, "terminal_real_right": 1}


def _mirror_joint14(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 14:
        raise ValueError(f"expected joint14, got {tuple(values.shape)}")
    sign = MIRROR_SIGN.to(dtype=values.dtype, device=values.device)
    output = torch.empty_like(values)
    output[..., :7] = values[..., 7:14] * sign
    output[..., 7:14] = values[..., :7] * sign
    return output


def _right_prompt(text: str) -> str:
    rewritten = re.sub(r"left arm", "right arm", str(text), flags=re.IGNORECASE)
    lowered = rewritten.lower()
    if "right arm" not in lowered or "left arm" in lowered:
        raise ValueError(f"mirrored prompt is not explicitly right-arm-only: {text!r}")
    return rewritten


class V432PublicMirrorWindows(base.MultiChunkWindows):
    """Deterministic virtual mixture over the fixed public train40 sequences."""

    def __init__(self, *args, **kwargs) -> None:
        # Build both arms first.  Runtime use is right-only after physical mirror.
        kwargs["arm_filter"] = "all"
        super().__init__(*args, **kwargs)
        source_sequences = list(self.sequences)
        source_arm_right = self.arm_right.copy()
        source_capture = self.capture_success.copy()
        source_start = self.start.copy()
        source_episode = self.episode_id.copy()
        source_seed = self.synthetic_seed.copy()

        max_start_by_episode: dict[int, int] = {}
        for episode, start, is_right in zip(source_episode, source_start, source_arm_right):
            if is_right:
                max_start_by_episode[int(episode)] = max(
                    int(start), max_start_by_episode.get(int(episode), -1)
                )
        terminal = np.asarray(
            [
                bool(is_right) and int(start) == max_start_by_episode[int(episode)]
                for episode, start, is_right in zip(
                    source_episode, source_start, source_arm_right
                )
            ],
            dtype=np.bool_,
        )
        pools = {
            "real_right": np.flatnonzero(source_arm_right & ~terminal).tolist(),
            "mirrored_left": np.flatnonzero(~source_arm_right).tolist(),
            "terminal_real_right": np.flatnonzero(terminal).tolist(),
        }
        if any(not values for values in pools.values()):
            raise ValueError(f"v432 requires three non-empty sampling pools: {pools}")
        unit = max(len(values) for values in pools.values())
        virtual: list[tuple[int, bool, str]] = []
        for category, units in MIX_UNITS.items():
            pool = pools[category]
            count = units * unit
            virtual.extend(
                (pool[index % len(pool)], category == "mirrored_left", category)
                for index in range(count)
            )

        # A fixed permutation avoids long category runs while preserving the
        # exact 70/20/10 virtual cardinality before WeightedRandomSampler.
        generator = np.random.default_rng(1582)
        order = generator.permutation(len(virtual))
        virtual = [virtual[int(index)] for index in order]
        self._mirrored = np.asarray([row[1] for row in virtual], dtype=np.bool_)
        self.v432_category = np.asarray([row[2] for row in virtual])
        source_index = np.asarray([row[0] for row in virtual], dtype=np.int64)
        self.sequences = [source_sequences[index] for index in source_index]
        self.arm_right = np.ones(len(virtual), dtype=np.bool_)
        self.capture_success = source_capture[source_index]
        self.start = source_start[source_index]
        self.episode_id = source_episode[source_index]
        self.synthetic_seed = source_seed[source_index]
        self.v432_contract = {
            "format": "strict-track2-v432-public-mirror-mixture-v1",
            "virtual_sequences": len(virtual),
            "unit": unit,
            "category_counts": {
                key: int((self.v432_category == key).sum()) for key in MIX_UNITS
            },
            "real_right_source_sequences": len(pools["real_right"]),
            "mirrored_left_source_sequences": len(pools["mirrored_left"]),
            "terminal_real_right_source_sequences": len(pools["terminal_real_right"]),
            "terminal_real_right_episodes": len(max_start_by_episode),
        }

    def __getitem__(self, index: int):
        context, history, future, target, arm_right, capture, prompt = super().__getitem__(index)
        if self._mirrored[index]:
            context = torch.flip(context, dims=(2,))
            target = torch.flip(target, dims=(2,))
            history = _mirror_joint14(history)
            future = _mirror_joint14(future)
            prompt = _right_prompt(prompt)
        lowered = str(prompt).lower()
        if "right arm" not in lowered or "left arm" in lowered:
            raise ValueError(f"v432 prompt/action contract failed: {prompt!r}")
        return context, history, future, target, torch.tensor(True), capture, prompt


def _argument_value(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    return int(sys.argv[sys.argv.index(name) + 1])


def main() -> None:
    chunks = _argument_value("--chunks", 4)
    if chunks != 4:
        raise SystemExit("v432 is preregistered for exactly four 8-frame chunks")
    original_visual_loss = base.visual_loss
    call_index = 0

    def first8_weighted_visual_loss(*args, **kwargs):
        nonlocal call_index
        chunk_index = call_index % chunks
        call_index += 1
        value = original_visual_loss(*args, **kwargs)
        return 2.0 * value if chunk_index == 0 else value

    base.MultiChunkWindows = V432PublicMirrorWindows
    base.visual_loss = first8_weighted_visual_loss
    base.main()


if __name__ == "__main__":
    main()
