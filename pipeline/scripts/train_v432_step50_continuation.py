#!/usr/bin/env python3
"""Deterministically replay v432 steps 1..50 from the immutable parent.

The base trainer does not serialize AdamW state, so a weight-only step25
resume cannot preserve optimizer semantics.  This wrapper instead runs one
uninterrupted optimizer and seed-1582 sampling stream from the original
parent, saving anchors at steps 25 and 50.  The launcher must byte-verify the
replayed step25 before it may accept or audit step50.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import train_multichunk_reward_aligned_autoregressive_unet as base
import train_v432_public_mirror_prompt_terminal as v432


EXPECTED_STEP25_MODEL_SHA256 = "dff072aff2f5c64261f9f968cd9ae436440132edbe06a6cf9e1bd2549cdadf7f"


def _argument(name: str) -> str:
    if name not in sys.argv or sys.argv.index(name) + 1 >= len(sys.argv):
        raise SystemExit(f"v432 replay requires {name}")
    return sys.argv[sys.argv.index(name) + 1]


def main() -> None:
    if int(_argument("--steps")) != 50:
        raise SystemExit("v432 replay must run exactly 50 total optimizer steps")
    if int(_argument("--checkpoint-interval")) != 25:
        raise SystemExit("v432 replay must save deterministic anchors at steps 25 and 50")
    if int(_argument("--seed")) != 1582:
        raise SystemExit("v432 replay must preserve seed 1582")
    initialization = Path(_argument("--init-checkpoint"))
    if initialization.name != "checkpoint_step_000150":
        raise SystemExit("v432 replay must initialize from the original parent step150")
    if not (initialization / "model.pt").is_file():
        raise FileNotFoundError(initialization / "model.pt")

    original_save = base.save

    def save_replay_anchor(output: Path, model, mean, std, metadata: dict) -> None:
        if output.name not in {"checkpoint_step_000025", "checkpoint_step_000050"}:
            raise RuntimeError(f"unexpected replay checkpoint request: {output}")
        amended = copy.deepcopy(metadata)
        amended["deterministic_replay"] = {
            "format": "strict-track2-v432-deterministic-replay50-v1",
            "initialization": "original immutable parent checkpoint_step_000150",
            "optimizer": "single uninterrupted AdamW across steps 1..50",
            "sampler": "single uninterrupted seed1582 sampling stream across steps 1..50",
            "checkpoint_interval": 25,
            "expected_step25_model_sha256": EXPECTED_STEP25_MODEL_SHA256,
            "step25_semantics": "identity anchor only; launcher must verify before accepting step50",
            "step50_semantics": "forbidden unless replayed step25 model hash is exact",
        }
        original_save(output, model, mean, std, amended)

    base.save = save_replay_anchor
    v432.main()


if __name__ == "__main__":
    main()
