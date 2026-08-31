#!/usr/bin/env python3
"""Static/data-boundary audit for the v432 public-only training view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_v432_public_mirror_prompt_terminal import (
    MIX_UNITS,
    V432PublicMirrorWindows,
    _mirror_joint14,
    _right_prompt,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    train_left = [episode for episode in train if arms[episode] == "left"]
    train_right = [episode for episode in train if arms[episode] == "right"]
    validation_right = [episode for episode in validation if arms[episode] == "right"]
    checks = {
        "format": split.get("format") == "strict-track2-v205-public-demo-40train-10holdout-v1",
        "train40": len(train) == 40,
        "holdout10": len(validation) == 10,
        "episode_disjoint": not bool(set(train) & set(validation)),
        "train_left25": len(train_left) == 25,
        "train_right15": len(train_right) == 15,
        "validation_right4": len(validation_right) == 4,
        "no_hidden_or_final": split.get("hidden_or_final_evaluation_data") is False,
        "all_train_prompts_present": set(train) <= set(prompts),
    }
    for episode in train:
        expected = f"{arms[episode]} arm"
        opposite = "right arm" if arms[episode] == "left" else "left arm"
        text = prompts[episode].lower()
        checks[f"prompt_{episode}"] = expected in text and opposite not in text
    if not all(checks.values()):
        raise RuntimeError("v432 fixed public split/prompt contract failed")

    dataset = V432PublicMirrorWindows(
        args.windows,
        train,
        chunks=4,
        chunk_stride=8,
        arm_filter="right",
        instruction_by_episode=prompts,
    )
    counts = dataset.v432_contract["category_counts"]
    total_units = sum(MIX_UNITS.values())
    proportions = {key: counts[key] / len(dataset) for key in MIX_UNITS}
    expected = {key: value / total_units for key, value in MIX_UNITS.items()}
    mix_exact = all(abs(proportions[key] - expected[key]) < 1e-12 for key in MIX_UNITS)
    terminal_coverage = dataset.v432_contract["terminal_real_right_episodes"] == 15

    # Pure transform involution and prompt checks do not read evaluation data.
    probe = torch.arange(3 * 14, dtype=torch.float32).reshape(3, 14)
    mirror_involution = torch.equal(_mirror_joint14(_mirror_joint14(probe)), probe)
    # Golden vector from the previously audited 7-DoF left/right physical
    # mirror convention: swap arm blocks and apply [-,+,+,+,-,-,+].
    golden_source = torch.arange(14, dtype=torch.float32)
    golden_expected = torch.tensor(
        [-7.0, 8.0, 9.0, 10.0, -11.0, -12.0, 13.0,
         -0.0, 1.0, 2.0, 3.0, -4.0, -5.0, 6.0],
        dtype=torch.float32,
    )
    mirror_golden_vector = torch.equal(
        _mirror_joint14(golden_source), golden_expected
    )
    prompt_probe = _right_prompt("Use the left arm to lift the bottle.")
    prompt_transform = "right arm" in prompt_probe.lower() and "left arm" not in prompt_probe.lower()
    passed = bool(
        mix_exact
        and terminal_coverage
        and mirror_involution
        and mirror_golden_vector
        and prompt_transform
    )
    report = {
        "format": "strict-track2-v432-public-mirror-contract-audit-v1",
        "passed": passed,
        "split_sha256": sha256(args.split),
        "fixed_public_train_episodes": train,
        "fixed_public_validation_episodes": validation,
        "checks": checks,
        "dataset": dataset.v432_contract,
        "mixture_proportions": proportions,
        "mixture_expected": expected,
        "mix_exact": mix_exact,
        "terminal_real_right_episode_coverage": terminal_coverage,
        "mirror_joint14_involution": mirror_involution,
        "mirror_joint14_golden_vector": mirror_golden_vector,
        "mirror_prompt_transform": prompt_transform,
        "guards": {
            "policy_modified": False,
            "reward_model_modified": False,
            "development_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "dataset": dataset.v432_contract}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
