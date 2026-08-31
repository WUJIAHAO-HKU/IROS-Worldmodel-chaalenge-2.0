#!/usr/bin/env python3
"""Regression contract for Track-2 motion-weighted SFT examples."""

from __future__ import annotations

import ast
import pathlib
import sys

import torch


source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(source)
weight_fn = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and node.name == "apply_track2_sft_motion_sample_weights"
)
module = ast.Module(body=[weight_fn], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"torch": torch}
exec(compile(module, "<motion-weight-contract>", "exec"), namespace)

actions = torch.zeros((3, 8, 32), dtype=torch.float32)
right_zero = torch.tensor(
    [-0.0675038738, -0.1382685700, -0.1605281059, -0.0354791340, 0.1411993263, -0.1048347060]
)
actions[:, :, 7:13] = right_zero
actions[1, :, 7:13] += 0.025
actions[2, :, 7:13] += 0.10
base = torch.zeros(32)
base[:7] = 2.0
base[7:13] = 4.0
base[13] = 1.0

weighted, sample_weights = namespace[
    "apply_track2_sft_motion_sample_weights"
](
    actions,
    base,
    active_arm=1,
    horizon=8,
    motion_reference=0.05,
    minimum_weight=0.05,
)
assert weighted.shape == (3, 32)
assert torch.allclose(sample_weights, torch.tensor([0.05, 0.525, 1.0]))
assert torch.allclose(weighted[0], base * 0.05)
assert torch.allclose(weighted[1], base * 0.525)
assert torch.allclose(weighted[2], base)
assert torch.count_nonzero(weighted[:, 14:]).item() == 0

for bad in (
    {"horizon": 0},
    {"motion_reference": 0.0},
    {"minimum_weight": 1.1},
    {"power": 0.0},
):
    try:
        namespace["apply_track2_sft_motion_sample_weights"](
            actions, base, active_arm=1, **bad
        )
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid parameters accepted: {bad}")

print("motion_weighted_sft_contract_ok")
