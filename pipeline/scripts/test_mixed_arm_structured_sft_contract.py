#!/usr/bin/env python3
"""Regression contract for per-example mixed-arm Track-2 SFT weighting."""

from __future__ import annotations

import ast
import pathlib
import sys

import torch


actor_source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
openpi_source = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
actor_tree = ast.parse(actor_source)
openpi_tree = ast.parse(openpi_source)


def extract(tree: ast.AST, name: str, namespace: dict):
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name]


build = extract(
    actor_tree,
    "build_mixed_track2_sft_action_loss_weights",
    {"torch": torch},
)
actions = torch.zeros(2, 8, 32)
joint_zero = torch.tensor([
    -0.0457539920, -0.0791058647, -0.0978789938, -0.0798672016,
    -0.1268338160, 0.1565935501, -0.0675038738, -0.1382685700,
    -0.1605281059, -0.0354791340, 0.1411993263, -0.1048347060,
])
actions[:, :, :6] = joint_zero[:6]
actions[:, :, 7:13] = joint_zero[6:]
actions[:, :, 6] = 1.0
actions[:, :, 13] = 1.0
actions[0, :, 0] += torch.linspace(0, 1, 8)
actions[0, 4:, 6] = -1.0
actions[1, :, 7] += torch.linspace(0, 1, 8)
actions[1, 4:, 13] = -1.0
weights, active_right = build(
    actions,
    active_joint_weight=1.0,
    active_gripper_weight=3.0,
    inactive_keep_weight=2.0,
    gripper_activity_weight=0.25,
)
assert active_right.tolist() == [False, True]
assert weights.shape == (2, 32)
assert torch.all(weights[0, :6] == 1.0) and weights[0, 6] == 3.0
assert torch.all(weights[0, 7:14] == 2.0)
assert torch.all(weights[1, :7] == 2.0)
assert torch.all(weights[1, 7:13] == 1.0) and weights[1, 13] == 3.0
assert torch.all(weights[:, 14:] == 0.0)
assert torch.allclose(weights.sum(dim=1), torch.tensor([23.0, 23.0]))

reduce_loss = extract(
    openpi_tree,
    "reduce_sft_action_loss",
    {"torch": torch},
)
loss = torch.arange(2 * 8 * 14, dtype=torch.float32).reshape(2, 8, 14)
actual = reduce_loss(loss, weights)
expected = (loss * weights[:, None, :14]).sum() / (weights[:, :14].sum() * 8)
assert torch.allclose(actual, expected)

global_weights = torch.ones(32)
assert torch.allclose(reduce_loss(loss, global_weights), loss.mean())
assert 'self.cfg.actor.get("use_mixed_arm_structured_sft_action_loss", False)' in actor_source
assert "per-example action_loss_weights must have shape" in openpi_source
print("mixed_arm_structured_sft_contract_ok")
