#!/usr/bin/env python3
"""Static regression contract for single-arm structured Track-2 SFT."""

from __future__ import annotations

import ast
import pathlib
import sys

import torch


source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(source)

assert "sft_single_active_arm" in source
assert "actor.sft_single_active_arm must be 0 or 1" in source
assert "self.sft_single_active_arm" in source

weight_fn = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and node.name == "build_track2_sft_action_loss_weights"
)
assert any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "zeros"
    for node in ast.walk(weight_fn)
), "padded model dimensions must start with zero loss weight"

module = ast.Module(body=[weight_fn], type_ignores=[])
ast.fix_missing_locations(module)
namespace = {"torch": torch}
exec(compile(module, "<weight-contract>", "exec"), namespace)
weights = namespace["build_track2_sft_action_loss_weights"](
    32,
    1,
    active_joint_weight=1.0,
    active_gripper_weight=3.0,
    inactive_keep_weight=0.25,
)
assert weights.shape == (32,)
assert torch.all(weights[:7] == 0.25)
assert torch.all(weights[7:13] == 1.0)
assert weights[13].item() == 3.0
assert torch.count_nonzero(weights[14:]).item() == 0

physical_weight_fn = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and node.name == "build_physical_sft_action_loss_weights"
)
physical_module = ast.Module(body=[physical_weight_fn], type_ignores=[])
ast.fix_missing_locations(physical_module)
physical_namespace = {"torch": torch}
exec(compile(physical_module, "<physical-weight-contract>", "exec"), physical_namespace)
physical_weights = physical_namespace["build_physical_sft_action_loss_weights"](32, 14)
assert physical_weights.shape == (32,)
assert torch.all(physical_weights[:14] == 1.0)
assert torch.count_nonzero(physical_weights[14:]).item() == 0
assert physical_weights.sum().item() == 14.0

for action_dim, physical_dim in ((32, 0), (14, 15)):
    try:
        physical_namespace["build_physical_sft_action_loss_weights"](
            action_dim, physical_dim
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid physical action dimensions must be rejected")

assert 'self.cfg.actor.get("sft_physical_action_dim", None)' in source
assert "single-arm, mixed-arm, and physical-mask SFT losses are mutually exclusive" in source

print("single_arm_structured_sft_contract_ok")
