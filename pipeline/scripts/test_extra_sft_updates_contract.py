#!/usr/bin/env python3
"""Static regression contract for rollout-free expert mini-batch updates."""

from __future__ import annotations

import ast
import pathlib
import sys


source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
tree = ast.parse(source)
methods = {
    node.name: node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

assert 'cfg.actor.get("sft_extra_updates_per_global_batch", 0)' in source
assert "sft_extra_updates_per_global_batch requires" in source
assert "actor.enable_sft_co_train=True" in source
assert "actor.sft_extra_loss_weight must be positive" in source

helper = methods["_run_extra_sft_updates"]
helper_source = ast.get_source_segment(source, helper)
assert helper_source is not None
assert helper_source.index("_rebind_handle_views") < helper_source.index(
    "self._next_sft_loss()"
), "FSDP parameter views must be refreshed before the next checkpointed graph"
assert helper_source.index("self.before_micro_batch(") < helper_source.index(
    "self._next_sft_loss()"
), "FSDP micro-batch context must be created before the SFT forward pass"
helper_calls = {
    node.func.attr
    for node in ast.walk(helper)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
}
assert "_next_sft_loss" in helper_calls
assert "backward" in helper_calls
assert "optimizer_step" in helper_calls
assert "empty_cache" in helper_calls

loops = [node for node in ast.walk(helper) if isinstance(node, ast.For)]
assert any(
    isinstance(node.iter, ast.Call)
    and isinstance(node.iter.func, ast.Name)
    and node.iter.func.id == "range"
    and any(
        isinstance(arg, ast.Attribute)
        and arg.attr == "sft_extra_updates_per_global_batch"
        for arg in node.iter.args
    )
    for node in loops
)

run_training = methods["run_training"]
assert any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "_run_extra_sft_updates"
    for node in ast.walk(run_training)
)

print("extra_sft_updates_contract_ok")
