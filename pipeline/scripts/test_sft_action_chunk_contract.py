#!/usr/bin/env python3
"""Static contract for aligning SFT supervision with deployed action chunks."""

from __future__ import annotations

import ast
import pathlib
import sys


actor_source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
runner_source = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
tree = ast.parse(actor_source)
methods = {
    node.name: node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
helper = methods["_next_sft_loss"]
calls = [node for node in ast.walk(helper) if isinstance(node, ast.Call)]
model_calls = [
    node
    for node in calls
    if isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "self"
    and node.func.attr == "model"
]
assert len(model_calls) == 1
keywords = {keyword.arg: keyword.value for keyword in model_calls[0].keywords}
assert "use_action_chunk_loss" in keywords
assert 'self.cfg.actor.get("sft_use_action_chunk_loss", False)' in actor_source
assert 'TRACK2_SFT_USE_ACTION_CHUNK_LOSS' in runner_source
assert '+actor.sft_use_action_chunk_loss=$SFT_USE_ACTION_CHUNK_LOSS' in runner_source
print("sft_action_chunk_contract_ok")
