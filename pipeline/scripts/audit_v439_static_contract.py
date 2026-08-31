#!/usr/bin/env python3
"""Static preregistration audit for the frozen v439 causal projection runtime.

This script deliberately does not import the runtime.  It can therefore run in
staging before CUDA/model dependencies are present.  It checks the public-data
boundary, frozen blend constants, public API, and the complete call graph below
``gate_decision`` for forbidden gate inputs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


FORMAT = "track2-v439-v169-action-causal-projection-release-v1"
CLASS = "Track2V439V169ActionCausalProjection"
ALPHA = [0.0, 0.0, 0.03661165, 0.125, 0.125, 0.03661165, 0.0, 0.0]
TRAIN40 = [
    0, 1, 3, 4, 8, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41,
    42, 43, 44, 45, 46, 47, 49,
]
HOLDOUT10 = [2, 5, 6, 7, 9, 16, 18, 22, 39, 48]
FORBIDDEN_GATE_NAME = re.compile(r"(?:reward|seed|request_?id)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def class_methods(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS:
            return {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return {}


def global_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def self_calls(node: ast.AST) -> set[str]:
    result = set()
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id == "self"
        ):
            result.add(item.func.attr)
    return result


def global_calls(node: ast.AST) -> set[str]:
    return {
        item.func.id
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
    }


def reachable_nodes(methods: dict, functions: dict, root: str) -> dict[str, ast.AST]:
    pending = [("class", root)]
    result: dict[str, ast.AST] = {}
    while pending:
        kind, name = pending.pop()
        key = f"{kind}:{name}"
        namespace = methods if kind == "class" else functions
        if key in result or name not in namespace:
            continue
        node = namespace[name]
        result[key] = node
        pending.extend(("class", value) for value in self_calls(node))
        pending.extend(("global", value) for value in global_calls(node))
    return result


def referenced_names(nodes: list[ast.AST]) -> set[str]:
    names = set()
    for node in nodes:
        for item in ast.walk(node):
            if isinstance(item, ast.Name):
                names.add(item.id)
            elif isinstance(item, ast.Attribute):
                names.add(item.attr)
            elif isinstance(item, ast.keyword) and item.arg:
                names.add(item.arg)
    return names


def manifest_alpha(document: dict) -> list[float] | None:
    for parent in (document, document.get("composition", {}), document.get("profile", {}), document.get("formula", {})):
        for key in ("alpha", "alpha_8", "horizon_alpha", "right_horizon_alpha"):
            if key in parent:
                return [float(value) for value in parent[key]]
    return None


def manifest_cap(document: dict) -> float | None:
    for parent in (document, document.get("composition", {}), document.get("profile", {}), document.get("formula", {})):
        for key in ("residual_cap_uint8", "uint8_residual_cap", "residual_cap"):
            if key in parent:
                return float(parent[key])
        if parent.get("teacher_delta_clip") == [-8, 8]:
            return 8.0
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = args.runtime_source.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(args.runtime_source))
    methods = class_methods(tree)
    functions = global_functions(tree)
    gate = methods.get("gate_decision")
    gate_args = [] if gate is None else [item.arg for item in gate.args.args]
    reachable_gate = reachable_nodes(methods, functions, "gate_decision")
    names = referenced_names(list(reachable_gate.values()))
    forbidden = sorted(name for name in names if FORBIDDEN_GATE_NAME.search(name))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train = sorted(int(value) for value in split.get("train_episodes", []))
    validation = sorted(int(value) for value in split.get("validation_episodes", []))
    alpha = manifest_alpha(manifest)
    cap = manifest_cap(manifest)
    guards = manifest.get("guards", {})
    gate_inputs = manifest.get("gate_inputs", manifest.get("composition", {}).get("gate_inputs"))

    checks = {
        "runtime_class_exact": bool(methods),
        "public_gate_api_present": gate is not None,
        "gate_signature_exact_history_future_instruction": gate_args in (
            ["history", "future", "instruction"],
            ["history_actions", "future_actions", "instruction"],
            ["self", "history", "future", "instruction"],
            ["self", "history_actions", "future_actions", "instruction"],
        ),
        "gate_call_graph_nonempty": bool(reachable_gate),
        "reward_seed_request_id_absent_from_gate_call_graph": not forbidden,
        "manifest_format_exact": manifest.get("format") == FORMAT,
        "uint8_residual_cap_exact_8": cap == 8.0,
        "horizon_alpha_exact": alpha is not None
        and len(alpha) == len(ALPHA)
        and all(abs(a - b) <= 1e-12 for a, b in zip(alpha, ALPHA)),
        "protected_t1_t2_t7_t8_zero": alpha is not None
        and all(alpha[index] == 0.0 for index in (0, 1, 6, 7)),
        "fixed_public_train40": train == TRAIN40,
        "fixed_public_holdout10_dev": validation == HOLDOUT10,
        "train_holdout_episode_disjoint": not set(train).intersection(validation),
        "no_hidden_or_final_data": guards.get("hidden_or_final_data") is False,
        "policy_unmodified": guards.get("policy_modified") is False,
        "official_reward_unmodified": guards.get("official_reward_modified") is False,
        "gate_inputs_declared_action_prompt_only": gate_inputs in (
            ["history", "future", "instruction"],
            ["history_actions", "future_actions", "instruction"],
        ),
    }
    passed = all(checks.values())
    report = {
        "format": "strict-track2-v439-static-contract-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "runtime": {
            "module": "wam_pipeline.v439_v169_action_causal_projection_runtime",
            "class": CLASS,
            "gate_signature": gate_args,
            "gate_reachable_call_graph": sorted(reachable_gate),
            "forbidden_gate_names": forbidden,
        },
        "frozen_formula": {
            "uint8_residual_cap": cap,
            "horizon_alpha": alpha,
            "protected_zero_based_frames": [0, 1, 6, 7],
        },
        "checks": checks,
        "decision": "may run offline S1 only" if passed else "reject before inference",
        "evidence_sha256": {
            "runtime_source": sha256(args.runtime_source),
            "manifest": sha256(args.manifest),
            "split": sha256(args.split),
        },
        "guards": {
            "runtime_imported": False,
            "models_loaded": False,
            "remote_execution": False,
            "policy_updates": 0,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
