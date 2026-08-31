#!/usr/bin/env python3
"""Independent package/runtime/interface audit for a passed v482 public-train release."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists(): raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class FakeV169:
    calls = []
    def __init__(self, *args): pass
    def predict(self, context, history, future, seed, instruction):
        context = np.asarray(context); history = np.asarray(history); future = np.asarray(future)
        self.calls.append((history.copy(), future.copy(), int(seed), instruction))
        value = np.empty((8, 256, 256, 3), np.uint8)
        base = int(np.asarray(context, np.uint64).sum() + np.asarray(future, np.float64).sum() * 1000 + int(seed)) % 251
        for k in range(8): value[k].fill((base + k) % 256)
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.release.resolve()
    manifest_path = root / "v482_temporal8_residual_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = {}
    expected = {row["relative"]: row for row in manifest["inventory_excluding_manifest"]}
    actual = {}
    no_symlinks = True
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): no_symlinks = False
        elif path.is_file() and path.name != manifest_path.name:
            actual[path.relative_to(root).as_posix()] = {"relative": path.relative_to(root).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
    checks["exact_release_inventory"] = no_symlinks and actual == expected
    checks["manifest_closure"] = bool(
        manifest.get("format") == "track2-v482-temporal8-residual-release-v1"
        and manifest.get("film_formula") == "x*(1+gamma)+beta"
        and manifest.get("ordered_scalar_v169") is True
        and manifest.get("reward_or_outcome_used") is False
        and manifest.get("guards") == {"s1_authorized": False, "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False}
        and manifest["sha256"]["runtime_source"] == actual["v482_temporal8_residual_runtime.py"]["sha256"]
        and manifest["sha256"]["checkpoint"] == actual["all200_action.pt"]["sha256"]
        and manifest["inventory_digest_sha256"] == hashlib.sha256(
            json.dumps([[expected[name]["relative"], expected[name]["sha256"]] for name in sorted(expected)], separators=(",", ":")).encode()
        ).hexdigest()
        and manifest["sha256"]["contract"] == actual["v482_design_contract.json"]["sha256"]
        and manifest["sha256"]["preregistration"] == actual["v482_execution_preregistration.json"]["sha256"]
        and manifest["sha256"]["report"] == actual["v482_s0_report.json"]["sha256"]
        and manifest["sha256"]["audit"] == actual["v482_s0_audit.json"]["sha256"]
        and manifest["sha256"]["packager"] == actual["package_v482_temporal8_residual_release.py"]["sha256"]
        and manifest["sha256"]["v169_closure"] == actual["v169_closure.json"]["sha256"]
        and sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) <= 536870912
    )
    runtime_path = root / "v482_temporal8_residual_runtime.py"
    spec = importlib.util.spec_from_file_location("v482_release_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.Track2V169ArmRoutedRuntime = FakeV169
    runtime = module.Track2V482Temporal8ResidualRuntime(root, "cpu")
    state = torch.load(root / "all200_action.pt", map_location="cpu", weights_only=False)
    checks["checkpoint_architecture"] = bool(
        state.get("format") == module.CHECKPOINT_FORMAT
        and state.get("training_scope") == "all200-action"
        and sum(parameter.numel() for parameter in runtime.model.parameters()) <= 1_200_000
        and runtime.lower.shape == (14,) and runtime.upper.shape == (14,)
        and runtime.lower[13] == runtime.upper[13] == 0
        and all(bool(torch.isfinite(value).all()) for value in state.get("model", {}).values())
    )
    rng = np.random.default_rng(482)
    n = 16
    contexts = rng.integers(0, 256, (n, 5, 256, 256, 3), dtype=np.uint8)
    history = rng.normal(size=(n, 4, 14)).astype(np.float64)
    future = rng.normal(size=(n, 8, 14)).astype(np.float64)
    history[:, :, 13] = 0.2; future[:, :, 13] = 0.2
    history[2, :, 13] = 0.8; future[2, :, 13] = 0.8
    instructions = ["move with right arm" if i % 3 else "move with left arm" for i in range(n)]
    seeds = np.arange(100, 100 + n, dtype=np.int64)
    FakeV169.calls = []
    scalar = [runtime.predict_one_with_baseline(contexts[i], history[i], future[i], seeds[i], instructions[i]) for i in range(n)]
    scalar_b = np.stack([x[0] for x in scalar]); scalar_y = np.stack([x[1] for x in scalar])
    batch_b, batch_y, batch_d = runtime.predict_batch_with_baseline(contexts, history, future, seeds, instructions)
    permutation = rng.permutation(n); inverse = np.argsort(permutation)
    perm_b, perm_y, perm_d = runtime.predict_batch_with_baseline(
        contexts[permutation], history[permutation], future[permutation], seeds[permutation],
        [instructions[i] for i in permutation]
    )
    repeat_b, repeat_y, repeat_d = runtime.predict_batch_with_baseline(contexts, history, future, seeds, instructions)
    checks["mixed_n16_scalar_batch_permutation_repeat"] = bool(
        np.array_equal(scalar_b, batch_b) and np.array_equal(scalar_y, batch_y)
        and np.array_equal(perm_b[inverse], batch_b) and np.array_equal(perm_y[inverse], batch_y)
        and np.array_equal(repeat_b, batch_b) and np.array_equal(repeat_y, batch_y)
        and [x[2] for x in scalar] == batch_d == [perm_d[i] for i in inverse] == repeat_d
    )
    checks["raw_actions_forwarded_unchanged"] = bool(
        all(np.array_equal(FakeV169.calls[i][0], history[i]) and np.array_equal(FakeV169.calls[i][1], future[i])
            and FakeV169.calls[i][0].dtype == np.float64 and FakeV169.calls[i][1].dtype == np.float64
            for i in range(n))
    )
    enabled_index = 1
    checks["zero_span_runtime_and_gate"] = bool(
        batch_d[enabled_index]["gate"] is True
        and batch_d[enabled_index]["zero_span_observation_counts"]["13"] == {"below": 0, "equal": 0, "above": 12}
        and np.all(
            module.causal_action_features(history[enabled_index], future[enabled_index], runtime.lower, runtime.upper)[..., :168][..., np.arange(168) % 14 == 13]
            == 0
        )
    )
    checks["inactive_left_bitexact"] = all(
        np.array_equal(batch_b[i], batch_y[i]) for i in range(n) if "left arm" in instructions[i]
    )
    checks["explicit_right_g0_bitexact"] = batch_d[2]["explicit_right"] is True and batch_d[2]["gate"] is False and np.array_equal(batch_b[2], batch_y[2])
    n8 = 8
    b8, y8, d8 = runtime.predict_batch_with_baseline(contexts[:n8], history[:n8], future[:n8], seeds[:n8], instructions[:n8])
    checks["mixed_n8_deploy_path"] = b8.shape == y8.shape == (8, 8, 256, 256, 3) and len(d8) == 8
    h = history[enabled_index].copy(); f = future[enabled_index].copy()
    features = module.causal_action_features(h, f, runtime.lower, runtime.upper)
    causal = True
    for k in range(8):
        changed = f.copy()
        if k + 1 < 8: changed[k + 1:] += 7.125
        alternate = module.causal_action_features(h, changed, runtime.lower, runtime.upper)
        causal &= np.array_equal(features[0, k], alternate[0, k])
    checks["prefix_feature_causality"] = bool(causal)
    fixed_base = torch.as_tensor(batch_b[enabled_index], dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    fixed_context = torch.as_tensor(np.repeat(contexts[enabled_index, -1][None], 8, axis=0), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    residual_causal = True
    with torch.inference_mode():
        reference_residual = runtime.model(fixed_base, fixed_context, torch.as_tensor(features.reshape(8, 178)))
        for k in range(8):
            changed = f.copy()
            if k + 1 < 8: changed[k + 1:] += 7.125
            alternate_features = module.causal_action_features(h, changed, runtime.lower, runtime.upper)
            alternate_residual = runtime.model(fixed_base, fixed_context, torch.as_tensor(alternate_features.reshape(8, 178)))
            residual_causal &= torch.equal(reference_residual[k], alternate_residual[k])
    checks["fixed_baseline_lowres_residual_prefix_causality"] = bool(residual_causal)
    checks["runtime_outputs_and_residual_finite"] = bool(
        torch.isfinite(reference_residual).all() and np.isfinite(batch_b).all() and np.isfinite(batch_y).all()
    )
    class NonfiniteResidual(torch.nn.Module):
        def forward(self, baseline, context, features):
            return torch.full((len(baseline), 3, 128, 128), float("nan"), device=baseline.device)
    original_model = runtime.model; runtime.model = NonfiniteResidual()
    nonfinite_failed_closed = False
    try:
        runtime.predict_one(contexts[enabled_index], history[enabled_index], future[enabled_index], seeds[enabled_index], instructions[enabled_index])
    except RuntimeError as error:
        nonfinite_failed_closed = "nonfinite runtime residual" in str(error)
    finally:
        runtime.model = original_model
    checks["injected_nonfinite_residual_fails_closed"] = nonfinite_failed_closed
    tree = ast.parse(runtime_path.read_text())
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    checks["forbidden_imports"] = not any(any(word in name.lower() for word in ("reward", "policy", "rlinf")) for name in imported)
    checks["constructor_nonfinite_checkpoint_guard_present"] = "not all(bool(torch.isfinite(value).all())" in runtime_path.read_text()
    passed = all(checks.values())
    receipt = {
        "format": "strict-track2-v482-temporal8-residual-release-audit-v1",
        "passed": passed, "checks": checks,
        "release_manifest_sha256": sha(manifest_path),
        "guards": {"s1_authorized": False, "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False},
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2)); return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
