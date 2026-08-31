#!/usr/bin/env python3
"""Pre-execution static/interface audit for the frozen v482 r5 implementation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import h5py


CONTRACT_SHA = "19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
    return digest.hexdigest()


def arrsha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def atomic_json(path, payload):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists(): raise FileExistsError(path)
    def canonical(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, default=canonical); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class FakeV169:
    def __init__(self): self.calls = []
    def predict(self, context, history, future, seed, instruction):
        history, future = np.asarray(history), np.asarray(future)
        self.calls.append((history.copy(), future.copy(), int(seed), instruction))
        base = (int(seed) + int(np.asarray(future, np.float64).sum() * 1000)) % 256
        result = np.empty((8, 256, 256, 3), np.uint8)
        for k in range(8): result[k].fill((base + k) % 256)
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "contract", "runtime", "trainer", "prepare", "auditor", "packager", "release-auditor", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(); contract = json.loads(args.contract.read_text()); pre = json.loads(args.preregistration.read_text())
    paths = {name: getattr(args, name.replace("-", "_")) for name in ("runtime", "trainer", "prepare", "auditor", "packager", "release-auditor")}
    source_keys = {
        "runtime": "runtime", "trainer": "trainer", "prepare": "prepare", "auditor": "auditor",
        "packager": "packager", "release-auditor": "release_auditor",
    }
    six_keys = {**source_keys, "auditor": "s0_auditor"}
    checks = {
        "contract_r5": sha(args.contract) == CONTRACT_SHA and contract.get("format") == "strict-track2-v482-public-train-temporal-film-residual-model-design-contract-v5",
        "execution_preregistration": pre.get("format") == "strict-track2-v482-temporal8-residual-preregistration-v1"
        and pre.get("status") == "preregistered_public_train_temporal_s0_authorized"
        and Path(pre.get("source", {}).get("contract_path", "")).resolve() == args.contract.resolve()
        and pre.get("source", {}).get("contract_sha256") == CONTRACT_SHA,
        "sources_regular_nonempty": all(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 for path in paths.values()),
        "sources_match_preregistration": all(
            Path(pre.get("source", {}).get(source_keys[name] + "_path", "")).resolve() == path.resolve()
            and pre.get("source", {}).get(source_keys[name] + "_sha256") == sha(path)
            for name, path in paths.items()
        ),
        "authorization_boundary": pre.get("guards") == {
            "public_train_only": True, "training_authorized": True, "s1_authorized": False,
            "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False,
            "dev_reward_success_outcome_final_hidden_used": False,
            "submission_authorized": False,
        },
        "six_source_independent_closure": pre.get("execution_source_six_count") == 6
        and set(pre.get("execution_source_six", {})) == {"prepare", "trainer", "runtime", "s0_auditor", "packager", "release_auditor"}
        and pre.get("execution_source_six_digest_sha256") == hashlib.sha256(
            json.dumps(pre.get("execution_source_six", {}), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        and all(
            Path(pre["execution_source_six"][six_keys[name]]["path"]).resolve() == path.resolve()
            and pre["execution_source_six"][six_keys[name]]["sha256"] == sha(path)
            and pre["execution_source_six"][six_keys[name]]["bytes"] == path.stat().st_size
            for name, path in paths.items()
        ),
    }
    framework = pre.get("framework", {})
    interpreter = framework.get("execution_interpreter", {})
    lexical = Path(interpreter.get("lexical_path", ""))
    middle = Path(interpreter.get("symlink_target", ""))
    try:
        resolved = lexical.resolve(strict=True)
        checks["framework_execution_interpreter_exact"] = bool(
            lexical == Path(sys.executable)
            and str(lexical) == "/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
            and lexical.is_symlink()
            and interpreter.get("lexical_is_symlink") is True
            and os.readlink(lexical) == interpreter.get("symlink_target") == "/root/autodl-tmp/conda_envs/isaacsim51/bin/python"
            and middle.is_symlink()
            and os.readlink(middle) == "python3.11"
            and str(resolved) == interpreter.get("resolved_path") == "/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11"
            and resolved.is_file()
            and not resolved.is_symlink()
            and interpreter.get("resolved_is_regular_file") is True
            and resolved.stat().st_size == interpreter.get("resolved_bytes") == 25555040
            and sha(resolved) == interpreter.get("resolved_sha256") == "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788"
            and platform.python_version() == interpreter.get("python_version") == "3.11.15"
            and np.__version__ == framework.get("numpy_version") == interpreter.get("numpy_version") == "1.26.4"
            and torch.__version__ == framework.get("torch_version") == interpreter.get("torch_version") == "2.7.0+cu128"
            and framework.get("pythonhashseed_required") == "0"
            and os.environ.get("PYTHONHASHSEED") == "0"
            and framework.get("cublas_workspace_config_required") == ":4096:8"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and framework.get("torch_deterministic_algorithms_required") is True
        )
    except (OSError, RuntimeError):
        checks["framework_execution_interpreter_exact"] = False
    supersession = pre.get("supersession", {})
    expected_old = [
        ("a70731adf473582fb63a06e70efc61eb938d2045c2879ebc1a55a4de7c54a4f6", "non_execution_interpreter_framework_provenance"),
        ("0431814403f8cace842765b17fbee098427187cdbfcb7cc05ffde986f0ece665", "unbound_postprocessing_materializer_provenance"),
    ]
    supersession_ok = bool(
        supersession.get("all_previous_executed") is False
        and supersession.get("only_authorized_preregistration") == "this native r3 preregistration after independent audit"
        and supersession.get("native_prepare_author") == {
            "path": str(args.prepare.resolve()),
            "sha256": sha(args.prepare),
            "postprocessing_materializer_used": False,
        }
        and len(supersession.get("previous_formals", [])) == 2
    )
    if supersession_ok:
        for entry, (expected_sha, expected_reason) in zip(supersession["previous_formals"], expected_old):
            old = Path(entry.get("path", ""))
            supersession_ok = bool(
                supersession_ok
                and entry.get("executed") is False
                and entry.get("superseded") is True
                and entry.get("sha256") == expected_sha
                and entry.get("supersession_reason") == expected_reason
                and entry.get("parent_directory_exact_files") == ["preregistration.json"]
                and old.is_file()
                and not old.is_symlink()
                and sha(old) == expected_sha
                and sorted(item.name for item in old.parent.iterdir()) == ["preregistration.json"]
            )
    checks["superseded_formals_unexecuted_prereg_only"] = bool(supersession_ok)
    spec = importlib.util.spec_from_file_location("v482_static_runtime", args.runtime)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    lower = np.array([0,0,0,0,0,0,1,.3740280866622925,.7809063196182251,.08595156669616699,-1.440514326095581,.0006023868918418884,-1.094338297843933,0], np.float32)
    upper = np.array([0,0,0,0,0,0,1,.9274010062217712,2.437008857727051,2.334188461303711,.9241482019424438,.6692923903465271,.08544458448886871,0], np.float32)
    frozen_bounds = pre.get("action_bounds", {})
    checks["frozen_action_bounds_bitexact"] = bool(
        np.array_equal(np.asarray(frozen_bounds.get("lower"), np.float32), lower)
        and np.array_equal(np.asarray(frozen_bounds.get("upper"), np.float32), upper)
        and frozen_bounds.get("lower_upper_sha256") == arrsha(np.concatenate((lower, upper)))
        and frozen_bounds.get("lower_float32_bits_hex") == [f"0x{int(x):08x}" for x in lower.astype("<f4").view("<u4")]
        and frozen_bounds.get("upper_float32_bits_hex") == [f"0x{int(x):08x}" for x in upper.astype("<f4").view("<u4")]
        and frozen_bounds.get("zero_span_dimensions") == np.flatnonzero(upper == lower).tolist()
        and frozen_bounds.get("dimension13_exact_zero_span") is True
    )
    selection_path = Path(pre["dataset"]["selection"]["path"])
    selection = json.loads(selection_path.read_text())
    hdf5_cache = {}
    raw_hdf5_ok = bool(
        sha(selection_path) == pre["dataset"]["selection"]["sha256"]
        and selection.get("format") == "strict-track2-v478-window-eligible-action-only-selection-v1"
        and len(selection.get("contexts", [])) == 200
    )
    for row in selection.get("contexts", []):
        source = Path(row["source_hdf5"])
        key = str(source.resolve())
        if key not in hdf5_cache:
            if not source.is_file() or source.is_symlink() or sha(source) != row["source_hdf5_sha256"]:
                raw_hdf5_ok = False; break
            with h5py.File(source, "r") as handle:
                raw = np.asarray(handle["joint_action/vector"])
            if raw.dtype != np.float64 or raw.ndim != 2 or raw.shape[1] != 14 or not np.isfinite(raw).all():
                raw_hdf5_ok = False; break
            hdf5_cache[key] = raw
        raw = hdf5_cache[key]; start = int(row["start"])
        history = np.ascontiguousarray(raw[start:start + 4].astype(np.float32))
        future = np.ascontiguousarray(raw[start + 4:start + 12].astype(np.float32))
        if history.shape != (4, 14) or future.shape != (8, 14) or arrsha(history) != row["history_action_sha256"] or arrsha(future) != row["future_action_sha256"]:
            raw_hdf5_ok = False; break
    checks["raw_hdf5_float64_to_canonical_float32_identity"] = bool(raw_hdf5_ok and len(hdf5_cache) > 0)
    model = module.TemporalResidualUNet128FiLM(16).eval()
    checks["architecture"] = sum(x.numel() for x in model.parameters()) == 878579
    history = np.tile(lower, (16, 4, 1)).astype(np.float64)
    future = np.tile(lower, (16, 8, 1)).astype(np.float64)
    history[..., 13] = .2; future[..., 13] = .2
    history[2, :, 13] = .8; future[2, :, 13] = .8
    contexts = np.zeros((16, 5, 256, 256, 3), np.uint8)
    seeds = np.arange(16, dtype=np.int64) + 900
    instructions = ["use right arm" if i % 3 else "use left arm" for i in range(16)]
    runtime = module.Track2V482Temporal8ResidualRuntime.__new__(module.Track2V482Temporal8ResidualRuntime)
    runtime.device = torch.device("cpu"); runtime.model = model; runtime.lower = lower; runtime.upper = upper; runtime.v169 = FakeV169()
    scalar = [runtime.predict_one_with_baseline(contexts[i], history[i], future[i], seeds[i], instructions[i]) for i in range(16)]
    sb, sy = np.stack([x[0] for x in scalar]), np.stack([x[1] for x in scalar])
    bb, by, bd = runtime.predict_batch_with_baseline(contexts, history, future, seeds, instructions)
    permutation = np.random.default_rng(482).permutation(16); inverse = np.argsort(permutation)
    pb, py, pd = runtime.predict_batch_with_baseline(contexts[permutation], history[permutation], future[permutation], seeds[permutation], [instructions[i] for i in permutation])
    rb, ry, rd = runtime.predict_batch_with_baseline(contexts, history, future, seeds, instructions)
    checks["mixed_n16_scalar_batch_permutation_repeat"] = bool(
        np.array_equal(sb, bb) and np.array_equal(sy, by) and np.array_equal(pb[inverse], bb)
        and np.array_equal(py[inverse], by) and np.array_equal(rb, bb) and np.array_equal(ry, by)
        and [x[2] for x in scalar] == bd == [pd[i] for i in inverse] == rd
    )
    b8, y8, d8 = runtime.predict_batch_with_baseline(contexts[:8], history[:8], future[:8], seeds[:8], instructions[:8])
    checks["mixed_n8_deploy"] = b8.shape == y8.shape == (8, 8, 256, 256, 3) and len(d8) == 8
    checks["raw_action_bytes_forwarded"] = all(
        runtime.v169.calls[i][0].dtype == history[i].dtype == np.dtype("float64")
        and runtime.v169.calls[i][1].dtype == future[i].dtype == np.dtype("float64")
        and np.array_equal(runtime.v169.calls[i][0], history[i])
        and np.array_equal(runtime.v169.calls[i][1], future[i])
        for i in range(16)
    )
    checks["inactive_exact"] = bd[2]["explicit_right"] is True and bd[2]["gate"] is False and np.array_equal(bb[2], by[2]) and all(np.array_equal(bb[i], by[i]) for i in range(0, 16, 3))
    h, f = history[1], future[1]
    feature = module.causal_action_features(h, f, lower, upper)
    checks["zero_span_runtime"] = np.all(feature[..., :168][..., np.arange(168) % 14 == 13] == 0)
    fixed_base = torch.zeros(8, 3, 256, 256); fixed_context = torch.zeros_like(fixed_base)
    with torch.inference_mode(): reference = model(fixed_base, fixed_context, torch.as_tensor(feature.reshape(8, 178)))
    causal = True
    for k in range(8):
        changed = f.copy()
        if k + 1 < 8: changed[k + 1:, :13] += 7.125
        alternate_feature = module.causal_action_features(h, changed, lower, upper)
        with torch.inference_mode(): alternate = model(fixed_base, fixed_context, torch.as_tensor(alternate_feature.reshape(8, 178)))
        causal &= np.array_equal(feature[0, k], alternate_feature[0, k]) and torch.equal(reference[k], alternate[k])
    checks["fixed_baseline_actual_residual_prefix_causality"] = bool(causal)
    for source in paths.values():
        tree = ast.parse(source.read_text())
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        checks[f"no_forbidden_import_{source.name}"] = not any(any(token in name.lower() for token in ("reward", "policy", "rlinf")) for name in imports)
    trainer_source = args.trainer.read_text(); runtime_source = args.runtime.read_text()
    checks["serial_oof_and_runtime_source_guards"] = (
        "for begin in range(len(ids))" in trainer_source
        and "for index in range(n)" in runtime_source
        and "v169.predict_batch(" not in runtime_source
    )
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    receipt = {
        "format": "strict-track2-v482-temporal8-residual-static-audit-v1", "passed": passed,
        "checks": checks, "contract_sha256": CONTRACT_SHA,
        "preregistration_sha256": sha(args.preregistration),
        "sources": {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in paths.items()},
        "guards": {"training_launched": False, "submission_authorized": False, "s1_authorized": False, "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False},
    }
    atomic_json(args.output, receipt); print(json.dumps(receipt, indent=2)); return 0 if passed else 3


if __name__ == "__main__": raise SystemExit(main())
