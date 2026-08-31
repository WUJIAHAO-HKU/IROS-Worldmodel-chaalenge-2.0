#!/usr/bin/env python3
"""One-call v510 design fixture child; never trains and never writes a cache."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
PIPELINE_ROOT = ROOT / "pipeline"
PACKAGE_ROOT = PIPELINE_ROOT / "wam_pipeline"
SCRIPTS_ROOT = PIPELINE_ROOT / "scripts"
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CANONICAL_INPUT_PATHS = {
    "self_source": SCRIPTS_ROOT / "verify_v519_v518_phase_a_sample0_rng_isolation.py",
    "v509_preregistration": J / "v509_v508_phase_a_cache_qualification_prereg_seed1650_20260825/preregistration.json",
    "parent_preregistration": J / "v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json",
    "parent_failure": J / "v482_temporal8_residual_s0_r3_seed1624_20260824/launcher_failure_receipt.json",
    "authority_receipt": J / "v509_v508_phase_a_wam_pipeline_import_repair_execution_authority_seed1650_20260825/authority_receipt.json",
    "scope_source": SCRIPTS_ROOT / "v485_v169_cache_scope.py",
    "worker_source": SCRIPTS_ROOT / "generate_v485_v169_cache_qualification_worker.py",
    "package_manifest": SCRIPTS_ROOT / "v509_v508_wam_pipeline_package_tree_manifest.json",
    "failure_forensic": SCRIPTS_ROOT / "v509_phase_a_deterministic_warning_mismatch_failure_forensic.json",
    "rng_failure_forensic": SCRIPTS_ROOT / "v518_v512_phase_a_sample0_rng_state_failure_forensic.json",
}
SPAWN_ENVIRONMENT_EXACT = {
    "PYTHONPATH": str(PIPELINE_ROOT),
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
CHILD_OBSERVED_ENVIRONMENT_EXACT = {
    **SPAWN_ENVIRONMENT_EXACT,
    "LC_CTYPE": "C.UTF-8",
}
INTERPRETER_ADDED_ENVIRONMENT_EXACT = {"LC_CTYPE": "C.UTF-8"}
EXPECTED_SAMPLE0_SEED = 1103331955
FORENSIC_REPAIR_SUPERSESSION = {
    "forensic_is_immutable": True,
    "original_recommendation": "fresh_single_call_runtime_proxy_restores_all_rng_families_before_returning_to_frozen_v485_scope",
    "original_seed_entries_exact": ["random.seed(exact_sample_seed)", "numpy.random.seed(exact_sample_seed_mod_2_pow_32)"],
    "rejected_by_target_review": True,
    "decision": "reject_py_numpy_seeding",
    "semantic_diff_count": 2,
    "semantic_diffs": [
        {"path": ["repair_research", "inside_context_seed_sequence", 0], "original": "random.seed(exact_sample_seed)", "corrected": None},
        {"path": ["repair_research", "inside_context_seed_sequence", 1], "original": "numpy.random.seed(exact_sample_seed_mod_2_pow_32)", "corrected": None},
    ],
    "semantic_diffs_canonical_sha256": "ccad265fe0de6e61e3a9ee6720a023ab9612f35ff76dd03782babd6952b90083",
    "corrected_semantics": {
        "python_random": "observe_only_no_seed_all_four_stages_bitexact",
        "numpy_random": "observe_only_no_seed_all_four_stages_bitexact",
        "torch": "fork_rng_then_exact_manual_seed_internal_change_allowed_exit_bitexact",
    },
}
EXPECTED_WARNING = {
    "category": "UserWarning",
    "message": (
        "median CUDA with indices output does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at "
        "https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support "
        "for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)"
    ),
}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CALL_ACCOUNTING = {"sample0_calls_started": 0, "sample0_calls_completed": 0}
RNG_ISOLATION_EVIDENCE = None


def child_failure_receipt(error: BaseException) -> dict:
    return {
        "format": "strict-track2-v519-v518-phase-a-sample0-rng-isolation-child-failure-v1",
        "passed": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "sample0_calls_started": CALL_ACCOUNTING["sample0_calls_started"],
        "sample0_calls_completed": CALL_ACCOUNTING["sample0_calls_completed"],
        "sample0_calls": CALL_ACCOUNTING["sample0_calls_started"],
        "rng_isolation_evidence": RNG_ISOLATION_EVIDENCE,
        "v518_forensic_repair_research_supersession": FORENSIC_REPAIR_SUPERSESSION,
        "training_launched": False,
        "cache_reused": False,
        "reward_read": False,
        "dev_hidden_final_outcome_read": False,
        "folds": 0,
        "policy_updates": 0,
        "retry_authorized": False,
    }


def _state_record(raw: bytes) -> dict:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "logical_bytes": len(raw)}


def _python_state_record(random_module) -> dict:
    raw = json.dumps(random_module.getstate(), separators=(",", ":")).encode()
    return {"format": "python_random_getstate_canonical_json", **_state_record(raw)}


def _numpy_state_record(np) -> dict:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    value = {
        "bit_generator": name,
        "state_uint32": [int(item) for item in keys.tolist()],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian_hex": float(cached_gaussian).hex(),
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"format": "numpy_random_get_state_canonical_json", **_state_record(raw)}


def _torch_state_record(torch) -> dict:
    cpu = torch.random.get_rng_state().detach().cpu().contiguous().numpy().tobytes(order="C")
    cuda = []
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            raw = state.detach().cpu().contiguous().numpy().tobytes(order="C")
            cuda.append({"device_index": index, **_state_record(raw)})
    return {"cpu": _state_record(cpu), "cuda": cuda}


def _all_rng_state(torch, np, random_module) -> dict:
    return {
        "torch": _torch_state_record(torch),
        "python": _python_state_record(random_module),
        "numpy": _numpy_state_record(np),
    }


class RngIsolationRuntimeProxy:
    def __init__(self, delegate, exact_seed: int, torch, np, random_module):
        self.delegate = delegate
        self.exact_seed = int(exact_seed)
        self.torch = torch
        self.np = np
        self.random = random_module
        self.evidence = None

    def predict(self, context, history, future, seed, instruction):
        global RNG_ISOLATION_EVIDENCE
        if type(seed) is not int or seed != self.exact_seed:
            raise RuntimeError("rng proxy exact seed")
        device_count = int(self.torch.cuda.device_count()) if self.torch.cuda.is_available() else 0
        device_indices = list(range(device_count))
        python_entry_state = self.random.getstate()
        numpy_entry_state = self.np.random.get_state()
        entry = _all_rng_state(self.torch, self.np, self.random)
        inside_before = internal_after = exit_restored = None
        output = None
        delegate_error = None
        try:
            with self.torch.random.fork_rng(devices=device_indices, enabled=True):
                self.torch.manual_seed(self.exact_seed)
                if device_count:
                    self.torch.cuda.manual_seed_all(self.exact_seed)
                inside_before = _all_rng_state(self.torch, self.np, self.random)
                try:
                    CALL_ACCOUNTING["sample0_calls_started"] = 1
                    output = self.delegate.predict(context, history, future, seed, instruction)
                    CALL_ACCOUNTING["sample0_calls_completed"] = 1
                except BaseException as error:
                    delegate_error = error
                finally:
                    internal_after = _all_rng_state(self.torch, self.np, self.random)
        finally:
            self.random.setstate(python_entry_state)
            self.np.random.set_state(numpy_entry_state)
            exit_restored = _all_rng_state(self.torch, self.np, self.random)
            self.evidence = {
                "format": "strict-track2-v519-rng-isolation-four-stage-v1",
                "exact_seed": self.exact_seed,
                "cuda_device_count": device_count,
                "cuda_device_indices": device_indices,
                "fork_rng_devices_exact_all_cuda_indices": True,
                "delegate_invocations_started": CALL_ACCOUNTING["sample0_calls_started"],
                "delegate_invocations_completed": CALL_ACCOUNTING["sample0_calls_completed"],
                "entry_external": entry,
                "inside_before_delegate": inside_before,
                "internal_after_delegate": internal_after,
                "exit_restored": exit_restored,
                "python_all_four_stages_equal": inside_before is not None and internal_after is not None and entry["python"] == inside_before["python"] == internal_after["python"] == exit_restored["python"],
                "numpy_all_four_stages_equal": inside_before is not None and internal_after is not None and entry["numpy"] == inside_before["numpy"] == internal_after["numpy"] == exit_restored["numpy"],
                "torch_internal_change_allowed": True,
                "python_exit_restored": exit_restored["python"] == entry["python"],
                "numpy_exit_restored": exit_restored["numpy"] == entry["numpy"],
                "torch_cpu_exit_restored": exit_restored["torch"]["cpu"] == entry["torch"]["cpu"],
                "torch_cuda_exit_restored": exit_restored["torch"]["cuda"] == entry["torch"]["cuda"],
                "exception_finally_restoration_complete": True,
            }
            RNG_ISOLATION_EVIDENCE = self.evidence
        if not self.evidence["python_all_four_stages_equal"] or not self.evidence["numpy_all_four_stages_equal"]:
            raise RuntimeError("rng proxy python/numpy internal drift")
        if not all(self.evidence[key] for key in ("python_exit_restored", "numpy_exit_restored", "torch_cpu_exit_restored", "torch_cuda_exit_restored")):
            raise RuntimeError("rng proxy external restoration")
        if delegate_error is not None:
            raise delegate_error
        return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_forensic_repair_supersession(rng_forensic: dict) -> dict:
    repair_research = rng_forensic.get("repair_research", {})
    original_sequence = [
        "random.seed(exact_sample_seed)",
        "numpy.random.seed(exact_sample_seed_mod_2_pow_32)",
        "torch.manual_seed(exact_sample_seed)",
        "torch.cuda.manual_seed_all(exact_sample_seed)_if_cuda_available",
        "delegate_runtime.predict_exactly_once",
    ]
    if repair_research.get("recommended_minimal_semantics") != FORENSIC_REPAIR_SUPERSESSION["original_recommendation"] or repair_research.get("inside_context_seed_sequence") != original_sequence:
        raise RuntimeError("v518 forensic original repair research")
    if set(FORENSIC_REPAIR_SUPERSESSION) != {"forensic_is_immutable", "original_recommendation", "original_seed_entries_exact", "rejected_by_target_review", "decision", "semantic_diff_count", "semantic_diffs", "semantic_diffs_canonical_sha256", "corrected_semantics"}:
        raise RuntimeError("v519 forensic repair supersession keyset")
    if FORENSIC_REPAIR_SUPERSESSION.get("decision") != "reject_py_numpy_seeding" or FORENSIC_REPAIR_SUPERSESSION.get("semantic_diff_count") != 2 or canonical_sha256(FORENSIC_REPAIR_SUPERSESSION["semantic_diffs"]) != FORENSIC_REPAIR_SUPERSESSION["semantic_diffs_canonical_sha256"]:
        raise RuntimeError("v519 forensic repair supersession")
    return FORENSIC_REPAIR_SUPERSESSION


def exact_record(path: Path, expected_sha: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"nonregular input: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"input drift: {path}")
    return {"path": str(path.resolve()), "sha256": actual_sha, "logical_bytes": path.stat().st_size}


def load_exact(path: Path, name: str, expected_sha: str):
    record = exact_record(path, expected_sha)
    source = path.read_bytes()
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, record


def tree_snapshot(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("package root not a regular directory")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"package member type: {path}")
        if path.is_file():
            inventory.append([path.relative_to(root).as_posix(), sha256_file(path), path.stat().st_size])
    lines = "".join(f"{digest}  {relative}\n" for relative, digest, _ in inventory).encode()
    return {
        "file_count": len(inventory),
        "logical_file_bytes": sum(row[2] for row in inventory),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": canonical_sha256(inventory),
        "inventory": inventory,
    }


def sample0_dataset_snapshot(parent: dict) -> dict:
    selection_record = parent["dataset"]["selection"]
    frozen = parent["temporal_contexts"][0]
    paths = [
        ("selection", Path(selection_record["path"]), selection_record["sha256"]),
        ("row_npz", Path(frozen["row_npz_path"]), frozen["row_npz_sha256"]),
        ("row_receipt", Path(frozen["row_receipt_path"]), frozen["row_receipt_sha256"]),
    ]
    records = []
    for role, path, expected_sha in paths:
        record = exact_record(path, expected_sha)
        records.append({"role": role, **record})
    return {"records": records, "canonical_json_digest_sha256": canonical_sha256(records)}


def package_manifest_matches(snapshot: dict, manifest: dict) -> bool:
    keys = (
        "file_count",
        "logical_file_bytes",
        "sha256sum_lines_digest_sha256",
        "canonical_json_triples_digest_sha256",
        "inventory",
    )
    return all(snapshot.get(key) == manifest.get(key) for key in keys)


def sample0(parent: dict, worker):
    import numpy as np

    selection_path = Path(parent["dataset"]["selection"]["path"])
    selection = json.loads(selection_path.read_text())
    spec = selection["contexts"][0]
    frozen = parent["temporal_contexts"][0]
    row = Path(frozen["row_npz_path"])
    with np.load(row, allow_pickle=False) as data:
        history = np.ascontiguousarray(data["history_actions"])
        future = np.ascontiguousarray(data["future_actions"][0])
        stored = np.asarray(data["pre_future_context_rgb"])
    context = np.repeat(np.ascontiguousarray(stored[0, 0])[None], 5, axis=0)
    seed = worker.stable_seed(spec["episode"], spec["start"])
    if seed != EXPECTED_SAMPLE0_SEED:
        raise RuntimeError("sample0 seed drift")
    if worker.raw_sha(context) != frozen["constructed_repeat5_context_sha256"]:
        raise RuntimeError("sample0 context drift")
    if worker.raw_sha(history) != spec["history_action_sha256"]:
        raise RuntimeError("sample0 history drift")
    if worker.raw_sha(future) != spec["branch_action_sha256"]["factual"]:
        raise RuntimeError("sample0 future drift")
    return context, history, future, seed, str(spec["instruction"])


def parse_args():
    parser = argparse.ArgumentParser()
    for name in (
        "self-source",
        "v509-preregistration",
        "parent-preregistration",
        "parent-failure",
        "authority-receipt",
        "scope-source",
        "worker-source",
        "package-manifest",
        "failure-forensic",
        "rng-failure-forensic",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "self-sha",
        "v509-preregistration-sha",
        "parent-preregistration-sha",
        "parent-failure-sha",
        "authority-receipt-sha",
        "scope-sha",
        "worker-sha",
        "package-manifest-sha",
        "failure-forensic-sha",
        "rng-failure-forensic-sha",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args()


def validate_v509_authority_actual_schema(authority: dict) -> None:
    authority_authorization = authority.get("authorization")
    authority_checks = authority.get("checks")
    authority_source_closure = authority.get("source_closure")
    authority_source_role_order = authority.get("source_role_order")
    if (
        authority.get("passed") is not True
        or authority.get("phase_a_cache_qualification_authorized") is not True
        or not isinstance(authority_authorization, dict)
        or authority_authorization.get("phase_a_cache_qualification_launcher_authorized") is not True
        or type(authority_authorization.get("launcher_invocations_authorized")) is not int
        or authority_authorization.get("launcher_invocations_authorized") != 1
        or type(authority_authorization.get("launcher_invocations_consumed")) is not int
        or authority_authorization.get("launcher_invocations_consumed") != 0
        or type(authority_authorization.get("nested_phase_a_driver_invocations_authorized")) is not int
        or authority_authorization.get("nested_phase_a_driver_invocations_authorized") != 1
        or authority_authorization.get("nested_phase_a_driver_only_via_launcher") is not True
        or authority_authorization.get("retry_authorized") is not False
        or len(authority) != 61
        or canonical_sha256(sorted(authority)) != "7bb77c508c1243582e394817dd55efbb7d7611cd875a30509c9bb4eca793058f"
        or len(authority_authorization) != 21
        or canonical_sha256(sorted(authority_authorization)) != "c9431aaab372d2ce2b1376a3096a5cb3b3cb4bc022aa2a5f8918a988014088ad"
        or not isinstance(authority_checks, dict)
        or len(authority_checks) != 33
        or canonical_sha256(sorted(authority_checks)) != "813fe53c5f288e2aef5c75fd2a6e14a3e0fb7a49e7c4e3f6a2738caa4a410bbb"
        or any(value is not True for value in authority_checks.values())
        or not isinstance(authority_source_closure, dict)
        or len(authority_source_closure) != 22
        or canonical_sha256(sorted(authority_source_closure)) != "a397e9bdbfb819d38c10800d65e195ffbeaf9eee97a9f69fdb08bbc2b95d596c"
        or not isinstance(authority_source_role_order, list)
        or len(authority_source_role_order) != 22
        or canonical_sha256(authority_source_role_order) != "996d1e54749b62f977d7be90e21a6dc856d5a38354cba5bc7a240ccc61ee6e0e"
        or set(authority_source_role_order) != set(authority_source_closure)
    ):
        raise RuntimeError("v509 authority actual-schema anchor")


def main() -> int:
    global RNG_ISOLATION_EVIDENCE
    CALL_ACCOUNTING.update({"sample0_calls_started": 0, "sample0_calls_completed": 0})
    RNG_ISOLATION_EVIDENCE = None
    # This gate intentionally precedes every import that can import torch.
    if dict(os.environ) != CHILD_OBSERVED_ENVIRONMENT_EXACT:
        raise RuntimeError(f"child observed environment is not exact3: {sorted(os.environ)}")
    if "torch" in sys.modules or "numpy" in sys.modules:
        raise RuntimeError("torch/numpy imported before exact2 environment gate")
    args = parse_args()
    for name, expected_path in CANONICAL_INPUT_PATHS.items():
        if getattr(args, name) != expected_path:
            raise RuntimeError(f"noncanonical input path: {name}")
    records = {
        "self_source": exact_record(args.self_source, args.self_sha),
        "v509_preregistration": exact_record(args.v509_preregistration, args.v509_preregistration_sha),
        "parent_preregistration": exact_record(args.parent_preregistration, args.parent_preregistration_sha),
        "parent_failure": exact_record(args.parent_failure, args.parent_failure_sha),
        "authority_receipt": exact_record(args.authority_receipt, args.authority_receipt_sha),
        "scope_source": exact_record(args.scope_source, args.scope_sha),
        "worker_source": exact_record(args.worker_source, args.worker_sha),
        "package_manifest": exact_record(args.package_manifest, args.package_manifest_sha),
        "failure_forensic": exact_record(args.failure_forensic, args.failure_forensic_sha),
        "rng_failure_forensic": exact_record(args.rng_failure_forensic, args.rng_failure_forensic_sha),
    }
    forensic = json.loads(args.failure_forensic.read_text())
    if forensic.get("status") != "immutable_failed_no_retry_read_only_forensic" or forensic.get("retry_authorized") is not False:
        raise RuntimeError("v509 failure forensic")
    rng_forensic = json.loads(args.rng_failure_forensic.read_text())
    if rng_forensic.get("status") != "immutable_failed_no_retry_read_only_forensic" or rng_forensic.get("failure_summary", {}).get("error") != "v485 v169 predict changed CPU/CUDA RNG state" or rng_forensic.get("authorization", {}).get("retry_authorized") is not False:
        raise RuntimeError("v518 rng failure forensic")
    validate_forensic_repair_supersession(rng_forensic)
    authority = json.loads(args.authority_receipt.read_text())
    validate_v509_authority_actual_schema(authority)
    formal = json.loads(args.v509_preregistration.read_text())
    parent = json.loads(args.parent_preregistration.read_text())
    if formal.get("qualification_output_root") != "/root/v509_v508_phase_a_cache_qualification_seed1650_20260825":
        raise RuntimeError("v509 preregistration anchor")
    package_manifest = json.loads(args.package_manifest.read_text())
    package_before = tree_snapshot(PACKAGE_ROOT)
    if not package_manifest_matches(package_before, package_manifest):
        raise RuntimeError("package manifest prestate")
    dataset_before = sample0_dataset_snapshot(parent)

    worker, _ = load_exact(args.worker_source, "v510_frozen_worker", args.worker_sha)
    scope, _ = load_exact(args.scope_source, "v510_frozen_scope", args.scope_sha)
    runtime_path = Path(parent["source"]["runtime_path"])
    runtime_sha = parent["source"]["runtime_sha256"]
    runtime_module, runtime_record = load_exact(runtime_path, "v510_parent_runtime", runtime_sha)
    closure_path = Path(parent["v169"]["closure_path"])
    closure_record = exact_record(closure_path, parent["v169"]["closure_sha256"])
    closure = json.loads(closure_path.read_text())
    if runtime_module.verify_v169(closure) != parent["v169"]["closure_digest"]:
        raise RuntimeError("v169 closure")

    import gc
    import random
    import numpy as np
    import torch

    context, history, future, seed, instruction = sample0(parent, worker)
    torch.use_deterministic_algorithms(True, warn_only=False)
    strict = {"enabled": True, "warn_only": False}
    if scope.strict_state() != strict:
        raise RuntimeError("strict deterministic prestate")
    runtime = None
    try:
        runtime = runtime_module.Track2V169ArmRoutedRuntime(
            parent["v169"]["release_path"], parent["v169"]["library_path"], "cuda:0"
        )
        proxy = RngIsolationRuntimeProxy(runtime, seed, torch, np, random)
        output, event = scope.predict_one(proxy, context, history, future, seed, instruction)
    finally:
        if runtime is not None:
            del runtime
        torch.use_deterministic_algorithms(True, warn_only=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if scope.strict_state() != strict or event["warning"] != EXPECTED_WARNING:
        raise RuntimeError("raw warning or strict-mode poststate")
    if output.shape != (8, 256, 256, 3) or output.dtype != np.uint8:
        raise RuntimeError("sample0 output schema")
    package_after = tree_snapshot(PACKAGE_ROOT)
    dataset_after = sample0_dataset_snapshot(parent)
    if package_after != package_before or dataset_after != dataset_before:
        raise RuntimeError("source/package/dataset drift")
    receipt = {
        "format": "strict-track2-v519-v518-phase-a-sample0-rng-isolation-child-v1",
        "status": "passed_one_isolated_sample0_call_no_cache",
        "passed": True,
        "spawn_environment_exact": dict(SPAWN_ENVIRONMENT_EXACT),
        "spawn_environment_exact_key_count": 2,
        "child_observed_environment": dict(CHILD_OBSERVED_ENVIRONMENT_EXACT),
        "child_observed_environment_key_count": 3,
        "interpreter_added_environment": dict(INTERPRETER_ADDED_ENVIRONMENT_EXACT),
        "interpreter_added_environment_key_count": 1,
        "environment_checked_before_torch_import": True,
        "sample_id": 0,
        "branch": "factual",
        "calls": 1,
        "sample0_calls_started": 1,
        "sample0_calls_completed": 1,
        "raw_warnings_ordered": [event["warning"]],
        "raw_warning_count": 1,
        "canonical_warning_count": 1,
        "filtered_warning_count": 0,
        "deduplicated_warning_count": 0,
        "other_warning_count": 0,
        "rng_unchanged": event["rng_unchanged"],
        "rng_isolation_evidence": proxy.evidence,
        "v518_forensic_repair_research_supersession": FORENSIC_REPAIR_SUPERSESSION,
        "strict_mode_before": event["mode_before"],
        "strict_mode_during": event["mode_during"],
        "strict_mode_after": event["mode_after"],
        "output_schema": {"shape": list(output.shape), "dtype": str(output.dtype), "sha256": worker.raw_sha(output)},
        "cache_artifact_created": False,
        "source_records": records,
        "runtime_source": runtime_record,
        "v169_closure": {**closure_record, "closure_digest": parent["v169"]["closure_digest"]},
        "package_tree_pre": package_before,
        "package_tree_post": package_after,
        "package_tree_prepost_exactly_equal": True,
        "sample0_dataset_pre": dataset_before,
        "sample0_dataset_post": dataset_after,
        "sample0_dataset_prepost_exactly_equal": True,
        "training_launched": False,
        "cache_reused": False,
        "reward_read": False,
        "dev_hidden_final_outcome_read": False,
        "folds": 0,
        "policy_updates": 0,
        "service_control_invocations": 0,
        "retry_authorized": False,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def entrypoint(main_fn=main):
    try:
        return main_fn()
    except Exception as error:
        print(json.dumps(child_failure_receipt(error), sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        return 79


if __name__ == "__main__":
    raise SystemExit(entrypoint())
