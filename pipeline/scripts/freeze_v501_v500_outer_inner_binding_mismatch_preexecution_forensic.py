#!/usr/bin/env python3
"""Freeze the v500 outer/inner binding mismatch before any reconciliation attempt."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import stat
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS = ROOT / "pipeline/scripts"
SELF = SCRIPTS / "freeze_v501_v500_outer_inner_binding_mismatch_preexecution_forensic.py"
FORENSIC_ROOT = J / "v501_v500_outer_inner_binding_mismatch_preexecution_forensic_seed1643_20260825"
FORENSIC_PREP = FORENSIC_ROOT.with_name(FORENSIC_ROOT.name + ".registration-prep")
RECEIPT = FORENSIC_ROOT / "preexecution_forensic.json"

CONTRACT = SCRIPTS / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority_contract.json"
CONTRACT_EXACT = ("55bc38e1e101e97761ac8b8f4e5200b0ccd5a94676f27fc7a3123ce179df2e1e", 111329)
MATERIALIZER = SCRIPTS / "materialize_v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority.py"
MATERIALIZER_EXACT = ("e717c1360e5691865dd299249630b3b4cef1965c68e8a457008b50415fe9afb5", 109256)
OLD_INNER = SCRIPTS / "launch_v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_inner.py"
OLD_INNER_EXACT = ("132a91069dc219f70adc6a01292084a97b1e547f16d4cc0d3a6f28df231005ab", 124674)
OLD_OUTER = SCRIPTS / "launch_v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_outer.py"
OLD_OUTER_EXACT = ("3d8949268c1eb3b8e9dfd84d9aea5084ea6a299fb48bb554972cfe0cc6f7d458", 85248)
HELPER = SCRIPTS / "invoke_v500_failure_tree_diagnostic_execution_authority_materializer_once.py"
HELPER_EXACT = ("c53d750c52db061b79280205c83c6303956e21c8762aa66f60dfcf873131c686", 38766)
SCRIPT = Path("/root/v500_failure_tree_diagnostic_authority_materialize_once.sh")
SCRIPT_EXACT = ("6b3cc44a26ba0129e8cc02db7aefc56ff51373e0b616bf09de12ac1e78f8c605", 2543)
DEPLOYER = Path("/root/v500_authority_sources_corrected_atomic_deployer.py")
DEPLOYER_EXACT = ("8666cfed7e924deedc9229dd6a6349b595012e9812de3b167fcf889c994310d2", 24395)
AUTHORITY_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority_seed1642_20260825"
AUTHORITY_RECEIPT = AUTHORITY_ROOT / "authority_receipt.json"
AUTHORITY_EXACT = ("8c57944a115bcb26fd3709a0a7911a2ea58be2e82e238c0b6fffad04f5b22179", 222774)
HELPER_EVIDENCE = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority_materialization_evidence_seed1642_20260825"
HELPER_PROCESS = HELPER_EVIDENCE / "process_receipt.json"
HELPER_PROCESS_EXACT = ("0fab24fca5e3b8a975f9cb4e32109297cb7b1094d0fb9a7c9289320633e46b11", 79870)
DEPLOY_EVIDENCE = J / "v500_authority_sources_corrected_atomic_deployment_evidence_seed1642_20260825"
DEPLOY_PROCESS = DEPLOY_EVIDENCE / "process_receipt.json"
DEPLOY_PROCESS_EXACT = ("7e92b12a68bc1c150fcbfa7b6f0ff6cde7788000f4b92ddfc31693663f94d893", 31333)
F813_ROOT = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824"
F813_EXACT2 = [
    ["immutable_evidence/v485_static_b73.log", "7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b", 6475],
    ["preregistration.json", "f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8", 21296],
]
OUTER_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_outer_execution_evidence_seed1642_20260825"
OUTER_PREP = OUTER_ROOT.with_name(OUTER_ROOT.name + ".outer-prep")
INNER_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_inner_attempt_seed1642_20260825"
INNER_PREP = INNER_ROOT.with_name(INNER_ROOT.name + ".attempt-prep")
TRANSPARENT = F813_ROOT / "transparent_static_audit.json"
TRANSPARENT_TMP = TRANSPARENT.with_name(TRANSPARENT.name + ".tmp")
QUALIFICATION = Path("/root/v485_v169_cache_qualification_seed1627_20260824")
OLD_LITERAL = {"sha256": "c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f", "logical_bytes": 101600}


class ControlledSignal(BaseException):
    pass


def cbytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def csha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: tuple[str, int] | None = None) -> dict:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(f"regular: {path}")
    row = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if expected is not None and (row["sha256"], row["logical_bytes"]) != expected:
        raise RuntimeError(f"record mismatch: {path}")
    return row


def exact_tree(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"tree root: {root}")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"symlink: {path}")
        if path.is_file():
            if not stat.S_ISREG(os.lstat(path).st_mode):
                raise RuntimeError(f"nonregular: {path}")
            inventory.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"member: {path}")
    lines = "".join(f"{digest}  {name}\n" for name, digest, _ in inventory).encode()
    return {"root": str(root), "inventory": inventory, "file_count": len(inventory),
            "logical_file_bytes": sum(row[2] for row in inventory),
            "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
            "canonical_json_triples_digest_sha256": csha(inventory)}


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0: raise RuntimeError("short write")
            view = view[count:]
        os.fsync(descriptor)
    finally: os.close(descriptor)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def immutable_snapshot() -> dict:
    sources = {
        "self": regular(SELF), "v500_contract": regular(CONTRACT, CONTRACT_EXACT),
        "v500_materializer": regular(MATERIALIZER, MATERIALIZER_EXACT),
        "v500_inner": regular(OLD_INNER, OLD_INNER_EXACT), "v500_outer": regular(OLD_OUTER, OLD_OUTER_EXACT),
        "v500_helper": regular(HELPER, HELPER_EXACT), "v500_script": regular(SCRIPT, SCRIPT_EXACT),
        "v500_deployer": regular(DEPLOYER, DEPLOYER_EXACT),
    }
    trees = {"authority": exact_tree(AUTHORITY_ROOT), "helper_evidence": exact_tree(HELPER_EVIDENCE),
             "deploy_evidence": exact_tree(DEPLOY_EVIDENCE), "f813": exact_tree(F813_ROOT)}
    absences = {str(path): not os.path.lexists(path) for path in
                [OUTER_ROOT, OUTER_PREP, INNER_ROOT, INNER_PREP, TRANSPARENT, TRANSPARENT_TMP, QUALIFICATION]}
    return {"sources": sources, "trees": trees, "absences": absences}


def build_receipt(snapshot_builder=immutable_snapshot) -> dict:
    input_pre_snapshot = snapshot_builder()
    outer = load("v500_outer_forensic", OLD_OUTER)
    deployer = load("v500_deployer_forensic", DEPLOYER)
    contract = json.loads(CONTRACT.read_text())
    authority = json.loads(AUTHORITY_RECEIPT.read_text())
    helper_process = json.loads(HELPER_PROCESS.read_text())
    deploy_process = json.loads(DEPLOY_PROCESS.read_text())
    derived_argv = outer.inner_command({"contract": regular(CONTRACT, CONTRACT_EXACT),
                                        "authority": regular(AUTHORITY_RECEIPT, AUTHORITY_EXACT)})
    stale_binding = {"path": str(OLD_INNER), **OLD_LITERAL}
    actual = regular(OLD_INNER, OLD_INNER_EXACT)
    contract_binding = contract["source_closure"]["corrected_inner_wrapper"]
    authority_binding = authority["corrected_inner_wrapper_source"]
    derived_sha = derived_argv[derived_argv.index("--wrapper-sha") + 1]
    f813 = exact_tree(F813_ROOT)
    checks = {
        "old_outer_current": regular(OLD_OUTER, OLD_OUTER_EXACT)["sha256"] == OLD_OUTER_EXACT[0],
        "old_outer_literal_stale": stale_binding != actual,
        "actual_contract_authority_inner_equal": actual == contract_binding == authority_binding,
        "derived_argv_uses_stale_sha": derived_sha == OLD_LITERAL["sha256"] and derived_sha != actual["sha256"],
        "authority_exact1_current": exact_tree(AUTHORITY_ROOT)["file_count"] == 1 and regular(AUTHORITY_RECEIPT, AUTHORITY_EXACT)["sha256"] == AUTHORITY_EXACT[0],
        "helper_evidence_exact6": exact_tree(HELPER_EVIDENCE)["file_count"] == 6 and regular(HELPER_PROCESS, HELPER_PROCESS_EXACT)["sha256"] == HELPER_PROCESS_EXACT[0],
        "helper_partition_zero_chain": helper_process["authority_materializer_invocations"] == 1 and helper_process["outer_wrapper_invocations"] == 0 and helper_process["corrected_inner_wrapper_invocations"] == 0 and helper_process["reconciler_r2_invocations"] == 0,
        "deploy_evidence_exact6": exact_tree(DEPLOY_EVIDENCE)["file_count"] == 6 and regular(DEPLOY_PROCESS, DEPLOY_PROCESS_EXACT)["sha256"] == DEPLOY_PROCESS_EXACT[0],
        "deploy_partition_zero_chain": deploy_process["deployer_invocations"] == 1 and deploy_process["authority_materializer_invocations"] == 0 and deploy_process["outer_wrapper_invocations"] == 0 and deploy_process["corrected_inner_wrapper_invocations"] == 0 and deploy_process["reconciler_r2_invocations"] == 0,
        "outer_inner_r2_calls_zero": all(not os.path.lexists(path) for path in [OUTER_ROOT, OUTER_PREP, INNER_ROOT, INNER_PREP, TRANSPARENT, TRANSPARENT_TMP]),
        "qualification_absent": not os.path.lexists(QUALIFICATION),
        "f813_exact2": f813["inventory"] == F813_EXACT2 and f813["file_count"] == 2,
        "pid_gpu_empty": not deployer.live_target_processes() and not deployer.gpu_processes(),
        "old_outer_invalid_unconsumed": True,
        "no_execution_authority": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    runtime = {"preexecution_forensic_materialized": True, "v500_outer_executed": False,
               "v500_inner_executed": False, "reconciler_r2_executed": False,
               "transparent_receipt_created": False, "phase_a_executed": False,
               "training_launched": False, "folds": 0, "policy_updates": 0}
    authorization = {"forensic_materialization_authorized": False, "v500_outer_authorized": False,
                     "v500_inner_authorized": False, "reconciler_r2_authorized": False,
                     "retry_authorized": False, "phase_a_authorized": False,
                     "training_authorized": False, "reward_read_authorized": False,
                     "dev_hidden_final_outcome_read_authorized": False}
    receipt = {
        "format": "strict-track2-v501-v500-outer-inner-binding-mismatch-preexecution-forensic-v1",
        "status": "frozen_invalid_unconsumed_v500_outer_inner_binding_mismatch_no_execution_authority",
        "passed": True, "seed": 1643,
        "forensic_source": regular(SELF), "v500_outer_source": regular(OLD_OUTER, OLD_OUTER_EXACT),
        "v500_inner_source": actual, "stale_inner_binding_literal": stale_binding,
        "contract_inner_binding": contract_binding, "authority_inner_binding": authority_binding,
        "derived_inner_argv": derived_argv, "derived_wrapper_sha": derived_sha,
        "v500_authority_receipt": regular(AUTHORITY_RECEIPT, AUTHORITY_EXACT),
        "v500_authority_registration_tree": exact_tree(AUTHORITY_ROOT),
        "v500_helper_process_receipt": regular(HELPER_PROCESS, HELPER_PROCESS_EXACT),
        "v500_helper_evidence_tree": exact_tree(HELPER_EVIDENCE),
        "v500_deploy_process_receipt": regular(DEPLOY_PROCESS, DEPLOY_PROCESS_EXACT),
        "v500_deploy_evidence_tree": exact_tree(DEPLOY_EVIDENCE),
        "f813_registration_tree": f813,
        "required_absences": {"v500_outer_root": str(OUTER_ROOT), "v500_outer_prep": str(OUTER_PREP),
                              "v500_inner_root": str(INNER_ROOT), "v500_inner_prep": str(INNER_PREP),
                              "transparent": str(TRANSPARENT), "transparent_tmp": str(TRANSPARENT_TMP),
                              "qualification": str(QUALIFICATION)},
        "checks": checks, "check_keys": sorted(checks), "check_key_set_sha256": csha(sorted(checks)),
        "checks_sha256": csha(checks), "runtime_observation": runtime, "authorization": authorization,
        "root_cause": "v500 outer frozen INNER_WRAPPER_SHA and INNER_WRAPPER_BYTES remained c9b3/101600 while actual, contract, and authority all bind 132a/124674",
        "corrected_boundary": "fresh v501 authority authorizes only fresh v501 outer to fresh v501 inner to exact 9efc r2 once; v500 outer is invalid_unconsumed and permanently forbidden",
        "read_only_probe_disclosure": {"first_probe_returncode": 1, "first_probe_error": "AttributeError: module outer has no attribute inner_argv", "first_probe_mutation": False, "corrected_probe_returncode": 0},
    }
    input_post_snapshot = snapshot_builder()
    if input_pre_snapshot != input_post_snapshot:
        raise RuntimeError("forensic input snapshot drift")
    receipt.update({
        "input_pre_snapshot": input_pre_snapshot,
        "input_post_snapshot": input_post_snapshot,
        "input_snapshots_exactly_equal": True,
        "input_pre_snapshot_sha256": csha(input_pre_snapshot),
        "input_post_snapshot_sha256": csha(input_post_snapshot),
    })
    return receipt


def main() -> int:
    if os.path.lexists(FORENSIC_ROOT) or os.path.lexists(FORENSIC_PREP):
        raise RuntimeError("forensic prestate")
    receipt = build_receipt()
    pre = receipt["input_pre_snapshot"]
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(signum, _frame): raise ControlledSignal(f"signal {signum}")
    for sig in old: signal.signal(sig, interrupted)
    baseline = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    prep_identity = None
    try:
        FORENSIC_PREP.mkdir(mode=0o700)
        observed = os.lstat(FORENSIC_PREP); prep_identity = (observed.st_dev, observed.st_ino)
        fsync_dir(FORENSIC_PREP.parent)
        write_exclusive(FORENSIC_PREP / "preexecution_forensic.json", cbytes(receipt))
        fsync_dir(FORENSIC_PREP)
        postwrite = immutable_snapshot()
        if (pre != postwrite or csha(postwrite) != receipt["input_post_snapshot_sha256"]
                or exact_tree(FORENSIC_PREP)["file_count"] != 1):
            raise RuntimeError("prepromote drift")
        os.replace(FORENSIC_PREP, FORENSIC_ROOT); fsync_dir(FORENSIC_ROOT.parent)
        tree = exact_tree(FORENSIC_ROOT)
        if tree["file_count"] != 1 or tree["inventory"][0][0] != "preexecution_forensic.json" or os.path.lexists(FORENSIC_PREP):
            raise RuntimeError("forensic exact1")
        for sig in old: signal.signal(sig, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
        print(json.dumps({"passed": True, "receipt": regular(RECEIPT), "registration_tree": tree}, sort_keys=True))
        return 0
    except BaseException:
        if prep_identity is not None and FORENSIC_PREP.is_dir() and not FORENSIC_PREP.is_symlink():
            current = os.lstat(FORENSIC_PREP)
            if (current.st_dev, current.st_ino) == prep_identity:
                for child in FORENSIC_PREP.iterdir(): child.unlink()
                FORENSIC_PREP.rmdir(); fsync_dir(FORENSIC_PREP.parent)
        raise
    finally:
        if not os.path.lexists(FORENSIC_ROOT):
            try: signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
            except BaseException: pass
            for sig, handler in old.items(): signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
