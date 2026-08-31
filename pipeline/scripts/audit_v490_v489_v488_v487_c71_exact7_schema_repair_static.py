#!/usr/bin/env python3
"""Read-only postregistration audit for the v488 c71 exact7-schema repair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
FORMAL_PATH = J / "v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json"
FORMAL_SHA = "b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b"
FORMAL_BYTES = 36181
CONTRACT_PATH = ROOT / "pipeline/scripts/v488_v487_c71_exact7_schema_repair_contract.json"
CONTRACT_SHA = "197e90ea10bd87d65a32314dc97a06559dd3f994c25cde95fd3622c852f03dfa"
CONTRACT_BYTES = 10874
MATERIALIZER_PATH = ROOT / "pipeline/scripts/materialize_v488_v487_c71_exact7_schema_repair_preregistration.py"
MATERIALIZER_SHA = "9181bfac0d66541ab9d2083617cde84e87bf7369a2630a5e29bf88c918934710"
MATERIALIZER_BYTES = 13802
R2_PATH = ROOT / "pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py"
R2_SHA = "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377"
R2_BYTES = 51845
STATIC_PATH = ROOT / "pipeline/scripts/audit_v490_v489_v488_v487_c71_exact7_schema_repair_static.py"
STATIC_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825"
STATIC_PREP = STATIC_ROOT.with_name(STATIC_ROOT.name + ".registration-prep")
OUTPUT_PATH = STATIC_ROOT / "static_audit.json"
EVIDENCE_ROOT = J / "v488_v487_c71_exact7_schema_repair_materialization_evidence_seed1630_20260825"
AUTHORITY_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_authority_seed1632_20260825"
ATTEMPT_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")
OLD_C71_PATH_LITERAL = "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/reconcile_v486_v485_static_false_positive.py"
OLD_C71_PATH = Path(OLD_C71_PATH_LITERAL)
OLD_C71_SHA = "c71ba00b0c92efda03f9df149263d0c476de86ea7338ba3719a72f3305959984"
OLD_C71_BYTES = 36229
SUPERSEDED_STATIC_SOURCE_PATH = ROOT / "pipeline/scripts/audit_v488_v487_c71_exact7_schema_repair_static.py"
SUPERSEDED_STATIC_SOURCE_SHA = "f88ea78918eef4474128740213b374c9642ed89a46c226e1f4cfbea47b0d8140"
SUPERSEDED_STATIC_SOURCE_BYTES = 34730
SUPERSEDED_STATIC_ROOT = J / "v488_v487_c71_exact7_schema_repair_static_audit_seed1630_20260825"
SUPERSEDED_STATIC_PREP = SUPERSEDED_STATIC_ROOT.with_name(SUPERSEDED_STATIC_ROOT.name + ".registration-prep")
FAILED_STATIC_EVIDENCE_ROOT = J / "v488_v487_c71_exact7_schema_repair_static_audit_execution_evidence_seed1630_20260825"
FAILED_STATIC_EVIDENCE_TREE = {"inventory":[["argv.json","a862aa93f3c653f1c042d534c77ccb231f762e07b6a514d3b76304052cc3c3be",8661],["intent.json","01974de97653ff41cb8a3c24caadfcd06f154673c5e5ec2c8a0c5ed6db40ff7d",161],["process_receipt.json","6e4ed26d978371ee138674d2923770edb5584cefce14bc726c5221b9fbd3eb11",356],["static_stderr.log","0eb74d5096993d6c480460b090baee16c38de01f0e5a295014702838fec76f6b",2070],["static_stdout.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["transport_helper.py","05c29a6df902baf54b41e9eebb13fd02fe0e7ec343ea0202af16c178f45bfe1c",9005]],"file_count":6,"logical_file_bytes":20253,"sha256sum_lines_digest_sha256":"15e3ca66299807edef3001470d3956ea36d10da33cffe8565084fd1042ea07f1","canonical_json_triples_digest_sha256":"8b2fcc83543367575a4b2aad0eea45855a9a565fd0dc02f1c497e22f8af6d8c9"}
FAILED_STATIC_PROCESS_RECEIPT_SHA = "6e4ed26d978371ee138674d2923770edb5584cefce14bc726c5221b9fbd3eb11"
FAILED_STATIC_PROCESS_RECEIPT_BYTES = 356
V489_FAILED_STATIC_SOURCE_PATH = ROOT / "pipeline/scripts/audit_v489_v488_v487_c71_exact7_schema_repair_static.py"
V489_FAILED_STATIC_SOURCE_SHA = "0e89d17d47a363a2d3a3b3891220bda28e46f311acfcb53a8eafa7cee66f9758"
V489_FAILED_STATIC_SOURCE_BYTES = 41914
V489_FAILED_STATIC_ROOT = J / "v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1631_20260825"
V489_FAILED_STATIC_PREP = V489_FAILED_STATIC_ROOT.with_name(V489_FAILED_STATIC_ROOT.name + ".registration-prep")
V489_FAILED_STATIC_EVIDENCE_ROOT = J / "v489_v488_v487_c71_exact7_schema_repair_static_audit_execution_evidence_seed1631_20260825"
V489_FAILED_STATIC_EVIDENCE_TREE = {"inventory":[["argv.json","f681ae04ce79caa6f063179f17cb4b10150c1865f436f608c9d8a202216eaf7b",9601],["intent.json","89969affbd9bf77106f7471248f98c6665f83f475e02e4e845234255946767b5",170],["process_receipt.json","65f6cb52f91d4c6e39697b70f3816e825c7836bf6fc4fa3153ed34f441796d85",11911],["static_stderr.log","cd509821caaeebb1732dce6966980823fbf1486ff734b89f48fb2452ca07ec54",654],["static_stdout.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["transport_helper.py","2a6f51542a0e82112506b310ba6ef36f82edf5c6ce562d2ab22c88eeb3ed270e",12472]],"file_count":6,"logical_file_bytes":34808,"sha256sum_lines_digest_sha256":"632abb3f7435642ee00570d76bf7575ae979fc52874e207bf4365078b86a3c28","canonical_json_triples_digest_sha256":"fe68fc07bf1691266166b8184c13a42a9e7bac6f43cbfe8b84eefbd1a59eb2e9"}
V489_FAILED_STATIC_PROCESS_RECEIPT_SHA = "65f6cb52f91d4c6e39697b70f3816e825c7836bf6fc4fa3153ed34f441796d85"
V489_FAILED_STATIC_PROCESS_RECEIPT_BYTES = 11911
V489_PREDEPLOY_TRANSPORT_FAILURE = {"corrected_transport_authorized":True,"deploy_temp_absent_after":True,"observed_error":"bash: line 1: test: too many arguments","occurred":True,"persistent_target_absent_after":True,"r2_invocations":0,"stage":"predeploy_remote_first_test_before_copy_or_static_invocation","state_mutations":0,"static_invocations":0,"transport":"PowerShell outer-double-quoted ssh command locally expanded remote dollar variables and command substitutions"}

FORMAT = "strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-static-audit-v1"
STATUS = "passed_no_execution_authority"
FORMAL_FORMAT = "strict-track2-v488-v487-c71-exact7-schema-repair-preregistration-v1"
FORMAL_STATUS = "preregistered_readonly_source_repair_pending_postregistration_authority"
CHECK_KEYS = sorted({
    "contract_current", "f813_tree_exact2", "failed_attempt_tree_exact4_no_retry",
    "formal_authority_all_false", "formal_record_exact", "formal_runtime_no_execution",
    "formal_schema_exact21", "materializer_current", "no_live_process",
    "no_training_reward_outcome", "observed_exact7_current", "r2_current",
    "r2_exact4_strict_no_fallback", "r2_forbidden_imports_calls_absent",
    "r2_old_c71_dual_binding", "r2_snapshot_dual_ancestry",
    "source_closure_digest_exact", "source_closure_exact8_current",
    "synthetic_tamper_suite_passed", "transparent_and_fresh_outputs_absent",
    "v487_authority_tree_exact1", "v487_helper_tree_exact5",
    "failed_static_tree_exact6", "failed_static_process_no_retry",
    "failed_static_invocation_partition", "failed_static_root_prep_absent",
    "path_literal_ast_repair_exact",
    "second_failed_static_tree_exact6", "second_failed_static_process_no_retry",
    "second_failed_static_invocation_partition", "second_failed_static_root_prep_absent",
    "predeploy_transport_failure_disclosed", "real_publish_callgraph_fixture_passed",
})
CHECK_KEYSET_SHA = "53b721e19caa9b6302986e65c78dac72e0026716d67852d1b886aba27492d7cf"
CHECKS_SHA = "50edf45d3a0021c3054ff80b2fcc311ebb848272def1e0732aa3aaf7c2079a58"
TOP_KEYS = {"format","status","passed","checks","check_keys","check_key_set_sha256","checks_sha256","repair_preregistration","design_contract","materializer_source","reconciler_r2_source","static_auditor_source","source_closure","source_closure_sha256","f813_registration_tree","v487_authority_tree","v487_helper_evidence_tree","failed_reconciliation_attempt_tree","ast_diff_proof","synthetic_evidence","required_absences","runtime_observation","readonly_reconciliation_authorized","training_authorized","submission_authorized","superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
FORMAL_TOP_KEYS = {"format","status","seed","design_contract","materializer_source","source_closure","source_closure_sha256","f813_registration_tree","v487_authority_tree","v487_helper_evidence_tree","failed_reconciliation_attempt_tree","failed_attempt_no_retry","transparent_static_receipt_path","exact7_schema_repair_proof","future_reconciler_r2_source","required_postregistration_authority","authorization","runtime_observation","input_pre_snapshot","input_post_snapshot","input_snapshots_exactly_equal"}
SOURCE_ROLES = {"parent_v485_preregistration","f813_reconciliation_preregistration","v487_authority_receipt","v487_authority_helper","v487_helper_process_receipt","old_reconciler_c71","old_v487_wrapper","future_reconciler_r2"}
FORMAL_AUTH = {"readonly_source_repair_authorized":False,"postregistration_authority_required":True,"attempts_authorized":0,"retry_old_v487_authorized":False,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
FORMAL_RUNTIME = {"repair_formal_registered":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
STATIC_RUNTIME = {"static_audit_executed":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
EVIDENCE_TREE = {"inventory":[["argv.json","7f2eed5ca469b82af447ecf90f9cca3546a8d7f8171ff2f4a2ece0805052643e",4920],["intent.json","cee2151eb334fd9bcfefaaedfa45d9cb49f8c18f191442b63d5121c2faecea34",488],["materializer_stderr.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["materializer_stdout.log","f856ab2c375b7be36c728ba4543d6dd54ca0ca0695f30ad337201051b81fca08",810],["process_receipt.json","0d44c2d77bafb75c2ad48aad8a655beda738c2872deb3911ef2c43def3ee9810",4678],["transport_helper.py","f99a4eacee73ad6d0a7637a850fb1414a4a7dc05aa0f2ce17ab72b3fff5df406",12663]],"file_count":6,"logical_file_bytes":23559,"sha256sum_lines_digest_sha256":"6a48564898da15021985dc82b201f3f10139de83f7dacafd306c86345d52e234","canonical_json_triples_digest_sha256":"fb0319b132d97da2744bc34d69db894f8f7d6def6d9e7ab5a88b4d609a876f68"}


class ControlledSignal(BaseException):pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def regular(path: Path | str, want_sha: str, want_bytes: int) -> dict:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or path.is_symlink() or sha(path) != want_sha or path.stat().st_size != want_bytes:
        raise RuntimeError(f"regular closure: {path}")
    return {"path":str(path),"sha256":want_sha,"logical_bytes":want_bytes}


def exact_tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root: {root}")
    rows=[]
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): raise RuntimeError(f"tree symlink: {path}")
        if path.is_file(): rows.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
        elif not path.is_dir(): raise RuntimeError(f"tree nonregular: {path}")
    lines="".join(f"{digest}  {rel}\n" for rel,digest,_ in rows).encode()
    triples=json.dumps(rows,separators=(",",":")).encode()
    return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(row[2] for row in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(triples).hexdigest()}


def fsync_dir(path: Path) -> None:
    if os.name=="nt":return
    fd=os.open(str(path),os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_json(path: Path,value) -> None:
    tmp=path.with_name(path.name+".tmp")
    if os.path.lexists(path) or os.path.lexists(tmp): raise FileExistsError(path)
    with tmp.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);fsync_dir(path.parent)


def directory_identity(path: Path) -> tuple[int,int]:
    stat=path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_dir():raise RuntimeError("directory ownership")
    return stat.st_dev,stat.st_ino


def cleanup_owned_prep(path: Path,identity: tuple[int,int]) -> None:
    if not path.exists() or path.is_symlink() or directory_identity(path)!=identity:return
    allowed={"static_audit.json","static_audit.json.tmp"}
    entries=list(path.iterdir())
    if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):return
    for entry in entries:entry.unlink()
    path.rmdir();fsync_dir(path.parent)


def literal_assignment(tree: ast.Module,name: str):
    matches=[node for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
    if len(matches)!=1 or not isinstance(matches[0].value,ast.Constant) or not isinstance(matches[0].value.value,(str,int,bool,type(None))):raise RuntimeError(f"literal assignment: {name}")
    return matches[0].value.value


def strict_path_assignment(tree:ast.Module,name:str,expected:str)->str:
    matches=[node for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
    if len(matches)!=1:raise RuntimeError(f"path assignment count: {name}")
    value=matches[0].value
    if not isinstance(value,ast.Call) or not isinstance(value.func,ast.Name) or value.func.id!="Path" or len(value.args)!=1 or value.keywords:raise RuntimeError(f"path assignment call: {name}")
    if not isinstance(value.args[0],ast.Constant) or not isinstance(value.args[0].value,str) or value.args[0].value!=expected:raise RuntimeError(f"path assignment value: {name}")
    return value.args[0].value


def main_function(tree: ast.Module) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="main")


def ast_call_name(node: ast.AST) -> str:
    if isinstance(node,ast.Name): return node.id
    if isinstance(node,ast.Attribute):
        prefix=ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def formal_boundary(value: dict) -> bool:
    if not isinstance(value,dict) or set(value)!=FORMAL_TOP_KEYS:raise RuntimeError("formal exact21")
    if value.get("format")!=FORMAL_FORMAT or value.get("status")!=FORMAL_STATUS or value.get("seed")!=1630:raise RuntimeError("formal identity")
    if value.get("authorization")!=FORMAL_AUTH or value.get("runtime_observation")!=FORMAL_RUNTIME:raise RuntimeError("formal state")
    if value.get("required_postregistration_authority") is not True or value.get("failed_attempt_no_retry") is not True:raise RuntimeError("formal authority boundary")
    return True


def exact4_boundary(value: dict,source_records: list[dict],source_map: dict) -> bool:
    keys={"all_records_must_equal_parent_formal_and_old_receipt_and_current_files","canonical_records_digest_sha256","records","roles_in_order"}
    if not isinstance(value,dict) or set(value)!=keys:raise RuntimeError("exact4 keyset")
    if value["all_records_must_equal_parent_formal_and_old_receipt_and_current_files"] is not True:raise RuntimeError("exact4 guard")
    if value["records"]!=source_records or value["roles_in_order"]!=[row["role"] for row in source_records]:raise RuntimeError("exact4 records/order")
    if value["canonical_records_digest_sha256"]!=csha(source_records):raise RuntimeError("exact4 digest")
    if {row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in value["records"]}!=source_map:raise RuntimeError("exact4 map")
    return True


def synthetic_suite(source_records: list[dict],source_map: dict,self_record: dict) -> dict:
    actual={"all_records_must_equal_parent_formal_and_old_receipt_and_current_files":True,"canonical_records_digest_sha256":csha(source_records),"records":source_records,"roles_in_order":[row["role"] for row in source_records]}
    old3={"execution_sources":source_map,"execution_source_records":source_records,"execution_sources_digest_sha256":csha(source_records)}
    checks={"actual4_pass":exact4_boundary(actual,source_records,source_map) is True}
    for name,mutated in {"old3_rejected":old3,"guard_false_rejected":{**actual,"all_records_must_equal_parent_formal_and_old_receipt_and_current_files":False},"role_reorder_rejected":{**actual,"roles_in_order":list(reversed(actual["roles_in_order"]))},"record_tamper_rejected":{**actual,"records":[*source_records[:-1],{**source_records[-1],"sha256":"0"*64}]},"digest_tamper_rejected":{**actual,"canonical_records_digest_sha256":"0"*64},"extra_key_rejected":{**actual,"fallback":False}}.items():
        try:exact4_boundary(mutated,source_records,source_map);checks[name]=False
        except RuntimeError:checks[name]=True
    old={"path":str(OLD_C71_PATH),"sha256":OLD_C71_SHA,"logical_bytes":OLD_C71_BYTES}
    formal={key:None for key in FORMAL_TOP_KEYS};formal.update({"format":FORMAL_FORMAT,"status":FORMAL_STATUS,"seed":1630,"authorization":FORMAL_AUTH,"runtime_observation":FORMAL_RUNTIME,"required_postregistration_authority":True,"failed_attempt_no_retry":True})
    checks.update({"f813_old_identity":old!=self_record,"repair_self_identity":self_record!=old,"formal_actual_pass":formal_boundary(formal) is True})
    for name,mutated in {"formal_missing_key_rejected":{key:value for key,value in formal.items() if key!="source_closure_sha256"},"formal_extra_key_rejected":{**formal,"unexpected":False},"formal_auth_tamper_rejected":{**formal,"authorization":{**FORMAL_AUTH,"attempts_authorized":1}}}.items():
        try:formal_boundary(mutated);checks[name]=False
        except RuntimeError:checks[name]=True
    return {"passed":all(checks.values()),"check_count":len(checks),"checks":checks,"evidence_sha256":csha(checks)}


def strict_path_parser_synthetic(r2_tree:ast.Module)->dict:
    checks={"real_9efc_pass":strict_path_assignment(r2_tree,"OLD_C71_PATH",OLD_C71_PATH_LITERAL)==OLD_C71_PATH_LITERAL}
    fixtures={"wrong_func":"OLD_C71_PATH=Other('/x')","wrong_argc":"OLD_C71_PATH=Path('/x','/y')","keyword":"OLD_C71_PATH=Path(value='/x')","nonconstant":"OLD_C71_PATH=Path(VALUE)","wrong_path":"OLD_C71_PATH=Path('/wrong')","duplicate":"OLD_C71_PATH=Path('/x')\nOLD_C71_PATH=Path('/x')"}
    for name,text in fixtures.items():
        try:strict_path_assignment(ast.parse(text),"OLD_C71_PATH",OLD_C71_PATH_LITERAL);checks[f"{name}_rejected"]=False
        except RuntimeError:checks[f"{name}_rejected"]=True
    return {"passed":all(checks.values()),"check_count":len(checks),"checks":checks,"evidence_sha256":csha(checks)}


def publish_exact1(root:Path,prep:Path,output_name:str,receipt:dict,immutable_snapshot,phase_hook=None)->dict:
    """The single publication path used by both production and the real control-flow fixtures."""
    if os.path.lexists(root) or os.path.lexists(prep):raise RuntimeError("publication prestate")
    phase_hook=phase_hook or (lambda _phase:None)
    immutable_pre=immutable_snapshot()
    prep_created=False;prep_identity=None;postcommit_verified=False;received_signal=None
    def on_signal(signum,_frame):
        nonlocal received_signal
        received_signal=signum
        if not postcommit_verified:raise ControlledSignal(signum)
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in previous_handlers:signal.signal(sig,on_signal)
    baseline_mask=None;signals_blocked=False
    if hasattr(signal,"pthread_sigmask"):
        baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
    try:
        prep.mkdir();prep_created=True;prep_identity=directory_identity(prep);fsync_dir(prep.parent)
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask);signals_blocked=False
        phase_hook("owned_prep_unblocked")
        atomic_json(prep/output_name,receipt);fsync_dir(prep)
        phase_hook("after_write")
        immutable_after_write=immutable_snapshot()
        if immutable_after_write!=immutable_pre:raise RuntimeError("post-write input drift")
        if directory_identity(prep)!=prep_identity or os.path.lexists(root):raise RuntimeError("publication ownership drift")
        if baseline_mask is not None:
            signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
        phase_hook("commit_window_blocked")
        os.replace(prep,root);fsync_dir(root.parent);prep_created=False
        phase_hook("after_promote_before_snapshot")
        immutable_after_promote=immutable_snapshot()
        root_tree=exact_tree(root);output=root/output_name
        if immutable_after_promote!=immutable_pre or root_tree["inventory"]!=[[output_name,sha(output),output.stat().st_size]] or root_tree["file_count"]!=1 or os.path.lexists(prep):raise RuntimeError("post-promote registration/input drift")
        postcommit_verified=True
    except BaseException:
        if prep_created and prep_identity is not None and not os.path.lexists(root):cleanup_owned_prep(prep,prep_identity)
        raise
    finally:
        if signals_blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    return {"committed_success":True,"deferred_signal":received_signal,"root_tree":root_tree,"immutable_input_snapshot":immutable_pre,"native_pthread_sigmask_exercised":baseline_mask is not None}


def real_publish_callgraph_fixture()->dict:
    runtime_symbols={"getsignal_int_observed":signal.getsignal(signal.SIGINT) is not None,"getsignal_term_observed":signal.getsignal(signal.SIGTERM) is not None,"pthread_sigmask_available":hasattr(signal,"pthread_sigmask")}
    cases={}
    with tempfile.TemporaryDirectory(prefix="v490-real-publish-") as directory:
        base=Path(directory)
        def make_case(name):
            case=base/name;case.mkdir();guard=case/"immutable.guard"
            with guard.open("xb") as stream:stream.write(b"frozen\n");stream.flush();os.fsync(stream.fileno())
            fsync_dir(case)
            root=case/"static";prep=case/"static.registration-prep"
            snapshot=lambda:{"guard":{"sha256":sha(guard),"logical_bytes":guard.stat().st_size}}
            return root,prep,guard,snapshot
        root,prep,guard,snapshot=make_case("normal")
        normal_phases=[];normal=publish_exact1(root,prep,"static_audit.json",{"passed":True},snapshot,normal_phases.append)
        cases["normal_commit"]={"passed":normal["committed_success"] is True and normal["deferred_signal"] is None and normal["root_tree"]["file_count"]==1 and not os.path.lexists(prep) and normal_phases==["owned_prep_unblocked","after_write","commit_window_blocked","after_promote_before_snapshot"],"evidence":normal,"phases":normal_phases}

        root,prep,guard,snapshot=make_case("precommit-signal")
        def inject_precommit_signal(phase):
            if phase=="owned_prep_unblocked":
                if hasattr(signal,"pthread_sigmask"):os.kill(os.getpid(),signal.SIGTERM)
                else:raise ControlledSignal(signal.SIGTERM)
        try:
            publish_exact1(root,prep,"static_audit.json",{"passed":True},snapshot,inject_precommit_signal)
            precommit_passed=False;precommit_error=None
        except ControlledSignal as exc:
            precommit_passed=not os.path.lexists(root) and not os.path.lexists(prep);precommit_error=type(exc).__name__
        cases["precommit_sigterm_owned_cleanup"]={"passed":precommit_passed,"error_type":precommit_error,"root_absent":not os.path.lexists(root),"prep_absent":not os.path.lexists(prep)}

        root,prep,guard,snapshot=make_case("commit-window-signal")
        if hasattr(signal,"pthread_sigmask"):
            deferred=publish_exact1(root,prep,"static_audit.json",{"passed":True},snapshot,lambda phase:os.kill(os.getpid(),signal.SIGTERM) if phase=="commit_window_blocked" else None)
            deferred_passed=deferred["committed_success"] is True and deferred["deferred_signal"]==signal.SIGTERM and deferred["native_pthread_sigmask_exercised"] is True and not os.path.lexists(prep)
        else:
            deferred={"platform_skip":"pthread_sigmask_unavailable"};deferred_passed=os.name=="nt"
        cases["commit_window_deferred_signal_success"]={"passed":deferred_passed,"evidence":deferred}

        root,prep,guard,snapshot=make_case("postpromote-drift")
        def drift_hook(phase):
            if phase=="after_promote_before_snapshot":
                with guard.open("ab") as stream:stream.write(b"drift\n");stream.flush();os.fsync(stream.fileno())
        try:
            publish_exact1(root,prep,"static_audit.json",{"passed":True},snapshot,drift_hook)
            drift_passed=False;drift_error=None
        except RuntimeError as exc:
            drift_passed=os.path.lexists(root) and not os.path.lexists(prep) and "post-promote" in str(exc);drift_error=str(exc)
        cases["postpromote_drift_rejected"]={"passed":drift_passed,"error":drift_error,"root_present":os.path.lexists(root),"prep_absent":not os.path.lexists(prep)}
    runtime_symbols_passed=runtime_symbols["getsignal_int_observed"] and runtime_symbols["getsignal_term_observed"] and (runtime_symbols["pthread_sigmask_available"] or os.name=="nt")
    result={"passed":all(row["passed"] for row in cases.values()) and runtime_symbols_passed,"runtime_symbols":runtime_symbols,"runtime_symbols_passed":runtime_symbols_passed,"cases":cases};result["evidence_sha256"]=csha(result)
    return result


def synthetic_self_test() -> int:
    records=[{"role":f"role{i}","path":f"/frozen/source{i}","sha256":f"{i+1:064x}","logical_bytes":100+i} for i in range(7)]
    source_map={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in records}
    self_record={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}
    tamper=synthetic_suite(records,source_map,self_record)
    r2_fixture_path=R2_PATH if R2_PATH.is_file() else Path(__file__).with_name("reconcile_v488_v487_c71_exact7_schema_repair.py")
    real_r2=ast.parse(r2_fixture_path.read_text());path_synthetic=strict_path_parser_synthetic(real_r2)
    publish_fixture=real_publish_callgraph_fixture()
    result={"passed":tamper["passed"] and path_synthetic["passed"] and publish_fixture["passed"],"tamper":tamper,"strict_path_parser":path_synthetic,"real_publish_callgraph_fixture":publish_fixture};result["evidence_sha256"]=csha(result)
    print(json.dumps(result,sort_keys=True))
    return 0 if result["passed"] and result["tamper"]["check_count"]==13 else 3


def main() -> int:
    if sys.argv[1:]==["--synthetic-self-test"]:return synthetic_self_test()
    parser=argparse.ArgumentParser()
    parser.add_argument("--repair-preregistration",type=Path,required=True);parser.add_argument("--repair-preregistration-sha",required=True)
    parser.add_argument("--design-contract",type=Path,required=True);parser.add_argument("--design-contract-sha",required=True)
    parser.add_argument("--materializer-source",type=Path,required=True);parser.add_argument("--materializer-sha",required=True)
    parser.add_argument("--reconciler-r2-source",type=Path,required=True);parser.add_argument("--reconciler-r2-sha",required=True)
    parser.add_argument("--static-auditor-source",type=Path,required=True);parser.add_argument("--static-auditor-sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if (args.repair_preregistration.resolve(),args.repair_preregistration_sha)!=(FORMAL_PATH,FORMAL_SHA):raise RuntimeError("formal arguments")
    if (args.design_contract.resolve(),args.design_contract_sha)!=(CONTRACT_PATH,CONTRACT_SHA):raise RuntimeError("contract arguments")
    if (args.materializer_source.resolve(),args.materializer_sha)!=(MATERIALIZER_PATH,MATERIALIZER_SHA):raise RuntimeError("materializer arguments")
    if (args.reconciler_r2_source.resolve(),args.reconciler_r2_sha)!=(R2_PATH,R2_SHA):raise RuntimeError("r2 arguments")
    if args.static_auditor_source.resolve()!=STATIC_PATH or args.static_auditor_source.resolve()!=Path(__file__).resolve() or args.static_auditor_sha!=sha(Path(__file__).resolve()):raise RuntimeError("static self")
    if args.output.resolve()!=OUTPUT_PATH or args.output.parent!=STATIC_ROOT:raise RuntimeError("output boundary")
    if os.path.lexists(STATIC_ROOT) or os.path.lexists(STATIC_PREP):raise RuntimeError("static registration prestate")

    formal_record=regular(args.repair_preregistration,FORMAL_SHA,FORMAL_BYTES)
    contract_record=regular(args.design_contract,CONTRACT_SHA,CONTRACT_BYTES)
    materializer_record=regular(args.materializer_source,MATERIALIZER_SHA,MATERIALIZER_BYTES)
    r2_record=regular(args.reconciler_r2_source,R2_SHA,R2_BYTES)
    static_record=regular(args.static_auditor_source,args.static_auditor_sha,args.static_auditor_source.stat().st_size)
    formal=json.loads(FORMAL_PATH.read_text());contract=json.loads(CONTRACT_PATH.read_text());r2_text=R2_PATH.read_text();r2_tree=ast.parse(r2_text)
    checks={key:False for key in CHECK_KEYS}
    formal_reg_before=exact_tree(FORMAL_PATH.parent)
    checks["formal_record_exact"]=formal_record=={"path":str(FORMAL_PATH),"sha256":FORMAL_SHA,"logical_bytes":FORMAL_BYTES} and formal_reg_before["inventory"]==[["preregistration.json",FORMAL_SHA,FORMAL_BYTES]] and formal_reg_before["file_count"]==1 and formal_reg_before["logical_file_bytes"]==FORMAL_BYTES
    checks["formal_schema_exact21"]=formal_boundary(formal) is True
    checks["formal_authority_all_false"]=formal["authorization"]==FORMAL_AUTH and formal["required_postregistration_authority"] is True and formal["failed_attempt_no_retry"] is True
    checks["formal_runtime_no_execution"]=formal["runtime_observation"]==FORMAL_RUNTIME
    checks["contract_current"]=formal["design_contract"]==contract_record and contract.get("materializer_source")==materializer_record
    checks["materializer_current"]=formal["materializer_source"]==materializer_record
    checks["r2_current"]=formal["future_reconciler_r2_source"]==r2_record

    source_closure=formal["source_closure"]
    observed_sources={}
    if isinstance(source_closure,dict) and set(source_closure)==SOURCE_ROLES:
        for role,row in source_closure.items():
            if isinstance(row,dict) and set(row)=={"path","sha256","logical_bytes"}:
                observed_sources[role]=regular(row["path"],row["sha256"],row["logical_bytes"])
    checks["source_closure_exact8_current"]=observed_sources==source_closure and contract.get("source_closure")==source_closure
    checks["source_closure_digest_exact"]=formal["source_closure_sha256"]==csha(source_closure)

    f813_tree=exact_tree(Path(source_closure["f813_reconciliation_preregistration"]["path"]).parent)
    authority_tree=exact_tree(Path(source_closure["v487_authority_receipt"]["path"]).parent)
    helper_tree=exact_tree(Path(source_closure["v487_helper_process_receipt"]["path"]).parent)
    failed_tree=exact_tree(Path(contract["failed_reconciliation_attempt_tree"]["root"]))
    checks["f813_tree_exact2"]=f813_tree==formal["f813_registration_tree"] and f813_tree["file_count"]==2
    checks["v487_authority_tree_exact1"]=authority_tree==formal["v487_authority_tree"] and authority_tree["file_count"]==1
    checks["v487_helper_tree_exact5"]=helper_tree==formal["v487_helper_evidence_tree"] and helper_tree["file_count"]==5
    terminal=json.loads(Path(contract["failed_reconciliation_attempt_tree"]["root"],"terminal_receipt.json").read_text())
    checks["failed_attempt_tree_exact4_no_retry"]=failed_tree==formal["failed_reconciliation_attempt_tree"] and failed_tree["file_count"]==4 and terminal.get("status")=="failed_no_retry" and terminal.get("passed") is False and terminal.get("retry_authorized") is False
    evidence_tree=exact_tree(EVIDENCE_ROOT)
    if evidence_tree!=EVIDENCE_TREE:raise RuntimeError("materialization evidence exact6")
    evidence_receipt=json.loads((EVIDENCE_ROOT/"process_receipt.json").read_text())
    if evidence_receipt.get("status")!="passed_exact_once_no_reconciler_execution" or evidence_receipt.get("passed") is not True or evidence_receipt.get("returncode")!=0 or evidence_receipt.get("materializer_invocations")!=1 or evidence_receipt.get("reconciler_r2_invocations")!=0 or evidence_receipt.get("formal")!=formal_record:raise RuntimeError("materialization evidence semantics")
    superseded_static_record=regular(SUPERSEDED_STATIC_SOURCE_PATH,SUPERSEDED_STATIC_SOURCE_SHA,SUPERSEDED_STATIC_SOURCE_BYTES)
    failed_static_tree=exact_tree(FAILED_STATIC_EVIDENCE_ROOT)
    failed_static_process_receipt_record=regular(FAILED_STATIC_EVIDENCE_ROOT/"process_receipt.json",FAILED_STATIC_PROCESS_RECEIPT_SHA,FAILED_STATIC_PROCESS_RECEIPT_BYTES)
    failed_static_receipt=json.loads((FAILED_STATIC_EVIDENCE_ROOT/"process_receipt.json").read_text());failed_static_stderr=(FAILED_STATIC_EVIDENCE_ROOT/"static_stderr.log").read_text(errors="replace")
    failed_receipt_semantics=failed_static_receipt=={"cleanup":{"group_empty":True,"reaped":True,"started":True},"error":"rc 1","error_type":"RuntimeError","format":"strict-track2-v488-poststatic-process-receipt-v1","passed":False,"r2_invocations":0,"received_signal":None,"retry_authorized":False,"static_invocations":1,"status":"failed_no_retry","wall_seconds":0.12024363805539906,"wrapper_invocations":0}
    checks["failed_static_tree_exact6"]=failed_static_tree==FAILED_STATIC_EVIDENCE_TREE and superseded_static_record=={"path":str(SUPERSEDED_STATIC_SOURCE_PATH),"sha256":SUPERSEDED_STATIC_SOURCE_SHA,"logical_bytes":SUPERSEDED_STATIC_SOURCE_BYTES}
    checks["failed_static_process_no_retry"]=failed_receipt_semantics and "literal_assignment" in failed_static_stderr and "OLD_C71_PATH" in failed_static_stderr and "malformed node or string" in failed_static_stderr
    checks["failed_static_invocation_partition"]=failed_static_receipt.get("static_invocations")==1 and failed_static_receipt.get("r2_invocations")==0 and failed_static_receipt.get("wrapper_invocations")==0
    checks["failed_static_root_prep_absent"]=not os.path.lexists(SUPERSEDED_STATIC_ROOT) and not os.path.lexists(SUPERSEDED_STATIC_PREP)
    v489_failed_source_record=regular(V489_FAILED_STATIC_SOURCE_PATH,V489_FAILED_STATIC_SOURCE_SHA,V489_FAILED_STATIC_SOURCE_BYTES)
    v489_failed_tree=exact_tree(V489_FAILED_STATIC_EVIDENCE_ROOT)
    v489_failed_process_record=regular(V489_FAILED_STATIC_EVIDENCE_ROOT/"process_receipt.json",V489_FAILED_STATIC_PROCESS_RECEIPT_SHA,V489_FAILED_STATIC_PROCESS_RECEIPT_BYTES)
    v489_failed_receipt=json.loads((V489_FAILED_STATIC_EVIDENCE_ROOT/"process_receipt.json").read_text());v489_failed_stderr=(V489_FAILED_STATIC_EVIDENCE_ROOT/"static_stderr.log").read_text(errors="replace")
    v489_receipt_keys={"argv_record","cleanup","error","error_type","format","passed","post_absences","prior_predeploy_transport_failure","r2_invocations","received_signal","repair_registration_tree","retry_authorized","returncode","static_invocations","static_registration_tree","status","stderr","stdout","wall_seconds","wrapper_invocations"}
    v489_failed_semantics=set(v489_failed_receipt)==v489_receipt_keys and v489_failed_receipt["format"]=="strict-track2-v489-poststatic-process-receipt-v1" and v489_failed_receipt["status"]=="failed_no_retry" and v489_failed_receipt["passed"] is False and v489_failed_receipt["returncode"]==1 and v489_failed_receipt["wall_seconds"]==0.11549800704233348 and v489_failed_receipt["cleanup"]=={"group_empty":True,"reaped":True,"started":True} and v489_failed_receipt["error"]=="rc 1" and v489_failed_receipt["error_type"]=="RuntimeError" and v489_failed_receipt["received_signal"] is None and v489_failed_receipt["retry_authorized"] is False and v489_failed_receipt["post_absences"] is None and v489_failed_receipt["static_registration_tree"] is None
    checks["second_failed_static_tree_exact6"]=v489_failed_tree==V489_FAILED_STATIC_EVIDENCE_TREE and v489_failed_source_record=={"path":str(V489_FAILED_STATIC_SOURCE_PATH),"sha256":V489_FAILED_STATIC_SOURCE_SHA,"logical_bytes":V489_FAILED_STATIC_SOURCE_BYTES}
    checks["second_failed_static_process_no_retry"]=v489_failed_semantics and "NameError: name 'signal' is not defined" in v489_failed_stderr and "line 372, in main" in v489_failed_stderr
    checks["second_failed_static_invocation_partition"]=v489_failed_receipt.get("static_invocations")==1 and v489_failed_receipt.get("r2_invocations")==0 and v489_failed_receipt.get("wrapper_invocations")==0
    checks["second_failed_static_root_prep_absent"]=not os.path.lexists(V489_FAILED_STATIC_ROOT) and not os.path.lexists(V489_FAILED_STATIC_PREP)
    checks["predeploy_transport_failure_disclosed"]=v489_failed_receipt.get("prior_predeploy_transport_failure")==V489_PREDEPLOY_TRANSPORT_FAILURE and v489_failed_receipt.get("argv_record",{}).get("prior_predeploy_transport_failure")==V489_PREDEPLOY_TRANSPORT_FAILURE

    parent=json.loads(Path(source_closure["parent_v485_preregistration"]["path"]).read_text());f813=json.loads(Path(source_closure["f813_reconciliation_preregistration"]["path"]).read_text())
    role_order=parent["execution_source_records"]
    observed_exact7=[]
    for row in role_order:
        observed=regular(row["path"],row["sha256"],row["logical_bytes"]);observed_exact7.append({"role":row["role"],**observed})
    proof=formal["exact7_schema_repair_proof"]
    checks["observed_exact7_current"]=observed_exact7==role_order==proof["observed_current_records"] and csha(observed_exact7)==proof["observed_current_records_digest_sha256"]=="9adc5bdbfaa0022b745c44abac2ac2bd02f801f751ebce35d0116c0a8e8e53e0"

    exact4=f813["exact7_source_closure"]
    derived_map={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in observed_exact7}
    strict_exact4=exact4_boundary(exact4,observed_exact7,derived_map) is True and derived_map==parent["execution_sources"]
    main_tree=main_function(r2_tree);main_source=ast.get_source_segment(r2_text,main_tree) or ""
    required_fragments=["set(exact7)!={\"all_records_must_equal_parent_formal_and_old_receipt_and_current_files\",\"canonical_records_digest_sha256\",\"records\",\"roles_in_order\"}","exact7[\"all_records_must_equal_parent_formal_and_old_receipt_and_current_files\"] is not True","derived_sources!=formal[\"execution_sources\"]"]
    checks["r2_exact4_strict_no_fallback"]=strict_exact4 and all(fragment in main_source for fragment in required_fragments) and "compatibility" not in main_source.lower() and "fallback" not in main_source.lower()
    checks["r2_old_c71_dual_binding"]=literal_assignment(r2_tree,"OLD_C71_SHA")==OLD_C71_SHA and strict_path_assignment(r2_tree,"OLD_C71_PATH",OLD_C71_PATH_LITERAL)==OLD_C71_PATH_LITERAL and observed_sources["old_reconciler_c71"]=={"path":OLD_C71_PATH_LITERAL,"sha256":OLD_C71_SHA,"logical_bytes":OLD_C71_BYTES} and observed_sources["future_reconciler_r2"]==r2_record and "f813 old reconciler identity" in r2_text and "repair r2 identity" in r2_text
    expanded_keys={"repair_format","repair_preregistration","repair_design_contract","repair_materializer_source","old_reconciler_c71_source","reconciler_r2_source","repair_source_closure","repair_registration_tree","f813_registration_tree","repair_formal_ancestry"}
    receipt_nodes=[node for node in ast.walk(main_tree) if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="receipt" for target in node.targets) and isinstance(node.value,ast.Dict)]
    receipt_keys={ast.literal_eval(key) for key in receipt_nodes[-1].value.keys}
    snapshot_names={"f813_reconciliation_preregistration","repair_preregistration","repair_design_contract","repair_materializer_source","reconciler_r2_cli_source","old_reconciler_c71"}
    checks["r2_snapshot_dual_ancestry"]=expanded_keys.issubset(receipt_keys) and all(name in r2_text for name in snapshot_names) and "v488_repair_registration" in r2_text and "failed_v487_attempt" in r2_text
    forbidden_imports={"torch","rlinf","wandb","transformers","requests"};imports={alias.name.split(".")[0] for node in ast.walk(r2_tree) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in (node.names if isinstance(node,ast.Import) else [ast.alias(name=node.module or "")])}
    forbidden_calls={"train","fit","backward","step","load_reward","evaluate_reward","predict_one_with_baseline"};calls={ast_call_name(node.func).split(".")[-1] for node in ast.walk(r2_tree) if isinstance(node,ast.Call)}
    checks["r2_forbidden_imports_calls_absent"]=not(imports&forbidden_imports) and not(calls&forbidden_calls)
    checks["no_training_reward_outcome"]=all(value is False for value in (formal["authorization"]["training_authorized"],formal["authorization"]["reward_read_authorized"],formal["authorization"]["dev_hidden_final_outcome_read_authorized"])) and formal["authorization"]["folds_authorized"]==0 and formal["authorization"]["policy_updates"]==0 and checks["r2_forbidden_imports_calls_absent"]

    tamper_synthetic=synthetic_suite(observed_exact7,derived_map,r2_record);path_synthetic=strict_path_parser_synthetic(r2_tree);publish_fixture=real_publish_callgraph_fixture()
    synthetic={"passed":tamper_synthetic["passed"] and path_synthetic["passed"] and publish_fixture["passed"],"tamper":tamper_synthetic,"strict_path_parser":path_synthetic,"real_publish_callgraph_fixture":publish_fixture};synthetic["evidence_sha256"]=csha(synthetic)
    checks["synthetic_tamper_suite_passed"]=tamper_synthetic["passed"] is True and tamper_synthetic["check_count"]==13 and path_synthetic["passed"] is True and path_synthetic["check_count"]==7
    checks["path_literal_ast_repair_exact"]=path_synthetic["passed"] is True and path_synthetic["check_count"]==7
    static_ast=ast.parse(STATIC_PATH.read_text());static_functions={node.name:node for node in static_ast.body if isinstance(node,ast.FunctionDef)}
    publish_callgraph={name:sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="publish_exact1" for node in ast.walk(static_functions[name])) for name in ("main","real_publish_callgraph_fixture")}
    checks["real_publish_callgraph_fixture_passed"]=publish_fixture["passed"] is True and publish_fixture["runtime_symbols"]=={"getsignal_int_observed":True,"getsignal_term_observed":True,"pthread_sigmask_available":True} and set(publish_fixture["cases"])=={"normal_commit","precommit_sigterm_owned_cleanup","commit_window_deferred_signal_success","postpromote_drift_rejected"} and all(row["passed"] is True for row in publish_fixture["cases"].values()) and publish_fixture["cases"]["normal_commit"]["phases"]==["owned_prep_unblocked","after_write","commit_window_blocked","after_promote_before_snapshot"] and publish_callgraph=={"main":1,"real_publish_callgraph_fixture":4}
    absences={"fresh_static_prep":{"path":str(STATIC_PREP),"absent":not os.path.lexists(STATIC_PREP)},"authority_root":{"path":str(AUTHORITY_ROOT),"absent":not os.path.lexists(AUTHORITY_ROOT)},"authority_prep":{"path":str(AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+".registration-prep")),"absent":not os.path.lexists(AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+".registration-prep"))},"attempt_root":{"path":str(ATTEMPT_ROOT),"absent":not os.path.lexists(ATTEMPT_ROOT)},"attempt_prep":{"path":str(ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+".attempt-prep")),"absent":not os.path.lexists(ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+".attempt-prep"))},"transparent_output":{"path":str(TRANSPARENT_PATH),"absent":not os.path.lexists(TRANSPARENT_PATH)},"transparent_tmp":{"path":str(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp")),"absent":not os.path.lexists(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp"))},"qualification_root":{"path":str(QUALIFICATION_ROOT),"absent":not os.path.lexists(QUALIFICATION_ROOT)},"superseded_static_root":{"path":str(SUPERSEDED_STATIC_ROOT),"absent":not os.path.lexists(SUPERSEDED_STATIC_ROOT)},"superseded_static_prep":{"path":str(SUPERSEDED_STATIC_PREP),"absent":not os.path.lexists(SUPERSEDED_STATIC_PREP)},"v489_superseded_static_root":{"path":str(V489_FAILED_STATIC_ROOT),"absent":not os.path.lexists(V489_FAILED_STATIC_ROOT)},"v489_superseded_static_prep":{"path":str(V489_FAILED_STATIC_PREP),"absent":not os.path.lexists(V489_FAILED_STATIC_PREP)}}
    checks["transparent_and_fresh_outputs_absent"]=all(row["absent"] is True for row in absences.values())
    live=[]
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():continue
        try: cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if int(proc.name)!=os.getpid() and (str(R2_PATH) in cmd or str(MATERIALIZER_PATH) in cmd or str(SUPERSEDED_STATIC_SOURCE_PATH) in cmd or str(V489_FAILED_STATIC_SOURCE_PATH) in cmd or str(STATIC_PATH) in cmd or "launch_v488_v487_c71_exact7_schema_repair.py" in cmd or "launch_v489_v488_v487_c71_exact7_schema_repair.py" in cmd or "launch_v490_v489_v488_v487_c71_exact7_schema_repair.py" in cmd):live.append({"pid":int(proc.name),"cmdline":cmd})
    checks["no_live_process"]=live==[]
    if set(checks)!=set(CHECK_KEYS) or not all(checks.values()) or csha(CHECK_KEYS)!=CHECK_KEYSET_SHA or csha(checks)!=CHECKS_SHA:raise RuntimeError("static checks")
    repair_registration_tree_before=formal_reg_before
    ast_proof={"materialization_evidence_tree":evidence_tree,"superseded_f88_source":superseded_static_record,"superseded_f88_failure_tree":failed_static_tree,"superseded_f88_failure_receipt":failed_static_receipt,"v489_failed_static_source":v489_failed_source_record,"v489_failed_static_execution_tree":v489_failed_tree,"v489_failed_static_receipt":v489_failed_receipt,"v489_predeploy_transport_failure":V489_PREDEPLOY_TRANSPORT_FAILURE,"repair_registration_tree_before":repair_registration_tree_before,"repair_registration_tree_after":repair_registration_tree_before,"repair_registration_tree_equal":True,"actual_exact4_keyset":sorted(exact4),"old_c71_exact3_keyset":["execution_source_records","execution_sources","execution_sources_digest_sha256"],"exact4_strict":strict_exact4,"strict_path_assignment_proof":path_synthetic,"real_publish_callgraph_fixture":publish_fixture,"publish_callgraph":publish_callgraph,"observed_exact7_current_records":observed_exact7,"observed_exact7_digest_sha256":csha(observed_exact7),"dual_formal":{"f813_old_reconciler":observed_sources["old_reconciler_c71"],"repair_r2_self":r2_record},"expanded_output_keys":sorted(expanded_keys),"forbidden_imports_observed":sorted(imports&forbidden_imports),"forbidden_calls_observed":sorted(calls&forbidden_calls),"live_processes":live}
    receipt={"format":FORMAT,"status":STATUS,"passed":True,"checks":checks,"check_keys":CHECK_KEYS,"check_key_set_sha256":CHECK_KEYSET_SHA,"checks_sha256":CHECKS_SHA,"repair_preregistration":formal_record,"design_contract":contract_record,"materializer_source":materializer_record,"reconciler_r2_source":r2_record,"static_auditor_source":static_record,"source_closure":observed_sources,"source_closure_sha256":csha(observed_sources),"f813_registration_tree":f813_tree,"v487_authority_tree":authority_tree,"v487_helper_evidence_tree":helper_tree,"failed_reconciliation_attempt_tree":failed_tree,"ast_diff_proof":ast_proof,"synthetic_evidence":synthetic,"required_absences":absences,"runtime_observation":STATIC_RUNTIME,"readonly_reconciliation_authorized":False,"training_authorized":False,"submission_authorized":False,"superseded_static_source":superseded_static_record,"failed_static_execution_tree":failed_static_tree,"failed_static_process_receipt":failed_static_process_receipt_record,"superseded_static_absences":{"superseded_static_root":absences["superseded_static_root"],"superseded_static_prep":absences["superseded_static_prep"]},"v489_failed_static_source":v489_failed_source_record,"v489_failed_static_execution_tree":v489_failed_tree,"v489_failed_static_process_receipt":v489_failed_process_record,"v489_superseded_static_absences":{"v489_superseded_static_root":absences["v489_superseded_static_root"],"v489_superseded_static_prep":absences["v489_superseded_static_prep"]},"v489_predeploy_transport_failure":V489_PREDEPLOY_TRANSPORT_FAILURE}
    if set(receipt)!=TOP_KEYS:raise RuntimeError("receipt schema")
    stable_absence_paths={name:Path(row["path"]) for name,row in absences.items() if name!="fresh_static_prep"}
    def immutable_input_snapshot():
        files={"formal":regular(FORMAL_PATH,FORMAL_SHA,FORMAL_BYTES),"contract":regular(CONTRACT_PATH,CONTRACT_SHA,CONTRACT_BYTES),"materializer":regular(MATERIALIZER_PATH,MATERIALIZER_SHA,MATERIALIZER_BYTES),"r2":regular(R2_PATH,R2_SHA,R2_BYTES),"static":regular(STATIC_PATH,args.static_auditor_sha,args.static_auditor_source.stat().st_size),"superseded_f88_static":regular(SUPERSEDED_STATIC_SOURCE_PATH,SUPERSEDED_STATIC_SOURCE_SHA,SUPERSEDED_STATIC_SOURCE_BYTES),"failed_v489_static":regular(V489_FAILED_STATIC_SOURCE_PATH,V489_FAILED_STATIC_SOURCE_SHA,V489_FAILED_STATIC_SOURCE_BYTES)}
        sources_now={role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in source_closure.items()}
        trees={"repair":exact_tree(FORMAL_PATH.parent),"evidence":exact_tree(EVIDENCE_ROOT),"superseded_f88_failure":exact_tree(FAILED_STATIC_EVIDENCE_ROOT),"failed_v489_static":exact_tree(V489_FAILED_STATIC_EVIDENCE_ROOT),"f813":exact_tree(Path(source_closure["f813_reconciliation_preregistration"]["path"]).parent),"v487_authority":exact_tree(Path(source_closure["v487_authority_receipt"]["path"]).parent),"v487_helper":exact_tree(Path(source_closure["v487_helper_process_receipt"]["path"]).parent),"failed_attempt":exact_tree(Path(contract["failed_reconciliation_attempt_tree"]["root"]))}
        absence_state={name:{"path":str(path),"absent":not os.path.lexists(path)} for name,path in stable_absence_paths.items()}
        if not all(row["absent"] for row in absence_state.values()):raise RuntimeError("stable absence drift")
        return {"files":files,"sources":sources_now,"trees":trees,"absences":absence_state}
    publication=publish_exact1(STATIC_ROOT,STATIC_PREP,"static_audit.json",receipt,immutable_input_snapshot)
    repair_registration_tree_after=exact_tree(FORMAL_PATH.parent);static_tree=publication["root_tree"]
    if repair_registration_tree_after!=repair_registration_tree_before or any(os.path.lexists(row["path"]) for row in absences.values()):raise RuntimeError("post-publication ancestry/absence drift")
    print(json.dumps({"committed_success":publication["committed_success"],"deferred_signal":publication["deferred_signal"],"native_pthread_sigmask_exercised":publication["native_pthread_sigmask_exercised"]},sort_keys=True),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
