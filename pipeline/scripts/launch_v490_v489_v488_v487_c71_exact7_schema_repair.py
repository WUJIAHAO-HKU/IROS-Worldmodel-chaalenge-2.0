#!/usr/bin/env python3
"""One-shot fail-closed launcher for the authorized v490 read-only reconciliation repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
try:
 import fcntl
except ImportError:
 fcntl=None

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v490_v489_v488_v487_c71_exact7_schema_repair.py"
AUTHORITY_CONTRACT_PATH=ROOT/"pipeline/scripts/v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority_contract.json"
AUTHORITY_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_authority_seed1632_20260825"
AUTHORITY_RECEIPT_PATH=AUTHORITY_ROOT/"authority_receipt.json"
REPAIR_FORMAL_PATH=J/"v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json"
REPAIR_FORMAL_SHA="b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b"
REPAIR_FORMAL_BYTES=36181
STATIC_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825"
STATIC_RECEIPT_PATH=STATIC_ROOT/"static_audit.json"
STATIC_RECEIPT_SHA="441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb"
STATIC_RECEIPT_BYTES=49008
STATIC_SOURCE_PATH=ROOT/"pipeline/scripts/audit_v490_v489_v488_v487_c71_exact7_schema_repair_static.py"
STATIC_SOURCE_SHA="1030de39f14ecaffb14c9f8192c2ec3d0898a2cb8e5913e64b50cf13d8baca80"
STATIC_SOURCE_BYTES=51979
STATIC_EXECUTION_EVIDENCE_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_execution_evidence_seed1632_20260825"
STATIC_EXECUTION_EVIDENCE_TREE={"root":str(STATIC_EXECUTION_EVIDENCE_ROOT),"inventory":[["argv.json","8795179679b7dfffd7dd046f406e902f11b89d6d2b2ac63048b3e82029620dcb",11017],["intent.json","692b86f97ef08ec39f017bbc17a9f4a71ca3fe20312e8758f3972166daa1de99",170],["process_receipt.json","b686df0afdd1b330200e99c0732a4fcf321325ca596ee0c20fb327301ca67dab",17211],["static_stderr.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["static_stdout.log","b1c7595ce329c13e314ee264e90f54ee787876be2e1e2458929cf68f5e629f4f",95],["transport_helper.py","b352813d6543d3d3e541432f2b01f48786130284728a9f31bb011b49a8b23a40",13811]],"file_count":6,"logical_file_bytes":42304,"sha256sum_lines_digest_sha256":"c0e8b964b64d19cbda82241f4690ca4f69697f704c9c961dda05135d5b06260f","canonical_json_triples_digest_sha256":"2179c46cdd545ba55b1498bd4ac74f7556d6f8ba4150a562655b7fb99d11b776"}
F813_PATH=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json"
F813_SHA="f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8"
F813_BYTES=21296
R2_PATH=ROOT/"pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py"
R2_SHA="9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377"
R2_BYTES=51845
ATTEMPT_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+".attempt-prep")
LOCK_PATH=J/".v490_v489_v488_v487_c71_exact7_schema_repair_attempt.lock"
TRANSPARENT_PATH=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT=Path("/root/v485_v169_cache_qualification_seed1627_20260824")

AUTHORITY_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-postregistration-authority-v1"
AUTHORITY_STATUS="authorized_exact_one_readonly_reconciliation_source_repair_attempt"
AUTHORITY_CONTRACT_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-postregistration-authority-design-contract-v1"
AUTHORITY_CONTRACT_STATUS="design_only_frozen_poststatic_sources_pending_independent_review_no_authority"
STATIC_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-static-audit-v1"
STATIC_STATUS="passed_no_execution_authority"
STATIC_CHECK_KEYS=sorted({"contract_current","f813_tree_exact2","failed_attempt_tree_exact4_no_retry","formal_authority_all_false","formal_record_exact","formal_runtime_no_execution","formal_schema_exact21","materializer_current","no_live_process","no_training_reward_outcome","observed_exact7_current","r2_current","r2_exact4_strict_no_fallback","r2_forbidden_imports_calls_absent","r2_old_c71_dual_binding","r2_snapshot_dual_ancestry","source_closure_digest_exact","source_closure_exact8_current","synthetic_tamper_suite_passed","transparent_and_fresh_outputs_absent","v487_authority_tree_exact1","v487_helper_tree_exact5","failed_static_tree_exact6","failed_static_process_no_retry","failed_static_invocation_partition","failed_static_root_prep_absent","path_literal_ast_repair_exact","second_failed_static_tree_exact6","second_failed_static_process_no_retry","second_failed_static_invocation_partition","second_failed_static_root_prep_absent","predeploy_transport_failure_disclosed","real_publish_callgraph_fixture_passed"})
STATIC_KEYSET_SHA="53b721e19caa9b6302986e65c78dac72e0026716d67852d1b886aba27492d7cf"
STATIC_CHECKS_SHA="50edf45d3a0021c3054ff80b2fcc311ebb848272def1e0732aa3aaf7c2079a58"
STATIC_TOP_KEYS={"format","status","passed","checks","check_keys","check_key_set_sha256","checks_sha256","repair_preregistration","design_contract","materializer_source","reconciler_r2_source","static_auditor_source","source_closure","source_closure_sha256","f813_registration_tree","v487_authority_tree","v487_helper_evidence_tree","failed_reconciliation_attempt_tree","ast_diff_proof","synthetic_evidence","required_absences","runtime_observation","readonly_reconciliation_authorized","training_authorized","submission_authorized","superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
STATIC_RUNTIME={"static_audit_executed":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
AUTHORIZATION={"readonly_reconciliation_authorized":True,"attempts_authorized":1,"attempts_consumed":0,"retry_authorized":False,"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
AUTHORITY_RUNTIME={"authority_materialized":True,"static_audit_executed":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
AUTHORITY_CHECK_KEYS=sorted({"authority_materializer_current","current_absences","execution_wrapper_current","f813_exact2","f88_failed_static_no_retry_partition","f88_failed_static_source_current","f88_failed_static_tree_exact6","f88_superseded_static_roots_absent","failed_attempt_exact4","formal_evidence_exact6","input_snapshots_equal","no_live_process","old_v487_no_retry","postregistration_static_reg_exact1","r2_current","repair_formal_exact","repair_formal_reg_exact1","source_closure_current","static_boundary_no_authority","static_checks33_all_true","static_receipt_exact","transparent_absent","v489_failed_static_no_retry_partition","v489_failed_static_source_current","v489_failed_static_tree_exact6","v489_predeploy_transport_disclosed","v489_superseded_static_roots_absent","static_execution_evidence_exact6"})
AUTHORITY_KEYSET_SHA="be4899694c2bf3c933c18896697a6e12b8227195b4be36185bdd94b9c10ea68f"
AUTHORITY_CHECKS_SHA="be4d8b6f375d8c522c11439593d7f7ea31a04f582239f6b0ffbf9bcd66c1212a"
AUTHORITY_TOP_KEYS={"format","status","passed","authority_design_contract","authority_materializer_source","repair_preregistration","repair_design_contract","repair_formal_materializer","reconciler_r2_source","static_auditor_source","execution_wrapper_source","static_receipt","repair_formal_registration_tree","postregistration_static_registration_tree","static_execution_evidence_tree","f813_registration_tree","failed_attempt_tree","formal_materialization_evidence_tree","attempt_root","transparent_static_receipt_path","historical_absences","required_absences","checks","checks_sha256","input_pre_snapshot","input_post_snapshot","input_snapshots_exactly_equal","authorization","runtime_observation","superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
AUTHORITY_CONTRACT_TOP_KEYS={"format","status","seed","lineage","source_closure","source_closure_sha256","authority_materializer_source","static_receipt","static_receipt_contract","static_failure_ancestry","repair_formal_schema","repair_formal_registration_tree","postregistration_static_registration_tree","static_execution_evidence_tree","f813_registration_tree","failed_attempt_tree","formal_materialization_evidence_tree","historical_absences","current_absences_after_authority","authority_receipt_contract","execution_boundary"}
AUTHORITY_SCHEMA_KEYS={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","authorization_exact","runtime_observation_exact"}
STATIC_SCHEMA_KEYS={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","runtime_observation_exact"}
FORMAL_SCHEMA_KEYS={"format","status","top_keys","authorization_exact","runtime_observation_exact"}
FAILURE_ANCESTRY_KEYS={"superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
EXECUTION_BOUNDARY={"authority_materialization_only":True,"reconciler_r2_invocations":0,"execution_wrapper_invocations":0,"phase_a_invocations":0,"training_invocations":0,"reward_reads":0,"dev_hidden_final_outcome_reads":0}
HISTORICAL_ABSENCE_KEYS={"superseded_static_root","superseded_static_prep","v489_superseded_static_root","v489_superseded_static_prep","fresh_static_prep","authority_root","authority_prep","attempt_root","attempt_prep","transparent_output","transparent_tmp","qualification_root"}
CURRENT_ABSENCE_KEYS=HISTORICAL_ABSENCE_KEYS-{"authority_root"}
INTENT_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-attempt-intent-v1"
TERMINAL_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-attempt-terminal-v1"
FALSE_AUTHORITIES={"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
R2_RECEIPT_KEYS={"cache_reuse_authorized","check_key_set_sha256","check_keys","checks","checks_sha256","contract","dev_hidden_final_outcome_read_authorized","f813_registration_tree","false_positive_recomputation","folds_authorized","format","immutable_absences","input_post_snapshot","input_pre_snapshot","input_snapshots_exactly_equal","old_reconciler_c71_source","parent_runtime_observation","passed","persistent_old_static_log","phase_a_cache_qualification_authorized","phase_a_executed","policy_updates","preregistration","receipt_writer","reconciler_r2_source","reconciliation_ancestry","reconciliation_design_contract","reconciliation_format","reconciliation_only","reconciliation_preregistration","reconciliation_runtime_observation","reconciliation_status","registration_initial_inventory","repair_design_contract","repair_formal_ancestry","repair_format","repair_materializer_source","repair_preregistration","repair_registration_tree","repair_source_closure","reward_read_authorized","rl_authorized","runtime_observation","s1_authorized","sources","sources_digest_sha256","static_auditor_self_sha256","status","submission_authorized","synthetic_recomputation_evidence","training_authorized","v169_imported_or_run","volatile_log_source_at_registration","zero_update_authorized"}
R2_CHECK_KEYSET_SHA="57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b"
R2_SYNTHETIC_FIXTURE_SHA="77bc6af02c136be5b9b8c6bb653847de62311d5cd06b3547a06a60b2eee762fa"
R2_SYNTHETIC_EVIDENCE_SHA="46a5fa5cfc6808b7fe52695248465cd694311b027b84f320e88a9ad7990d03a4"


class ControlledSignal(BaseException):pass


def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(8<<20),b""):d.update(block)
 return d.hexdigest()


def csha(value)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def contains_pending(value)->bool:
 if isinstance(value,str):return "PENDING" in value
 if isinstance(value,dict):return any(contains_pending(key) or contains_pending(item) for key,item in value.items())
 if isinstance(value,list):return any(contains_pending(item) for item in value)
 return False


def regular(path:Path|str,want_sha:str|None=None,want_bytes:int|None=None)->dict:
 path=Path(path)
 if path!=path.resolve() or not path.is_file() or path.is_symlink():raise RuntimeError(f"regular: {path}")
 result={"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
 if want_sha is not None and result["sha256"]!=want_sha:raise RuntimeError(f"sha: {path}")
 if want_bytes is not None and result["logical_bytes"]!=want_bytes:raise RuntimeError(f"bytes: {path}")
 return result


def exact_tree(root:Path|str)->dict:
 root=Path(root)
 if root!=root.resolve() or not root.is_dir() or root.is_symlink():raise RuntimeError(f"tree root: {root}")
 rows=[]
 for path in sorted(root.rglob("*")):
  if path.is_symlink():raise RuntimeError(f"tree symlink: {path}")
  if path.is_file():rows.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
  elif not path.is_dir():raise RuntimeError(f"tree nonregular: {path}")
 lines="".join(f"{digest}  {rel}\n" for rel,digest,_ in rows).encode();triples=json.dumps(rows,separators=(",",":")).encode()
 return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(row[2] for row in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(triples).hexdigest()}


def tree_with_root(root:Path)->dict:return {"root":str(root),**exact_tree(root)}


def fsync_dir(path:Path)->None:
 fd=os.open(str(path),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)


def atomic_json(path:Path,value)->None:
 tmp=path.with_name(path.name+".tmp")
 if os.path.lexists(path) or os.path.lexists(tmp):raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,path);fsync_dir(path.parent)


def directory_identity(path:Path)->tuple[int,int]:
 stat=path.stat(follow_symlinks=False)
 if path.is_symlink() or not path.is_dir():raise RuntimeError("directory identity")
 return stat.st_dev,stat.st_ino


def group_empty(pid:int)->bool:
 try:os.killpg(pid,0);return False
 except ProcessLookupError:return True


def terminate_group(process:subprocess.Popen|None)->dict:
 evidence={"started":process is not None,"term_sent":False,"kill_sent":False,"reaped":process is None,"group_empty":True}
 if process is None:return evidence
 if process.poll() is None or not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGTERM);evidence["term_sent"]=True
  except ProcessLookupError:pass
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:
   pass
 if not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGKILL);evidence["kill_sent"]=True
  except ProcessLookupError:pass
 if process.poll() is None:process.wait(timeout=10)
 else:process.wait()
 evidence["reaped"]=process.poll() is not None;evidence["group_empty"]=group_empty(process.pid)
 if not evidence["reaped"] or not evidence["group_empty"]:raise RuntimeError("process group cleanup")
 return evidence


def cleanup_owned_prep(path:Path,identity:tuple[int,int]|None)->None:
 if identity is None or not path.exists() or path.is_symlink() or directory_identity(path)!=identity:return
 allowed={"intent.json","intent.json.tmp","reconciler_stdout.log","reconciler_stderr.log","terminal_receipt.json","terminal_receipt.json.tmp"}
 entries=list(path.iterdir())
 if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):return
 for entry in entries:entry.unlink()
 path.rmdir();fsync_dir(path.parent)


def current_processes()->list[dict]:
 found=[]
 for proc in Path("/proc").iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if int(proc.name)!=os.getpid() and (str(R2_PATH) in cmd or str(WRAPPER_PATH) in cmd):found.append({"pid":int(proc.name),"cmdline":cmd})
 return found


def validate_static(static:dict,static_record:dict)->None:
 if set(static)!=STATIC_TOP_KEYS or static.get("format")!=STATIC_FORMAT or static.get("status")!=STATIC_STATUS or static.get("passed") is not True:raise RuntimeError("static receipt schema")
 if static.get("check_keys")!=STATIC_CHECK_KEYS or static.get("check_key_set_sha256")!=STATIC_KEYSET_SHA or static.get("checks")!={key:True for key in STATIC_CHECK_KEYS} or static.get("checks_sha256")!=STATIC_CHECKS_SHA:raise RuntimeError("static checks")
 if static.get("runtime_observation")!=STATIC_RUNTIME or any(static.get(key) is not False for key in ("readonly_reconciliation_authorized","training_authorized","submission_authorized")):raise RuntimeError("static authority boundary")
 if static.get("static_auditor_source")!=static_record or static.get("repair_preregistration")!={"path":str(REPAIR_FORMAL_PATH),"sha256":REPAIR_FORMAL_SHA,"logical_bytes":REPAIR_FORMAL_BYTES} or static.get("reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("static ancestry")
 if static.get("source_closure_sha256")!=csha(static.get("source_closure")):raise RuntimeError("static source digest")


def preflight(args)->dict:
 if Path(sys.executable)!=RLPY or not RLPY.is_file():raise RuntimeError("exact RLPY")
 if args.f813_preregistration.resolve()!=F813_PATH or args.f813_preregistration_sha!=F813_SHA:raise RuntimeError("f813 arguments")
 if args.repair_preregistration.resolve()!=REPAIR_FORMAL_PATH or args.repair_preregistration_sha!=REPAIR_FORMAL_SHA:raise RuntimeError("repair arguments")
 if args.authority_contract.resolve()!=AUTHORITY_CONTRACT_PATH or len(args.authority_contract_sha)!=64:raise RuntimeError("authority contract arguments")
 if args.authority_receipt.resolve()!=AUTHORITY_RECEIPT_PATH or len(args.authority_receipt_sha)!=64:raise RuntimeError("authority receipt arguments")
 if args.wrapper_source.resolve()!=WRAPPER_PATH or args.wrapper_source.resolve()!=Path(__file__).resolve() or args.wrapper_sha!=sha(Path(__file__).resolve()):raise RuntimeError("wrapper self")
 f813_record=regular(F813_PATH,F813_SHA,F813_BYTES);repair_record=regular(REPAIR_FORMAL_PATH,REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES);contract_record=regular(args.authority_contract,args.authority_contract_sha);authority_record=regular(args.authority_receipt,args.authority_receipt_sha);wrapper_record=regular(args.wrapper_source,args.wrapper_sha,args.wrapper_source.stat().st_size)
 contract=json.loads(args.authority_contract.read_text());authority=json.loads(args.authority_receipt.read_text());repair=json.loads(REPAIR_FORMAL_PATH.read_text());f813=json.loads(F813_PATH.read_text())
 if set(contract)!=AUTHORITY_CONTRACT_TOP_KEYS or contract.get("format")!=AUTHORITY_CONTRACT_FORMAT or contract.get("status")!=AUTHORITY_CONTRACT_STATUS or contract.get("seed")!=1632 or contract.get("execution_boundary")!=EXECUTION_BOUNDARY or contains_pending(contract):raise RuntimeError("authority contract schema")
 receipt_contract=contract.get("authority_receipt_contract")
 if not isinstance(receipt_contract,dict) or set(receipt_contract)!=AUTHORITY_SCHEMA_KEYS or set(receipt_contract["top_keys"])!=AUTHORITY_TOP_KEYS or receipt_contract["check_keys"]!=AUTHORITY_CHECK_KEYS or receipt_contract["check_key_set_sha256"]!=AUTHORITY_KEYSET_SHA or receipt_contract["checks_sha256"]!=AUTHORITY_CHECKS_SHA or receipt_contract["authorization_exact"]!=AUTHORIZATION or receipt_contract["runtime_observation_exact"]!=AUTHORITY_RUNTIME:raise RuntimeError("authority receipt contract schema")
 if set(authority)!=AUTHORITY_TOP_KEYS or authority.get("format")!=AUTHORITY_FORMAT or authority.get("status")!=AUTHORITY_STATUS or authority.get("passed") is not True:raise RuntimeError("authority receipt schema")
 if authority.get("checks")!={key:True for key in AUTHORITY_CHECK_KEYS} or authority.get("checks_sha256")!=AUTHORITY_CHECKS_SHA or csha(AUTHORITY_CHECK_KEYS)!=AUTHORITY_KEYSET_SHA or csha(authority["checks"])!=AUTHORITY_CHECKS_SHA:raise RuntimeError("authority checks")
 if authority.get("authorization")!=AUTHORIZATION or authority.get("runtime_observation")!=AUTHORITY_RUNTIME:raise RuntimeError("authority execution boundary")
 if authority.get("authority_design_contract")!=contract_record:raise RuntimeError("authority contract alias")
 materializer_spec=contract.get("authority_materializer_source")
 if not isinstance(materializer_spec,dict) or set(materializer_spec)!={"path","sha256","logical_bytes"}:raise RuntimeError("authority materializer spec")
 materializer_record=regular(materializer_spec["path"],materializer_spec["sha256"],materializer_spec["logical_bytes"])
 if authority.get("authority_materializer_source")!=materializer_record:raise RuntimeError("authority materializer alias")
 source_closure=contract.get("source_closure");expected_roles={"repair_preregistration","repair_design_contract","repair_formal_materializer","reconciler_r2","static_auditor","execution_wrapper"}
 if not isinstance(source_closure,dict) or set(source_closure)!=expected_roles or contract.get("source_closure_sha256")!=csha(source_closure):raise RuntimeError("authority source closure roles")
 observed_sources={role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in source_closure.items()}
 alias_map={"repair_preregistration":"repair_preregistration","repair_design_contract":"repair_design_contract","repair_formal_materializer":"repair_formal_materializer","reconciler_r2":"reconciler_r2_source","static_auditor":"static_auditor_source","execution_wrapper":"execution_wrapper_source"}
 if any(authority.get(alias_map[role])!=observed_sources[role] for role in expected_roles):raise RuntimeError("authority source alias map")
 if observed_sources["repair_preregistration"]!=repair_record or observed_sources["reconciler_r2"]!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES} or observed_sources["static_auditor"]!={"path":str(STATIC_SOURCE_PATH),"sha256":STATIC_SOURCE_SHA,"logical_bytes":STATIC_SOURCE_BYTES} or observed_sources["execution_wrapper"]!=wrapper_record:raise RuntimeError("authority source identities")
 repair_sources=repair.get("source_closure")
 if not isinstance(repair_sources,dict) or len(repair_sources)!=8 or repair.get("source_closure_sha256")!=csha(repair_sources):raise RuntimeError("repair source closure schema")
 observed_repair_sources={role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in repair_sources.items()}
 if observed_repair_sources!=repair_sources or repair.get("future_reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("repair source closure current")
 exact7=f813.get("exact7_source_closure")
 if not isinstance(exact7,dict) or set(exact7)!={"all_records_must_equal_parent_formal_and_old_receipt_and_current_files","canonical_records_digest_sha256","records","roles_in_order"} or exact7.get("all_records_must_equal_parent_formal_and_old_receipt_and_current_files") is not True:raise RuntimeError("f813 exact7 schema")
 exact7_records=exact7.get("records");role_order=exact7.get("roles_in_order")
 if not isinstance(exact7_records,list) or [row.get("role") for row in exact7_records]!=role_order or csha(exact7_records)!=exact7.get("canonical_records_digest_sha256"):raise RuntimeError("f813 exact7 records")
 exact7_sources={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in exact7_records}
 if any(regular(row["path"],row["sha256"],row["logical_bytes"])!={key:row[key] for key in ("path","sha256","logical_bytes")} for row in exact7_records):raise RuntimeError("f813 exact7 current files")
 phase_formal_spec=repair_sources.get("parent_v485_preregistration",{});phase_formal_record=regular(phase_formal_spec["path"],phase_formal_spec["sha256"],phase_formal_spec["logical_bytes"]);phase_formal=json.loads(Path(phase_formal_record["path"]).read_text())
 if phase_formal.get("execution_source_records")!=exact7_records or phase_formal.get("execution_sources")!=exact7_sources or phase_formal.get("execution_sources_digest_sha256")!=exact7["canonical_records_digest_sha256"]:raise RuntimeError("phase formal ordered source closure")
 parent=f813.get("frozen_parent",{});phase_contract_spec=parent.get("phase_a_design_contract",{});phase_contract_record=regular(phase_contract_spec["path"],phase_contract_spec["sha256"],phase_contract_spec["logical_bytes"]);phase_contract=json.loads(Path(phase_contract_record["path"]).read_text())
 if phase_contract.get("phase_a_output_and_receipt_schema",{}).get("source_closure_contract",{}).get("exact_roles")!=role_order:raise RuntimeError("phase contract exact role order")
 old_receipt_spec=parent.get("old_failed_static_receipt",{});old_receipt_record=regular(old_receipt_spec["path"],old_receipt_spec["sha256"],old_receipt_spec["logical_bytes"]);old_receipt=json.loads(Path(old_receipt_record["path"]).read_text())
 if old_receipt.get("sources")!=exact7_sources or old_receipt.get("sources_digest_sha256")!=exact7["canonical_records_digest_sha256"]:raise RuntimeError("old receipt ordered source closure")
 old_source_spec=parent.get("old_static_source",{});old_source_record=regular(old_source_spec["path"],old_source_spec["sha256"],old_source_spec["logical_bytes"])
 persistent_log_record=regular(f813["persistent_old_static_log"]["path"],f813["persistent_old_static_log"]["sha256"],f813["persistent_old_static_log"]["logical_bytes"]);volatile_log_record=regular(f813["volatile_log_source_at_registration"]["path"],f813["volatile_log_source_at_registration"]["sha256"],f813["volatile_log_source_at_registration"]["logical_bytes"])
 if persistent_log_record["sha256"]!=volatile_log_record["sha256"] or persistent_log_record["logical_bytes"]!=volatile_log_record["logical_bytes"]:raise RuntimeError("f813 log provenance")
 static_record=regular(STATIC_RECEIPT_PATH,STATIC_RECEIPT_SHA,STATIC_RECEIPT_BYTES);static=json.loads(STATIC_RECEIPT_PATH.read_text());validate_static(static,observed_sources["static_auditor"])
 if authority.get("static_receipt")!=static_record or contract.get("static_receipt")!=static_record:raise RuntimeError("static receipt alias")
 static_schema=contract.get("static_receipt_contract")
 if not isinstance(static_schema,dict) or set(static_schema)!=STATIC_SCHEMA_KEYS or set(static_schema["top_keys"])!=STATIC_TOP_KEYS or static_schema["format"]!=STATIC_FORMAT or static_schema["status"]!=STATIC_STATUS or static_schema["check_keys"]!=STATIC_CHECK_KEYS or static_schema["check_key_set_sha256"]!=STATIC_KEYSET_SHA or static_schema["checks_sha256"]!=STATIC_CHECKS_SHA or static_schema["runtime_observation_exact"]!=STATIC_RUNTIME:raise RuntimeError("static contract schema")
 formal_schema=contract.get("repair_formal_schema")
 if not isinstance(formal_schema,dict) or set(formal_schema)!=FORMAL_SCHEMA_KEYS or set(repair)!=set(formal_schema["top_keys"]) or repair.get("format")!=formal_schema["format"] or repair.get("status")!=formal_schema["status"] or repair.get("authorization")!=formal_schema["authorization_exact"] or repair.get("runtime_observation")!=formal_schema["runtime_observation_exact"]:raise RuntimeError("repair formal schema")
 repair_tree=tree_with_root(REPAIR_FORMAL_PATH.parent);static_tree=tree_with_root(STATIC_ROOT);authority_tree=tree_with_root(AUTHORITY_ROOT);f813_tree_rooted=tree_with_root(F813_PATH.parent);f813_tree=exact_tree(F813_PATH.parent)
 if repair_tree!=authority.get("repair_formal_registration_tree") or repair_tree["inventory"]!=[["preregistration.json",REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES]]:raise RuntimeError("repair REG exact1")
 if static_tree!=authority.get("postregistration_static_registration_tree") or static_tree["inventory"]!=[["static_audit.json",STATIC_RECEIPT_SHA,STATIC_RECEIPT_BYTES]]:raise RuntimeError("static REG exact1")
 if authority_tree["inventory"]!=[["authority_receipt.json",authority_record["sha256"],authority_record["logical_bytes"]]] or authority_tree["file_count"]!=1:raise RuntimeError("authority REG exact1")
 if f813_tree_rooted!=authority.get("f813_registration_tree") or f813_tree_rooted["file_count"]!=2:raise RuntimeError("f813 exact2")
 if tree_with_root(STATIC_EXECUTION_EVIDENCE_ROOT)!=STATIC_EXECUTION_EVIDENCE_TREE or authority.get("static_execution_evidence_tree")!=STATIC_EXECUTION_EVIDENCE_TREE:raise RuntimeError("static execution evidence exact6")
 for field,actual in (("repair_formal_registration_tree",repair_tree),("postregistration_static_registration_tree",static_tree),("static_execution_evidence_tree",STATIC_EXECUTION_EVIDENCE_TREE),("f813_registration_tree",f813_tree_rooted)):
  if contract.get(field)!=actual:raise RuntimeError(f"authority contract tree: {field}")
 if authority.get("input_snapshots_exactly_equal") is not True or authority.get("input_pre_snapshot")!=authority.get("input_post_snapshot"):raise RuntimeError("authority snapshots")
 if authority.get("attempt_root")!=str(ATTEMPT_ROOT) or authority.get("transparent_static_receipt_path")!=str(TRANSPARENT_PATH):raise RuntimeError("authority output paths")
 historical=authority.get("historical_absences");required=authority.get("required_absences")
 if historical!=static.get("required_absences") or set(historical)!=HISTORICAL_ABSENCE_KEYS or set(required)!=CURRENT_ABSENCE_KEYS:raise RuntimeError("authority absence schema")
 if contract.get("historical_absences")!={name:row["path"] for name,row in historical.items()} or contract.get("current_absences_after_authority")!={name:row["path"] for name,row in required.items()}:raise RuntimeError("authority contract absence crossbind")
 if any(row.get("absent") is not True for row in historical.values()) or any(row.get("absent") is not True or os.path.lexists(row.get("path","")) for row in required.values()):raise RuntimeError("authority absences")
 for key in ("superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"):
  if authority.get(key)!=static.get(key) or contract.get("static_failure_ancestry",{}).get(key)!=static.get(key):raise RuntimeError(f"failed lineage alias: {key}")
 if set(contract.get("static_failure_ancestry",{}))!=FAILURE_ANCESTRY_KEYS:raise RuntimeError("failure ancestry schema")
 for field,root in (("failed_attempt_tree",Path(repair["failed_reconciliation_attempt_tree"]["root"])),("formal_materialization_evidence_tree",J/"v488_v487_c71_exact7_schema_repair_materialization_evidence_seed1630_20260825")):
  if authority.get(field)!=tree_with_root(root):raise RuntimeError(field)
 if os.path.lexists(ATTEMPT_ROOT) or os.path.lexists(ATTEMPT_PREP) or os.path.lexists(TRANSPARENT_PATH) or os.path.lexists(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp")) or os.path.lexists(QUALIFICATION_ROOT):raise RuntimeError("execution outputs preexist")
 lineage=contract.get("lineage")
 if lineage!={"authority_root":str(AUTHORITY_ROOT),"attempt_root":str(ATTEMPT_ROOT),"attempt_prep":str(ATTEMPT_PREP),"transparent_static_receipt_path":str(TRANSPARENT_PATH),"qualification_root":str(QUALIFICATION_ROOT),"postregistration_static_root":str(STATIC_ROOT)}:raise RuntimeError("authority lineage")
 if current_processes():raise RuntimeError("live reconciliation process")
 return {"contract_record":contract_record,"authority_record":authority_record,"wrapper_record":wrapper_record,"materializer_record":materializer_record,"contract":contract,"authority":authority,"repair":repair,"f813":f813,"repair_record":repair_record,"f813_record":f813_record,"source_closure":observed_sources,"static_record":static_record,"static":static,"repair_tree":repair_tree,"static_tree":static_tree,"authority_tree":authority_tree,"f813_tree":f813_tree,"f813_tree_rooted":f813_tree_rooted,"exact7_records":exact7_records,"exact7_sources":exact7_sources,"exact7_digest":exact7["canonical_records_digest_sha256"],"phase_formal_record":phase_formal_record,"phase_contract_record":phase_contract_record,"old_receipt_record":old_receipt_record,"old_receipt":old_receipt,"old_source_record":old_source_record,"persistent_log_record":persistent_log_record,"volatile_log_record":volatile_log_record}


def immutable_snapshot(context:dict)->dict:
 authority=context["authority"]
 files={"authority_contract":context["contract_record"],"authority_receipt":context["authority_record"],"authority_materializer":regular(context["materializer_record"]["path"],context["materializer_record"]["sha256"],context["materializer_record"]["logical_bytes"]),"repair_formal":regular(REPAIR_FORMAL_PATH,REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES),"f813_formal":regular(F813_PATH,F813_SHA,F813_BYTES),"static_receipt":regular(STATIC_RECEIPT_PATH,context["static_record"]["sha256"],context["static_record"]["logical_bytes"]),"phase_formal":regular(context["phase_formal_record"]["path"],context["phase_formal_record"]["sha256"],context["phase_formal_record"]["logical_bytes"]),"phase_contract":regular(context["phase_contract_record"]["path"],context["phase_contract_record"]["sha256"],context["phase_contract_record"]["logical_bytes"]),"old_static_source":regular(context["old_source_record"]["path"],context["old_source_record"]["sha256"],context["old_source_record"]["logical_bytes"]),"old_static_receipt":regular(context["old_receipt_record"]["path"],context["old_receipt_record"]["sha256"],context["old_receipt_record"]["logical_bytes"]),"persistent_old_log":regular(context["persistent_log_record"]["path"],context["persistent_log_record"]["sha256"],context["persistent_log_record"]["logical_bytes"]),"volatile_old_log":regular(context["volatile_log_record"]["path"],context["volatile_log_record"]["sha256"],context["volatile_log_record"]["logical_bytes"])}
 sources={"authority":{role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in context["source_closure"].items()},"repair":{role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in context["repair"]["source_closure"].items()},"phase_exact7":{row["role"]:regular(row["path"],row["sha256"],row["logical_bytes"]) for row in context["exact7_records"]}}
 tree_specs={"repair":authority["repair_formal_registration_tree"],"static":authority["postregistration_static_registration_tree"],"authority":tree_with_root(AUTHORITY_ROOT),"static_execution":authority["static_execution_evidence_tree"],"formal_evidence":authority["formal_materialization_evidence_tree"],"failed_v487":authority["failed_attempt_tree"]}
 trees={name:tree_with_root(Path(spec["root"])) for name,spec in tree_specs.items()}
 if trees!=tree_specs:raise RuntimeError("immutable rooted trees")
 for name,tree_key,receipt_key in (("failed_f88","failed_static_execution_tree","failed_static_process_receipt"),("failed_v489","v489_failed_static_execution_tree","v489_failed_static_process_receipt")):
  root=Path(authority[receipt_key]["path"]).parent;observed=exact_tree(root)
  if observed!=authority[tree_key]:raise RuntimeError(f"immutable tree: {name}")
  trees[name]={"root":str(root),**observed}
 controlled={"attempt_root","attempt_prep","transparent_output","transparent_tmp"}
 absences={name:{"path":row["path"],"absent":not os.path.lexists(row["path"])} for name,row in authority["required_absences"].items() if name not in controlled}
 if not all(row["absent"] for row in absences.values()):raise RuntimeError("immutable absences")
 return {"files":files,"sources":sources,"trees":trees,"absences":absences}


def build_r2_command(context:dict)->list[str]:
 repair=context["repair"];f813=context["f813"];sources=repair["source_closure"];parent=f813["frozen_parent"]
 phase_contract=parent["phase_a_design_contract"];old_source=parent["old_static_source"];old_receipt=parent["old_failed_static_receipt"];old_log=parent["old_static_log_volatile_source"];recon_contract=f813["design_contract"]
 for row in (phase_contract,old_source,old_receipt,old_log,recon_contract):
  regular(row["path"],row["sha256"],row.get("logical_bytes"))
 return [str(RLPY),str(R2_PATH),"--preregistration",sources["parent_v485_preregistration"]["path"],"--preregistration-sha",sources["parent_v485_preregistration"]["sha256"],"--contract",phase_contract["path"],"--contract-sha",phase_contract["sha256"],"--old-static-source",old_source["path"],"--old-static-sha",old_source["sha256"],"--old-static-receipt",old_receipt["path"],"--old-static-receipt-sha",old_receipt["sha256"],"--old-static-log",old_log["path"],"--old-static-log-sha",old_log["sha256"],"--reconciliation-preregistration",str(F813_PATH),"--reconciliation-preregistration-sha",F813_SHA,"--repair-preregistration",str(REPAIR_FORMAL_PATH),"--repair-preregistration-sha",REPAIR_FORMAL_SHA,"--reconciliation-contract",recon_contract["path"],"--reconciliation-contract-sha",recon_contract["sha256"],"--reconciler-source",str(R2_PATH),"--reconciler-sha",R2_SHA,"--output",str(TRANSPARENT_PATH)]


def commit_intent_at(attempt_root:Path,attempt_prep:Path,intent:dict):
 prep_identity=None;stdout_stream=None;stderr_stream=None;baseline_mask=None;signals_blocked=False
 try:
  if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
  attempt_prep.mkdir();prep_identity=directory_identity(attempt_prep);fsync_dir(attempt_prep.parent)
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask);signals_blocked=False
  atomic_json(attempt_prep/"intent.json",intent)
  stdout_stream=(attempt_prep/"reconciler_stdout.log").open("xb",buffering=0);stderr_stream=(attempt_prep/"reconciler_stderr.log").open("xb",buffering=0)
  os.fsync(stdout_stream.fileno());os.fsync(stderr_stream.fileno());fsync_dir(attempt_prep)
  if directory_identity(attempt_prep)!=prep_identity or os.path.lexists(attempt_root):raise RuntimeError("attempt prep ownership")
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
  os.replace(attempt_prep,attempt_root);fsync_dir(attempt_root.parent)
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask);signals_blocked=False
  return stdout_stream,stderr_stream
 except BaseException as error:
  for stream in (stdout_stream,stderr_stream):
   if stream is not None and not stream.closed:stream.close()
  if prep_identity is not None and not os.path.lexists(attempt_root) and attempt_prep.is_dir() and not attempt_prep.is_symlink() and directory_identity(attempt_prep)==prep_identity:
   failure_path=attempt_prep/"terminal_receipt.json"
   if not os.path.lexists(failure_path) and not os.path.lexists(failure_path.with_name(failure_path.name+".tmp")):
    intent_path=attempt_prep/"intent.json";failure={"format":TERMINAL_FORMAT,"status":"failed_before_attempt_promotion_no_retry","passed":False,"attempt_nonce":intent["attempt_nonce"],"intent":regular(intent_path) if intent_path.is_file() and not intent_path.is_symlink() else None,"error_type":type(error).__name__,"error":str(error),"retry_authorized":False,**FALSE_AUTHORITIES}
    atomic_json(failure_path,failure);fsync_dir(attempt_prep)
   if directory_identity(attempt_prep)!=prep_identity or os.path.lexists(attempt_root):raise RuntimeError("attempt prep failure ownership")
   os.replace(attempt_prep,attempt_root);fsync_dir(attempt_root.parent)
  raise
 finally:
  if signals_blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)


def commit_intent(intent:dict):return commit_intent_at(ATTEMPT_ROOT,ATTEMPT_PREP,intent)


def validate_f813_transition(before:dict,after:dict,output_record:dict)->bool:
 expected=[*before["inventory"],["transparent_static_audit.json",output_record["sha256"],output_record["logical_bytes"]]]
 return before["file_count"]==2 and after["inventory"]==expected and after["file_count"]==3 and after["logical_file_bytes"]==before["logical_file_bytes"]+output_record["logical_bytes"]


def validate_transparent(context:dict)->tuple[dict,dict,dict]:
 output_record=regular(TRANSPARENT_PATH);receipt=json.loads(TRANSPARENT_PATH.read_text())
 if set(receipt)!=R2_RECEIPT_KEYS or receipt.get("format")!="strict-track2-v485-v169-cache-qualification-static-audit-v1" or receipt.get("status")!="passed_no_execution_authority" or receipt.get("passed") is not True:raise RuntimeError("transparent receipt schema")
 check_keys=receipt.get("check_keys");checks=receipt.get("checks")
 if not isinstance(check_keys,list) or len(check_keys)!=47 or check_keys!=sorted(checks) or csha(check_keys)!=R2_CHECK_KEYSET_SHA or checks!={key:True for key in check_keys} or receipt.get("checks_sha256")!=csha(checks):raise RuntimeError("transparent checks")
 if receipt.get("repair_preregistration")!=context["repair_record"] or receipt.get("reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES} or receipt.get("receipt_writer")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("transparent repair ancestry")
 repair=context["repair"];repair_sources=repair["source_closure"]
 expected_repair_ancestry={"f813_reconciliation_preregistration":repair_sources["f813_reconciliation_preregistration"],"repair_preregistration":context["repair_record"],"failed_v487_attempt_tree":repair["failed_reconciliation_attempt_tree"],"v487_authority_tree":repair["v487_authority_tree"],"v487_helper_evidence_tree":repair["v487_helper_evidence_tree"],"exact7_schema_repair_proof":repair["exact7_schema_repair_proof"]}
 if receipt.get("repair_format")!="strict-track2-v488-v487-c71-exact7-schema-repair-transparent-reconciliation-v1" or receipt.get("repair_design_contract")!=context["source_closure"]["repair_design_contract"] or receipt.get("repair_materializer_source")!=context["source_closure"]["repair_formal_materializer"] or receipt.get("old_reconciler_c71_source")!=repair_sources["old_reconciler_c71"] or receipt.get("repair_source_closure")!=repair_sources or receipt.get("repair_formal_ancestry")!=expected_repair_ancestry:raise RuntimeError("transparent expanded repair ancestry")
 if receipt.get("repair_registration_tree")!=exact_tree(REPAIR_FORMAL_PATH.parent) or receipt.get("f813_registration_tree")!=context["f813_tree"] or receipt.get("registration_initial_inventory")!=context["f813_tree"]:raise RuntimeError("transparent initial trees")
 if receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot")!=receipt.get("input_post_snapshot"):raise RuntimeError("transparent snapshots")
 if receipt.get("sources")!=context["exact7_sources"] or receipt.get("sources_digest_sha256")!=context["exact7_digest"] or csha(context["exact7_records"])!=context["exact7_digest"]:raise RuntimeError("transparent source digest")
 old=context["old_receipt"];f813=context["f813"]
 if receipt.get("contract")!=old.get("contract") or receipt.get("preregistration")!=old.get("preregistration") or receipt.get("static_auditor_self_sha256")!=old.get("static_auditor_self_sha256"):raise RuntimeError("transparent old receipt aliases")
 if receipt.get("reconciliation_preregistration")!={"path":context["f813_record"]["path"],"sha256":context["f813_record"]["sha256"]} or receipt.get("reconciliation_design_contract")!=f813.get("design_contract"):raise RuntimeError("transparent f813 aliases")
 if receipt.get("persistent_old_static_log")!=context["persistent_log_record"] or receipt.get("volatile_log_source_at_registration")!=context["volatile_log_record"]:raise RuntimeError("transparent log ancestry")
 old_rooted=f813["frozen_parent"]["old_registration_tree"];old_unrooted={key:old_rooted[key] for key in ("inventory","file_count","logical_file_bytes","sha256sum_lines_digest_sha256","canonical_json_triples_digest_sha256")}
 expected_old_ancestry={"old_static_source":context["old_source_record"],"old_failed_static_receipt":context["old_receipt_record"],"old_registration_tree":old_unrooted,"old_true_check_count":44,"old_false_check_names":["determinism_scope_exact","driver_owned_full_lifetime_logs_and_completion","no_training_reward_outcome"]}
 if receipt.get("reconciliation_ancestry")!=expected_old_ancestry:raise RuntimeError("transparent old ancestry")
 if receipt.get("runtime_observation")!=old.get("runtime_observation") or receipt.get("parent_runtime_observation")!=old.get("runtime_observation") or receipt.get("reconciliation_runtime_observation")!={"reconciliation_executed":True,"phase_a_executed":False,"v169_imported_or_run":False,"qualification_root_created":False,"training_launched":False,"folds":0,"policy_updates":0,"reward_read_or_loaded":False,"dev_hidden_final_outcome_read":False}:raise RuntimeError("transparent runtime observations")
 parent_prereg=Path(context["repair"]["source_closure"]["parent_v485_preregistration"]["path"])
 expected_absences=[str(QUALIFICATION_ROOT.resolve()),str((parent_prereg.parent/"phase_a_launcher_attempts").resolve()),str((parent_prereg.parent/"authority_receipt.json").resolve())]
 if receipt.get("immutable_absences")!=expected_absences or any(os.path.lexists(path) for path in expected_absences):raise RuntimeError("transparent immutable absences")
 proofs=receipt.get("false_positive_recomputation")
 if not isinstance(proofs,dict) or set(proofs)!={"determinism_scope_exact","driver_owned_full_lifetime_logs_and_completion","no_training_reward_outcome"} or any(not isinstance(row,dict) or row.get("passed") is not True for row in proofs.values()):raise RuntimeError("transparent structured proofs")
 if receipt.get("reconciliation_only") is not True or receipt.get("phase_a_executed") is not False or receipt.get("v169_imported_or_run") is not False:raise RuntimeError("transparent runtime")
 for key in ("phase_a_cache_qualification_authorized","cache_reuse_authorized","training_authorized","s1_authorized","zero_update_authorized","rl_authorized","submission_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized"):
  if receipt.get(key) is not False:raise RuntimeError(f"transparent unsafe: {key}")
 if receipt.get("folds_authorized")!=0 or receipt.get("policy_updates")!=0:raise RuntimeError("transparent unsafe counts")
 expanded={"repair_format","repair_preregistration","repair_design_contract","repair_materializer_source","old_reconciler_c71_source","reconciler_r2_source","repair_source_closure","repair_registration_tree","f813_registration_tree","repair_formal_ancestry"}
 synthetic=receipt.get("synthetic_recomputation_evidence",{})
 expected_synthetic_checks={"literal_not_call":True,"false_call_detected":True,"two_argument_driver_accepted":True,"false_provenance_names_allowed":True}
 if not expanded.issubset(receipt) or synthetic!={"passed":True,"check_count":4,"checks":expected_synthetic_checks,"fixture_sources_sha256":R2_SYNTHETIC_FIXTURE_SHA,"evidence_sha256":R2_SYNTHETIC_EVIDENCE_SHA}:raise RuntimeError("transparent expanded evidence")
 after=exact_tree(F813_PATH.parent)
 if not validate_f813_transition(context["f813_tree"],after,output_record) or os.path.lexists(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp")):raise RuntimeError("f813 exact2 to exact3")
 return receipt,after,output_record


def acquire_execution_lock(path:Path)->int:
 if fcntl is None:raise RuntimeError("fcntl required")
 fd=os.open(str(path),os.O_RDONLY)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BaseException:os.close(fd);raise
 return fd


def release_execution_lock(fd:int|None)->None:
 if fd is None:return
 try:
  if fcntl is not None:fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)


def process_identity()->dict:
 stat_fields=(Path("/proc/self/stat").read_text().split() if Path("/proc/self/stat").is_file() else [])
 return {"pid":os.getpid(),"pgid":os.getpgrp(),"sid":os.getsid(0),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":int(stat_fields[21]),"executable":regular(Path(sys.executable))}


def fsync_close(stream)->None:
 if stream is None or stream.closed:return
 stream.flush();os.fsync(stream.fileno());stream.close()


def safe_snapshot(context:dict):
 try:return immutable_snapshot(context)
 except BaseException as error:return {"snapshot_error_type":type(error).__name__,"snapshot_error":str(error)}


def commit_terminal_exact4(root:Path,receipt:dict,immutable_snapshot,phase_hook=None)->dict:
 phase_hook=phase_hook or (lambda _phase:None);terminal_path=root/"terminal_receipt.json";tmp=terminal_path.with_name(terminal_path.name+".tmp")
 if root!=root.resolve() or not root.is_dir() or root.is_symlink() or os.path.lexists(terminal_path) or os.path.lexists(tmp):raise RuntimeError("terminal prestate")
 identity=directory_identity(root);pretree=exact_tree(root);expected_pre=["intent.json","reconciler_stderr.log","reconciler_stdout.log"]
 if [row[0] for row in pretree["inventory"]]!=expected_pre or pretree["file_count"]!=3:raise RuntimeError("attempt preterminal exact3")
 immutable_pre=immutable_snapshot();received_signal=None;committed=False;baseline_mask=None;blocked=False
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGINT,signal.SIGTERM)}
 if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});blocked=True
 def on_signal(signum,_frame):
  nonlocal received_signal
  received_signal=signum
  if not committed:raise ControlledSignal(f"terminal signal {signum}")
 for signum in previous:signal.signal(signum,on_signal)
 try:
  phase_hook("terminal_signals_blocked")
  if directory_identity(root)!=identity or exact_tree(root)!=pretree or immutable_snapshot()!=immutable_pre:raise RuntimeError("terminal precommit drift")
  payload=dict(receipt);payload["attempt_preterminal_tree"]=pretree;payload["terminal_commit_expected_sole_fourth_file"]="terminal_receipt.json";payload["terminal_signal_policy"]="deferred_until_exact4_postcheck_then_committed_terminal_priority"
  atomic_json(terminal_path,payload);fsync_dir(root)
  posttree=exact_tree(root)
  if directory_identity(root)!=identity or [row[0] for row in posttree["inventory"]]!=expected_pre+["terminal_receipt.json"] or posttree["inventory"][:3]!=pretree["inventory"] or posttree["file_count"]!=4 or os.path.lexists(tmp) or immutable_snapshot()!=immutable_pre:raise RuntimeError("terminal exact4 postcheck")
  committed=True
 finally:
  if blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)
  for signum,handler in previous.items():signal.signal(signum,handler)
 return {"terminal_receipt":regular(terminal_path),"attempt_tree":posttree,"immutable_snapshot":immutable_pre,"deferred_signal":received_signal,"committed":True}


def commit_terminal_with_priority(root:Path,receipt:dict,immutable_snapshot,terminal_state:dict,inner_phase_hook=None,caller_phase_hook=None)->dict:
 if terminal_state!={"committed":False}:raise RuntimeError("terminal caller state")
 caller_phase_hook=caller_phase_hook or (lambda _phase:None);received_signal=None;baseline_mask=None;blocked=False
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGINT,signal.SIGTERM)}
 if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});blocked=True
 def caller_signal(signum,_frame):
  nonlocal received_signal
  received_signal=signum
  if not terminal_state["committed"]:raise ControlledSignal(f"terminal caller signal {signum}")
 for signum in previous:signal.signal(signum,caller_signal)
 try:
  result=commit_terminal_exact4(root,receipt,immutable_snapshot,inner_phase_hook)
  caller_phase_hook("after_inner_return_before_flag")
  terminal_state["committed"]=True
 finally:
  if blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)
  for signum,handler in previous.items():signal.signal(signum,handler)
 result["caller_deferred_signal"]=received_signal;result["caller_committed_priority"]=terminal_state["committed"]
 return result


def synthetic_self_test()->int:
 checks={
  "static_schema_digests":csha(STATIC_CHECK_KEYS)==STATIC_KEYSET_SHA and csha({key:True for key in STATIC_CHECK_KEYS})==STATIC_CHECKS_SHA,
  "authority_schema_digests":csha(AUTHORITY_CHECK_KEYS)==AUTHORITY_KEYSET_SHA and csha({key:True for key in AUTHORITY_CHECK_KEYS})==AUTHORITY_CHECKS_SHA,
  "authority_top38":len(AUTHORITY_TOP_KEYS)==38 and len(AUTHORITY_CHECK_KEYS)==28,
 "dual_formal_command_literals":str(F813_PATH)!=str(REPAIR_FORMAL_PATH) and "--reconciliation-preregistration" in build_r2_command.__code__.co_consts and "--repair-preregistration" in build_r2_command.__code__.co_consts,
 }
 if F813_PATH.is_file() and REPAIR_FORMAL_PATH.is_file():
  f813=json.loads(F813_PATH.read_text());repair=json.loads(REPAIR_FORMAL_PATH.read_text());exact7=f813["exact7_source_closure"];records=exact7["records"];roles=exact7["roles_in_order"];derived={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in records};phase=json.loads(Path(repair["source_closure"]["parent_v485_preregistration"]["path"]).read_text());phase_contract=json.loads(Path(f813["frozen_parent"]["phase_a_design_contract"]["path"]).read_text());old=json.loads(Path(f813["frozen_parent"]["old_failed_static_receipt"]["path"]).read_text())
  checks["actual_frozen_ordered_records_consumer"]=phase_contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]==roles and [row["role"] for row in records]==roles and csha(records)==exact7["canonical_records_digest_sha256"]==phase["execution_sources_digest_sha256"]==old["sources_digest_sha256"] and phase["execution_source_records"]==records and phase["execution_sources"]==derived and old["sources"]==derived
 else:checks["actual_frozen_ordered_records_consumer"]=True
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);owner=root/"owner";prep=root/"owner.prep";intent={"attempt_nonce":"1"*64}
  out,err=commit_intent_at(owner,prep,intent);fsync_close(out);fsync_close(err)
  checks["atomic_intent_exact3_before_terminal"]=exact_tree(owner)["inventory"]==[["intent.json",sha(owner/"intent.json"),(owner/"intent.json").stat().st_size],["reconciler_stderr.log",hashlib.sha256(b"").hexdigest(),0],["reconciler_stdout.log",hashlib.sha256(b"").hexdigest(),0]] and not os.path.lexists(prep)
  success_state={"committed":False};success_commit=commit_terminal_with_priority(owner,{"format":TERMINAL_FORMAT,"status":"synthetic_passed","passed":True},lambda:{"stable":True},success_state)
  checks["success_terminal_exact4"]=success_commit["committed"] and success_commit["attempt_tree"]["file_count"]==4
  failure_root=root/"failure";failure_prep=root/"failure.prep";out,err=commit_intent_at(failure_root,failure_prep,intent);fsync_close(out);fsync_close(err);failure_state={"committed":False};failure_commit=commit_terminal_with_priority(failure_root,{"format":TERMINAL_FORMAT,"status":"failed_no_retry","passed":False},lambda:{"stable":True},failure_state)
  checks["failure_terminal_exact4"]=failure_commit["committed"] and failure_commit["attempt_tree"]["file_count"]==4
  extra_root=root/"extra";extra_prep=root/"extra.prep";out,err=commit_intent_at(extra_root,extra_prep,intent);fsync_close(out);fsync_close(err);(extra_root/"foreign").write_text("x")
  try:commit_terminal_with_priority(extra_root,{"format":TERMINAL_FORMAT,"status":"synthetic","passed":False},lambda:{"stable":True},{"committed":False});extra_rejected=False
  except RuntimeError:extra_rejected=True
  checks["foreign_preterminal_extra_rejected"]=extra_rejected and not os.path.lexists(extra_root/"terminal_receipt.json")
  foreign=root/"foreign.prep";foreign.mkdir();(foreign/"marker").write_text("foreign")
  try:commit_intent_at(root/"foreign",foreign,intent);rejected=False
  except FileExistsError:rejected=True
  checks["foreign_prep_loser_untouched"]=rejected and (foreign/"marker").read_text()=="foreign" and not os.path.lexists(root/"foreign")
  tree_root=root/"reg";tree_root.mkdir();(tree_root/"a").write_bytes(b"a");(tree_root/"b").write_bytes(b"bb");before=exact_tree(tree_root);(tree_root/"transparent_static_audit.json").write_bytes(b"ccc");record=regular(tree_root/"transparent_static_audit.json")
  checks["exact2_to_sole_exact3"]=validate_f813_transition(before,exact_tree(tree_root),record)
  if os.name=="posix":
   signal_root=root/"signal";signal_prep=root/"signal.prep";out,err=commit_intent_at(signal_root,signal_prep,intent);fsync_close(out);fsync_close(err)
   signal_state={"committed":False};signaled=commit_terminal_with_priority(signal_root,{"format":TERMINAL_FORMAT,"status":"synthetic_signal_commit","passed":True},lambda:{"stable":True},signal_state,None,lambda phase:os.kill(os.getpid(),signal.SIGTERM) if phase=="after_inner_return_before_flag" else None)
   checks["terminal_deferred_signal_commit_priority"]=signaled["committed"] and signaled["caller_deferred_signal"]==signal.SIGTERM and signaled["caller_committed_priority"] is True and signal_state["committed"] is True and signaled["attempt_tree"]["file_count"]==4
   sleeper=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);cleanup=terminate_group(sleeper)
   checks["bounded_group_cleanup"]=cleanup["reaped"] and cleanup["group_empty"]
   lock_path=root/"lock";lock_path.write_bytes(b"x");fd=acquire_execution_lock(lock_path)
   contender=subprocess.run([sys.executable,"-c","import fcntl,os,sys;f=os.open(sys.argv[1],os.O_RDONLY)\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);sys.exit(1)\nexcept BlockingIOError:sys.exit(0)",str(lock_path)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
   release_execution_lock(fd);checks["external_flock_contender_rejected"]=contender.returncode==0
  else:
   checks["terminal_deferred_signal_commit_priority"]=True;checks["bounded_group_cleanup"]=True;checks["external_flock_contender_rejected"]=True
 import ast
 tree=ast.parse(Path(__file__).read_text());main_node=next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="main")
 calls=[node for node in ast.walk(main_node) if isinstance(node,ast.Call)]
 checks["main_unique_popen"] = sum(isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess" and node.func.attr=="Popen" for node in calls)==1
 checks["main_under_lock_preflight"] = sum(isinstance(node.func,ast.Name) and node.func.id=="preflight" for node in calls)==2 and sum(isinstance(node.func,ast.Name) and node.func.id=="acquire_execution_lock" for node in calls)==1
 result={"passed":set(checks.values())=={True},"checks":checks,"checks_sha256":csha(checks)};print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 3


def parse_args():
 parser=argparse.ArgumentParser()
 parser.add_argument("--f813-preregistration",type=Path,required=True);parser.add_argument("--f813-preregistration-sha",required=True)
 parser.add_argument("--repair-preregistration",type=Path,required=True);parser.add_argument("--repair-preregistration-sha",required=True)
 parser.add_argument("--authority-contract",type=Path,required=True);parser.add_argument("--authority-contract-sha",required=True)
 parser.add_argument("--authority-receipt",type=Path,required=True);parser.add_argument("--authority-receipt-sha",required=True)
 parser.add_argument("--wrapper-source",type=Path,required=True);parser.add_argument("--wrapper-sha",required=True)
 return parser.parse_args()


def main()->int:
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic_self_test()
 args=parse_args()
 if args.f813_preregistration_sha!=F813_SHA or args.repair_preregistration_sha!=REPAIR_FORMAL_SHA:raise RuntimeError("frozen formal SHA arguments")
 context=preflight(args)
 lock_fd=acquire_execution_lock(AUTHORITY_RECEIPT_PATH)
 process=None;stdout_stream=None;stderr_stream=None;terminal_state={"committed":False};intent_record=None;old_handlers={};started=time.monotonic_ns();cleanup={"started":False,"term_sent":False,"kill_sent":False,"reaped":True,"group_empty":True}
 def interrupted(signum,_frame):
  if terminal_state["committed"]:return
  raise ControlledSignal(f"wrapper signal {signum}")
 try:
  context=preflight(args)
  command=build_r2_command(context);core_pre=immutable_snapshot(context)
  intent={"format":INTENT_FORMAT,"status":"committed_before_reconciler_r2_start","attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"process_identity":process_identity(),"command_argv":command,"command_argv_sha256":csha(command),"authority_receipt":context["authority_record"],"repair_registration_tree":context["repair_tree"],"postregistration_static_registration_tree":context["static_tree"],"static_execution_evidence_tree":STATIC_EXECUTION_EVIDENCE_TREE,"f813_registration_tree_before":context["f813_tree_rooted"],"immutable_input_snapshot":core_pre,"timeout_seconds":300,"retry_authorized":False,**FALSE_AUTHORITIES}
  for signum in (signal.SIGINT,signal.SIGTERM):old_handlers[signum]=signal.getsignal(signum);signal.signal(signum,interrupted)
  stdout_stream,stderr_stream=commit_intent(intent);intent_record=regular(ATTEMPT_ROOT/"intent.json")
  process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout_stream,stderr=stderr_stream,start_new_session=True)
  try:returncode=process.wait(timeout=300)
  except subprocess.TimeoutExpired as error:raise RuntimeError("reconciler r2 timeout") from error
  fsync_close(stdout_stream);stdout_stream=None;fsync_close(stderr_stream);stderr_stream=None
  cleanup=terminate_group(process)
  if returncode!=0 or not cleanup["reaped"] or not cleanup["group_empty"]:raise RuntimeError("reconciler r2 nonzero or group retained")
  transparent,f813_after,output_record=validate_transparent(context)
  core_post=immutable_snapshot(context)
  if core_post!=core_pre:raise RuntimeError("immutable inputs changed")
  if current_processes() or os.path.lexists(QUALIFICATION_ROOT) or os.path.lexists(ATTEMPT_PREP):raise RuntimeError("post execution boundary")
  terminal={"format":TERMINAL_FORMAT,"status":"passed_readonly_reconciliation_source_repair_no_phase_a_execution_authority","passed":True,"attempt_nonce":intent["attempt_nonce"],"wall_seconds":float((time.monotonic_ns()-started)/1e9),"intent":intent_record,"reconciler_r2_source":{"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES},"reconciler_exit_code":0,"reconciler_process_cleanup":cleanup,"reconciler_stdout":regular(ATTEMPT_ROOT/"reconciler_stdout.log"),"reconciler_stderr":regular(ATTEMPT_ROOT/"reconciler_stderr.log"),"transparent_static_receipt":output_record,"transparent_receipt_checks_sha256":transparent["checks_sha256"],"repair_registration_tree":tree_with_root(REPAIR_FORMAL_PATH.parent),"postregistration_static_registration_tree":tree_with_root(STATIC_ROOT),"static_execution_evidence_tree":tree_with_root(STATIC_EXECUTION_EVIDENCE_ROOT),"authority_registration_tree":tree_with_root(AUTHORITY_ROOT),"f813_registration_tree_before":context["f813_tree_rooted"],"f813_registration_tree_after":{"root":str(F813_PATH.parent),**f813_after},"f813_exact2_to_sole_exact3":True,"immutable_input_pre_snapshot":core_pre,"immutable_input_post_snapshot":core_post,"immutable_inputs_exactly_equal":True,"failed_v487_attempt_tree":context["authority"]["failed_attempt_tree"],"superseded_static_source":context["authority"]["superseded_static_source"],"failed_static_execution_tree":context["authority"]["failed_static_execution_tree"],"failed_static_process_receipt":context["authority"]["failed_static_process_receipt"],"superseded_static_absences":context["authority"]["superseded_static_absences"],"v489_failed_static_source":context["authority"]["v489_failed_static_source"],"v489_failed_static_execution_tree":context["authority"]["v489_failed_static_execution_tree"],"v489_failed_static_process_receipt":context["authority"]["v489_failed_static_process_receipt"],"v489_superseded_static_absences":context["authority"]["v489_superseded_static_absences"],"v489_predeploy_transport_failure":context["authority"]["v489_predeploy_transport_failure"],"retry_authorized":False,**FALSE_AUTHORITIES}
  if not math.isfinite(terminal["wall_seconds"]) or terminal["wall_seconds"]<0:raise RuntimeError("terminal wall")
  commit_terminal_with_priority(ATTEMPT_ROOT,terminal,lambda:immutable_snapshot(context),terminal_state)
  return 0
 except BaseException as error:
  try:fsync_close(stdout_stream)
  except BaseException:pass
  try:fsync_close(stderr_stream)
  except BaseException:pass
  try:cleanup=terminate_group(process)
  except BaseException as cleanup_error:cleanup={"cleanup_error_type":type(cleanup_error).__name__,"cleanup_error":str(cleanup_error),"group_empty":False,"reaped":False}
  terminal_path=ATTEMPT_ROOT/"terminal_receipt.json"
  if terminal_state["committed"] or terminal_path.is_file():raise
  if ATTEMPT_ROOT.is_dir() and not ATTEMPT_ROOT.is_symlink():
   if intent_record is None and (ATTEMPT_ROOT/"intent.json").is_file():intent_record=regular(ATTEMPT_ROOT/"intent.json")
   failure={"format":TERMINAL_FORMAT,"status":"failed_no_retry","passed":False,"wall_seconds":float((time.monotonic_ns()-started)/1e9),"intent":intent_record,"error_type":type(error).__name__,"error":str(error),"cleanup":cleanup,"reconciler_stdout":regular(ATTEMPT_ROOT/"reconciler_stdout.log") if (ATTEMPT_ROOT/"reconciler_stdout.log").is_file() else None,"reconciler_stderr":regular(ATTEMPT_ROOT/"reconciler_stderr.log") if (ATTEMPT_ROOT/"reconciler_stderr.log").is_file() else None,"transparent_output_present":TRANSPARENT_PATH.is_file() and not TRANSPARENT_PATH.is_symlink(),"immutable_post_failure_snapshot":safe_snapshot(context),"retry_authorized":False,**FALSE_AUTHORITIES}
   commit_terminal_with_priority(ATTEMPT_ROOT,failure,lambda:safe_snapshot(context),terminal_state)
  raise
 finally:
  try:
   for signum,handler in old_handlers.items():signal.signal(signum,handler)
  except BaseException:
   if not terminal_state["committed"]:raise
  try:release_execution_lock(lock_fd)
  except BaseException:
   if not terminal_state["committed"]:raise


if __name__=="__main__":raise SystemExit(main())
