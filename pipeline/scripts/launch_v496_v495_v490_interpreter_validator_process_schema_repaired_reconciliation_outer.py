#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,fcntl,hashlib,json,math,os,signal,stat,subprocess,sys,tempfile,time
from pathlib import Path

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge");J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810";RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
SELF_PATH=ROOT/"pipeline/scripts/launch_v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer.py"
AUTH_CONTRACT=ROOT/"pipeline/scripts/v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_contract.json"
AUTH_ROOT=J/"v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_seed1638_20260825";AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+".registration-prep");AUTH_RECEIPT=AUTH_ROOT/"authority_receipt.json"
V493_AUTH_CONTRACT=ROOT/"pipeline/scripts/v493_v492_v490_corrected_reconciliation_execution_authority_contract.json";V493_AUTH_CONTRACT_SHA="0db696b01c8a561ce717b500b54bb3fd7c73e2f81b1ae6c117d92ed93de85e9f";V493_AUTH_CONTRACT_BYTES=26569
V493_AUTH_MATERIALIZER=ROOT/"pipeline/scripts/materialize_v493_v492_v490_corrected_reconciliation_execution_authority.py";V493_AUTH_MATERIALIZER_SHA="1d9a3fa94aa04c87afe309c0a7662c2ee51eb8eeda9d307132476252097699ad";V493_AUTH_MATERIALIZER_BYTES=33691
V493_AUTH_RECEIPT=J/"v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825/authority_receipt.json";V493_AUTH_RECEIPT_SHA="2a93aebbfb968c8134b92b4ed7bf2e809a32085483d1e745a9393997f55167c7";V493_AUTH_RECEIPT_BYTES=60534
V493_OUTER_WRAPPER=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_outer.py";V493_OUTER_WRAPPER_SHA="658c13857de127e4661fb1a010c2736951de5bd972e53c56a577c7cda882fa0a";V493_OUTER_WRAPPER_BYTES=39225
V493_INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_inner.py";V493_INNER_WRAPPER_SHA="343577ec4050a63ecf86ba082b23b5c5b869f90de8abb9f9d10f22cc47510597";V493_INNER_WRAPPER_BYTES=75190
V493_OUTER_FAILURE_ROOT=J/"v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825"
V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT=J/"v493_v492_v490_corrected_reconciliation_execution_authority_materialization_evidence_seed1635_20260825"
V494_AUTH_CONTRACT=ROOT/"pipeline/scripts/v494_v493_v490_interpreter_symlink_repair_execution_authority_contract.json";V494_AUTH_CONTRACT_SHA="2f4cef2671dfd5c443d5a7f53dc0ea7b770b520da84cba3512a363884d56bd4a";V494_AUTH_CONTRACT_BYTES=29803
V494_AUTH_MATERIALIZER=ROOT/"pipeline/scripts/materialize_v494_v493_v490_interpreter_symlink_repair_execution_authority.py";V494_AUTH_MATERIALIZER_SHA="5b9396fd74e1d161fa41610108d823f356313992364cba708ed869519fa0a421";V494_AUTH_MATERIALIZER_BYTES=39236
V494_OUTER_WRAPPER=ROOT/"pipeline/scripts/launch_v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer.py";V494_OUTER_WRAPPER_SHA="bea4b6df050afb22c833908725a6af7417b292984d8ddb4fbb972b99128d3df5";V494_OUTER_WRAPPER_BYTES=43024
V494_INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner.py";V494_INNER_WRAPPER_SHA="1f2c9de60703f12c22966b7aeb3526a6c32e0d5a0e0258094c9d208b9a56f4bf";V494_INNER_WRAPPER_BYTES=79760
V494_FAILURE_FORENSIC=ROOT/"pipeline/scripts/v495_v494_authority_materializer_failure_forensic_reconstructed.json";V494_FAILURE_FORENSIC_SHA="248c4a94a24c3b793bb29b6cc07a5955a93694f7e22b1a9363ef5bb7ee0ba0dc";V494_FAILURE_FORENSIC_BYTES=6409
V494_FAILURE_TRANSPORT=ROOT/"pipeline/scripts/v495_v494_failed_authority_materialization_transport_script.sh";V494_FAILURE_TRANSPORT_SHA="0d4669e9a1fa52ee98fe2c459e9e42d880116e4428270edcefb0f50e5b95d49c";V494_FAILURE_TRANSPORT_BYTES=942
V494_AUTH_ROOT=J/"v494_v493_v490_interpreter_symlink_repair_execution_authority_seed1636_20260825";V494_OUTER_ROOT=J/"v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer_execution_evidence_seed1636_20260825";V494_INNER_ROOT=J/"v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner_attempt_seed1636_20260825"
V495_AUTH_CONTRACT=ROOT/"pipeline/scripts/v495_v494_v490_interpreter_validator_repair_execution_authority_contract.json";V495_AUTH_CONTRACT_SHA="60d978a803bd148f6fe68906e71113cf33e61a3ae5bf77e22c8f2855c68c2823";V495_AUTH_CONTRACT_BYTES=37084
V495_AUTH_MATERIALIZER=ROOT/"pipeline/scripts/materialize_v495_v494_v490_interpreter_validator_repair_execution_authority.py";V495_AUTH_MATERIALIZER_SHA="b0f4b59212ea67754a12394bf1f5a8e7d2c5389c93c44207e0099d9f2850fe18";V495_AUTH_MATERIALIZER_BYTES=50978
V495_INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v495_v494_v490_interpreter_validator_repaired_reconciliation_inner.py";V495_INNER_WRAPPER_SHA="b52e7eb29870141bda85eead81cbc1c3a47522ad86e6a7b6829f7052766a8f76";V495_INNER_WRAPPER_BYTES=90261
V495_OUTER_WRAPPER=ROOT/"pipeline/scripts/launch_v495_v494_v490_interpreter_validator_repaired_reconciliation_outer.py";V495_OUTER_WRAPPER_SHA="3aae6deb4059e143cc40a3e10bdf6f1cf51e20e884c0459b4aa950e0d480d526";V495_OUTER_WRAPPER_BYTES=50749
V495_FAILURE_ROOT=J/"v495_v494_v490_interpreter_validator_repair_execution_authority_materialization_evidence_seed1637_20260825";V495_FAILURE_PROCESS=V495_FAILURE_ROOT/"process_receipt.json";V495_FAILURE_PROCESS_SHA="72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4";V495_FAILURE_PROCESS_BYTES=411
V495_FAILURE_TREE={"root":str(V495_FAILURE_ROOT),"inventory":[["argv.json","54cda5c2853e9b9d74c1abc11b3c56ef51de5a1650487d8813c4454fdc17b8a3",16203],["intent.json","e339933407261a528b516d9bb3759fffdc853fc7998416b5f1e66b745a032828",542],["materializer_stderr.log","bfe41948b189d59af0aa4644bc3be1d32d4a2dea8537d6b867ba3c51a4cdbc75",544],["materializer_stdout.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["process_receipt.json",V495_FAILURE_PROCESS_SHA,V495_FAILURE_PROCESS_BYTES],["transport_helper.py","f2ad3a5a10fb419c0fdd1595997e212c5ec1ccfbb6f6ec3c23410d7ab9b31727",14439]],"file_count":6,"logical_file_bytes":32139,"sha256sum_lines_digest_sha256":"21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1b","canonical_json_triples_digest_sha256":"9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290"}
INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner.py";INNER_WRAPPER_SHA="c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f";INNER_WRAPPER_BYTES=101600
R2=ROOT/"pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py";R2_SHA="9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377";R2_BYTES=51845
F813=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json";F813_SHA="f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8";F813_BYTES=21296
REPAIR=J/"v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json";REPAIR_SHA="b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b";REPAIR_BYTES=36181
STATIC=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json";STATIC_SHA="441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb";STATIC_BYTES=49008
PHASE_CONTRACT=ROOT/"pipeline/scripts/v485_v482_v169_cache_determinism_scope_repair_contract.json";PHASE_CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64";PHASE_CONTRACT_BYTES=43960
INNER_ATTEMPT=J/"v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner_attempt_seed1638_20260825";INNER_PREP=INNER_ATTEMPT.with_name(INNER_ATTEMPT.name+".attempt-prep")
TRANSPARENT=F813.parent/"transparent_static_audit.json";TRANSPARENT_TMP=TRANSPARENT.with_name(TRANSPARENT.name+".tmp")
OUTER_EVIDENCE=J/"v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer_execution_evidence_seed1638_20260825";OUTER_PREP=OUTER_EVIDENCE.with_name(OUTER_EVIDENCE.name+".outer-prep")
QUALIFICATION=Path("/root/v485_v169_cache_qualification_seed1627_20260824")
AUTH_FORMAT="strict-track2-v496-v495-v490-interpreter-validator-process-schema-repair-execution-authority-v1";AUTH_STATUS="authorized_exact_one_external_v496_interpreter_validator_process_schema_repaired_outer_attempt"
CONTRACT_FORMAT="strict-track2-v496-v495-v490-interpreter-validator-process-schema-repair-execution-authority-design-contract-v1";CONTRACT_STATUS="design_only_frozen_sources_pending_independent_review_no_authority"
OUTER_INTENT_FORMAT="strict-track2-v496-v495-v490-interpreter-validator-process-schema-repaired-reconciliation-outer-attempt-intent-v1";OUTER_TERMINAL_FORMAT="strict-track2-v496-v495-v490-interpreter-validator-process-schema-repaired-reconciliation-outer-attempt-terminal-v1"
AUTHORIZATION={"outer_execution_wrapper_authorized":True,"outer_attempts_authorized":1,"outer_attempts_consumed":0,"retry_authorized":False,"direct_corrected_inner_authorized":False,"direct_r2_authorized":False,"nested_corrected_inner_invocations_authorized":1,"nested_r2_invocations_authorized":1,"nested_corrected_inner_only_via_outer":True,"nested_r2_only_via_corrected_inner":True,"phase_a_authorized":False,"cache_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
RUNTIME={"execution_authority_materialized":True,"outer_execution_wrapper_executed":False,"corrected_inner_wrapper_executed":False,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
EXECUTION_BOUNDARY={"authority_materialization_only":True,"outer_execution_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"phase_a_invocations":0,"training_invocations":0,"reward_reads":0,"dev_hidden_final_outcome_reads":0}
AUTH_TOP_KEYS={"authority_design_contract","authority_materializer_source","authorization","check_key_set_sha256","check_keys","checks","checks_sha256","corrected_inner_wrapper_source","execution_boundary","execution_interpreter_evidence","f813_registration_tree","failed_v493_outer_execution_tree","failed_v493_outer_terminal_receipt","format","historical_absences","inner_attempt_root","input_post_snapshot","input_pre_snapshot","input_snapshots_exactly_equal","outer_evidence_root","outer_execution_wrapper_source","passed","phase_a_design_contract_source","postregistration_static_registration_tree","reconciler_r2_source","repair_formal_registration_tree","required_absences","runtime_observation","source_closure","source_closure_sha256","status","transparent_static_receipt_path","v493_authority_contract","v493_authority_materialization_evidence_tree","v493_authority_materializer_process_receipt","v493_authority_materializer_source","v493_authority_receipt","v493_authority_registration_tree","v493_inner_wrapper_source","v493_outer_wrapper_source","v494_authority_contract_source","v494_authority_materializer_source","v494_outer_wrapper_source","v494_inner_wrapper_source","v494_materializer_failure_forensic","v494_materializer_failure_transport_script","v494_materializer_failure_ancestry","v495_authority_contract_source","v495_authority_materializer_source","v495_inner_wrapper_source","v495_outer_wrapper_source","v495_materializer_failure_tree","v495_materializer_failure_process_receipt","v495_materializer_failure_ancestry","v493_authority_materializer_process_schema"}
SOURCE_ROLES={"v493_authority_contract","v493_authority_materializer","v493_outer_wrapper","v493_inner_wrapper","v494_authority_contract","v494_authority_materializer","v494_outer_wrapper","v494_inner_wrapper","corrected_inner_wrapper","outer_execution_wrapper","reconciler_r2","phase_a_design_contract","v494_failure_forensic","v494_failure_transport_script","v495_authority_contract","v495_authority_materializer","v495_inner_wrapper","v495_outer_wrapper"}
AUTH_CHECK_KEYS=sorted({"authority_contract_current","authority_materializer_current","corrected_inner_current","corrected_inner_diff_exact","current_absences","execution_boundary","execution_interpreter_chain_exact","execution_interpreter_runtime_exact","f813_exact2","failed_v493_outer_exact4","failed_v493_outer_no_retry_partition","failed_v493_outer_terminal_exact","gpu_empty","historical_absences","input_snapshots_equal","no_live_process","outer_wrapper_current","phase_a_contract_current","phase_a_contract_spec_exact2","postregistration_static_exact1","r2_current","repair_formal_exact1","source_closure_current","v493_authority_contract_current","v493_authority_exact3","v493_authority_materialization_evidence_exact6","v493_authority_materializer_process_exact","v493_outer_wrapper_current","v494_authority_contract_current","v494_authority_materializer_current","v494_inner_wrapper_current","v494_outer_wrapper_current","v494_failure_forensic_exact","v494_failure_transport_script_current","v494_materializer_failure_partition_exact","validator_torch_version_scope_exact","validator_torch_version_tamper_suite_passed","v495_authority_contract_current","v495_authority_materializer_current","v495_inner_wrapper_current","v495_outer_wrapper_current","v495_materializer_failure_exact6","v495_materializer_failure_process_exact","v493_process_schema_exact24","validator_process_schema_tamper_suite_passed"})
AUTH_KEYSET_SHA="f889bafe02388706482cf6dc649cc024d53cd771eae9c60757e52619a5212cfc";AUTH_CHECKS_SHA="85328543d1455c211715bb9af1eefadaaaf9be6609b701a0e3a2f810e9b6a69a"
CONTRACT_TOP_KEYS={"authority_materializer_source","authority_receipt_contract","current_absences_after_authority","execution_boundary","execution_interpreter_contract","f813_registration_tree","format","historical_absences","lineage","phase_a_design_contract_record","postregistration_static_registration_tree","repair_formal_registration_tree","seed","source_closure","source_closure_sha256","status","v493_authority_contract","v493_authority_materialization_evidence_tree","v493_authority_materializer_process_receipt","v493_authority_receipt","v493_authority_registration_tree","v493_outer_failure_ancestry","v493_outer_failure_tree","v493_outer_terminal_receipt","v494_authority_contract","v494_materializer_failure_forensic","v494_materializer_failure_transport_script","v494_materializer_failure_ancestry","v495_authority_contract","v495_materializer_failure_tree","v495_materializer_failure_process_receipt","v495_materializer_failure_ancestry","v493_authority_materializer_process_schema"}
HISTORICAL_ABSENCE_KEYS={"superseded_static_root","superseded_static_prep","v489_superseded_static_root","v489_superseded_static_prep","fresh_static_prep","v490_authority_prep","old_v490_inner_attempt_root","old_v490_inner_attempt_prep","transparent_output","transparent_tmp","qualification_root","superseded_v491_authority_root","superseded_v491_authority_prep","superseded_v491_outer_evidence_root","superseded_v491_outer_evidence_prep","v492_authority_prep","failed_v492_outer_evidence_prep","v493_authority_prep","failed_v493_outer_evidence_prep","failed_v493_inner_attempt_root","failed_v493_inner_attempt_prep","failed_v494_authority_root","failed_v494_authority_prep","failed_v494_outer_evidence_root","failed_v494_outer_evidence_prep","failed_v494_inner_attempt_root","failed_v494_inner_attempt_prep","execution_authority_root","execution_authority_prep","outer_evidence_root","outer_evidence_prep","failed_v495_authority_root","failed_v495_authority_prep","failed_v495_outer_evidence_root","failed_v495_outer_evidence_prep","failed_v495_inner_attempt_root","failed_v495_inner_attempt_prep","execution_authority_root","execution_authority_prep","outer_evidence_root","outer_evidence_prep","corrected_inner_attempt_root","corrected_inner_attempt_prep"}
CURRENT_ABSENCE_KEYS=HISTORICAL_ABSENCE_KEYS-{"execution_authority_root"}
FALSE_BOUNDARY={"phase_a_executed":False,"cache_executed":False,"training_launched":False,"folds":0,"policy_updates":0,"reward_read":False,"dev_hidden_final_outcome_read":False,"retry_authorized":False}
F813_EXACT2=[["immutable_evidence/v485_static_b73.log","7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b",6475],["preregistration.json",F813_SHA,F813_BYTES]]

INTERPRETER_INTERMEDIATE=Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python");INTERPRETER_RESOLVED=Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
EXECUTION_INTERPRETER_CONTRACT={"lexical":{"path":str(RLPY),"lstat":{"device":2304,"inode":17217118070,"mode":41471,"size":49},"readlink":str(INTERPRETER_INTERMEDIATE)},"intermediate":{"path":str(INTERPRETER_INTERMEDIATE),"lstat":{"device":2304,"inode":7529246267,"mode":41471,"size":10},"readlink":"python3.11"},"resolved":{"path":str(INTERPRETER_RESOLVED),"sha256":"11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788","logical_bytes":25555040,"lstat":{"device":2304,"inode":7529484239,"mode":33277,"size":25555040}},"runtime":{"sys_executable":str(RLPY),"python_version":"3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]","numpy_version":"1.26.4","torch_version":"2.7.0+cu128"}}
V494_MATERIALIZER_FAILURE_ANCESTRY={"failed_status":"frozen_reconstructed_v494_materializer_prestate_failure_no_retry","materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"native_exit_code":1,"native_persistent_capture_available":False,"tool_capture_sha256":"4a65e33de2de1da29e8a36d39e5374e43b66435aa5e6bc3722ccfca574fb4a4c","tool_capture_logical_bytes":787,"root_cause":"corrected inner forbidden import","corrected_rule":"allow exact one import torch and exact one load torch.__version__ only inside execution_interpreter_evidence","retry_authorized":False}
V493_PROCESS_EXACT_KEYS=sorted({"argv","authority_receipt","authority_tree","cleanup","corrected_inner_wrapper_invocations","failed_v492_outer_terminal_receipt","format","helper_returncode","intent","materializer_invocations","materializer_returncode","outer_wrapper_invocations","passed","post_snapshot","pre_post_snapshots_exactly_equal","pre_snapshot","r2_invocations","retry_authorized","status","stderr","stdout","transport_helper","transport_helper_copy","wall_seconds"})
V493_PROCESS_REQUIRED={"status":"passed_exact_once_no_outer_or_inner_execution","passed":True,"materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"r2_invocations":0,"retry_authorized":False,"pre_post_snapshots_exactly_equal":True,"cleanup":{"reaped":True,"group_empty":True}}
V493_PROCESS_SCHEMA={"record":{"path":str(V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json"),"sha256":"babcfda7fad625ea7d0b7a34592e119d6287fa4b34dcfbda743f7473a0fe4789","logical_bytes":19613},"exact_keys":V493_PROCESS_EXACT_KEYS,"key_set_sha256":"e477a875bc016fad9215afb62b61d8f1878d61aad2b734afa90eba2d3c897b71","required_values":V493_PROCESS_REQUIRED,"forbidden_keys":["inner_wrapper_invocations"]}
V495_MATERIALIZER_FAILURE_ANCESTRY={"failed_status":"failed_no_retry","materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"native_exit_code":1,"retry_authorized":False,"root_cause":"v495 materializer required stale inner_wrapper_invocations absent from immutable v493 process receipt","corrected_rule":"validate exact24 process receipt with corrected_inner_wrapper_invocations int zero and reject stale, missing, both, extra, nonzero, and bool variants"}

class ControlledSignal(BaseException):pass
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def csha(v)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def validate_v493_process_schema(value)->bool:
 if not isinstance(value,dict) or sorted(value)!=V493_PROCESS_EXACT_KEYS or csha(sorted(value))!=V493_PROCESS_SCHEMA["key_set_sha256"] or any(key in value for key in V493_PROCESS_SCHEMA["forbidden_keys"]):return False
 if type(value.get("corrected_inner_wrapper_invocations")) is not int or value["corrected_inner_wrapper_invocations"]!=0:return False
 for key,expected in V493_PROCESS_REQUIRED.items():
  if key=="cleanup":
   cleanup=value.get("cleanup")
   if not isinstance(cleanup,dict) or any(cleanup.get(name) is not required for name,required in expected.items()):return False
  elif value.get(key)!=expected:return False
 return True
def cbytes(v)->bytes:return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def regular(path:Path|str,want_sha=None,want_bytes=None):
 path=Path(path)
 if path!=path.resolve() or not path.is_file() or path.is_symlink():raise RuntimeError(f"regular: {path}")
 row={"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
 if want_sha is not None and row["sha256"]!=want_sha:raise RuntimeError(f"sha: {path}")
 if want_bytes is not None and row["logical_bytes"]!=want_bytes:raise RuntimeError(f"bytes: {path}")
 return row
def execution_interpreter_evidence():
 if sys.executable!=str(RLPY):raise RuntimeError("interpreter lexical executable")
 def lstat_row(path):
  fields=os.lstat(path);return {"device":fields.st_dev,"inode":fields.st_ino,"mode":fields.st_mode,"size":fields.st_size}
 lexical=lstat_row(RLPY);lexical_target=os.readlink(RLPY)
 if not stat.S_ISLNK(lexical["mode"]) or not os.path.isabs(lexical_target):raise RuntimeError("interpreter lexical symlink")
 intermediate=Path(lexical_target);intermediate_stat=lstat_row(intermediate);intermediate_target=os.readlink(intermediate)
 if not stat.S_ISLNK(intermediate_stat["mode"]) or os.path.isabs(intermediate_target):raise RuntimeError("interpreter intermediate symlink")
 resolved=(intermediate.parent/intermediate_target).resolve(strict=True);resolved_stat=lstat_row(resolved)
 if resolved!=INTERPRETER_RESOLVED or stat.S_ISLNK(resolved_stat["mode"]) or not stat.S_ISREG(resolved_stat["mode"]):raise RuntimeError("interpreter resolved regular")
 import numpy
 import torch
 evidence={"lexical":{"path":str(RLPY),"lstat":lexical,"readlink":lexical_target},"intermediate":{"path":str(intermediate),"lstat":intermediate_stat,"readlink":intermediate_target},"resolved":{**regular(resolved),"lstat":resolved_stat},"runtime":{"sys_executable":sys.executable,"python_version":sys.version,"numpy_version":numpy.__version__,"torch_version":torch.__version__}}
 if evidence!=EXECUTION_INTERPRETER_CONTRACT:raise RuntimeError("execution interpreter evidence")
 return evidence
def exact_tree(root:Path|str):
 root=Path(root)
 if root!=root.resolve() or not root.is_dir() or root.is_symlink():raise RuntimeError(f"tree: {root}")
 rows=[]
 for p in sorted(root.rglob("*")):
  if p.is_symlink():raise RuntimeError(f"symlink: {p}")
  if p.is_file():rows.append([p.relative_to(root).as_posix(),sha(p),p.stat().st_size])
  elif not p.is_dir():raise RuntimeError(f"nonregular: {p}")
 lines="".join(f"{h}  {n}\n" for n,h,_ in rows).encode();triples=json.dumps(rows,separators=(",",":")).encode()
 return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(r[2] for r in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(triples).hexdigest()}
def rooted_tree(root:Path|str):return {"root":str(Path(root)),**exact_tree(root)}
def fsync_dir(path:Path):
 fd=os.open(str(path),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_exclusive(path:Path,data:bytes):
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  view=memoryview(data)
  while view:
   n=os.write(fd,view)
   if n<=0:raise RuntimeError("short write")
   view=view[n:]
  os.fsync(fd)
 finally:os.close(fd)
def atomic_json(path:Path,value):
 tmp=path.with_name(path.name+".tmp")
 if os.path.lexists(path) or os.path.lexists(tmp):raise FileExistsError(path)
 write_exclusive(tmp,cbytes(value));os.replace(tmp,path);fsync_dir(path.parent)
def close_fsync(stream):
 if stream is not None and not stream.closed:stream.flush();os.fsync(stream.fileno());stream.close()
def directory_identity(path:Path):
 st=path.stat(follow_symlinks=False);return [st.st_dev,st.st_ino]
def group_empty(pid:int):
 try:os.killpg(pid,0);return False
 except ProcessLookupError:return True
def terminate_group(process,grace_seconds=10):
 out={"started":process is not None,"term_sent":False,"kill_sent":False,"reaped":process is None,"group_empty":True}
 if process is None:return out
 if process.poll() is None or not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGTERM);out["term_sent"]=True
  except ProcessLookupError:pass
  try:process.wait(timeout=grace_seconds)
  except subprocess.TimeoutExpired:pass
 if not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGKILL);out["kill_sent"]=True
  except ProcessLookupError:pass
 if process.poll() is None:process.wait(timeout=grace_seconds)
 else:process.wait()
 out["reaped"]=process.poll() is not None;out["group_empty"]=group_empty(process.pid);return out
def spawn_owned(command,stdout,stderr,owner,phase_hook=None):
 if owner!={"process":None,"started":False}:raise RuntimeError("spawn ownership prestate")
 phase_hook=phase_hook or (lambda _process:None);baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
 try:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
  owner["process"]=child;owner["started"]=True;phase_hook(child)
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)
 return owner["process"]
def live_processes():
 found=[]
 for p in Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(x in cmd for x in (str(INNER_WRAPPER),str(R2),str(SELF_PATH))):found.append({"pid":int(p.name),"cmdline":cmd})
 return found
def acquire_lock():
 fd=os.open(str(AUTH_RECEIPT),os.O_RDONLY)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BaseException:os.close(fd);raise
 return fd
def decode_path_map(value,count):
 if not isinstance(value,dict) or len(value)!=count:raise RuntimeError("absence schema")
 out={}
 for key,row in value.items():
  if not isinstance(key,str) or not isinstance(row,dict) or set(row)!={"path"} or not isinstance(row["path"],str):raise RuntimeError("absence row")
  p=Path(row["path"])
  if p!=p.resolve():raise RuntimeError("absence canonical")
  out[key]=p
 return out
def decode_absence_receipt(value,count):
 if not isinstance(value,dict) or len(value)!=count:raise RuntimeError("receipt absence schema")
 out={}
 for key,row in value.items():
  if not isinstance(key,str) or not isinstance(row,dict) or set(row)!={"path","absent"} or row["absent"] is not True or not isinstance(row["path"],str):raise RuntimeError("receipt absence row")
  path=Path(row["path"])
  if path!=path.resolve():raise RuntimeError("receipt absence canonical")
  out[key]=path
 return out

def authority_preflight(args):
 interpreter_evidence=execution_interpreter_evidence()
 if Path(__file__).resolve()!=SELF_PATH:raise RuntimeError("runtime/self")
 if args.authority_contract!=AUTH_CONTRACT or args.authority_contract.resolve()!=AUTH_CONTRACT or args.authority_receipt!=AUTH_RECEIPT or args.authority_receipt.resolve()!=AUTH_RECEIPT or args.wrapper_source!=SELF_PATH or args.wrapper_source.resolve()!=SELF_PATH:raise RuntimeError("canonical args")
 self_record=regular(SELF_PATH,args.wrapper_sha,args.wrapper_source.stat().st_size);contract_record=regular(AUTH_CONTRACT,args.authority_contract_sha);receipt_record=regular(AUTH_RECEIPT,args.authority_receipt_sha)
 contract=json.loads(AUTH_CONTRACT.read_text());receipt=json.loads(AUTH_RECEIPT.read_text())
 if set(contract)!=CONTRACT_TOP_KEYS or contract.get("format")!=CONTRACT_FORMAT or contract.get("status")!=CONTRACT_STATUS or contract.get("seed")!=1638 or contract.get("execution_boundary")!=EXECUTION_BOUNDARY:raise RuntimeError("contract identity")
 schema=contract.get("authority_receipt_contract")
 if not isinstance(schema,dict) or set(schema)!={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","authorization_exact","runtime_observation_exact"}:raise RuntimeError("receipt contract")
 if schema["format"]!=AUTH_FORMAT or schema["status"]!=AUTH_STATUS or schema["authorization_exact"]!=AUTHORIZATION or schema["runtime_observation_exact"]!=RUNTIME or set(schema["top_keys"])!=AUTH_TOP_KEYS or schema["check_keys"]!=AUTH_CHECK_KEYS or schema["check_key_set_sha256"]!=AUTH_KEYSET_SHA or schema["checks_sha256"]!=AUTH_CHECKS_SHA:raise RuntimeError("receipt contract constants")
 if set(receipt)!=AUTH_TOP_KEYS or receipt.get("format")!=AUTH_FORMAT or receipt.get("status")!=AUTH_STATUS or receipt.get("passed") is not True or receipt.get("check_keys")!=AUTH_CHECK_KEYS or receipt.get("check_key_set_sha256")!=AUTH_KEYSET_SHA or receipt.get("checks")!={k:True for k in AUTH_CHECK_KEYS} or receipt.get("checks_sha256")!=AUTH_CHECKS_SHA:raise RuntimeError("authority schema")
 if receipt.get("authorization")!=AUTHORIZATION or receipt.get("runtime_observation")!=RUNTIME or receipt.get("execution_boundary")!=EXECUTION_BOUNDARY or receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot")!=receipt.get("input_post_snapshot"):raise RuntimeError("authority boundary")
 if contract.get("execution_interpreter_contract")!=EXECUTION_INTERPRETER_CONTRACT or receipt.get("execution_interpreter_evidence")!=interpreter_evidence:raise RuntimeError("interpreter crossbind")
 if receipt.get("authority_design_contract")!=contract_record or receipt.get("outer_execution_wrapper_source")!=self_record:raise RuntimeError("authority aliases")
 materializer_spec=contract.get("authority_materializer_source");materializer_record=regular(materializer_spec["path"],materializer_spec["sha256"],materializer_spec["logical_bytes"])
 if receipt.get("authority_materializer_source")!=materializer_record:raise RuntimeError("authority materializer")
 closure=contract.get("source_closure")
 if not isinstance(closure,dict) or set(closure)!=SOURCE_ROLES or contract.get("source_closure_sha256")!=csha(closure) or receipt.get("source_closure")!=closure or receipt.get("source_closure_sha256")!=csha(closure) or {k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in closure.items()}!=closure:raise RuntimeError("source closure")
 aliases={"v493_authority_contract":"v493_authority_contract","v493_authority_materializer":"v493_authority_materializer_source","v493_outer_wrapper":"v493_outer_wrapper_source","v493_inner_wrapper":"v493_inner_wrapper_source","v494_authority_contract":"v494_authority_contract_source","v494_authority_materializer":"v494_authority_materializer_source","v494_outer_wrapper":"v494_outer_wrapper_source","v494_inner_wrapper":"v494_inner_wrapper_source","corrected_inner_wrapper":"corrected_inner_wrapper_source","outer_execution_wrapper":"outer_execution_wrapper_source","reconciler_r2":"reconciler_r2_source","phase_a_design_contract":"phase_a_design_contract_source","v494_failure_forensic":"v494_materializer_failure_forensic","v494_failure_transport_script":"v494_materializer_failure_transport_script","v495_authority_contract":"v495_authority_contract_source","v495_authority_materializer":"v495_authority_materializer_source","v495_inner_wrapper":"v495_inner_wrapper_source","v495_outer_wrapper":"v495_outer_wrapper_source"}
 if any(receipt.get(alias)!=closure[role] for role,alias in aliases.items()):raise RuntimeError("source aliases")
 expected={"v493_authority_contract":regular(V493_AUTH_CONTRACT,V493_AUTH_CONTRACT_SHA,V493_AUTH_CONTRACT_BYTES),"v493_authority_materializer":regular(V493_AUTH_MATERIALIZER,V493_AUTH_MATERIALIZER_SHA,V493_AUTH_MATERIALIZER_BYTES),"v493_outer_wrapper":regular(V493_OUTER_WRAPPER,V493_OUTER_WRAPPER_SHA,V493_OUTER_WRAPPER_BYTES),"v493_inner_wrapper":regular(V493_INNER_WRAPPER,V493_INNER_WRAPPER_SHA,V493_INNER_WRAPPER_BYTES),"v494_authority_contract":regular(V494_AUTH_CONTRACT,V494_AUTH_CONTRACT_SHA,V494_AUTH_CONTRACT_BYTES),"v494_authority_materializer":regular(V494_AUTH_MATERIALIZER,V494_AUTH_MATERIALIZER_SHA,V494_AUTH_MATERIALIZER_BYTES),"v494_outer_wrapper":regular(V494_OUTER_WRAPPER,V494_OUTER_WRAPPER_SHA,V494_OUTER_WRAPPER_BYTES),"v494_inner_wrapper":regular(V494_INNER_WRAPPER,V494_INNER_WRAPPER_SHA,V494_INNER_WRAPPER_BYTES),"corrected_inner_wrapper":regular(INNER_WRAPPER,INNER_WRAPPER_SHA,INNER_WRAPPER_BYTES),"outer_execution_wrapper":self_record,"reconciler_r2":regular(R2,R2_SHA,R2_BYTES),"phase_a_design_contract":regular(PHASE_CONTRACT,PHASE_CONTRACT_SHA,PHASE_CONTRACT_BYTES),"v494_failure_forensic":regular(V494_FAILURE_FORENSIC,V494_FAILURE_FORENSIC_SHA,V494_FAILURE_FORENSIC_BYTES),"v494_failure_transport_script":regular(V494_FAILURE_TRANSPORT,V494_FAILURE_TRANSPORT_SHA,V494_FAILURE_TRANSPORT_BYTES),"v495_authority_contract":regular(V495_AUTH_CONTRACT,V495_AUTH_CONTRACT_SHA,V495_AUTH_CONTRACT_BYTES),"v495_authority_materializer":regular(V495_AUTH_MATERIALIZER,V495_AUTH_MATERIALIZER_SHA,V495_AUTH_MATERIALIZER_BYTES),"v495_inner_wrapper":regular(V495_INNER_WRAPPER,V495_INNER_WRAPPER_SHA,V495_INNER_WRAPPER_BYTES),"v495_outer_wrapper":regular(V495_OUTER_WRAPPER,V495_OUTER_WRAPPER_SHA,V495_OUTER_WRAPPER_BYTES)}
 if closure!=expected or contract.get("phase_a_design_contract_record")!=expected["phase_a_design_contract"]:raise RuntimeError("source identities")
 v493_authority=regular(V493_AUTH_RECEIPT,V493_AUTH_RECEIPT_SHA,V493_AUTH_RECEIPT_BYTES);v493_tree=rooted_tree(V493_AUTH_RECEIPT.parent);v493_mat_tree=rooted_tree(V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT);v493_process=regular(V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json")
 if any(receipt.get(k)!=v for k,v in (("v493_authority_receipt",v493_authority),("v493_authority_registration_tree",v493_tree),("v493_authority_materialization_evidence_tree",v493_mat_tree),("v493_authority_materializer_process_receipt",v493_process))) or any(contract.get(k)!=v for k,v in (("v493_authority_receipt",v493_authority),("v493_authority_registration_tree",v493_tree),("v493_authority_materialization_evidence_tree",v493_mat_tree),("v493_authority_materializer_process_receipt",v493_process))):raise RuntimeError("v492 authority ancestry")
 process_value=json.loads(Path(v493_process["path"]).read_text())
 if contract.get("v493_authority_materializer_process_schema")!=V493_PROCESS_SCHEMA or receipt.get("v493_authority_materializer_process_schema")!=V493_PROCESS_SCHEMA or v493_process!=V493_PROCESS_SCHEMA["record"] or not validate_v493_process_schema(process_value):raise RuntimeError("v493 authority process schema")
 failed_tree=rooted_tree(V493_OUTER_FAILURE_ROOT);failed_terminal=regular(V493_OUTER_FAILURE_ROOT/"terminal_receipt.json");failed_value=json.loads((V493_OUTER_FAILURE_ROOT/"terminal_receipt.json").read_text())
 if receipt.get("failed_v493_outer_execution_tree")!=failed_tree or receipt.get("failed_v493_outer_terminal_receipt")!=failed_terminal or contract.get("v493_outer_failure_tree")!=failed_tree or contract.get("v493_outer_terminal_receipt")!=failed_terminal:raise RuntimeError("v492 outer failure ancestry")
 if failed_value.get("status")!="failed_no_retry" or failed_value.get("passed") is not False or failed_value.get("nested_corrected_inner_invocations")!=1 or failed_value.get("transparent_present") is not False or failed_value.get("cleanup",{}).get("reaped") is not True or failed_value.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v493 outer failure semantics")
 forensic=json.loads(V494_FAILURE_FORENSIC.read_text());forensic_record=expected["v494_failure_forensic"];transport_record=expected["v494_failure_transport_script"]
 if contract.get("v494_authority_contract")!=expected["v494_authority_contract"] or contract.get("v494_materializer_failure_forensic")!=forensic_record or contract.get("v494_materializer_failure_transport_script")!=transport_record or contract.get("v494_materializer_failure_ancestry")!=V494_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v494 failure contract ancestry")
 if receipt.get("v494_materializer_failure_forensic")!=forensic_record or receipt.get("v494_materializer_failure_transport_script")!=transport_record or receipt.get("v494_materializer_failure_ancestry")!=V494_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v494 failure receipt ancestry")
 invocation=forensic.get("invocation",{});tool_capture=forensic.get("tool_capture",{});state=forensic.get("state_after_failure",{});expected_forensic_sources={"authority_contract":expected["v494_authority_contract"],"authority_materializer":expected["v494_authority_materializer"],"outer_execution_wrapper":expected["v494_outer_wrapper"],"corrected_inner_wrapper":expected["v494_inner_wrapper"]}
 if forensic.get("format")!="strict-track2-v495-v494-authority-materializer-failure-forensic-reconstructed-v1" or forensic.get("status")!=V494_MATERIALIZER_FAILURE_ANCESTRY["failed_status"] or forensic.get("source_records")!=expected_forensic_sources or invocation.get("materializer_invocations")!=1 or any(invocation.get(key)!=0 for key in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or tool_capture.get("combined_output_sha256")!=V494_MATERIALIZER_FAILURE_ANCESTRY["tool_capture_sha256"] or tool_capture.get("combined_output_logical_bytes")!=787 or state.get("all_absent") is not True or any(os.path.lexists(path) for path in state.get("paths",{}).values()):raise RuntimeError("v494 failure forensic")
 v495_failure_tree=rooted_tree(V495_FAILURE_ROOT);v495_failure_process=regular(V495_FAILURE_PROCESS,V495_FAILURE_PROCESS_SHA,V495_FAILURE_PROCESS_BYTES);v495_failure_value=json.loads(V495_FAILURE_PROCESS.read_text())
 if v495_failure_tree!=V495_FAILURE_TREE or contract.get("v495_materializer_failure_tree")!=V495_FAILURE_TREE or receipt.get("v495_materializer_failure_tree")!=V495_FAILURE_TREE or contract.get("v495_materializer_failure_process_receipt")!=v495_failure_process or receipt.get("v495_materializer_failure_process_receipt")!=v495_failure_process:raise RuntimeError("v495 failure exact6 ancestry")
 if contract.get("v495_authority_contract")!=expected["v495_authority_contract"] or contract.get("v495_materializer_failure_ancestry")!=V495_MATERIALIZER_FAILURE_ANCESTRY or receipt.get("v495_materializer_failure_ancestry")!=V495_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v495 failure contract ancestry")
 if set(v495_failure_value)!={"cleanup","corrected_inner_wrapper_invocations","error","error_type","format","materializer_invocations","outer_wrapper_invocations","passed","r2_invocations","retry_authorized","status"} or v495_failure_value.get("status")!="failed_no_retry" or v495_failure_value.get("passed") is not False or v495_failure_value.get("materializer_invocations")!=1 or any(v495_failure_value.get(key)!=0 for key in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","r2_invocations")) or v495_failure_value.get("retry_authorized") is not False or v495_failure_value.get("cleanup",{}).get("reaped") is not True or v495_failure_value.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v495 failure process partition")
 repair_tree=rooted_tree(REPAIR.parent);static_tree=rooted_tree(STATIC.parent);f813_tree=rooted_tree(F813.parent);auth_tree=rooted_tree(AUTH_ROOT)
 if exact_tree(F813.parent)["inventory"]!=F813_EXACT2 or receipt.get("repair_formal_registration_tree")!=repair_tree or receipt.get("postregistration_static_registration_tree")!=static_tree or receipt.get("f813_registration_tree")!=f813_tree or contract.get("repair_formal_registration_tree")!=repair_tree or contract.get("postregistration_static_registration_tree")!=static_tree or contract.get("f813_registration_tree")!=f813_tree:raise RuntimeError("input trees")
 if auth_tree["file_count"]!=1 or auth_tree["inventory"][0][0]!="authority_receipt.json":raise RuntimeError("authority exact1")
 historical=decode_absence_receipt(receipt.get("historical_absences"),39);required=decode_absence_receipt(receipt.get("required_absences"),38);contract_historical=decode_path_map(contract.get("historical_absences"),39);contract_current=decode_path_map(contract.get("current_absences_after_authority"),38)
 if set(historical)!=HISTORICAL_ABSENCE_KEYS or set(required)!=CURRENT_ABSENCE_KEYS or historical!=contract_historical or required!=contract_current or any(os.path.lexists(path) for path in required.values()):raise RuntimeError("absence crossbind")
 for required_path in (AUTH_PREP,OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,TRANSPARENT,TRANSPARENT_TMP,QUALIFICATION):
  if required_path not in set(required.values()):raise RuntimeError(f"required absence mapping: {required_path}")
 if receipt.get("outer_evidence_root")!=str(OUTER_EVIDENCE) or receipt.get("inner_attempt_root")!=str(INNER_ATTEMPT) or receipt.get("transparent_static_receipt_path")!=str(TRANSPARENT):raise RuntimeError("execution paths")
 if contract.get("lineage")!={"execution_authority_root":str(AUTH_ROOT),"authority_prep":str(AUTH_PREP),"outer_evidence_root":str(OUTER_EVIDENCE),"outer_evidence_prep":str(OUTER_PREP),"inner_attempt_root":str(INNER_ATTEMPT),"inner_attempt_prep":str(INNER_PREP),"transparent_static_receipt_path":str(TRANSPARENT),"qualification_root":str(QUALIFICATION)}:raise RuntimeError("lineage")
 if live_processes():raise RuntimeError("live process")
 return {"contract":contract_record,"authority":receipt_record,"authority_tree":auth_tree,"self":self_record,"materializer":materializer_record,"v493_authority":v493_authority,"v493_authority_tree":v493_tree,"v493_materialization_tree":v493_mat_tree,"v493_process":v493_process,"failed_v493_terminal":failed_terminal,"failed_v493_tree":failed_tree,"v495_failure_process":v495_failure_process,"v495_failure_tree":v495_failure_tree,"inner_wrapper":expected["corrected_inner_wrapper"],"r2":expected["reconciler_r2"],"repair":regular(REPAIR,REPAIR_SHA,REPAIR_BYTES),"static":regular(STATIC,STATIC_SHA,STATIC_BYTES),"f813":regular(F813,F813_SHA,F813_BYTES),"phase_contract":expected["phase_a_design_contract"],"source_closure":closure,"execution_interpreter_evidence":interpreter_evidence}

def immutable_snapshot(context):
 files={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context.items() if isinstance(v,dict) and set(v)=={"path","sha256","logical_bytes"}}
 sources={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context["source_closure"].items()}
 trees={"authority":rooted_tree(AUTH_ROOT),"repair":rooted_tree(REPAIR.parent),"static":rooted_tree(STATIC.parent),"v493_authority":rooted_tree(V493_AUTH_RECEIPT.parent),"v493_authority_materialization":rooted_tree(V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT),"failed_v493_outer":rooted_tree(V493_OUTER_FAILURE_ROOT),"failed_v495_materializer":rooted_tree(V495_FAILURE_ROOT)}
 absences={"qualification":not os.path.lexists(QUALIFICATION)}
 if not all(absences.values()):raise RuntimeError("immutable absence")
 interpreter_evidence=execution_interpreter_evidence()
 if interpreter_evidence!=context["execution_interpreter_evidence"]:raise RuntimeError("interpreter snapshot drift")
 return {"files":files,"sources":sources,"trees":trees,"absences":absences,"execution_interpreter_evidence":interpreter_evidence}
def inner_command(context=None):
 contract_sha=context["contract"]["sha256"] if context is not None else "0"*64;receipt_sha=context["authority"]["sha256"] if context is not None else "0"*64
 return [str(RLPY),str(INNER_WRAPPER),"--f813-preregistration",str(F813),"--f813-preregistration-sha",F813_SHA,"--repair-preregistration",str(REPAIR),"--repair-preregistration-sha",REPAIR_SHA,"--authority-contract",str(AUTH_CONTRACT),"--authority-contract-sha",contract_sha,"--authority-receipt",str(AUTH_RECEIPT),"--authority-receipt-sha",receipt_sha,"--wrapper-source",str(INNER_WRAPPER),"--wrapper-sha",INNER_WRAPPER_SHA]
def commit_outer_intent(intent):
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None;identity=None
 try:
  if os.path.lexists(OUTER_EVIDENCE) or os.path.lexists(OUTER_PREP):raise RuntimeError("outer state exists")
  OUTER_PREP.mkdir();identity=directory_identity(OUTER_PREP);fsync_dir(OUTER_PREP.parent)
  write_exclusive(OUTER_PREP/"intent.json",cbytes(intent));write_exclusive(OUTER_PREP/"corrected_inner_stdout.log",b"");write_exclusive(OUTER_PREP/"corrected_inner_stderr.log",b"");fsync_dir(OUTER_PREP)
  if directory_identity(OUTER_PREP)!=identity or os.path.lexists(OUTER_EVIDENCE):raise RuntimeError("outer prep ownership")
  os.replace(OUTER_PREP,OUTER_EVIDENCE);fsync_dir(OUTER_EVIDENCE.parent)
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)
 return (OUTER_EVIDENCE/"corrected_inner_stdout.log").open("ab",buffering=0),(OUTER_EVIDENCE/"corrected_inner_stderr.log").open("ab",buffering=0)
def commit_terminal(value,immutable_before,state):
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
 try:
  if state["committed"] or exact_tree(OUTER_EVIDENCE)["file_count"]!=3:raise RuntimeError("preterminal exact3")
  if immutable_snapshot(state["context"])!=immutable_before:raise RuntimeError("preterminal drift")
  atomic_json(OUTER_EVIDENCE/"terminal_receipt.json",value);fsync_dir(OUTER_EVIDENCE)
  final=exact_tree(OUTER_EVIDENCE)
  if final["file_count"]!=4 or [r[0] for r in final["inventory"]]!=["corrected_inner_stderr.log","corrected_inner_stdout.log","intent.json","terminal_receipt.json"] or os.path.lexists(OUTER_EVIDENCE/"terminal_receipt.json.tmp"):raise RuntimeError("terminal exact4")
  if immutable_snapshot(state["context"])!=immutable_before:raise RuntimeError("postterminal drift")
  state["committed"]=True
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)

def execute(args):
 lock=acquire_lock();process_state={"process":None,"started":False};stdout=None;stderr=None;context=None;pre=None;intent=None;terminal_state={"committed":False,"context":None};old_handlers={};started=time.monotonic();cleanup={"started":False,"term_sent":False,"kill_sent":False,"reaped":True,"group_empty":True}
 def interrupted(signum,_frame):
  if not terminal_state["committed"]:raise ControlledSignal(signum)
 try:
  context=authority_preflight(args);terminal_state["context"]=context;pre=immutable_snapshot(context);command=inner_command(context)
  intent={"format":OUTER_INTENT_FORMAT,"status":"committed_before_corrected_inner_start","attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"command_argv":command,"command_argv_sha256":csha(command),"authority_receipt":context["authority"],"execution_interpreter_evidence":context["execution_interpreter_evidence"],"failed_v493_outer_execution_tree":context["failed_v493_tree"],"v494_materializer_failure_forensic":context["source_closure"]["v494_failure_forensic"],"v494_materializer_failure_transport_script":context["source_closure"]["v494_failure_transport_script"],"v494_materializer_failure_ancestry":V494_MATERIALIZER_FAILURE_ANCESTRY,"v495_materializer_failure_tree":context["v495_failure_tree"],"v495_materializer_failure_process_receipt":context["v495_failure_process"],"v495_materializer_failure_ancestry":V495_MATERIALIZER_FAILURE_ANCESTRY,"v493_authority_materializer_process_schema":V493_PROCESS_SCHEMA,"immutable_pre_snapshot":pre,"nested_corrected_inner_invocations_before":0,"nested_r2_invocations_before":0,"retry_authorized":False,**FALSE_BOUNDARY}
  for s in (signal.SIGINT,signal.SIGTERM):old_handlers[s]=signal.getsignal(s);signal.signal(s,interrupted)
  stdout,stderr=commit_outer_intent(intent)
  process=spawn_owned(command,stdout,stderr,process_state)
  try:rc=process.wait(timeout=600)
  except subprocess.TimeoutExpired as error:raise RuntimeError("corrected inner timeout") from error
  close_fsync(stdout);stdout=None;close_fsync(stderr);stderr=None;cleanup=terminate_group(process)
  if rc!=0 or not cleanup["reaped"] or not cleanup["group_empty"]:raise RuntimeError(f"corrected inner rc: {rc}")
  inner_tree=rooted_tree(INNER_ATTEMPT)
  if inner_tree["file_count"]!=4 or [r[0] for r in inner_tree["inventory"]]!=["intent.json","reconciler_stderr.log","reconciler_stdout.log","terminal_receipt.json"]:raise RuntimeError("inner exact4")
  inner_terminal=json.loads((INNER_ATTEMPT/"terminal_receipt.json").read_text())
  if inner_terminal.get("format")!="strict-track2-v496-v495-v490-interpreter-validator-process-schema-repaired-reconciliation-inner-attempt-terminal-v1" or inner_terminal.get("status")!="passed_exact_one_corrected_inner_and_r2_readonly_reconciliation" or inner_terminal.get("passed") is not True or inner_terminal.get("reconciler_exit_code")!=0 or inner_terminal.get("f813_exact2_to_sole_exact3") is not True or inner_terminal.get("failed_v493_outer_execution_tree")!=context["failed_v493_tree"] or inner_terminal.get("failed_v493_outer_terminal_receipt")!=context["failed_v493_terminal"] or inner_terminal.get("execution_interpreter_evidence")!=context["execution_interpreter_evidence"] or inner_terminal.get("v494_materializer_failure_forensic")!=context["source_closure"]["v494_failure_forensic"] or inner_terminal.get("v494_materializer_failure_transport_script")!=context["source_closure"]["v494_failure_transport_script"] or inner_terminal.get("v494_materializer_failure_ancestry")!=V494_MATERIALIZER_FAILURE_ANCESTRY or inner_terminal.get("v495_materializer_failure_tree")!=context["v495_failure_tree"] or inner_terminal.get("v495_materializer_failure_process_receipt")!=context["v495_failure_process"] or inner_terminal.get("v495_materializer_failure_ancestry")!=V495_MATERIALIZER_FAILURE_ANCESTRY or inner_terminal.get("v493_authority_materializer_process_schema")!=V493_PROCESS_SCHEMA:raise RuntimeError("inner terminal")
  f813_after=rooted_tree(F813.parent);expected_names=[r[0] for r in F813_EXACT2]+["transparent_static_audit.json"]
  if f813_after["file_count"]!=3 or [r[0] for r in f813_after["inventory"]]!=expected_names or f813_after["inventory"][:2]!=F813_EXACT2:raise RuntimeError("F813 exact2 to exact3")
  transparent=regular(TRANSPARENT);terminal_output=inner_terminal.get("transparent_static_receipt")
  if terminal_output!=transparent or os.path.lexists(TRANSPARENT_TMP):raise RuntimeError("transparent output")
  post=immutable_snapshot(context)
  if post!=pre or live_processes():raise RuntimeError("immutable post")
  terminal={"format":OUTER_TERMINAL_FORMAT,"status":"passed_exact_one_nested_corrected_inner_and_r2_readonly_reconciliation","passed":True,"attempt_nonce":intent["attempt_nonce"],"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json"),"authority_receipt":context["authority"],"execution_interpreter_evidence":context["execution_interpreter_evidence"],"nested_corrected_inner_source":context["inner_wrapper"],"nested_r2_source":context["r2"],"nested_corrected_inner_returncode":0,"nested_corrected_inner_invocations":1,"nested_r2_invocations":1,"cleanup":cleanup,"stdout":regular(OUTER_EVIDENCE/"corrected_inner_stdout.log"),"stderr":regular(OUTER_EVIDENCE/"corrected_inner_stderr.log"),"inner_attempt_tree":inner_tree,"inner_terminal_receipt":regular(INNER_ATTEMPT/"terminal_receipt.json"),"transparent_static_receipt":transparent,"f813_registration_tree_before":{"root":str(F813.parent),**exact_tree_before(F813_EXACT2)},"f813_registration_tree_after":f813_after,"f813_exact2_to_sole_exact3":True,"failed_v493_outer_execution_tree":context["failed_v493_tree"],"failed_v493_outer_terminal_receipt":context["failed_v493_terminal"],"v494_materializer_failure_forensic":context["source_closure"]["v494_failure_forensic"],"v494_materializer_failure_transport_script":context["source_closure"]["v494_failure_transport_script"],"v494_materializer_failure_ancestry":V494_MATERIALIZER_FAILURE_ANCESTRY,"v495_materializer_failure_tree":context["v495_failure_tree"],"v495_materializer_failure_process_receipt":context["v495_failure_process"],"v495_materializer_failure_ancestry":V495_MATERIALIZER_FAILURE_ANCESTRY,"v493_authority_materializer_process_schema":V493_PROCESS_SCHEMA,"immutable_pre_snapshot":pre,"immutable_post_snapshot":post,"immutable_inputs_exactly_equal":True,"retry_authorized":False,**FALSE_BOUNDARY}
  if not math.isfinite(terminal["wall_seconds"]) or terminal["wall_seconds"]<0:raise RuntimeError("wall")
  commit_terminal(terminal,pre,terminal_state);return 0
 except BaseException as error:
  for s in old_handlers:signal.signal(s,signal.SIG_IGN)
  try:close_fsync(stdout)
  except BaseException:pass
  try:close_fsync(stderr)
  except BaseException:pass
  try:cleanup=terminate_group(process_state["process"])
  except BaseException as ce:cleanup={"cleanup_error":f"{type(ce).__name__}: {ce}","reaped":False,"group_empty":False}
  if OUTER_EVIDENCE.is_dir() and not OUTER_EVIDENCE.is_symlink() and not terminal_state["committed"] and not (OUTER_EVIDENCE/"terminal_receipt.json").exists():
   failure={"format":OUTER_TERMINAL_FORMAT,"status":"failed_no_retry","passed":False,"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json") if (OUTER_EVIDENCE/"intent.json").is_file() else None,"error_type":type(error).__name__,"error":str(error),"execution_interpreter_evidence":context["execution_interpreter_evidence"] if context is not None else None,"cleanup":cleanup,"nested_corrected_inner_invocations":1 if process_state["started"] else 0,"nested_r2_invocations":"bounded_by_inner_terminal_or_zero","transparent_present":TRANSPARENT.is_file() and not TRANSPARENT.is_symlink(),"retry_authorized":False,**FALSE_BOUNDARY}
   if context is not None and pre is not None:commit_terminal(failure,pre,terminal_state)
  raise
 finally:
  for s,h in old_handlers.items():signal.signal(s,h)
  fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)
def exact_tree_before(rows):
 lines="".join(f"{h}  {n}\n" for n,h,_ in rows).encode();triples=json.dumps(rows,separators=(",",":")).encode()
 return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(r[2] for r in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(triples).hexdigest()}

def synthetic():
 command=inner_command()
 tree=ast.parse(Path(__file__).read_text());popen_count=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="subprocess" and n.func.attr=="Popen" for n in ast.walk(tree))
 checks={"command_unique_corrected_inner":command[0:2]==[str(RLPY),str(INNER_WRAPPER)] and command.count("--wrapper-source")==1 and str(R2) not in command,"single_popen":popen_count==1,"authority_schema_digests":csha(AUTH_CHECK_KEYS)==AUTH_KEYSET_SHA and csha({k:True for k in AUTH_CHECK_KEYS})==AUTH_CHECKS_SHA and len(AUTH_TOP_KEYS)==55 and len(AUTH_CHECK_KEYS)==45,"unsafe_false":all(AUTHORIZATION[k] is False for k in ("retry_authorized","direct_corrected_inner_authorized","direct_r2_authorized","phase_a_authorized","training_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized")),"nested_exact1":AUTHORIZATION["nested_corrected_inner_invocations_authorized"]==AUTHORIZATION["nested_r2_invocations_authorized"]==1,"roots_distinct":len({OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,AUTH_ROOT})==5,"phase_contract_current":regular(PHASE_CONTRACT,PHASE_CONTRACT_SHA,PHASE_CONTRACT_BYTES)["logical_bytes"]==PHASE_CONTRACT_BYTES,"failed_v493_exact4":rooted_tree(V493_OUTER_FAILURE_ROOT)["file_count"]==4 and rooted_tree(V493_OUTER_FAILURE_ROOT)["sha256sum_lines_digest_sha256"]=="1369b6ed7c053b1bf283ece7b3bb400bf9c0dc37b4322c1d8f4c08a96e0a77a0" and rooted_tree(V493_OUTER_FAILURE_ROOT)["canonical_json_triples_digest_sha256"]=="e09a1b83fca92c950c1e7b79d373bc3c63a65e6f9974011dc38806496ee64a9a","execution_interpreter_runtime":execution_interpreter_evidence()==EXECUTION_INTERPRETER_CONTRACT,"v494_failure_forensic_current":regular(V494_FAILURE_FORENSIC,V494_FAILURE_FORENSIC_SHA,V494_FAILURE_FORENSIC_BYTES)["logical_bytes"]==V494_FAILURE_FORENSIC_BYTES and regular(V494_FAILURE_TRANSPORT,V494_FAILURE_TRANSPORT_SHA,V494_FAILURE_TRANSPORT_BYTES)["logical_bytes"]==V494_FAILURE_TRANSPORT_BYTES,"v495_failure_exact6":rooted_tree(V495_FAILURE_ROOT)==V495_FAILURE_TREE}
 actual_process=json.loads((V493_AUTH_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json").read_text());checks["v493_process_schema_exact24"]=validate_v493_process_schema(actual_process)
 tampers=[]
 missing=dict(actual_process);missing.pop("corrected_inner_wrapper_invocations");tampers.append(missing)
 stale=dict(actual_process);stale.pop("corrected_inner_wrapper_invocations");stale["inner_wrapper_invocations"]=0;tampers.append(stale)
 both=dict(actual_process);both["inner_wrapper_invocations"]=0;tampers.append(both)
 extra=dict(actual_process);extra["unexpected"]=0;tampers.append(extra)
 nonzero=dict(actual_process);nonzero["corrected_inner_wrapper_invocations"]=1;tampers.append(nonzero)
 boolean=dict(actual_process);boolean["corrected_inner_wrapper_invocations"]=False;tampers.append(boolean)
 checks["validator_process_schema_tamper_suite_passed"]=all(not validate_v493_process_schema(tamper) for tamper in tampers)
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);p=root/"x";write_exclusive(p,b"x");checks["exclusive_fsync_write"]=p.read_bytes()==b"x"
 owner={"process":None,"started":False};previous=signal.getsignal(signal.SIGTERM);caught=False;fixture_cleanup={}
 def fixture_signal(signum,_frame):raise ControlledSignal(signum)
 def fixture_window(child):
  if child.stdout.readline()!=b"ready\n":raise RuntimeError("fixture child handshake")
  os.kill(os.getpid(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,fixture_signal)
 try:
  spawn_owned([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"],subprocess.PIPE,subprocess.DEVNULL,owner,fixture_window)
 except ControlledSignal:caught=True
 finally:
  signal.signal(signal.SIGTERM,signal.SIG_IGN)
  try:
   fixture_cleanup=terminate_group(owner["process"],0.1)
   if owner["process"] is not None and owner["process"].stdout is not None:owner["process"].stdout.close()
  finally:signal.signal(signal.SIGTERM,previous)
 checks["spawn_signal_window_owned_cleanup"]=(caught and owner["started"] and fixture_cleanup["term_sent"] and fixture_cleanup["kill_sent"] and fixture_cleanup["reaped"] and fixture_cleanup["group_empty"])
 fixture_path=Path("/dev/shm/v496_authority_contract_candidate.json")
 if fixture_path.is_file():
  fixture=json.loads(fixture_path.read_text());historical=decode_path_map(fixture["historical_absences"],39);current=decode_path_map(fixture["current_absences_after_authority"],38)
  receipt_h={k:{"path":str(v),"absent":True} for k,v in historical.items()};receipt_c={k:{"path":str(v),"absent":True} for k,v in current.items()}
  checks["live_contract_schema"]=(set(fixture)==CONTRACT_TOP_KEYS and fixture["format"]==CONTRACT_FORMAT and fixture["status"]==CONTRACT_STATUS and fixture["seed"]==1638 and set(fixture["authority_receipt_contract"]["top_keys"])==AUTH_TOP_KEYS and fixture["authority_receipt_contract"]["check_keys"]==AUTH_CHECK_KEYS and fixture["authority_receipt_contract"]["authorization_exact"]==AUTHORIZATION and fixture["authority_receipt_contract"]["runtime_observation_exact"]==RUNTIME and fixture["execution_boundary"]==EXECUTION_BOUNDARY and fixture["execution_interpreter_contract"]==EXECUTION_INTERPRETER_CONTRACT and set(fixture["source_closure"])==SOURCE_ROLES)
  checks["live_absence_schema_transform"]=(decode_absence_receipt(receipt_h,39)==historical and decode_absence_receipt(receipt_c,38)==current and current=={k:v for k,v in historical.items() if k!="execution_authority_root"} and historical["outer_evidence_prep"]==OUTER_PREP and historical["corrected_inner_attempt_root"]==INNER_ATTEMPT and historical["qualification_root"]==QUALIFICATION)
 print(json.dumps({"passed":all(checks.values()),"checks":checks,"checks_sha256":csha(checks)},sort_keys=True));return 0 if all(checks.values()) else 3
def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--authority-contract",type=Path,required=True);p.add_argument("--authority-contract-sha",required=True);p.add_argument("--authority-receipt",type=Path,required=True);p.add_argument("--authority-receipt-sha",required=True);p.add_argument("--wrapper-source",type=Path,required=True);p.add_argument("--wrapper-sha",required=True);return p.parse_args()
def main():
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic()
 return execute(parse_args())
if __name__=="__main__":raise SystemExit(main())
