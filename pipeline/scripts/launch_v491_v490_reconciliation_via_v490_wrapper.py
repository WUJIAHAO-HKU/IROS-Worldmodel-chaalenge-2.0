#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,fcntl,hashlib,json,math,os,signal,subprocess,sys,tempfile,time
from pathlib import Path

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge");J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810";RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
SELF_PATH=ROOT/"pipeline/scripts/launch_v491_v490_reconciliation_via_v490_wrapper.py"
AUTH_CONTRACT=ROOT/"pipeline/scripts/v491_v490_reconciliation_execution_authority_contract.json"
AUTH_ROOT=J/"v491_v490_reconciliation_execution_authority_seed1633_20260825";AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+".registration-prep");AUTH_RECEIPT=AUTH_ROOT/"authority_receipt.json"
FORENSIC=ROOT/"pipeline/scripts/v491_v490_prehelper_rc2_transport_forensic_reconstructed.json";FORENSIC_SHA="ce8346717d2e07c995d384b5f3cb6eab40bf837227b69cfa7e4b21a1d72bade5";FORENSIC_BYTES=7833
OLD_AUTH_CONTRACT=ROOT/"pipeline/scripts/v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority_contract.json";OLD_AUTH_CONTRACT_SHA="f37dc074bff5cffa25c13d0b4c16530f1f02e82362187fb7bb116ad4250e8058";OLD_AUTH_CONTRACT_BYTES=28754
OLD_AUTH_MATERIALIZER=ROOT/"pipeline/scripts/materialize_v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority.py";OLD_AUTH_MATERIALIZER_SHA="6618ec3266c46a3320f93caa447234109ab301082352b7da9e2e1aaf2af67218";OLD_AUTH_MATERIALIZER_BYTES=36797
OLD_AUTH_RECEIPT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_authority_seed1632_20260825/authority_receipt.json";OLD_AUTH_RECEIPT_SHA="66874ce32b9cf8a01d5e720cf3814267c2cb0f9a0e68147241c0269bc7f93190";OLD_AUTH_RECEIPT_BYTES=51803
INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v490_v489_v488_v487_c71_exact7_schema_repair.py";INNER_WRAPPER_SHA="733931fa162c7e2559fcbf45b7ce8b98960db61df958143c7df3f4c15f3bc877";INNER_WRAPPER_BYTES=67698
R2=ROOT/"pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py";R2_SHA="9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377";R2_BYTES=51845
F813=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json";F813_SHA="f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8";F813_BYTES=21296
REPAIR=J/"v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json";REPAIR_SHA="b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b";REPAIR_BYTES=36181
STATIC=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json";STATIC_SHA="441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb";STATIC_BYTES=49008
INNER_ATTEMPT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825";INNER_PREP=INNER_ATTEMPT.with_name(INNER_ATTEMPT.name+".attempt-prep")
TRANSPARENT=F813.parent/"transparent_static_audit.json";TRANSPARENT_TMP=TRANSPARENT.with_name(TRANSPARENT.name+".tmp")
OUTER_EVIDENCE=J/"v491_v490_reconciliation_outer_execution_evidence_seed1633_20260825";OUTER_PREP=OUTER_EVIDENCE.with_name(OUTER_EVIDENCE.name+".outer-prep")
QUALIFICATION=Path("/root/v485_v169_cache_qualification_seed1627_20260824")
AUTH_FORMAT="strict-track2-v491-v490-reconciliation-execution-authority-v1";AUTH_STATUS="authorized_exact_one_external_v490_reconciliation_wrapper_attempt"
CONTRACT_FORMAT="strict-track2-v491-v490-reconciliation-execution-authority-design-contract-v1";CONTRACT_STATUS="design_only_frozen_sources_pending_independent_review_no_authority"
OUTER_INTENT_FORMAT="strict-track2-v491-v490-reconciliation-outer-attempt-intent-v1";OUTER_TERMINAL_FORMAT="strict-track2-v491-v490-reconciliation-outer-attempt-terminal-v1"
AUTHORIZATION={"outer_execution_wrapper_authorized":True,"outer_attempts_authorized":1,"outer_attempts_consumed":0,"retry_authorized":False,"direct_v490_wrapper_authorized":False,"direct_r2_authorized":False,"nested_v490_wrapper_invocations_authorized":1,"nested_r2_invocations_authorized":1,"nested_v490_wrapper_only_via_outer":True,"nested_r2_only_via_v490_wrapper":True,"phase_a_authorized":False,"cache_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
RUNTIME={"execution_authority_materialized":True,"outer_execution_wrapper_executed":False,"v490_wrapper_executed":False,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
EXECUTION_BOUNDARY={"authority_materialization_only":True,"dev_hidden_final_outcome_reads":0,"outer_execution_wrapper_invocations":0,"phase_a_invocations":0,"reconciler_r2_invocations":0,"reward_reads":0,"training_invocations":0,"v490_wrapper_invocations":0}
AUTH_TOP_KEYS={"format","status","passed","authority_design_contract","authority_materializer_source","outer_execution_wrapper_source","rc2_forensic_source","rc2_forensic_copy","corrected_transport_script_source","corrected_transport_script_copy","v490_authority_contract","v490_authority_materializer_source","v490_wrapper_source","reconciler_r2_source","v490_authority_receipt","v490_authority_registration_tree","v490_helper_process_receipt","v490_helper_evidence_tree","repair_formal_registration_tree","postregistration_static_registration_tree","f813_registration_tree","source_closure","source_closure_sha256","outer_evidence_root","inner_attempt_root","transparent_static_receipt_path","historical_absences","required_absences","checks","check_keys","check_key_set_sha256","checks_sha256","input_pre_snapshot","input_post_snapshot","input_snapshots_exactly_equal","authorization","runtime_observation","execution_boundary"}
SOURCE_ROLES={"rc2_forensic_source","corrected_transport_script","v490_authority_contract","v490_authority_materializer","v490_wrapper","reconciler_r2","outer_execution_wrapper"}
HISTORICAL_ABSENCE_KEYS={"superseded_static_root","superseded_static_prep","v489_superseded_static_root","v489_superseded_static_prep","fresh_static_prep","v490_authority_prep","inner_attempt_root","inner_attempt_prep","transparent_output","transparent_tmp","qualification_root","execution_authority_root","execution_authority_prep","outer_evidence_root","outer_evidence_prep"}
CURRENT_ABSENCE_KEYS=HISTORICAL_ABSENCE_KEYS-{"execution_authority_root"}
FALSE_BOUNDARY={"phase_a_executed":False,"cache_executed":False,"training_launched":False,"folds":0,"policy_updates":0,"reward_read":False,"dev_hidden_final_outcome_read":False,"retry_authorized":False}
F813_EXACT2=[["immutable_evidence/v485_static_b73.log","7a3e5aa7b627ff92ec82cb7073cac70dea35e26774bda76ef302c547a3ab070b",6475],["preregistration.json",F813_SHA,F813_BYTES]]

class ControlledSignal(BaseException):pass
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def csha(v)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def cbytes(v)->bytes:return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def regular(path:Path|str,want_sha=None,want_bytes=None):
 path=Path(path)
 if path!=path.resolve() or not path.is_file() or path.is_symlink():raise RuntimeError(f"regular: {path}")
 row={"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
 if want_sha is not None and row["sha256"]!=want_sha:raise RuntimeError(f"sha: {path}")
 if want_bytes is not None and row["logical_bytes"]!=want_bytes:raise RuntimeError(f"bytes: {path}")
 return row
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

def validate_forensic():
 record=regular(FORENSIC,FORENSIC_SHA,FORENSIC_BYTES);value=json.loads(FORENSIC.read_text())
 expected_top={"format","status","occurred","passed","native_persistent_capture_available","evidence_class","failure_stage","native_exit_code","transport","tool_capture_disclosure","reconstructed_observations","state_at_event","source_records","current_crosscheck","unsafe_authorization"}
 if set(value)!=expected_top or value.get("format")!="strict-track2-v491-v490-prehelper-rc2-transport-forensic-reconstructed-v1" or value.get("status")!="frozen_reconstructed_prehelper_rc2_zero_state_no_attempt_consumed":raise RuntimeError("forensic schema")
 if value.get("occurred") is not True or value.get("passed") is not False or value.get("native_persistent_capture_available") is not False or value.get("native_exit_code")!=2:raise RuntimeError("forensic classification")
 disclosure=value.get("tool_capture_disclosure",{});observed=value.get("reconstructed_observations",{});state=value.get("state_at_event",{});unsafe=value.get("unsafe_authorization",{})
 if any(disclosure.get(k) is not None for k in ("native_argv_bytes","native_stdout_bytes","native_stderr_bytes","native_wall_seconds")) or disclosure.get("native_stream_partition_available") is not False or disclosure.get("binary_operator_expected_observation_count")!=4 or disclosure.get("four_test_exit_codes")!=[2,2,2,2]:raise RuntimeError("forensic disclosure")
 split=disclosure.get("wrong_split_target",{});regular(split.get("path"),split.get("sha256"),split.get("logical_bytes"))
 if any(observed.get(k)!=0 for k in ("write_command_count","fresh_helper_entry_count","authority_materializer_invocation_count","v490_wrapper_invocation_count","r2_invocation_count")) or observed.get("attempt_consumed") is not False or observed.get("retry_consumed") is not False:raise RuntimeError("forensic zero calls")
 if any(state.get(k) is not True for k in ("fresh_helper_target_absent","authority_root_absent","authority_prep_absent","authority_materialization_evidence_root_absent","inner_attempt_root_absent","inner_attempt_prep_absent","transparent_output_absent","qualification_root_absent")) or state.get("live_relevant_processes")!=[] or state.get("gpu_compute_processes")!=[]:raise RuntimeError("forensic historical zero state")
 if not isinstance(unsafe,dict) or not unsafe or any(v is not False for v in unsafe.values()):raise RuntimeError("forensic unsafe boundary")
 sources=value.get("source_records",{});expected_roles={"old_helper","fresh_helper","corrected_script","v490_authority_contract","v490_authority_materializer","v490_wrapper","r2"}
 if set(sources)!=expected_roles or {k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in sources.items()}!=sources:raise RuntimeError("forensic sources")
 current=value.get("current_crosscheck",{})
 if current.get("v490_authority_receipt")!=regular(OLD_AUTH_RECEIPT,OLD_AUTH_RECEIPT_SHA,OLD_AUTH_RECEIPT_BYTES) or current.get("v490_authority_tree")!=rooted_tree(OLD_AUTH_RECEIPT.parent):raise RuntimeError("forensic authority crosscheck")
 helper_receipt=current.get("authority_helper_process_receipt",{});helper_tree=current.get("authority_helper_evidence_tree",{});helper_root=Path(helper_tree.get("root","/invalid"))
 if regular(helper_receipt.get("path"),helper_receipt.get("sha256"),helper_receipt.get("logical_bytes"))!=helper_receipt or rooted_tree(helper_root)!=helper_tree:raise RuntimeError("forensic helper evidence")
 if current.get("inner_attempt_root_absent") is not True or current.get("inner_attempt_prep_absent") is not True or current.get("transparent_output_absent") is not True or current.get("relevant_processes")!=[] or current.get("gpu_compute_processes")!=[]:raise RuntimeError("forensic current crosscheck")
 return record

def authority_preflight(args):
 if Path(sys.executable)!=RLPY or Path(__file__).resolve()!=SELF_PATH:raise RuntimeError("runtime/self")
 if args.authority_contract!=AUTH_CONTRACT or args.authority_contract.resolve()!=AUTH_CONTRACT or args.authority_receipt!=AUTH_RECEIPT or args.authority_receipt.resolve()!=AUTH_RECEIPT or args.wrapper_source!=SELF_PATH or args.wrapper_source.resolve()!=SELF_PATH:raise RuntimeError("canonical args")
 self_record=regular(SELF_PATH,args.wrapper_sha,args.wrapper_source.stat().st_size);contract_record=regular(AUTH_CONTRACT,args.authority_contract_sha);receipt_record=regular(AUTH_RECEIPT,args.authority_receipt_sha)
 contract=json.loads(AUTH_CONTRACT.read_text());receipt=json.loads(AUTH_RECEIPT.read_text())
 if contract.get("format")!=CONTRACT_FORMAT or contract.get("status")!=CONTRACT_STATUS or contract.get("seed")!=1633:raise RuntimeError("contract identity")
 schema=contract.get("authority_receipt_contract")
 if not isinstance(schema,dict) or set(schema)!={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","authorization_exact","runtime_observation_exact"}:raise RuntimeError("receipt contract")
 if schema["format"]!=AUTH_FORMAT or schema["status"]!=AUTH_STATUS or schema["authorization_exact"]!=AUTHORIZATION or schema["runtime_observation_exact"]!=RUNTIME:raise RuntimeError("receipt constants")
 if set(schema["top_keys"])!=AUTH_TOP_KEYS or set(receipt)!=AUTH_TOP_KEYS or len(receipt)!=38 or receipt.get("format")!=AUTH_FORMAT or receipt.get("status")!=AUTH_STATUS or receipt.get("passed") is not True:raise RuntimeError("authority schema")
 if not isinstance(schema["check_keys"],list) or len(schema["check_keys"])!=27 or receipt.get("checks")!={k:True for k in schema["check_keys"]} or receipt.get("check_keys")!=schema["check_keys"] or receipt.get("check_key_set_sha256")!=csha(schema["check_keys"]) or receipt.get("checks_sha256")!=csha(receipt["checks"]):raise RuntimeError("authority checks")
 if receipt["check_key_set_sha256"]!=schema["check_key_set_sha256"] or receipt["checks_sha256"]!=schema["checks_sha256"] or receipt.get("authorization")!=AUTHORIZATION or receipt.get("runtime_observation")!=RUNTIME:raise RuntimeError("authority boundary")
 if receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot")!=receipt.get("input_post_snapshot"):raise RuntimeError("authority snapshots")
 if receipt.get("authority_design_contract")!=contract_record or receipt.get("outer_execution_wrapper_source")!=self_record:raise RuntimeError("authority aliases")
 materializer_spec=contract.get("authority_materializer_source")
 if not isinstance(materializer_spec,dict) or receipt.get("authority_materializer_source")!=regular(materializer_spec["path"],materializer_spec["sha256"],materializer_spec["logical_bytes"]):raise RuntimeError("authority materializer")
 forensic_record=validate_forensic()
 if receipt.get("rc2_forensic_source")!=forensic_record:raise RuntimeError("forensic source")
 copy=receipt.get("rc2_forensic_copy")
 if copy!=regular(AUTH_ROOT/"immutable_evidence/prehelper_rc2_transport_forensic_reconstructed.json",FORENSIC_SHA,FORENSIC_BYTES):raise RuntimeError("forensic copy")
 script_source=receipt.get("corrected_transport_script_source");script_copy=receipt.get("corrected_transport_script_copy")
 if not isinstance(script_source,dict) or regular(script_source["path"],script_source["sha256"],script_source["logical_bytes"])!=script_source:raise RuntimeError("script source")
 if script_copy!=regular(AUTH_ROOT/"immutable_evidence/v490_authority_materialize_corrected.sh",script_source["sha256"],script_source["logical_bytes"]):raise RuntimeError("script copy")
 if receipt.get("v490_authority_contract")!=regular(OLD_AUTH_CONTRACT,OLD_AUTH_CONTRACT_SHA,OLD_AUTH_CONTRACT_BYTES) or receipt.get("v490_authority_materializer_source")!=regular(OLD_AUTH_MATERIALIZER,OLD_AUTH_MATERIALIZER_SHA,OLD_AUTH_MATERIALIZER_BYTES) or receipt.get("v490_authority_receipt")!=regular(OLD_AUTH_RECEIPT,OLD_AUTH_RECEIPT_SHA,OLD_AUTH_RECEIPT_BYTES):raise RuntimeError("old authority")
 forensic_value=json.loads(FORENSIC.read_text());cross=forensic_value["current_crosscheck"]
 if receipt.get("v490_helper_process_receipt")!=cross["authority_helper_process_receipt"] or receipt.get("v490_helper_evidence_tree")!=cross["authority_helper_evidence_tree"]:raise RuntimeError("v490 helper ancestry")
 if receipt.get("v490_wrapper_source")!=regular(INNER_WRAPPER,INNER_WRAPPER_SHA,INNER_WRAPPER_BYTES) or receipt.get("reconciler_r2_source")!=regular(R2,R2_SHA,R2_BYTES):raise RuntimeError("nested sources")
 closure=contract.get("source_closure")
 if not isinstance(closure,dict) or set(closure)!=SOURCE_ROLES or contract.get("source_closure_sha256")!=csha(closure) or receipt.get("source_closure")!=closure or receipt.get("source_closure_sha256")!=csha(closure):raise RuntimeError("source closure")
 if {k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in closure.items()}!=closure:raise RuntimeError("source closure current")
 aliases={"rc2_forensic_source":"rc2_forensic_source","corrected_transport_script":"corrected_transport_script_source","v490_authority_contract":"v490_authority_contract","v490_authority_materializer":"v490_authority_materializer_source","v490_wrapper":"v490_wrapper_source","reconciler_r2":"reconciler_r2_source","outer_execution_wrapper":"outer_execution_wrapper_source"}
 if any(receipt.get(alias)!=closure[role] for role,alias in aliases.items()):raise RuntimeError("source aliases")
 auth_tree=rooted_tree(AUTH_ROOT)
 if auth_tree["file_count"]!=3 or [r[0] for r in auth_tree["inventory"]]!=["authority_receipt.json","immutable_evidence/prehelper_rc2_transport_forensic_reconstructed.json","immutable_evidence/v490_authority_materialize_corrected.sh"]:raise RuntimeError("authority exact3")
 if receipt.get("v490_authority_registration_tree")!=rooted_tree(OLD_AUTH_RECEIPT.parent):raise RuntimeError("old authority tree")
 if receipt.get("repair_formal_registration_tree")!=rooted_tree(REPAIR.parent) or receipt.get("postregistration_static_registration_tree")!=rooted_tree(STATIC.parent) or receipt.get("f813_registration_tree")!={"root":str(F813.parent),**exact_tree(F813.parent)}:raise RuntimeError("input trees")
 if exact_tree(F813.parent)["inventory"]!=F813_EXACT2:raise RuntimeError("F813 not exact2")
 historical=decode_absence_receipt(receipt.get("historical_absences"),15);required=decode_absence_receipt(receipt.get("required_absences"),14)
 if set(historical)!=HISTORICAL_ABSENCE_KEYS or set(required)!=CURRENT_ABSENCE_KEYS:raise RuntimeError("absence keysets")
 contract_historical=decode_path_map(contract.get("historical_absences"),15);contract_current=decode_path_map(contract.get("current_absences_after_authority"),14)
 if historical!=contract_historical or required!=contract_current:raise RuntimeError("absence crossbind")
 if any(os.path.lexists(path) for path in required.values()):raise RuntimeError("required absence present")
 for required_path in (AUTH_PREP,OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,TRANSPARENT,TRANSPARENT_TMP,QUALIFICATION):
  if required_path not in set(required.values()):raise RuntimeError(f"required absence mapping: {required_path}")
 if receipt.get("outer_evidence_root")!=str(OUTER_EVIDENCE) or receipt.get("inner_attempt_root")!=str(INNER_ATTEMPT) or receipt.get("transparent_static_receipt_path")!=str(TRANSPARENT):raise RuntimeError("execution paths")
 boundary=receipt.get("execution_boundary")
 if boundary!=EXECUTION_BOUNDARY or contract.get("execution_boundary")!=EXECUTION_BOUNDARY:raise RuntimeError("materialization execution boundary")
 if live_processes():raise RuntimeError("live process")
 return {"contract":contract_record,"authority":receipt_record,"authority_tree":auth_tree,"forensic":forensic_record,"self":self_record,"old_authority":regular(OLD_AUTH_RECEIPT,OLD_AUTH_RECEIPT_SHA,OLD_AUTH_RECEIPT_BYTES),"inner_wrapper":regular(INNER_WRAPPER,INNER_WRAPPER_SHA,INNER_WRAPPER_BYTES),"r2":regular(R2,R2_SHA,R2_BYTES),"repair":regular(REPAIR,REPAIR_SHA,REPAIR_BYTES),"static":regular(STATIC,STATIC_SHA,STATIC_BYTES),"f813":regular(F813,F813_SHA,F813_BYTES),"source_closure":closure}

def immutable_snapshot(context):
 files={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context.items() if isinstance(v,dict) and set(v)=={"path","sha256","logical_bytes"}}
 sources={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context["source_closure"].items()}
 trees={"authority":rooted_tree(AUTH_ROOT),"repair":rooted_tree(REPAIR.parent),"static":rooted_tree(STATIC.parent)}
 absences={"qualification":not os.path.lexists(QUALIFICATION)}
 if not all(absences.values()):raise RuntimeError("immutable absence")
 return {"files":files,"sources":sources,"trees":trees,"absences":absences}
def inner_command():return [str(RLPY),str(INNER_WRAPPER),"--f813-preregistration",str(F813),"--f813-preregistration-sha",F813_SHA,"--repair-preregistration",str(REPAIR),"--repair-preregistration-sha",REPAIR_SHA,"--authority-contract",str(OLD_AUTH_CONTRACT),"--authority-contract-sha",OLD_AUTH_CONTRACT_SHA,"--authority-receipt",str(OLD_AUTH_RECEIPT),"--authority-receipt-sha",OLD_AUTH_RECEIPT_SHA,"--wrapper-source",str(INNER_WRAPPER),"--wrapper-sha",INNER_WRAPPER_SHA]
def commit_outer_intent(intent):
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None;identity=None
 try:
  if os.path.lexists(OUTER_EVIDENCE) or os.path.lexists(OUTER_PREP):raise RuntimeError("outer state exists")
  OUTER_PREP.mkdir();identity=directory_identity(OUTER_PREP);fsync_dir(OUTER_PREP.parent)
  write_exclusive(OUTER_PREP/"intent.json",cbytes(intent));write_exclusive(OUTER_PREP/"v490_wrapper_stdout.log",b"");write_exclusive(OUTER_PREP/"v490_wrapper_stderr.log",b"");fsync_dir(OUTER_PREP)
  if directory_identity(OUTER_PREP)!=identity or os.path.lexists(OUTER_EVIDENCE):raise RuntimeError("outer prep ownership")
  os.replace(OUTER_PREP,OUTER_EVIDENCE);fsync_dir(OUTER_EVIDENCE.parent)
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)
 return (OUTER_EVIDENCE/"v490_wrapper_stdout.log").open("ab",buffering=0),(OUTER_EVIDENCE/"v490_wrapper_stderr.log").open("ab",buffering=0)
def commit_terminal(value,immutable_before,state):
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
 try:
  if state["committed"] or exact_tree(OUTER_EVIDENCE)["file_count"]!=3:raise RuntimeError("preterminal exact3")
  if immutable_snapshot(state["context"])!=immutable_before:raise RuntimeError("preterminal drift")
  atomic_json(OUTER_EVIDENCE/"terminal_receipt.json",value);fsync_dir(OUTER_EVIDENCE)
  final=exact_tree(OUTER_EVIDENCE)
  if final["file_count"]!=4 or [r[0] for r in final["inventory"]]!=["intent.json","terminal_receipt.json","v490_wrapper_stderr.log","v490_wrapper_stdout.log"] or os.path.lexists(OUTER_EVIDENCE/"terminal_receipt.json.tmp"):raise RuntimeError("terminal exact4")
  if immutable_snapshot(state["context"])!=immutable_before:raise RuntimeError("postterminal drift")
  state["committed"]=True
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)

def execute(args):
 lock=acquire_lock();process_state={"process":None,"started":False};stdout=None;stderr=None;context=None;pre=None;intent=None;terminal_state={"committed":False,"context":None};old_handlers={};started=time.monotonic();cleanup={"started":False,"term_sent":False,"kill_sent":False,"reaped":True,"group_empty":True}
 def interrupted(signum,_frame):
  if not terminal_state["committed"]:raise ControlledSignal(signum)
 try:
  context=authority_preflight(args);terminal_state["context"]=context;pre=immutable_snapshot(context);command=inner_command()
  intent={"format":OUTER_INTENT_FORMAT,"status":"committed_before_unique_nested_v490_wrapper","attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"command":command,"command_sha256":csha(command),"outer_authority":context["authority"],"rc2_forensic":context["forensic"],"immutable_pre_snapshot":pre,"nested_v490_wrapper_invocations_before":0,"nested_r2_invocations_before":0,"retry_authorized":False,**FALSE_BOUNDARY}
  for s in (signal.SIGINT,signal.SIGTERM):old_handlers[s]=signal.getsignal(s);signal.signal(s,interrupted)
  stdout,stderr=commit_outer_intent(intent)
  process=spawn_owned(command,stdout,stderr,process_state)
  try:rc=process.wait(timeout=600)
  except subprocess.TimeoutExpired as error:raise RuntimeError("nested v490 wrapper timeout") from error
  close_fsync(stdout);stdout=None;close_fsync(stderr);stderr=None;cleanup=terminate_group(process)
  if rc!=0 or not cleanup["reaped"] or not cleanup["group_empty"]:raise RuntimeError(f"nested wrapper rc: {rc}")
  inner_tree=rooted_tree(INNER_ATTEMPT)
  if inner_tree["file_count"]!=4 or [r[0] for r in inner_tree["inventory"]]!=["intent.json","reconciler_stderr.log","reconciler_stdout.log","terminal_receipt.json"]:raise RuntimeError("inner exact4")
  inner_terminal=json.loads((INNER_ATTEMPT/"terminal_receipt.json").read_text())
  if inner_terminal.get("format")!="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-attempt-terminal-v1" or inner_terminal.get("status")!="passed_readonly_reconciliation_source_repair_no_phase_a_execution_authority" or inner_terminal.get("passed") is not True or inner_terminal.get("reconciler_exit_code")!=0:raise RuntimeError("inner terminal")
  f813_after=rooted_tree(F813.parent);expected_names=[r[0] for r in F813_EXACT2]+["transparent_static_audit.json"]
  if f813_after["file_count"]!=3 or [r[0] for r in f813_after["inventory"]]!=expected_names or f813_after["inventory"][:2]!=F813_EXACT2:raise RuntimeError("F813 exact2 to exact3")
  transparent=regular(TRANSPARENT);terminal_output=inner_terminal.get("transparent_static_receipt")
  if terminal_output!=transparent or os.path.lexists(TRANSPARENT_TMP):raise RuntimeError("transparent output")
  post=immutable_snapshot(context)
  if post!=pre or live_processes():raise RuntimeError("immutable post")
  terminal={"format":OUTER_TERMINAL_FORMAT,"status":"passed_exact_one_nested_v490_wrapper_and_r2_readonly_reconciliation","passed":True,"attempt_nonce":intent["attempt_nonce"],"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json"),"outer_authority":context["authority"],"rc2_forensic":context["forensic"],"nested_v490_wrapper_source":context["inner_wrapper"],"nested_r2_source":context["r2"],"nested_v490_wrapper_returncode":0,"nested_v490_wrapper_invocations":1,"nested_r2_invocations":1,"cleanup":cleanup,"stdout":regular(OUTER_EVIDENCE/"v490_wrapper_stdout.log"),"stderr":regular(OUTER_EVIDENCE/"v490_wrapper_stderr.log"),"inner_attempt_tree":inner_tree,"inner_terminal_receipt":regular(INNER_ATTEMPT/"terminal_receipt.json"),"transparent_static_receipt":transparent,"f813_registration_tree_before":{"root":str(F813.parent),**exact_tree_before(F813_EXACT2)},"f813_registration_tree_after":f813_after,"f813_exact2_to_sole_exact3":True,"immutable_pre_snapshot":pre,"immutable_post_snapshot":post,"immutable_inputs_exactly_equal":True,"retry_authorized":False,**FALSE_BOUNDARY}
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
   failure={"format":OUTER_TERMINAL_FORMAT,"status":"failed_no_retry","passed":False,"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json") if (OUTER_EVIDENCE/"intent.json").is_file() else None,"error_type":type(error).__name__,"error":str(error),"cleanup":cleanup,"nested_v490_wrapper_invocations":1 if process_state["started"] else 0,"nested_r2_invocations":"bounded_by_inner_terminal_or_zero","transparent_present":TRANSPARENT.is_file() and not TRANSPARENT.is_symlink(),"retry_authorized":False,**FALSE_BOUNDARY}
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
 checks={"command_unique_inner":command[0:2]==[str(RLPY),str(INNER_WRAPPER)] and command.count("--wrapper-source")==1 and str(R2) not in command,"single_popen":popen_count==1,"forensic_frozen":FORENSIC_SHA=="ce8346717d2e07c995d384b5f3cb6eab40bf837227b69cfa7e4b21a1d72bade5" and FORENSIC_BYTES==7833,"unsafe_false":all(AUTHORIZATION[k] is False for k in ("retry_authorized","direct_v490_wrapper_authorized","direct_r2_authorized","phase_a_authorized","training_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized")),"nested_exact1":AUTHORIZATION["nested_v490_wrapper_invocations_authorized"]==AUTHORIZATION["nested_r2_invocations_authorized"]==1,"roots_distinct":len({OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,AUTH_ROOT})==5}
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
  fixture_cleanup=terminate_group(owner["process"],0.1)
  if owner["process"] is not None and owner["process"].stdout is not None:owner["process"].stdout.close()
  signal.signal(signal.SIGTERM,previous)
 checks["spawn_signal_window_owned_cleanup"]=(caught and owner["started"] and fixture_cleanup["term_sent"] and fixture_cleanup["kill_sent"] and fixture_cleanup["reaped"] and fixture_cleanup["group_empty"])
 fixture_path=Path("/dev/shm/v491_authority_contract_candidate.json")
 if fixture_path.is_file():
  fixture=json.loads(fixture_path.read_text());historical=decode_path_map(fixture["historical_absences"],15);current=decode_path_map(fixture["current_absences_after_authority"],14)
  receipt_h={k:{"path":str(v),"absent":True} for k,v in historical.items()};receipt_c={k:{"path":str(v),"absent":True} for k,v in current.items()}
  checks["live_contract_schema"]=(fixture["format"]==CONTRACT_FORMAT and fixture["status"]==CONTRACT_STATUS and fixture["seed"]==1633 and set(fixture["authority_receipt_contract"]["top_keys"])==AUTH_TOP_KEYS and len(fixture["authority_receipt_contract"]["check_keys"])==27 and fixture["authority_receipt_contract"]["authorization_exact"]==AUTHORIZATION and fixture["authority_receipt_contract"]["runtime_observation_exact"]==RUNTIME and fixture["execution_boundary"]==EXECUTION_BOUNDARY)
  checks["live_absence_schema_transform"]=(decode_absence_receipt(receipt_h,15)==historical and decode_absence_receipt(receipt_c,14)==current and current=={k:v for k,v in historical.items() if k!="execution_authority_root"} and historical["outer_evidence_prep"]==OUTER_PREP and historical["qualification_root"]==QUALIFICATION)
 print(json.dumps({"passed":all(checks.values()),"checks":checks,"checks_sha256":csha(checks)},sort_keys=True));return 0 if all(checks.values()) else 3
def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--authority-contract",type=Path,required=True);p.add_argument("--authority-contract-sha",required=True);p.add_argument("--authority-receipt",type=Path,required=True);p.add_argument("--authority-receipt-sha",required=True);p.add_argument("--wrapper-source",type=Path,required=True);p.add_argument("--wrapper-sha",required=True);return p.parse_args()
def main():
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic()
 return execute(parse_args())
if __name__=="__main__":raise SystemExit(main())
