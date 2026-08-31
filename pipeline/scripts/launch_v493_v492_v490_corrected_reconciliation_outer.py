#!/usr/bin/env python3
from __future__ import annotations
import argparse,ast,fcntl,hashlib,json,math,os,signal,subprocess,sys,tempfile,time
from pathlib import Path

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge");J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810";RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
SELF_PATH=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_outer.py"
AUTH_CONTRACT=ROOT/"pipeline/scripts/v493_v492_v490_corrected_reconciliation_execution_authority_contract.json"
AUTH_ROOT=J/"v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825";AUTH_PREP=AUTH_ROOT.with_name(AUTH_ROOT.name+".registration-prep");AUTH_RECEIPT=AUTH_ROOT/"authority_receipt.json"
V492_AUTH_CONTRACT=ROOT/"pipeline/scripts/v492_v491_v490_reconciliation_execution_authority_contract.json";V492_AUTH_CONTRACT_SHA="32adf6dcfed2e8e9d4e500eec4d55847fe304b8f96633763a85f90017f1fb07b";V492_AUTH_CONTRACT_BYTES=25772
V492_AUTH_MATERIALIZER=ROOT/"pipeline/scripts/materialize_v492_v491_v490_reconciliation_execution_authority.py";V492_AUTH_MATERIALIZER_SHA="dc9ebcc9d24f3b327cda3f0730fb5d8528f2d38b95d0f54c001a057174b0e532";V492_AUTH_MATERIALIZER_BYTES=38380
V492_AUTH_RECEIPT=J/"v492_v491_v490_reconciliation_execution_authority_seed1634_20260825/authority_receipt.json";V492_AUTH_RECEIPT_SHA="a0be9be3de912188cb7654b07fb2177bb9b79008c4558805147efb79c5e443b2";V492_AUTH_RECEIPT_BYTES=59593
V492_OUTER_WRAPPER=ROOT/"pipeline/scripts/launch_v492_v491_v490_reconciliation_via_v490_wrapper.py";V492_OUTER_WRAPPER_SHA="bd212226de02eb646ed6cfb41dac5f504a954de8b7f2862fef85e9ee4a05895b";V492_OUTER_WRAPPER_BYTES=43036
V492_OUTER_FAILURE_ROOT=J/"v492_v491_v490_reconciliation_outer_execution_evidence_seed1634_20260825"
V492_AUTH_MATERIALIZATION_EVIDENCE_ROOT=J/"v492_v491_v490_reconciliation_execution_authority_materialization_evidence_seed1634_20260825"
SUPERSEDED_V490_INNER=ROOT/"pipeline/scripts/launch_v490_v489_v488_v487_c71_exact7_schema_repair.py";SUPERSEDED_V490_INNER_SHA="733931fa162c7e2559fcbf45b7ce8b98960db61df958143c7df3f4c15f3bc877";SUPERSEDED_V490_INNER_BYTES=67698
INNER_WRAPPER=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_inner.py";INNER_WRAPPER_SHA="343577ec4050a63ecf86ba082b23b5c5b869f90de8abb9f9d10f22cc47510597";INNER_WRAPPER_BYTES=75190
R2=ROOT/"pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py";R2_SHA="9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377";R2_BYTES=51845
F813=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json";F813_SHA="f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8";F813_BYTES=21296
REPAIR=J/"v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json";REPAIR_SHA="b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b";REPAIR_BYTES=36181
STATIC=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json";STATIC_SHA="441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb";STATIC_BYTES=49008
PHASE_CONTRACT=ROOT/"pipeline/scripts/v485_v482_v169_cache_determinism_scope_repair_contract.json";PHASE_CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64";PHASE_CONTRACT_BYTES=43960
INNER_ATTEMPT=J/"v493_v492_v490_corrected_reconciliation_inner_attempt_seed1635_20260825";INNER_PREP=INNER_ATTEMPT.with_name(INNER_ATTEMPT.name+".attempt-prep")
TRANSPARENT=F813.parent/"transparent_static_audit.json";TRANSPARENT_TMP=TRANSPARENT.with_name(TRANSPARENT.name+".tmp")
OUTER_EVIDENCE=J/"v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825";OUTER_PREP=OUTER_EVIDENCE.with_name(OUTER_EVIDENCE.name+".outer-prep")
QUALIFICATION=Path("/root/v485_v169_cache_qualification_seed1627_20260824")
AUTH_FORMAT="strict-track2-v493-v492-v490-corrected-reconciliation-execution-authority-v1";AUTH_STATUS="authorized_exact_one_external_v493_corrected_reconciliation_outer_attempt"
CONTRACT_FORMAT="strict-track2-v493-v492-v490-corrected-reconciliation-execution-authority-design-contract-v1";CONTRACT_STATUS="design_only_frozen_sources_pending_independent_review_no_authority"
OUTER_INTENT_FORMAT="strict-track2-v493-v492-v490-corrected-reconciliation-outer-attempt-intent-v1";OUTER_TERMINAL_FORMAT="strict-track2-v493-v492-v490-corrected-reconciliation-outer-attempt-terminal-v1"
AUTHORIZATION={"outer_execution_wrapper_authorized":True,"outer_attempts_authorized":1,"outer_attempts_consumed":0,"retry_authorized":False,"direct_corrected_inner_authorized":False,"direct_r2_authorized":False,"nested_corrected_inner_invocations_authorized":1,"nested_r2_invocations_authorized":1,"nested_corrected_inner_only_via_outer":True,"nested_r2_only_via_corrected_inner":True,"phase_a_authorized":False,"cache_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
RUNTIME={"execution_authority_materialized":True,"outer_execution_wrapper_executed":False,"corrected_inner_wrapper_executed":False,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
EXECUTION_BOUNDARY={"authority_materialization_only":True,"outer_execution_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"phase_a_invocations":0,"training_invocations":0,"reward_reads":0,"dev_hidden_final_outcome_reads":0}
AUTH_TOP_KEYS={"authority_design_contract","authority_materializer_source","authorization","check_key_set_sha256","check_keys","checks","checks_sha256","corrected_inner_wrapper_source","execution_boundary","f813_registration_tree","failed_v492_outer_execution_tree","failed_v492_outer_terminal_receipt","format","historical_absences","inner_attempt_root","input_post_snapshot","input_pre_snapshot","input_snapshots_exactly_equal","outer_evidence_root","outer_execution_wrapper_source","passed","phase_a_design_contract_source","postregistration_static_registration_tree","reconciler_r2_source","repair_formal_registration_tree","required_absences","runtime_observation","source_closure","source_closure_sha256","status","superseded_v490_inner_wrapper_source","transparent_static_receipt_path","v492_authority_contract","v492_authority_materialization_evidence_tree","v492_authority_materializer_process_receipt","v492_authority_materializer_source","v492_authority_receipt","v492_authority_registration_tree","v492_outer_wrapper_source"}
SOURCE_ROLES={"v492_authority_contract","v492_authority_materializer","v492_outer_wrapper","superseded_v490_inner_wrapper","corrected_inner_wrapper","reconciler_r2","outer_execution_wrapper","phase_a_design_contract"}
AUTH_CHECK_KEYS=sorted({"authority_contract_current","authority_materializer_current","corrected_inner_current","corrected_inner_diff_exact","current_absences","execution_boundary","f813_exact2","failed_v492_outer_exact4","failed_v492_outer_no_retry_partition","failed_v492_outer_terminal_exact","gpu_empty","historical_absences","input_snapshots_equal","no_live_process","outer_wrapper_current","phase_a_contract_current","phase_a_contract_spec_exact2","postregistration_static_exact1","r2_current","repair_formal_exact1","source_closure_current","v492_authority_contract_current","v492_authority_exact3","v492_authority_materialization_evidence_exact6","v492_authority_materializer_process_exact","v492_outer_wrapper_current"})
AUTH_KEYSET_SHA="55b257b1d6837b0b6240361a7a233526b17dd006e8534314ab5689dba6f5ae6f";AUTH_CHECKS_SHA="cd1cda09f441f917823c33102aaaf02b0580d14d8bb2f356c3dd574afa92dae4"
CONTRACT_TOP_KEYS={"authority_materializer_source","authority_receipt_contract","current_absences_after_authority","execution_boundary","f813_registration_tree","format","historical_absences","lineage","phase_a_design_contract_record","postregistration_static_registration_tree","repair_formal_registration_tree","seed","source_closure","source_closure_sha256","status","v492_authority_contract","v492_authority_materialization_evidence_tree","v492_authority_materializer_process_receipt","v492_authority_receipt","v492_authority_registration_tree","v492_outer_failure_ancestry","v492_outer_failure_tree","v492_outer_terminal_receipt"}
HISTORICAL_ABSENCE_KEYS={"superseded_static_root","superseded_static_prep","v489_superseded_static_root","v489_superseded_static_prep","fresh_static_prep","v490_authority_prep","old_v490_inner_attempt_root","old_v490_inner_attempt_prep","transparent_output","transparent_tmp","qualification_root","superseded_v491_authority_root","superseded_v491_authority_prep","superseded_v491_outer_evidence_root","superseded_v491_outer_evidence_prep","v492_authority_prep","failed_v492_outer_evidence_prep","execution_authority_root","execution_authority_prep","outer_evidence_root","outer_evidence_prep","corrected_inner_attempt_root","corrected_inner_attempt_prep"}
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

def authority_preflight(args):
 if Path(sys.executable)!=RLPY or Path(__file__).resolve()!=SELF_PATH:raise RuntimeError("runtime/self")
 if args.authority_contract!=AUTH_CONTRACT or args.authority_contract.resolve()!=AUTH_CONTRACT or args.authority_receipt!=AUTH_RECEIPT or args.authority_receipt.resolve()!=AUTH_RECEIPT or args.wrapper_source!=SELF_PATH or args.wrapper_source.resolve()!=SELF_PATH:raise RuntimeError("canonical args")
 self_record=regular(SELF_PATH,args.wrapper_sha,args.wrapper_source.stat().st_size);contract_record=regular(AUTH_CONTRACT,args.authority_contract_sha);receipt_record=regular(AUTH_RECEIPT,args.authority_receipt_sha)
 contract=json.loads(AUTH_CONTRACT.read_text());receipt=json.loads(AUTH_RECEIPT.read_text())
 if set(contract)!=CONTRACT_TOP_KEYS or contract.get("format")!=CONTRACT_FORMAT or contract.get("status")!=CONTRACT_STATUS or contract.get("seed")!=1635 or contract.get("execution_boundary")!=EXECUTION_BOUNDARY:raise RuntimeError("contract identity")
 schema=contract.get("authority_receipt_contract")
 if not isinstance(schema,dict) or set(schema)!={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","authorization_exact","runtime_observation_exact"}:raise RuntimeError("receipt contract")
 if schema["format"]!=AUTH_FORMAT or schema["status"]!=AUTH_STATUS or schema["authorization_exact"]!=AUTHORIZATION or schema["runtime_observation_exact"]!=RUNTIME or set(schema["top_keys"])!=AUTH_TOP_KEYS or schema["check_keys"]!=AUTH_CHECK_KEYS or schema["check_key_set_sha256"]!=AUTH_KEYSET_SHA or schema["checks_sha256"]!=AUTH_CHECKS_SHA:raise RuntimeError("receipt contract constants")
 if set(receipt)!=AUTH_TOP_KEYS or receipt.get("format")!=AUTH_FORMAT or receipt.get("status")!=AUTH_STATUS or receipt.get("passed") is not True or receipt.get("check_keys")!=AUTH_CHECK_KEYS or receipt.get("check_key_set_sha256")!=AUTH_KEYSET_SHA or receipt.get("checks")!={k:True for k in AUTH_CHECK_KEYS} or receipt.get("checks_sha256")!=AUTH_CHECKS_SHA:raise RuntimeError("authority schema")
 if receipt.get("authorization")!=AUTHORIZATION or receipt.get("runtime_observation")!=RUNTIME or receipt.get("execution_boundary")!=EXECUTION_BOUNDARY or receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot")!=receipt.get("input_post_snapshot"):raise RuntimeError("authority boundary")
 if receipt.get("authority_design_contract")!=contract_record or receipt.get("outer_execution_wrapper_source")!=self_record:raise RuntimeError("authority aliases")
 materializer_spec=contract.get("authority_materializer_source");materializer_record=regular(materializer_spec["path"],materializer_spec["sha256"],materializer_spec["logical_bytes"])
 if receipt.get("authority_materializer_source")!=materializer_record:raise RuntimeError("authority materializer")
 closure=contract.get("source_closure")
 if not isinstance(closure,dict) or set(closure)!=SOURCE_ROLES or contract.get("source_closure_sha256")!=csha(closure) or receipt.get("source_closure")!=closure or receipt.get("source_closure_sha256")!=csha(closure) or {k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in closure.items()}!=closure:raise RuntimeError("source closure")
 aliases={"v492_authority_contract":"v492_authority_contract","v492_authority_materializer":"v492_authority_materializer_source","v492_outer_wrapper":"v492_outer_wrapper_source","superseded_v490_inner_wrapper":"superseded_v490_inner_wrapper_source","corrected_inner_wrapper":"corrected_inner_wrapper_source","reconciler_r2":"reconciler_r2_source","outer_execution_wrapper":"outer_execution_wrapper_source","phase_a_design_contract":"phase_a_design_contract_source"}
 if any(receipt.get(alias)!=closure[role] for role,alias in aliases.items()):raise RuntimeError("source aliases")
 expected={"v492_authority_contract":regular(V492_AUTH_CONTRACT,V492_AUTH_CONTRACT_SHA,V492_AUTH_CONTRACT_BYTES),"v492_authority_materializer":regular(V492_AUTH_MATERIALIZER,V492_AUTH_MATERIALIZER_SHA,V492_AUTH_MATERIALIZER_BYTES),"v492_outer_wrapper":regular(V492_OUTER_WRAPPER,V492_OUTER_WRAPPER_SHA,V492_OUTER_WRAPPER_BYTES),"superseded_v490_inner_wrapper":regular(SUPERSEDED_V490_INNER,SUPERSEDED_V490_INNER_SHA,SUPERSEDED_V490_INNER_BYTES),"corrected_inner_wrapper":regular(INNER_WRAPPER,INNER_WRAPPER_SHA,INNER_WRAPPER_BYTES),"reconciler_r2":regular(R2,R2_SHA,R2_BYTES),"outer_execution_wrapper":self_record,"phase_a_design_contract":regular(PHASE_CONTRACT,PHASE_CONTRACT_SHA,PHASE_CONTRACT_BYTES)}
 if closure!=expected or contract.get("phase_a_design_contract_record")!=expected["phase_a_design_contract"]:raise RuntimeError("source identities")
 v492_authority=regular(V492_AUTH_RECEIPT,V492_AUTH_RECEIPT_SHA,V492_AUTH_RECEIPT_BYTES);v492_tree=rooted_tree(V492_AUTH_RECEIPT.parent);v492_mat_tree=rooted_tree(V492_AUTH_MATERIALIZATION_EVIDENCE_ROOT);v492_process=regular(V492_AUTH_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json")
 if any(receipt.get(k)!=v for k,v in (("v492_authority_receipt",v492_authority),("v492_authority_registration_tree",v492_tree),("v492_authority_materialization_evidence_tree",v492_mat_tree),("v492_authority_materializer_process_receipt",v492_process))) or any(contract.get(k)!=v for k,v in (("v492_authority_receipt",v492_authority),("v492_authority_registration_tree",v492_tree),("v492_authority_materialization_evidence_tree",v492_mat_tree),("v492_authority_materializer_process_receipt",v492_process))):raise RuntimeError("v492 authority ancestry")
 process_value=json.loads(Path(v492_process["path"]).read_text())
 if process_value.get("status")!="passed_exact_once_no_outer_or_inner_execution" or process_value.get("materializer_invocations")!=1 or any(process_value.get(k)!=0 for k in ("outer_wrapper_invocations","inner_wrapper_invocations","r2_invocations")):raise RuntimeError("v492 authority process")
 failed_tree=rooted_tree(V492_OUTER_FAILURE_ROOT);failed_terminal=regular(V492_OUTER_FAILURE_ROOT/"terminal_receipt.json");failed_value=json.loads((V492_OUTER_FAILURE_ROOT/"terminal_receipt.json").read_text())
 if receipt.get("failed_v492_outer_execution_tree")!=failed_tree or receipt.get("failed_v492_outer_terminal_receipt")!=failed_terminal or contract.get("v492_outer_failure_tree")!=failed_tree or contract.get("v492_outer_terminal_receipt")!=failed_terminal:raise RuntimeError("v492 outer failure ancestry")
 if failed_value.get("status")!="failed_no_retry" or failed_value.get("passed") is not False or failed_value.get("nested_v490_wrapper_invocations")!=1 or failed_value.get("transparent_present") is not False or failed_value.get("cleanup",{}).get("reaped") is not True or failed_value.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v492 outer failure semantics")
 repair_tree=rooted_tree(REPAIR.parent);static_tree=rooted_tree(STATIC.parent);f813_tree=rooted_tree(F813.parent);auth_tree=rooted_tree(AUTH_ROOT)
 if exact_tree(F813.parent)["inventory"]!=F813_EXACT2 or receipt.get("repair_formal_registration_tree")!=repair_tree or receipt.get("postregistration_static_registration_tree")!=static_tree or receipt.get("f813_registration_tree")!=f813_tree or contract.get("repair_formal_registration_tree")!=repair_tree or contract.get("postregistration_static_registration_tree")!=static_tree or contract.get("f813_registration_tree")!=f813_tree:raise RuntimeError("input trees")
 if auth_tree["file_count"]!=1 or auth_tree["inventory"][0][0]!="authority_receipt.json":raise RuntimeError("authority exact1")
 historical=decode_absence_receipt(receipt.get("historical_absences"),23);required=decode_absence_receipt(receipt.get("required_absences"),22);contract_historical=decode_path_map(contract.get("historical_absences"),23);contract_current=decode_path_map(contract.get("current_absences_after_authority"),22)
 if set(historical)!=HISTORICAL_ABSENCE_KEYS or set(required)!=CURRENT_ABSENCE_KEYS or historical!=contract_historical or required!=contract_current or any(os.path.lexists(path) for path in required.values()):raise RuntimeError("absence crossbind")
 for required_path in (AUTH_PREP,OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,TRANSPARENT,TRANSPARENT_TMP,QUALIFICATION):
  if required_path not in set(required.values()):raise RuntimeError(f"required absence mapping: {required_path}")
 if receipt.get("outer_evidence_root")!=str(OUTER_EVIDENCE) or receipt.get("inner_attempt_root")!=str(INNER_ATTEMPT) or receipt.get("transparent_static_receipt_path")!=str(TRANSPARENT):raise RuntimeError("execution paths")
 if contract.get("lineage")!={"execution_authority_root":str(AUTH_ROOT),"authority_prep":str(AUTH_PREP),"outer_evidence_root":str(OUTER_EVIDENCE),"outer_evidence_prep":str(OUTER_PREP),"inner_attempt_root":str(INNER_ATTEMPT),"inner_attempt_prep":str(INNER_PREP),"transparent_static_receipt_path":str(TRANSPARENT),"qualification_root":str(QUALIFICATION)}:raise RuntimeError("lineage")
 if live_processes():raise RuntimeError("live process")
 return {"contract":contract_record,"authority":receipt_record,"authority_tree":auth_tree,"self":self_record,"materializer":materializer_record,"v492_authority":v492_authority,"v492_authority_tree":v492_tree,"v492_materialization_tree":v492_mat_tree,"v492_process":v492_process,"failed_v492_terminal":failed_terminal,"failed_v492_tree":failed_tree,"inner_wrapper":expected["corrected_inner_wrapper"],"r2":expected["reconciler_r2"],"repair":regular(REPAIR,REPAIR_SHA,REPAIR_BYTES),"static":regular(STATIC,STATIC_SHA,STATIC_BYTES),"f813":regular(F813,F813_SHA,F813_BYTES),"phase_contract":expected["phase_a_design_contract"],"source_closure":closure}

def immutable_snapshot(context):
 files={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context.items() if isinstance(v,dict) and set(v)=={"path","sha256","logical_bytes"}}
 sources={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in context["source_closure"].items()}
 trees={"authority":rooted_tree(AUTH_ROOT),"repair":rooted_tree(REPAIR.parent),"static":rooted_tree(STATIC.parent),"v492_authority":rooted_tree(V492_AUTH_RECEIPT.parent),"v492_authority_materialization":rooted_tree(V492_AUTH_MATERIALIZATION_EVIDENCE_ROOT),"failed_v492_outer":rooted_tree(V492_OUTER_FAILURE_ROOT)}
 absences={"qualification":not os.path.lexists(QUALIFICATION)}
 if not all(absences.values()):raise RuntimeError("immutable absence")
 return {"files":files,"sources":sources,"trees":trees,"absences":absences}
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
  intent={"format":OUTER_INTENT_FORMAT,"status":"committed_before_corrected_inner_start","attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"command_argv":command,"command_argv_sha256":csha(command),"authority_receipt":context["authority"],"failed_v492_outer_execution_tree":context["failed_v492_tree"],"immutable_pre_snapshot":pre,"nested_corrected_inner_invocations_before":0,"nested_r2_invocations_before":0,"retry_authorized":False,**FALSE_BOUNDARY}
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
  if inner_terminal.get("format")!="strict-track2-v493-v492-v490-corrected-reconciliation-inner-attempt-terminal-v1" or inner_terminal.get("status")!="passed_exact_one_corrected_inner_and_r2_readonly_reconciliation" or inner_terminal.get("passed") is not True or inner_terminal.get("reconciler_exit_code")!=0 or inner_terminal.get("f813_exact2_to_sole_exact3") is not True or inner_terminal.get("failed_v492_outer_execution_tree")!=context["failed_v492_tree"] or inner_terminal.get("failed_v492_outer_terminal_receipt")!=context["failed_v492_terminal"]:raise RuntimeError("inner terminal")
  f813_after=rooted_tree(F813.parent);expected_names=[r[0] for r in F813_EXACT2]+["transparent_static_audit.json"]
  if f813_after["file_count"]!=3 or [r[0] for r in f813_after["inventory"]]!=expected_names or f813_after["inventory"][:2]!=F813_EXACT2:raise RuntimeError("F813 exact2 to exact3")
  transparent=regular(TRANSPARENT);terminal_output=inner_terminal.get("transparent_static_receipt")
  if terminal_output!=transparent or os.path.lexists(TRANSPARENT_TMP):raise RuntimeError("transparent output")
  post=immutable_snapshot(context)
  if post!=pre or live_processes():raise RuntimeError("immutable post")
  terminal={"format":OUTER_TERMINAL_FORMAT,"status":"passed_exact_one_nested_corrected_inner_and_r2_readonly_reconciliation","passed":True,"attempt_nonce":intent["attempt_nonce"],"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json"),"authority_receipt":context["authority"],"nested_corrected_inner_source":context["inner_wrapper"],"nested_r2_source":context["r2"],"nested_corrected_inner_returncode":0,"nested_corrected_inner_invocations":1,"nested_r2_invocations":1,"cleanup":cleanup,"stdout":regular(OUTER_EVIDENCE/"corrected_inner_stdout.log"),"stderr":regular(OUTER_EVIDENCE/"corrected_inner_stderr.log"),"inner_attempt_tree":inner_tree,"inner_terminal_receipt":regular(INNER_ATTEMPT/"terminal_receipt.json"),"transparent_static_receipt":transparent,"f813_registration_tree_before":{"root":str(F813.parent),**exact_tree_before(F813_EXACT2)},"f813_registration_tree_after":f813_after,"f813_exact2_to_sole_exact3":True,"failed_v492_outer_execution_tree":context["failed_v492_tree"],"failed_v492_outer_terminal_receipt":context["failed_v492_terminal"],"immutable_pre_snapshot":pre,"immutable_post_snapshot":post,"immutable_inputs_exactly_equal":True,"retry_authorized":False,**FALSE_BOUNDARY}
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
   failure={"format":OUTER_TERMINAL_FORMAT,"status":"failed_no_retry","passed":False,"wall_seconds":time.monotonic()-started,"intent":regular(OUTER_EVIDENCE/"intent.json") if (OUTER_EVIDENCE/"intent.json").is_file() else None,"error_type":type(error).__name__,"error":str(error),"cleanup":cleanup,"nested_corrected_inner_invocations":1 if process_state["started"] else 0,"nested_r2_invocations":"bounded_by_inner_terminal_or_zero","transparent_present":TRANSPARENT.is_file() and not TRANSPARENT.is_symlink(),"retry_authorized":False,**FALSE_BOUNDARY}
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
 checks={"command_unique_corrected_inner":command[0:2]==[str(RLPY),str(INNER_WRAPPER)] and command.count("--wrapper-source")==1 and str(R2) not in command,"single_popen":popen_count==1,"authority_schema_digests":csha(AUTH_CHECK_KEYS)==AUTH_KEYSET_SHA and csha({k:True for k in AUTH_CHECK_KEYS})==AUTH_CHECKS_SHA and len(AUTH_TOP_KEYS)==39 and len(AUTH_CHECK_KEYS)==26,"unsafe_false":all(AUTHORIZATION[k] is False for k in ("retry_authorized","direct_corrected_inner_authorized","direct_r2_authorized","phase_a_authorized","training_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized")),"nested_exact1":AUTHORIZATION["nested_corrected_inner_invocations_authorized"]==AUTHORIZATION["nested_r2_invocations_authorized"]==1,"roots_distinct":len({OUTER_EVIDENCE,OUTER_PREP,INNER_ATTEMPT,INNER_PREP,AUTH_ROOT})==5,"phase_contract_current":regular(PHASE_CONTRACT,PHASE_CONTRACT_SHA,PHASE_CONTRACT_BYTES)["logical_bytes"]==PHASE_CONTRACT_BYTES,"failed_v492_exact4":rooted_tree(V492_OUTER_FAILURE_ROOT)["file_count"]==4 and rooted_tree(V492_OUTER_FAILURE_ROOT)["sha256sum_lines_digest_sha256"]=="02903695aa916eeceac2c416811250cf9deef58d7e5ba64cfcde84930f10690a" and rooted_tree(V492_OUTER_FAILURE_ROOT)["canonical_json_triples_digest_sha256"]=="a851b3d415ccf57c8a43eb4fafb2efc3c482fe23199f6f820257b63968c26277"}
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
 fixture_path=Path("/dev/shm/v493_authority_contract_candidate.json")
 if fixture_path.is_file():
  fixture=json.loads(fixture_path.read_text());historical=decode_path_map(fixture["historical_absences"],23);current=decode_path_map(fixture["current_absences_after_authority"],22)
  receipt_h={k:{"path":str(v),"absent":True} for k,v in historical.items()};receipt_c={k:{"path":str(v),"absent":True} for k,v in current.items()}
  checks["live_contract_schema"]=(set(fixture)==CONTRACT_TOP_KEYS and fixture["format"]==CONTRACT_FORMAT and fixture["status"]==CONTRACT_STATUS and fixture["seed"]==1635 and set(fixture["authority_receipt_contract"]["top_keys"])==AUTH_TOP_KEYS and fixture["authority_receipt_contract"]["check_keys"]==AUTH_CHECK_KEYS and fixture["authority_receipt_contract"]["authorization_exact"]==AUTHORIZATION and fixture["authority_receipt_contract"]["runtime_observation_exact"]==RUNTIME and fixture["execution_boundary"]==EXECUTION_BOUNDARY and set(fixture["source_closure"])==SOURCE_ROLES)
  checks["live_absence_schema_transform"]=(decode_absence_receipt(receipt_h,23)==historical and decode_absence_receipt(receipt_c,22)==current and current=={k:v for k,v in historical.items() if k!="execution_authority_root"} and historical["outer_evidence_prep"]==OUTER_PREP and historical["corrected_inner_attempt_root"]==INNER_ATTEMPT and historical["qualification_root"]==QUALIFICATION)
 print(json.dumps({"passed":all(checks.values()),"checks":checks,"checks_sha256":csha(checks)},sort_keys=True));return 0 if all(checks.values()) else 3
def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--authority-contract",type=Path,required=True);p.add_argument("--authority-contract-sha",required=True);p.add_argument("--authority-receipt",type=Path,required=True);p.add_argument("--authority-receipt-sha",required=True);p.add_argument("--wrapper-source",type=Path,required=True);p.add_argument("--wrapper-sha",required=True);return p.parse_args()
def main():
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic()
 return execute(parse_args())
if __name__=="__main__":raise SystemExit(main())
