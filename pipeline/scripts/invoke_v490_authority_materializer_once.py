#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,os,signal,subprocess,sys,tempfile,time
from pathlib import Path

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge");J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810";RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
CONTRACT=ROOT/"pipeline/scripts/v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority_contract.json";CONTRACT_SHA="f37dc074bff5cffa25c13d0b4c16530f1f02e82362187fb7bb116ad4250e8058";CONTRACT_BYTES=28754
MATERIALIZER=ROOT/"pipeline/scripts/materialize_v490_v489_v488_v487_c71_exact7_schema_repair_postregistration_authority.py";MATERIALIZER_SHA="6618ec3266c46a3320f93caa447234109ab301082352b7da9e2e1aaf2af67218";MATERIALIZER_BYTES=36797
WRAPPER=ROOT/"pipeline/scripts/launch_v490_v489_v488_v487_c71_exact7_schema_repair.py";WRAPPER_SHA="733931fa162c7e2559fcbf45b7ce8b98960db61df958143c7df3f4c15f3bc877";WRAPPER_BYTES=67698
STATIC_RECEIPT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825/static_audit.json"
AUTHORITY_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_authority_seed1632_20260825";AUTHORITY_PREP=AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+".registration-prep")
EVIDENCE=J/"v490_v489_v488_v487_c71_exact7_schema_repair_authority_materialization_evidence_seed1632_20260825";SELF_PATH=ROOT/"pipeline/scripts/invoke_v490_authority_materializer_once.py"
SOURCE_ROLES={"repair_preregistration","repair_design_contract","repair_formal_materializer","reconciler_r2","static_auditor","execution_wrapper"}

class ControlledSignal(BaseException):pass
def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):d.update(b)
 return d.hexdigest()
def cbytes(v)->bytes:return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def csha(v)->str:return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def fsync_dir(path:Path):
 fd=os.open(str(path),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def regular(path:Path|str,want_sha=None,want_bytes=None):
 path=Path(path)
 if path!=path.resolve() or not path.is_file() or path.is_symlink():raise RuntimeError(f"regular: {path}")
 r={"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
 if want_sha is not None and r["sha256"]!=want_sha:raise RuntimeError(f"sha: {path}")
 if want_bytes is not None and r["logical_bytes"]!=want_bytes:raise RuntimeError(f"bytes: {path}")
 return r
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
def atomic_bytes(path:Path,data:bytes):
 tmp=path.with_name(path.name+".tmp")
 if os.path.lexists(path) or os.path.lexists(tmp):raise FileExistsError(path)
 fd=os.open(str(tmp),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 try:
  view=memoryview(data)
  while view:
   n=os.write(fd,view)
   if n<=0:raise RuntimeError("short write")
   view=view[n:]
  os.fsync(fd)
 finally:os.close(fd)
 os.replace(tmp,path);fsync_dir(path.parent)
def atomic_json(path:Path,value):atomic_bytes(path,cbytes(value))
def copy_fsync(source:Path,target:Path):
 if os.path.lexists(target):raise FileExistsError(target)
 with source.open("rb") as s,target.open("xb") as t:
  for block in iter(lambda:s.read(8<<20),b""):t.write(block)
  t.flush();os.fsync(t.fileno())
 fsync_dir(target.parent)
def close_fsync(stream):
 if stream is not None and not stream.closed:stream.flush();os.fsync(stream.fileno());stream.close()
def group_empty(pid:int):
 try:os.killpg(pid,0);return False
 except ProcessLookupError:return True
def terminate(process):
 out={"started":process is not None,"term_sent":False,"kill_sent":False,"reaped":process is None,"group_empty":True}
 if process is None:return out
 if process.poll() is None or not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGTERM);out["term_sent"]=True
  except ProcessLookupError:pass
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:pass
 if not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGKILL);out["kill_sent"]=True
  except ProcessLookupError:pass
 if process.poll() is None:process.wait(timeout=10)
 else:process.wait()
 out["reaped"]=process.poll() is not None;out["group_empty"]=group_empty(process.pid)
 return out
def live_processes():
 found=[];fragments=(str(MATERIALIZER),str(WRAPPER),"reconcile_v488_v487_c71_exact7_schema_repair.py")
 for p in Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(x in cmd for x in fragments):found.append({"pid":int(p.name),"cmdline":cmd})
 return found
def materializer_argv():return [str(RLPY),str(MATERIALIZER),"--contract",str(CONTRACT),"--contract-sha",CONTRACT_SHA,"--materializer-source",str(MATERIALIZER),"--materializer-sha",MATERIALIZER_SHA,"--static-receipt",str(STATIC_RECEIPT),"--authority-root",str(AUTHORITY_ROOT)]
def load_contract(self_record):
 contract_record=regular(CONTRACT,CONTRACT_SHA,CONTRACT_BYTES);materializer_record=regular(MATERIALIZER,MATERIALIZER_SHA,MATERIALIZER_BYTES);wrapper_record=regular(WRAPPER,WRAPPER_SHA,WRAPPER_BYTES);contract=json.loads(CONTRACT.read_text())
 if contract.get("authority_materializer_source")!=materializer_record or set(contract.get("source_closure",{}))!=SOURCE_ROLES or contract["source_closure"]["execution_wrapper"]!=wrapper_record or contract.get("source_closure_sha256")!=csha(contract["source_closure"]):raise RuntimeError("contract source closure")
 sources={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in contract["source_closure"].items()}
 if sources!=contract["source_closure"]:raise RuntimeError("current source closure")
 static_record=regular(STATIC_RECEIPT,contract["static_receipt"]["sha256"],contract["static_receipt"]["logical_bytes"])
 return contract,{"contract":contract_record,"materializer":materializer_record,"wrapper":wrapper_record,"helper":self_record,"static_receipt":static_record},sources
def stable_snapshot(contract,files,sources):
 observed_files={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in files.items()};observed_sources={k:regular(v["path"],v["sha256"],v["logical_bytes"]) for k,v in sources.items()}
 rooted_fields=("repair_formal_registration_tree","postregistration_static_registration_tree","static_execution_evidence_tree","f813_registration_tree","failed_attempt_tree","formal_materialization_evidence_tree")
 trees={k:rooted_tree(contract[k]["root"]) for k in rooted_fields}
 if any(trees[k]!=contract[k] for k in rooted_fields):raise RuntimeError("rooted input tree")
 failure=contract["static_failure_ancestry"];extra={}
 for label,receipt_key,tree_key in (("f88","failed_static_process_receipt","failed_static_execution_tree"),("v489","v489_failed_static_process_receipt","v489_failed_static_execution_tree")):
  root=Path(failure[receipt_key]["path"]).parent;actual=exact_tree(root)
  if actual!=failure[tree_key]:raise RuntimeError(f"{label} failure tree")
  extra[label]={"root":str(root),**actual}
 current=contract["current_absences_after_authority"]
 if len(current)!=11:raise RuntimeError("current absence count")
 absences={k:{"path":v,"absent":not os.path.lexists(v)} for k,v in sorted(current.items())}
 if not all(v["absent"] for v in absences.values()):raise RuntimeError("current absence")
 return {"files":observed_files,"sources":observed_sources,"trees":trees,"failure_trees":extra,"current_absences":absences}
def validate_authority(contract):
 receipt_path=AUTHORITY_ROOT/"authority_receipt.json";record=regular(receipt_path);receipt=json.loads(receipt_path.read_text());schema=contract["authority_receipt_contract"]
 if set(receipt)!=set(schema["top_keys"]) or receipt.get("format")!=schema["format"] or receipt.get("status")!=schema["status"] or receipt.get("passed") is not True:raise RuntimeError("authority schema")
 if receipt.get("checks")!={k:True for k in schema["check_keys"]} or receipt.get("checks_sha256")!=schema["checks_sha256"] or receipt.get("authorization")!=schema["authorization_exact"] or receipt.get("runtime_observation")!=schema["runtime_observation_exact"]:raise RuntimeError("authority boundary")
 tree=rooted_tree(AUTHORITY_ROOT)
 if tree["inventory"]!=[["authority_receipt.json",record["sha256"],record["logical_bytes"]]] or tree["file_count"]!=1 or os.path.lexists(AUTHORITY_PREP):raise RuntimeError("authority exact1")
 return record,tree,receipt
def cleanup_preintent(created):
 if not created or not EVIDENCE.is_dir() or EVIDENCE.is_symlink():return
 allowed={"transport_helper.py","argv.json","argv.json.tmp","materializer_stdout.log","materializer_stderr.log","intent.json","intent.json.tmp"}
 entries=list(EVIDENCE.iterdir())
 if any(e.name not in allowed or e.is_dir() or e.is_symlink() for e in entries):return
 for e in entries:e.unlink()
 EVIDENCE.rmdir();fsync_dir(EVIDENCE.parent)

def execute(self_sha,self_bytes):
 self_record=regular(Path(__file__).resolve(),self_sha,self_bytes);contract,files,sources=load_contract(self_record)
 if Path(sys.executable)!=RLPY or os.path.lexists(EVIDENCE) or os.path.lexists(AUTHORITY_ROOT) or os.path.lexists(AUTHORITY_PREP) or live_processes():raise RuntimeError("prestate")
 historical=contract["historical_absences"]
 if len(historical)!=12 or any(os.path.lexists(v) for v in historical.values()):raise RuntimeError("historical absence")
 pre=stable_snapshot(contract,files,sources);pre_digest=csha(pre);created=False;intent_committed=False;process=None;stdout=None;stderr=None;started=time.monotonic();calls=0;cleanup={"started":False,"term_sent":False,"kill_sent":False,"reaped":True,"group_empty":True};old_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def on_signal(signum,_frame):raise ControlledSignal(signum)
 for s in old_handlers:signal.signal(s,on_signal)
 try:
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
  try:EVIDENCE.mkdir();created=True;fsync_dir(EVIDENCE.parent)
  finally:
   if mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
  copy_fsync(Path(__file__).resolve(),EVIDENCE/"transport_helper.py");persistent_helper=regular(EVIDENCE/"transport_helper.py",self_sha,self_bytes);argv=materializer_argv()
  argv_payload={"format":"strict-track2-v490-authority-materializer-argv-evidence-v1","argv":argv,"argv_repr":[repr(x) for x in argv],"argv_utf8_hex":[x.encode().hex() for x in argv],"all_tokens_no_cr_lf":all("\r" not in x and "\n" not in x for x in argv),"source_records":files,"source_closure":sources,"transport_helper":self_record,"persistent_transport_helper":persistent_helper,"pre_snapshot":pre,"pre_snapshot_sha256":pre_digest,"materializer_invocations_before":0,"wrapper_invocations_before":0,"r2_invocations_before":0}
  atomic_json(EVIDENCE/"argv.json",argv_payload)
  if json.loads((EVIDENCE/"argv.json").read_text())!=argv_payload or (EVIDENCE/"argv.json").read_bytes()!=cbytes(argv_payload) or argv_payload["argv"]!=materializer_argv() or not argv_payload["all_tokens_no_cr_lf"]:raise RuntimeError("argv readback")
  atomic_bytes(EVIDENCE/"materializer_stdout.log",b"");atomic_bytes(EVIDENCE/"materializer_stderr.log",b"")
  intent={"format":"strict-track2-v490-authority-materializer-invocation-intent-v1","argv":regular(EVIDENCE/"argv.json"),"attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"materializer_invocations_before":0,"wrapper_invocations":0,"r2_invocations":0,"retry_authorized":False}
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
  try:atomic_json(EVIDENCE/"intent.json",intent);intent_committed=True
  finally:
   if mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
  stdout=(EVIDENCE/"materializer_stdout.log").open("ab",buffering=0);stderr=(EVIDENCE/"materializer_stderr.log").open("ab",buffering=0)
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
  try:process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True);calls=1
  finally:
   if mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
  try:returncode=process.wait(timeout=300)
  except subprocess.TimeoutExpired as error:raise RuntimeError("materializer timeout") from error
  close_fsync(stdout);stdout=None;close_fsync(stderr);stderr=None;cleanup=terminate(process)
  if returncode!=0 or not cleanup["reaped"] or not cleanup["group_empty"]:raise RuntimeError(f"materializer rc/cleanup: {returncode} {cleanup}")
  post=stable_snapshot(contract,files,sources);authority_record,authority_tree,authority=validate_authority(contract)
  if post!=pre or csha(post)!=pre_digest or live_processes():raise RuntimeError("poststate drift")
  wall=time.monotonic()-started
  receipt={"format":"strict-track2-v490-authority-materializer-process-receipt-v1","status":"passed_exact_once_no_wrapper_or_r2_execution","passed":True,"helper_returncode":0,"materializer_returncode":0,"materializer_invocations":1,"wrapper_invocations":0,"r2_invocations":0,"retry_authorized":False,"timed_out":False,"wall_seconds":wall,"cleanup":cleanup,"argv":regular(EVIDENCE/"argv.json"),"intent":regular(EVIDENCE/"intent.json"),"stdout":regular(EVIDENCE/"materializer_stdout.log"),"stderr":regular(EVIDENCE/"materializer_stderr.log"),"transport_helper":self_record,"persistent_transport_helper":persistent_helper,"pre_snapshot":pre,"post_snapshot":post,"pre_post_snapshots_exactly_equal":True,"pre_snapshot_sha256":pre_digest,"post_snapshot_sha256":csha(post),"authority_receipt":authority_record,"authority_registration_tree":authority_tree,"authority_top_keys":sorted(authority),"authority_check_count":len(authority["checks"]),"current_absences":post["current_absences"]}
  if not math.isfinite(wall) or wall<0:raise RuntimeError("wall")
  for s in old_handlers:signal.signal(s,signal.SIG_IGN)
  atomic_json(EVIDENCE/"process_receipt.json",receipt);return 0
 except BaseException as error:
  for s in old_handlers:signal.signal(s,signal.SIG_IGN)
  try:close_fsync(stdout)
  except BaseException:pass
  try:close_fsync(stderr)
  except BaseException:pass
  try:cleanup=terminate(process)
  except BaseException as ce:cleanup={"cleanup_error":f"{type(ce).__name__}: {ce}","group_empty":False,"reaped":False}
  if intent_committed and EVIDENCE.is_dir() and not os.path.lexists(EVIDENCE/"process_receipt.json"):
   failure={"format":"strict-track2-v490-authority-materializer-process-receipt-v1","status":"failed_no_retry","passed":False,"helper_returncode":125,"materializer_returncode":None if process is None else process.returncode,"materializer_invocations":calls,"wrapper_invocations":0,"r2_invocations":0,"retry_authorized":False,"wall_seconds":time.monotonic()-started,"error_type":type(error).__name__,"error":str(error),"cleanup":cleanup,"argv":regular(EVIDENCE/"argv.json"),"intent":regular(EVIDENCE/"intent.json"),"stdout":regular(EVIDENCE/"materializer_stdout.log"),"stderr":regular(EVIDENCE/"materializer_stderr.log"),"transport_helper":self_record}
   atomic_json(EVIDENCE/"process_receipt.json",failure)
  elif not intent_committed:cleanup_preintent(created)
  raise
 finally:
  for s,h in old_handlers.items():signal.signal(s,h)

def synthetic():
 checks={"argv_exact14":len(materializer_argv())==14,"source_constants":len(CONTRACT_SHA)==len(MATERIALIZER_SHA)==len(WRAPPER_SHA)==64,"fresh_roots_distinct":EVIDENCE!=AUTHORITY_ROOT and AUTHORITY_PREP!=AUTHORITY_ROOT}
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/"x.json";v={"argv":["a","b c"]};atomic_json(p,v);checks["canonical_json"]=(p.read_bytes()==cbytes(v) and json.loads(p.read_text())==v)
 print(json.dumps({"passed":all(checks.values()),"checks":checks},sort_keys=True));return 0 if all(checks.values()) else 3
def main():
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic()
 p=argparse.ArgumentParser();p.add_argument("--self-sha",required=True);p.add_argument("--self-bytes",required=True,type=int);a=p.parse_args();return execute(a.self_sha,a.self_bytes)
if __name__=="__main__":raise SystemExit(main())
