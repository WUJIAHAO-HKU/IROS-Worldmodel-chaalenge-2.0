#!/usr/bin/env python3
"""One-shot endpoint-oracle collection for v478's twelve newly selected contexts."""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,multiprocessing as mp,os,queue,subprocess,sys,time,traceback
from pathlib import Path
import h5py,numpy as np

FORMAT="strict-track2-v478-endpoint-oracle12-preregistration-v1"
REPORT_FORMAT="strict-track2-v478-endpoint-oracle12-generation-report-v1"
VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT=VARIANTS[1:5]
KEYS={"episode","dataset_seed","start","variants","instruction","branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","history_actions","future_actions","endpoint_rgb","endpoint_state","endpoint_bottle_position","context_rgb_sha256","context_state_sha256","executed_action_sha256"}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def module(path):
 s=importlib.util.spec_from_file_location("v478_endpoint_collector",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def gpu():
 try:
  r=subprocess.run(["nvidia-smi","--query-compute-apps=used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
  if r.returncode:return None
  return sum(int(x) for x in r.stdout.split() if x.strip())
 except Exception:return None
def support_tree(root):
 root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)
def stem(s):return f"episode{int(s['episode'])}_start{int(s['start']):05d}"
def branches(history,future):
 a=history[-1,7:13];d=future[:,7:13]-a;out={}
 for n,k in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  x=future.copy();x[:,7:13]=a+k*d;out[n]=x
 return {"factual":future.copy(),**out,"factual_duplicate":future.copy()}
def validate_source(spec):
 if sha(spec["source_hdf5"])!=spec["source_hdf5_sha256"]:raise RuntimeError("source HDF5 drift")
 with h5py.File(spec["source_hdf5"],"r") as h:raw=np.asarray(h["joint_action/vector"])
 if raw.dtype!=np.float64 or raw.ndim!=2 or raw.shape[1]!=14 or not np.isfinite(raw).all():raise RuntimeError("source raw action schema")
 actions=raw.astype(np.float32);start=int(spec["start"]);history=actions[start:start+4].copy();future=actions[start+4:start+12].copy();b=branches(history,future)
 if sha(spec["source_window"])!=spec["source_window_sha256"]:raise RuntimeError("source window drift")
 with np.load(spec["source_window"],allow_pickle=False) as z:
  wh=np.asarray(z["history_actions"]);wf=np.asarray(z["future_actions"])
  if wh.dtype!=np.float32 or wh.shape!=(4,14) or wf.dtype!=np.float32 or wf.shape!=(8,14) or not np.array_equal(wh,history) or not np.array_equal(wf,future):raise RuntimeError("source window action identity")
 endpoint={n:float(np.linalg.norm(b[n][-1,7:13]-b["factual"][-1,7:13])) for n in TRANSPORT};path={n:float(np.linalg.norm(b[n][:,7:13]-b["factual"][:,7:13])) for n in TRANSPORT}
 if arrsha(history)!=spec["history_action_sha256"] or arrsha(future)!=spec["factual_future8_action_sha256"] or any(arrsha(b[n])!=spec["branch_action_sha256"][n] for n in VARIANTS) or not all(endpoint[n]>=.01 and path[n]>=.01 for n in TRANSPORT) or any(not np.isclose(endpoint[n],float(spec["endpoint_diffs"][n]),rtol=1e-7,atol=0) or not np.isclose(path[n],float(spec["path_diffs"][n]),rtol=1e-7,atol=0) for n in TRANSPORT):raise RuntimeError("source action identity/difference gate")
 return history,b
def validate_npz(path,spec):
 with np.load(path,allow_pickle=False) as z:
  if set(z.files)!=KEYS:raise RuntimeError("oracle keys")
  if int(z["episode"])!=int(spec["episode"]) or int(z["start"])!=int(spec["start"]) or int(z["dataset_seed"])!=int(spec["dataset_seed"]):raise RuntimeError("oracle identity")
  exact=(("branch_context_rgb",(6,256,256,3),np.uint8),("branch_context_state",(6,14),np.float32),("branch_context_pose",(6,16),np.float64),("branch_context_bottle_position",(6,7),np.float64),("history_actions",(4,14),np.float32),("future_actions",(6,8,14),np.float32),("endpoint_rgb",(6,256,256,3),np.uint8),("endpoint_state",(6,14),np.float32),("endpoint_bottle_position",(6,7),np.float64))
  if any(z[k].shape!=s or z[k].dtype!=d for k,s,d in exact):raise RuntimeError("oracle schema")
  history,b=validate_source(spec);future=np.stack([b[n] for n in VARIANTS])
  if not np.array_equal(z["history_actions"],history) or not np.array_equal(z["future_actions"],future):raise RuntimeError("oracle actions")
  if list(z["variants"].astype("U"))!=list(VARIANTS) or not np.array_equal(z["executed_action_sha256"].astype("U"),np.asarray([arrsha(x) for x in future])):raise RuntimeError("oracle action SHA")
  if not np.array_equal(z["context_rgb_sha256"].astype("U"),np.asarray([arrsha(x) for x in z["branch_context_rgb"]])) or not np.array_equal(z["context_state_sha256"].astype("U"),np.asarray([arrsha(x) for x in z["branch_context_state"]])):raise RuntimeError("oracle context SHA")
  if not all(np.array_equal(z["branch_context_rgb"][0],z["branch_context_rgb"][i]) and np.array_equal(z["branch_context_state"][0],z["branch_context_state"][i]) and np.array_equal(z["branch_context_pose"][0],z["branch_context_pose"][i]) and np.array_equal(z["branch_context_bottle_position"][0],z["branch_context_bottle_position"][i]) for i in range(1,6)):raise RuntimeError("oracle context")
  if not np.array_equal(z["endpoint_rgb"][0],z["endpoint_rgb"][5]) or not np.array_equal(z["endpoint_state"][0],z["endpoint_state"][5]) or not np.array_equal(z["endpoint_bottle_position"][0],z["endpoint_bottle_position"][5]):raise RuntimeError("oracle duplicate")
 return sha(path)
def collect_worker(lane,specs,collector,support,task,root,lower,upper,q):
 os.sched_setaffinity(0,set(range(lane*6,(lane+1)*6)));os.environ.update({"OMP_NUM_THREADS":"3","MKL_NUM_THREADS":"3","OPENBLAS_NUM_THREADS":"3"})
 try:
  m=module(Path(collector))
  for spec in specs:
   validate_source(spec);name=stem(spec);work=Path(root)/"work"/(name+f".{os.getpid()}");partial=Path(root)/"rows"/("."+name+f".partial.{os.getpid()}");final=Path(root)/"rows"/name
   if any(p.exists() for p in (work,partial,final)):raise FileExistsError(name)
   work.mkdir(parents=True);row,technical=m.collect_one(spec,Path(support),Path(task),work,lower,upper)
   if row is None or technical:raise RuntimeError(f"oracle technical failure {technical}")
   src=work/row["file"];npz_sha=validate_npz(src,spec);partial.mkdir();os.replace(src,partial/"endpoint.npz")
   receipt={**row,"format":"strict-track2-v478-endpoint-oracle12-row-v1","npz":"endpoint.npz","npz_sha256":npz_sha,"source_hdf5_sha256":spec["source_hdf5_sha256"],"source_action_raw_dtype":"float64","canonical_action_dtype":"float32","history_action_sha256":spec["history_action_sha256"],"branch_action_sha256":spec["branch_action_sha256"],"all_selected_rows_branches_retained":True,"effect_used_for_retry_or_filter":False,"task_reward_success_outcome_consumed":False}
   atomic(partial/"receipt.json",receipt);os.rmdir(work);os.replace(partial,final);fd=os.open(str(final.parent),os.O_RDONLY);os.fsync(fd);os.close(fd);q.put({"kind":"row","row":receipt})
  q.put({"kind":"done","lane":lane})
 except BaseException as e:q.put({"kind":"error","lane":lane,"error":repr(e),"traceback":traceback.format_exc()})
def stop(ps):
 for p in ps:
  if p.is_alive():p.terminate()
 end=time.monotonic()+15
 for p in ps:p.join(max(0,end-time.monotonic()))
 for p in ps:
  if p.is_alive():p.kill()
 for p in ps:p.join(5)
 return {"pids":[p.pid for p in ps],"exitcodes":[p.exitcode for p in ps],"alive_after":[p.pid for p in ps if p.is_alive()]}
def _main():
 ap=argparse.ArgumentParser()
 for n in ("preregistration","collector","support-root","task-config","resize-source","output"):ap.add_argument("--"+n,type=Path,required=True)
 a=ap.parse_args();pre=json.loads(a.preregistration.read_text());closure=pre["execution_closure"]
 if a.output.exists() or pre.get("format")!=FORMAT or pre.get("status")!="preregistered_public_train_endpoint_oracle_collection_authorized":raise RuntimeError("oracle prereg")
 if len(pre.get("contexts",[]))!=12 or pre.get("oracle_reuse",{}).get("selected_new")!=12:raise RuntimeError("oracle exact12")
 for k,p in (("generator",Path(__file__).resolve()),("collector",a.collector)):
  if Path(closure[k]["path"]).resolve()!=p.resolve() or closure[k]["sha256"]!=sha(p):raise RuntimeError(k+" closure")
 support_digest,support_count=support_tree(a.support_root);collector_module=module(a.collector);resize_path=Path(inspect.getsourcefile(collector_module.resize_rgb)).resolve()
 if Path(pre["oracle_root"]).resolve()!=a.output.resolve() or Path(closure["support_root"]["path"]).resolve()!=a.support_root.resolve() or support_digest!=closure["support_root"]["source_config_tree_sha256"] or support_count!=closure["support_root"]["source_config_file_count"] or Path(closure["task_config"]["path"]).resolve()!=a.task_config.resolve() or closure["task_config"]["sha256"]!=sha(a.task_config) or resize_path!=a.resize_source.resolve() or Path(closure["resize_source"]["path"]).resolve()!=resize_path or closure["resize_source"]["sha256"]!=sha(resize_path):raise RuntimeError("oracle environment closure")
 a.output.mkdir(parents=True);(a.output/"rows").mkdir();(a.output/"work").mkdir();start=time.monotonic();peak=gpu()
 if peak is None:raise RuntimeError("GPU query")
 ctx=mp.get_context("spawn");q=ctx.Queue();lanes=[pre["contexts"][0::2],pre["contexts"][1::2]];ps=[ctx.Process(target=collect_worker,args=(i,lanes[i],str(a.collector),str(a.support_root),str(a.task_config),str(a.output),pre["per_dim_action_lower"],pre["per_dim_action_upper"],q)) for i in range(2)];done=set();rows=[];fatal=None;all_dead_since=None
 try:
  for p in ps:p.start()
  while len(done)<2 and fatal is None:
   if time.monotonic()-start>1800:fatal=TimeoutError("oracle12 timeout");break
   used=gpu()
   if used is None or used>24576:fatal=RuntimeError(f"oracle GPU {used}");break
   peak=max(peak,used)
   try:item=q.get(timeout=1)
   except queue.Empty:
    bad=[p.exitcode for p in ps if not p.is_alive() and p.exitcode not in (None,0)]
    if bad:fatal=RuntimeError(f"oracle worker exit {bad}")
    elif all(not p.is_alive() for p in ps):
     if all_dead_since is None:all_dead_since=time.monotonic()
     elif time.monotonic()-all_dead_since>5:fatal=RuntimeError("oracle workers exited without completion receipts")
    else:all_dead_since=None
    continue
   if item["kind"]=="done":done.add(item["lane"])
   elif item["kind"]=="error":fatal=RuntimeError(item["error"]+"\n"+item["traceback"])
   else:rows.append(item["row"])
 finally:cleanup=stop(ps)
 if fatal:
  atomic(a.output/"failure_receipt.json",{"format":"strict-track2-v478-endpoint-oracle12-failure-v1","passed":False,"error":repr(fatal),"cleanup":cleanup,"rows_retained":len(list((a.output/"rows").glob("*/receipt.json"))),"retry_under_same_lineage_authorized":False,"phase_b_authorized":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False});raise fatal
 try:
  if any((a.output/"work").iterdir()):raise RuntimeError("oracle work remains")
  (a.output/"work").rmdir()
  # The committed tree is authoritative across the rename/q.put crash window.
  rows=[json.loads((a.output/"rows"/stem(spec)/"receipt.json").read_text()) for spec in pre["contexts"]]
  for spec,row in zip(pre["contexts"],rows):
   if row["npz_sha256"]!=validate_npz(a.output/"rows"/stem(spec)/"endpoint.npz",spec):raise RuntimeError("committed oracle row drift")
  counts={n:sum(bool(r["effects"][n]["passed"]) for r in rows) for n in TRANSPORT};checks={"exact12":len(rows)==12,"identity_order":[(r["episode"],r["start"]) for r in rows]==[(int(x["episode"]),int(x["start"])) for x in pre["contexts"]],"row_integrity":all(r["context_bitexact"] and r["executed_actions_exact"] and r["action_bounds_passed"] and r["action_diff_passed"] and r["duplicate_passed"] for r in rows),"all_rows_retained":len(list((a.output/"rows").iterdir()))==12,"resource_bounds":time.monotonic()-start<=1800 and peak<=24576}
 except BaseException as exc:
  atomic(a.output/"failure_receipt.json",{"format":"strict-track2-v478-endpoint-oracle12-failure-v1","passed":False,"error":repr(exc),"cleanup":cleanup,"rows_retained":len(list((a.output/"rows").glob("*/receipt.json"))),"retry_under_same_lineage_authorized":False,"phase_b_authorized":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False});raise
 provenance=collector_module.runtime_provenance()
 report={"format":REPORT_FORMAT,"passed":all(checks.values()),"checks":checks,"rows":[{"episode":r["episode"],"start":r["start"],"npz_sha256":r["npz_sha256"]} for r in rows],"transport_effect_context_counts_diagnostic":counts,"wall_seconds":time.monotonic()-start,"gpu_peak_mib":peak,"preregistration_sha256":sha(a.preregistration),"generator_sha256":sha(Path(__file__).resolve()),"collector_sha256":sha(a.collector),"support_source_config_tree_sha256":support_digest,"support_source_config_file_count":support_count,"task_config_sha256":sha(a.task_config),"resize_source_sha256":sha(resize_path),"runtime_provenance":provenance,"guards":{"effect_used_for_retry_or_filter":False,"phase_b_temporal_authorized":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}}
 atomic(a.output/"generation_report.json",report);return 0 if report["passed"] else 2
def main():
 try:return _main()
 except BaseException as exc:
  try:
   argv=sys.argv;root=Path(argv[argv.index("--output")+1])
   receipt=root/"failure_receipt.json"
   if root.is_dir() and not receipt.exists() and not (root/"generation_report.json").exists():
    atomic(receipt,{"format":"strict-track2-v478-endpoint-oracle12-failure-v1","passed":False,"error":repr(exc),"traceback_sha256":hashlib.sha256(traceback.format_exc().encode()).hexdigest(),"rows_retained":len(list((root/"rows").glob("*/receipt.json"))) if (root/"rows").is_dir() else 0,"retry_under_same_lineage_authorized":False,"phase_b_authorized":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False})
  except Exception:pass
  raise
if __name__=="__main__":raise SystemExit(main())
