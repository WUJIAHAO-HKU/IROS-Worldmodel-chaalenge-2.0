#!/usr/bin/env python3
"""Run the frozen four-context/192-replay v477 temporal simulator pilot."""
from __future__ import annotations
import argparse,hashlib,importlib.util,inspect,json,multiprocessing as mp,os,queue,signal,subprocess,time,traceback
from pathlib import Path

FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-preregistration-v1"
REPORT_FORMAT="strict-track2-v477-public-train-paired-temporal8-pilot-generation-report-v1"
CONTRACT_SHA="9fbb494f55115786b2b22e77d91c172ec1a52df77cec224083ff6e82391e53a1"
ORDER=((25,70),(49,79),(40,118),(28,121));TRANSPORT=("no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def support_tree(root):
 root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as f:json.dump(obj,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def module(path):
 spec=importlib.util.spec_from_file_location("v477_frozen_collector",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def gpu_mib():
 result=subprocess.run(["nvidia-smi","--query-compute-apps=used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
 if result.returncode:return None
 values=[]
 for line in result.stdout.splitlines():
  line=line.strip()
  if line:
   if not line.isdigit():return None
   values.append(int(line))
 return sum(values)
def worker(lane,specs,collector_path,support_root,task_config,rows_dir,lower,upper,outq):
 try:
  os.sched_setaffinity(0,set(range(lane*6,(lane+1)*6)))
  for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="3"
  collect=module(collector_path)
  for item in specs:
   row,technical=collect.collect_one(item,support_root,task_config,rows_dir,lower,upper,900)
   outq.put({"kind":"row","lane":lane,"row":row,"technical":technical})
   if row is None or technical:return
  outq.put({"kind":"done","lane":lane})
 except BaseException as exc:outq.put({"kind":"error","lane":lane,"error":repr(exc),"traceback":traceback.format_exc()})
def stop(processes):
 for p in processes:
  if p.is_alive():p.terminate()
 deadline=time.monotonic()+15
 for p in processes:p.join(max(0,deadline-time.monotonic()))
 for p in processes:
  if p.is_alive():p.kill()
 for p in processes:p.join(5)
def main():
 ap=argparse.ArgumentParser()
 for name in ("preregistration","collector","support-root","task-config","output"):ap.add_argument("--"+name,required=True,type=Path)
 a=ap.parse_args();pre=json.loads(a.preregistration.read_text())
 if a.output.exists() or pre.get("format")!=FORMAT or pre.get("status")!="preregistered_public_train_paired_temporal8_pilot_authorized" or pre.get("contract",{}).get("sha256")!=CONTRACT_SHA:raise RuntimeError("v477 preregistration drift")
 closure=pre["execution_closure"]
 support_digest,support_count=support_tree(a.support_root)
 collector_module=module(a.collector);resize_path=Path(inspect.getsourcefile(collector_module.resize_rgb)).resolve()
 if sha(Path(__file__).resolve())!=pre["evidence_sha256"]["generator"] or sha(a.collector)!=pre["evidence_sha256"]["collector"] or resize_path!=Path(closure["resize_source"]["path"]).resolve() or sha(resize_path)!=closure["resize_source"]["sha256"] or Path(closure["support_root"]["path"]).resolve()!=a.support_root.resolve() or support_digest!=closure["support_root"]["source_config_tree_sha256"] or support_count!=closure["support_root"]["source_config_file_count"] or Path(closure["task_config"]["path"]).resolve()!=a.task_config.resolve() or sha(a.task_config)!=closure["task_config"]["sha256"] or [(int(x["episode"]),int(x["start"])) for x in pre["contexts"]]!=list(ORDER):raise RuntimeError("v477 source/context closure")
 a.output.mkdir(parents=True);rows_dir=a.output/"rows";rows_dir.mkdir();start=time.monotonic();peak=gpu_mib()
 if peak is None:raise RuntimeError("v477 GPU query failed")
 ctx=mp.get_context("spawn");outq=ctx.Queue();lanes=[pre["contexts"][0::2],pre["contexts"][1::2]];processes=[ctx.Process(target=worker,args=(i,lanes[i],str(a.collector.resolve()),str(a.support_root.resolve()),str(a.task_config.resolve()),str(rows_dir),pre["per_dim_action_lower"],pre["per_dim_action_upper"],outq)) for i in range(2)]
 rows=[];technical=[];done=set();fatal=None;all_dead_since=None
 try:
  for p in processes:p.start()
  while len(done)<2 and fatal is None:
   wall=time.monotonic()-start
   if wall>1800:fatal={"error":"v477 wall timeout","wall_seconds":wall};break
   used=gpu_mib()
   if used is None or used>24576:fatal={"error":"v477 GPU gate","gpu_mib":used};break
   peak=max(peak,used)
   try:item=outq.get(timeout=1)
   except queue.Empty:
    dead=[(i,p.exitcode) for i,p in enumerate(processes) if not p.is_alive() and i not in done]
    if any(code not in (None,0) for _i,code in dead):fatal={"error":"v477 worker abnormal exit","workers":dead}
    elif len(dead)==2-len(done):
     if all_dead_since is None:all_dead_since=time.monotonic()
     elif time.monotonic()-all_dead_since>5:fatal={"error":"v477 worker exited without completion receipt","workers":dead}
    else:all_dead_since=None
    continue
   if item["kind"]=="done":done.add(item["lane"])
   elif item["kind"]=="error":fatal=item
   else:
    technical.extend(item["technical"])
    if item["row"] is None or item["technical"]:fatal={"error":"v477 technical row failure","item":item}
    else:rows.append(item["row"])
 finally:
  stop(processes)
 wall=time.monotonic()-start
 if fatal is not None:
  atomic(a.output/"failure_receipt.json",{"format":"strict-track2-v477-temporal8-pilot-failure-v1","passed":False,"fatal":fatal,"technical_failures":technical,"rows_completed":len(rows),"wall_seconds":wall,"gpu_peak_mib":peak,"parent_training_authorized":False,"full200_collection_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False});return 2
 rows.sort(key=lambda x:ORDER.index((int(x["episode"]),int(x["start"]))))
 counts={name:sum(bool(row["effects"][name]["passed"]) for row in rows) for name in TRANSPORT};output_bytes=sum(p.stat().st_size for p in a.output.rglob("*") if p.is_file());collector=module(a.collector);provenance=collector.runtime_provenance()
 checks={"exact_contexts_replays":len(rows)==4 and sum(48 for _ in rows)==192,"row_identity_order":[(x["episode"],x["start"]) for x in rows]==list(ORDER),"all_row_integrity":all(x["context_48way_bitexact"] and x["context_v461_oracle_bitexact"] and x["k8_v461_rgb_state_bottle_bitexact"] and x["executed_prefix_actions_exact"] and x["action_bounds_passed"] and x["factual_duplicate_temporal_rgb_state_pose_bottle_bitexact"] for x in rows),"technical_effect_per_context":all(x["effective_transport_gate_passed"] for x in rows),"technical_effect_per_transport":set(counts)==set(TRANSPORT) and all(v>=3 for v in counts.values()),"all_rows_branches_retained":all(x["all_fixed_rows_branches_retained"] for x in rows),"native_runtime":all(x["open3d_import_mode"]=="native" for x in rows),"no_task_outcome_or_policy":all(x["task_reward_success_done_outcome_consumed"] is False and x["technical_intervention_effect_qualification_only"] is True for x in rows),"resource_bounds":wall<=1800 and peak<=24576 and output_bytes<=268435456,"no_technical_failures":not technical}
 passed=all(checks.values());report={"format":REPORT_FORMAT,"passed":passed,"checks":checks,"rows":rows,"transport_effect_context_counts":counts,"wall_seconds":wall,"gpu_peak_mib":peak,"output_bytes":output_bytes,"runtime_provenance":provenance,"technical_failures":technical,"preregistration_sha256":sha(a.preregistration),"collector_sha256":sha(a.collector),"generator_sha256":sha(Path(__file__).resolve()),"execution_closure_digest":hashlib.sha256(json.dumps(closure,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"support_source_config_tree_sha256":support_digest,"task_config_sha256":sha(a.task_config),"resize_source_sha256":sha(resize_path),"guards":{"technical_intervention_effect_whole_pilot_gate":True,"task_reward_success_outcome_selection":False,"all_fixed_rows_branches_frames_retained":True,"public_factual_diagnostic_only":True,"parent_training_authorized":False,"full200_temporal_design_authorized_if_pass":passed,"full200_collection_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False}}
 atomic(a.output/"generation_report.json",report);print(json.dumps({"passed":passed,"checks":checks,"transport_effect_context_counts":counts},sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
