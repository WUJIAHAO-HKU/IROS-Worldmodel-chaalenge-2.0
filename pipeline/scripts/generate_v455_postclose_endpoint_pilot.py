#!/usr/bin/env python3
"""Collect true within-simulator paired endpoints for v455."""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, importlib, json, multiprocessing as mp, os, signal, subprocess, sys, time, types
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import h5py, numpy as np, yaml
from wam_pipeline.data import resize_rgb

VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT=VARIANTS[1:5]
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
def ensure_open3d_importable():
 existing=sys.modules.get("open3d")
 if existing is not None and getattr(existing,"__v455_rgb_only_stub__",False):return True
 try:__import__("open3d");return False
 except ModuleNotFoundError as exc:
  if getattr(exc,"name",None)!="open3d":raise
  stub=types.ModuleType("open3d");stub.__v455_rgb_only_stub__=True;stub.__all__=[];sys.modules["open3d"]=stub;return True
def runtime_provenance():
 modules={}
 for name in ("open3d","toppra","mplib","sapien"):
  module=importlib.import_module(name); version=getattr(module,"__version__",None)
  if version is None:
   try:version=metadata.version(name)
   except metadata.PackageNotFoundError:version="unknown"
  modules[name]={"file":str(Path(module.__file__).resolve()),"version":str(version)}
 return {"sys_executable":sys.executable,"sys_prefix":sys.prefix,"modules":modules}
def image(obs):
 raw=np.asarray(obs["full_image"])
 if raw.shape!=(240,320,3) or raw.dtype!=np.uint8:raise RuntimeError(f"bad raw RGB {raw.shape} {raw.dtype}")
 value=resize_rgb(raw,256)
 if value.shape!=(256,256,3) or value.dtype!=np.uint8:raise RuntimeError("bad resized RGB")
 return value
def pose(task):
 raw=task.get_obs()["endpose"]
 return np.concatenate((np.asarray(raw["left_endpose"]).reshape(-1),np.asarray(raw["left_gripper"]).reshape(-1),np.asarray(raw["right_endpose"]).reshape(-1),np.asarray(raw["right_gripper"]).reshape(-1))).astype(np.float64)
def bottle(task):return np.asarray(task.bottle.get_functional_point(0),dtype=np.float64).reshape(-1)
def make_branches(history,future):
 anchor=history[-1,7:13];delta=future[:,7:13]-anchor;result={}
 for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
  value=future.copy();value[:,7:13]=anchor+scale*delta;result[name]=value
 return {"factual":future.copy(),**result,"factual_duplicate":future.copy()}
def alarm(_s,_f):raise TimeoutError("v455 context exceeded 300 seconds")
def collect_one(spec,support_root,task_config,out_dir,lower,upper):
 signal.signal(signal.SIGALRM,alarm);signal.alarm(300);out=Path(out_dir);cfg=yaml.safe_load(Path(task_config).read_text());cfg.update({"task_name":"adjust_bottle","planner_backend":"mplib","step_lim":100000,"clear_cache_freq":1});os.environ["ASSETS_PATH"]=str(Path(support_root).resolve());stub=ensure_open3d_importable()
 from robotwin.envs.vector_env import VectorEnv
 env=None;technical=[]
 try:
  env=VectorEnv(task_config=cfg,n_envs=1,env_seeds=[int(spec["dataset_seed"])]);start=int(spec["start"])
  with h5py.File(spec["source_hdf5"],"r") as h5:actions=np.asarray(h5["joint_action/vector"],dtype=np.float32)
  history=actions[start:start+4].copy();future=actions[start+4:start+12].copy();action_map=make_branches(history,future);contexts=[];states=[];poses=[];bottles=[];end_rgb=[];end_state=[];end_bottle=[];executed=[]
  for name in VARIANTS:
   task=None;original=None
   try:
    env.reset(env_idx=[0],env_seeds=[int(spec["dataset_seed"])]);task=env.envs[0].task;task.eval_success=False;original=task.check_success;task.check_success=types.MethodType(lambda self:False,task)
    if start:env.step(actions[:start][None])
    env.step(history[None]);obs=env.get_obs()[0];contexts.append(image(obs));states.append(np.asarray(obs["state"],dtype=np.float32).copy());poses.append(pose(task));bottles.append(bottle(task))
    env.step(action_map[name][None]);obs=env.get_obs()[0];end_rgb.append(image(obs));end_state.append(np.asarray(obs["state"],dtype=np.float32).copy());end_bottle.append(bottle(task));executed.append(digest(action_map[name]))
   except Exception as exc:technical.append({"episode":int(spec["episode"]),"variant":name,"error":repr(exc)});break
   finally:
    if task is not None and original is not None:task.check_success=original
  if len(end_rgb)!=6:return None,technical
  contexts=np.stack(contexts);states=np.stack(states);poses=np.stack(poses);bottles=np.stack(bottles);end_rgb=np.stack(end_rgb);end_state=np.stack(end_state);end_bottle=np.stack(end_bottle);futures=np.stack([action_map[n] for n in VARIANTS]);lower=np.asarray(lower,np.float32);upper=np.asarray(upper,np.float32)
  context_exact=all(np.array_equal(contexts[0],contexts[i]) and np.array_equal(states[0],states[i]) and np.array_equal(poses[0],poses[i]) and np.array_equal(bottles[0],bottles[i]) for i in range(1,6));action_bounds=bool(np.all(futures>=lower) and np.all(futures<=upper));action_diff={name:float(np.linalg.norm(futures[i,-1,7:13]-futures[0,-1,7:13])) for i,name in enumerate(TRANSPORT,1)}
  duplicate=float(np.abs(end_rgb[0].astype(np.float32)-end_rgb[5].astype(np.float32)).mean());effects={}
  for i,name in enumerate(TRANSPORT,1):
   rgb=float(np.abs(end_rgb[i].astype(np.float32)-end_rgb[0].astype(np.float32)).mean());qpos=float(np.linalg.norm(end_state[i]-end_state[0]));obj=float(np.linalg.norm(end_bottle[i]-end_bottle[0]));effects[name]={"endpoint_rgb_mae":rgb,"endpoint_qpos_l2":qpos,"bottle_position_l2":obj,"passed":rgb>=1. and (qpos>=.01 or obj>=.005)}
  with np.load(spec["source_window"],allow_pickle=False) as source:public_endpoint=source["target_frames"][-1].astype(np.uint8)
  public_mae=float(np.abs(end_rgb[0].astype(np.float32)-public_endpoint.astype(np.float32)).mean());path=out/f"episode{int(spec['episode'])}_start{start:05d}.npz"
  np.savez_compressed(path,episode=np.int64(spec["episode"]),dataset_seed=np.int64(spec["dataset_seed"]),start=np.int64(start),variants=np.asarray(VARIANTS),instruction=np.asarray(spec["instruction"]),branch_context_rgb=contexts,branch_context_state=states,branch_context_pose=poses,branch_context_bottle_position=bottles,history_actions=history,future_actions=futures,endpoint_rgb=end_rgb,endpoint_state=end_state,endpoint_bottle_position=end_bottle,context_rgb_sha256=np.asarray([digest(x) for x in contexts]),context_state_sha256=np.asarray([digest(x) for x in states]),executed_action_sha256=np.asarray(executed),public_factual_endpoint_sha256=np.asarray(digest(public_endpoint)),factual_public_endpoint_rgb_mae=np.float64(public_mae))
  row={"episode":int(spec["episode"]),"start":start,"file":path.name,"context_bitexact":context_exact,"executed_actions_exact":executed==[digest(x) for x in futures],"action_bounds_passed":action_bounds,"action_endpoint_diff_l2":action_diff,"action_diff_passed":min(action_diff.values())>=.01,"factual_duplicate_rgb_mae":duplicate,"duplicate_passed":duplicate<=.5,"factual_public_endpoint_rgb_mae_diagnostic":public_mae,"effects":effects,"effect_count":sum(x["passed"] for x in effects.values()),"effect_passed":sum(x["passed"] for x in effects.values())>=3,"open3d_import_mode":"rgb_only_stub" if stub else "native"}
  return row,technical
 finally:
  signal.alarm(0)
  if env is not None:env.close(clear_cache=True)
def gpu_mib():
 try:
  text=subprocess.check_output(["nvidia-smi","--query-compute-apps=used_memory","--format=csv,noheader,nounits"],text=True,timeout=5);return sum(int(x.strip()) for x in text.splitlines() if x.strip())
 except Exception:return -1
def main():
 p=argparse.ArgumentParser()
 for name in ("preregistration","support-root","task-config","output-dir"):p.add_argument(f"--{name}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());out=a.output_dir
 if pre.get("format")!="strict-track2-v455-postclose-endpoint-pilot-preregistration-v1" or out.exists():raise RuntimeError("v455 prereg/output contract failed")
 out.mkdir(parents=True);started=time.monotonic();rows=[];technical=[];peak=max(0,gpu_mib());ctx=mp.get_context("spawn");pool=cf.ProcessPoolExecutor(max_workers=2,mp_context=ctx);pending={pool.submit(collect_one,spec,str(a.support_root),str(a.task_config),str(out),pre["public_train"]["per_dim_action_lower"],pre["public_train"]["per_dim_action_upper"]):spec for spec in pre["public_train"]["contexts"]};aborted=False
 while pending:
  if time.monotonic()-started>1200:technical.append({"error":"wall timeout"});aborted=True;break
  done,_=cf.wait(pending,timeout=1,return_when=cf.FIRST_COMPLETED);sample=gpu_mib()
  if sample<0 or sample>24576:technical.append({"error":f"GPU monitor/limit {sample}"});aborted=True;break
  peak=max(peak,sample)
  for future in done:
   spec=pending.pop(future)
   try:row,errors=future.result();technical.extend(errors);rows.append(row) if row is not None else None;aborted=bool(errors or row is None)
   except Exception as exc:technical.append({"episode":spec["episode"],"error":repr(exc)});aborted=True
   if aborted:break
  if aborted:break
 if aborted:
  for f in pending:f.cancel()
  for proc in list(getattr(pool,"_processes",{}).values()):proc.terminate()
  pool.shutdown(wait=False,cancel_futures=True)
 else:pool.shutdown(wait=True)
 rows.sort(key=lambda x:x["episode"]);wall=time.monotonic()-started;counts={name:sum(row["effects"][name]["passed"] for row in rows) for name in TRANSPORT};output_bytes=sum(x.stat().st_size for x in out.glob("episode*_start*.npz"));passed=len(rows)==4 and not technical and peak<=24576 and wall<=1200 and output_bytes<=16777216 and all(row["context_bitexact"] and row["executed_actions_exact"] and row["action_bounds_passed"] and row["action_diff_passed"] and row["duplicate_passed"] and row["effect_passed"] and row["open3d_import_mode"]=="native" for row in rows) and all(value>=3 for value in counts.values())
 report={"format":"strict-track2-v455-postclose-endpoint-pilot-generation-report-v1","created_at":datetime.now(timezone.utc).isoformat(),"passed":passed,"rows":rows,"transport_effect_context_counts":counts,"gpu_peak_mib":peak,"wall_seconds":wall,"output_bytes":output_bytes,"technical_failures":technical,"runtime_provenance":runtime_provenance(),"official_rgb_preprocess":pre["preprocessing"],"guards":{"paired_endpoint_residual_only":True,"context_workers":2,"branches_serial":True,"future_single_chunk_call":True,"endpoint_only":True,"intermediate_frames_fabricated":False,"planner_backend":"mplib","reward_success_done_consumed":False,"factual_public_endpoint_mae_diagnostic_only":True,"policy_updates":0,"endpoint_parent_data_authorized_if_pass":True,"rl_authorized":False}}
 (out/"generation_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
