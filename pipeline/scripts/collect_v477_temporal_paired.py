#!/usr/bin/env python3
"""Four-context temporal paired-simulator collector for v477.

For every fixed branch and every cumulative prefix length k=1..8, this kernel
independently resets the same simulator seed, replays the same prefix/history,
executes that branch's k-action prefix in one chunk, and stores its real RGB
endpoint as temporal frame k.  It never fabricates intermediate frames and
never reads reward, success, done, policy targets, dev, hidden, or final data.
"""
from __future__ import annotations

import hashlib, importlib, json, os, signal, sys, types
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
import yaml
from wam_pipeline.data import resize_rgb

VARIANTS=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT=VARIANTS[1:5]

def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    with tmp.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def ensure_open3d_importable():
    existing=sys.modules.get("open3d")
    if existing is not None and getattr(existing,"__v477_rgb_only_stub__",False): return True
    try:
        __import__("open3d"); return False
    except ModuleNotFoundError as exc:
        if getattr(exc,"name",None)!="open3d": raise
        stub=types.ModuleType("open3d"); stub.__v477_rgb_only_stub__=True; stub.__all__=[]
        sys.modules["open3d"]=stub; return True

def runtime_provenance():
    modules={}
    for name in ("open3d","toppra","mplib","sapien"):
        module=importlib.import_module(name); version=getattr(module,"__version__",None)
        if version is None:
            try: version=metadata.version(name)
            except metadata.PackageNotFoundError: version="unknown"
        modules[name]={"file":str(Path(module.__file__).resolve()),"version":str(version)}
    return {"sys_executable":sys.executable,"sys_prefix":sys.prefix,"modules":modules}

def image(obs):
    raw=np.asarray(obs["full_image"])
    if raw.shape!=(240,320,3) or raw.dtype!=np.uint8: raise RuntimeError(f"bad raw RGB {raw.shape} {raw.dtype}")
    value=resize_rgb(raw,256)
    if value.shape!=(256,256,3) or value.dtype!=np.uint8: raise RuntimeError("bad resized RGB")
    return value

def pose(task):
    raw=task.get_obs()["endpose"]
    return np.concatenate((np.asarray(raw["left_endpose"]).reshape(-1),np.asarray(raw["left_gripper"]).reshape(-1),np.asarray(raw["right_endpose"]).reshape(-1),np.asarray(raw["right_gripper"]).reshape(-1))).astype(np.float64)

def bottle(task):
    return np.asarray(task.bottle.get_functional_point(0),dtype=np.float64).reshape(-1)

def make_branches(history,future):
    anchor=history[-1,7:13]; delta=future[:,7:13]-anchor; result={}
    for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
        value=future.copy(); value[:,7:13]=anchor+scale*delta; result[name]=value
    return {"factual":future.copy(),**result,"factual_duplicate":future.copy()}

def alarm(_signum,_frame):
    raise TimeoutError("v477 context exceeded frozen timeout")

def collect_one(spec,support_root,task_config,out_dir,lower,upper,timeout_seconds=900):
    """Collect one preregistered context; returns (row, technical_failures)."""
    signal.signal(signal.SIGALRM,alarm); signal.alarm(int(timeout_seconds))
    out=Path(out_dir); cfg=yaml.safe_load(Path(task_config).read_text())
    cfg.update({"task_name":"adjust_bottle","planner_backend":"mplib","step_lim":100000,"clear_cache_freq":1})
    os.environ["ASSETS_PATH"]=str(Path(support_root).resolve()); stub=ensure_open3d_importable()
    from robotwin.envs.vector_env import VectorEnv
    env=None; technical=[]
    try:
        env=VectorEnv(task_config=cfg,n_envs=1,env_seeds=[int(spec["dataset_seed"])]); start=int(spec["start"])
        if file_sha(spec["source_hdf5"])!=spec["source_hdf5_sha256"]:raise RuntimeError("v477 source HDF5 SHA drift")
        oracle_path=Path(spec["v461_endpoint_npz"]).resolve()
        if file_sha(oracle_path)!=spec["v461_endpoint_npz_sha256"]:raise RuntimeError("v477 endpoint oracle SHA drift")
        with h5py.File(spec["source_hdf5"],"r") as h5: actions=np.asarray(h5["joint_action/vector"],dtype=np.float32)
        history=actions[start:start+4].copy(); future=actions[start+4:start+12].copy(); action_map=make_branches(history,future)
        with np.load(oracle_path,allow_pickle=False) as z:
            oracle={key:np.asarray(z[key]).copy() for key in ("variants","history_actions","future_actions","branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","endpoint_rgb","endpoint_state","endpoint_bottle_position")}
        oracle_futures=np.stack([action_map[name] for name in VARIANTS])
        if list(oracle["variants"].astype("U"))!=list(VARIANTS) or not np.array_equal(history,oracle["history_actions"]) or not np.array_equal(oracle_futures,oracle["future_actions"]):raise RuntimeError("v477 branch/action oracle drift")
        if digest(history)!=spec["history_action_sha256"] or any(digest(oracle_futures[i])!=spec["branch_action_sha256"][name] for i,name in enumerate(VARIANTS)):raise RuntimeError("v477 frozen action SHA drift")
        contexts=[]; states=[]; poses=[]; bottles=[]; target_rgb=[]; target_state=[]; target_pose=[]; target_bottle=[]; executed=[]
        for name in VARIANTS:
            for k in range(1,9):
                task=None; original=None
                try:
                    env.reset(env_idx=[0],env_seeds=[int(spec["dataset_seed"])]); task=env.envs[0].task; task.eval_success=False
                    original=task.check_success; task.check_success=types.MethodType(lambda self:False,task)
                    if start: env.step(actions[:start][None])
                    env.step(history[None]); obs=env.get_obs()[0]
                    contexts.append(image(obs)); states.append(np.asarray(obs["state"],dtype=np.float32).copy()); poses.append(pose(task)); bottles.append(bottle(task))
                    prefix=action_map[name][:k].copy();env.step(prefix[None]);obs=env.get_obs()[0]
                    target_rgb.append(image(obs));target_state.append(np.asarray(obs["state"],dtype=np.float32).copy());target_pose.append(pose(task));target_bottle.append(bottle(task));executed.append(digest(prefix))
                except Exception as exc:
                    technical.append({"episode":int(spec["episode"]),"variant":name,"prefix_length":k,"error":repr(exc)});break
                finally:
                    if task is not None and original is not None: task.check_success=original
            if technical:break
        if len(target_rgb)!=len(VARIANTS)*8:return None,technical
        contexts=np.stack(contexts).reshape(6,8,256,256,3);states=np.stack(states).reshape(6,8,-1);poses=np.stack(poses).reshape(6,8,-1);bottles=np.stack(bottles).reshape(6,8,-1)
        target_rgb=np.stack(target_rgb).reshape(6,8,256,256,3);target_state=np.stack(target_state).reshape(6,8,-1);target_pose=np.stack(target_pose).reshape(6,8,-1);target_bottle=np.stack(target_bottle).reshape(6,8,-1)
        futures=np.stack([action_map[name] for name in VARIANTS]); lower=np.asarray(lower,np.float32); upper=np.asarray(upper,np.float32)
        context_exact=all(np.array_equal(contexts[0,0],contexts.reshape(48,*contexts.shape[2:])[i]) and np.array_equal(states[0,0],states.reshape(48,-1)[i]) and np.array_equal(poses[0,0],poses.reshape(48,-1)[i]) and np.array_equal(bottles[0,0],bottles.reshape(48,-1)[i]) for i in range(1,48))
        context_oracle=all(np.array_equal(contexts[i],np.repeat(oracle["branch_context_rgb"][i,None],8,axis=0)) and np.array_equal(states[i],np.repeat(oracle["branch_context_state"][i,None],8,axis=0)) and np.array_equal(poses[i],np.repeat(oracle["branch_context_pose"][i,None],8,axis=0)) and np.array_equal(bottles[i],np.repeat(oracle["branch_context_bottle_position"][i,None],8,axis=0)) for i in range(6))
        k8_oracle=np.array_equal(target_rgb[:,-1],oracle["endpoint_rgb"]) and np.array_equal(target_state[:,-1],oracle["endpoint_state"]) and np.array_equal(target_bottle[:,-1],oracle["endpoint_bottle_position"])
        action_bounds=bool(np.all(futures>=lower) and np.all(futures<=upper))
        action_diff={name:[float(np.linalg.norm(futures[i,k,7:13]-futures[0,k,7:13])) for k in range(8)] for i,name in enumerate(TRANSPORT,1)}
        duplicate=[float(np.abs(target_rgb[0,k].astype(np.float32)-target_rgb[5,k].astype(np.float32)).mean()) for k in range(8)];duplicate_exact=np.array_equal(target_rgb[0],target_rgb[5]) and np.array_equal(target_state[0],target_state[5]) and np.array_equal(target_pose[0],target_pose[5]) and np.array_equal(target_bottle[0],target_bottle[5]);effects={}
        for i,name in enumerate(TRANSPORT,1):
            rgb=[float(np.abs(target_rgb[i,k].astype(np.float32)-target_rgb[0,k].astype(np.float32)).mean()) for k in range(8)];state=[float(np.linalg.norm(target_state[i,k]-target_state[0,k])) for k in range(8)];obj=[float(np.linalg.norm(target_bottle[i,k]-target_bottle[0,k])) for k in range(8)];effective=[rgb[k]>=1. and (state[k]>=.01 or obj[k]>=.005) for k in range(8)]
            effects[name]={"rgb_mae":rgb,"state_l2":state,"bottle_position_l2":obj,"effective_prefixes":effective,"effective_prefix_count":sum(effective),"passed":sum(effective)>=2}
        source_window=Path(spec["source_window"]).resolve()
        if file_sha(source_window)!=spec["source_window_sha256"]:raise RuntimeError("v477 public diagnostic window SHA drift")
        with np.load(source_window,allow_pickle=False) as source:
            public_history=np.asarray(source["history_actions"],np.float32).copy();public_future=np.asarray(source["future_actions"],np.float32).copy();public_factual=np.asarray(source["target_frames"],np.uint8).copy()
        if not np.array_equal(public_history,history) or not np.array_equal(public_future,future):raise RuntimeError("v477 public diagnostic window action alignment drift")
        if public_factual.shape!=(8,256,256,3):raise RuntimeError("v477 public factual frame shape")
        public_mae=[float(np.abs(target_rgb[0,k].astype(np.float32)-public_factual[k].astype(np.float32)).mean()) for k in range(8)]
        expected_shapes=((contexts,(6,8,256,256,3),np.uint8),(states,(6,8,14),np.float32),(poses,(6,8,16),np.float64),(bottles,(6,8,7),np.float64),(target_rgb,(6,8,256,256,3),np.uint8),(target_state,(6,8,14),np.float32),(target_pose,(6,8,16),np.float64),(target_bottle,(6,8,7),np.float64),(futures,(6,8,14),np.float32))
        if any(value.shape!=shape or value.dtype!=dtype or (value.dtype!=np.uint8 and not np.isfinite(value).all()) for value,shape,dtype in expected_shapes):raise RuntimeError("v477 exact array schema drift")
        stem=f"episode{int(spec['episode'])}_start{start:05d}";final=out/stem;partial=out/f".{stem}.partial.{os.getpid()}"
        if final.exists() or partial.exists():raise FileExistsError(stem)
        partial.mkdir(parents=False);path=partial/"temporal.npz"
        expected_prefix=[digest(futures[bi,:k]) for bi in range(6) for k in range(1,9)]
        with path.open("xb") as stream:
            np.savez_compressed(stream,episode=np.int64(spec["episode"]),dataset_seed=np.int64(spec["dataset_seed"]),start=np.int64(start),variants=np.asarray(VARIANTS),prefix_lengths=np.arange(1,9,dtype=np.int64),instruction=np.asarray(spec["instruction"]),pre_future_context_rgb=contexts,pre_future_context_state=states,pre_future_context_pose=poses,pre_future_context_bottle_position=bottles,history_actions=history,future_actions=futures,temporal_rgb=target_rgb,temporal_state=target_state,temporal_pose=target_pose,temporal_bottle_position=target_bottle,context_rgb_sha256=np.asarray([[digest(x) for x in row] for row in contexts]),context_state_sha256=np.asarray([[digest(x) for x in row] for row in states]),executed_prefix_action_sha256=np.asarray(executed).reshape(6,8),public_factual_target_rgb=public_factual,public_factual_target_sha256=np.asarray([digest(x) for x in public_factual]),factual_public_frame_rgb_mae=np.asarray(public_mae,np.float64));stream.flush();os.fsync(stream.fileno())
        row={"episode":int(spec["episode"]),"start":start,"row_dir":stem,"npz":"temporal.npz","npz_sha256":file_sha(path),"context_48way_bitexact":context_exact,"context_v461_oracle_bitexact":context_oracle,"k8_v461_rgb_state_bottle_bitexact":k8_oracle,"executed_prefix_actions_exact":executed==expected_prefix,"action_bounds_passed":action_bounds,"action_prefix_endpoint_diff_l2":action_diff,"factual_duplicate_frame_rgb_mae":duplicate,"factual_duplicate_temporal_rgb_state_pose_bottle_bitexact":duplicate_exact,"effects":effects,"effective_transport_count":sum(v["passed"] for v in effects.values()),"effective_transport_gate_passed":sum(v["passed"] for v in effects.values())>=3,"technical_intervention_effect_qualification_only":True,"all_fixed_rows_branches_retained":True,"factual_public_frame_rgb_mae_diagnostic":public_mae,"open3d_import_mode":"rgb_only_stub" if stub else "native","temporal_real_simulator_frames":True,"cumulative_prefix_single_chunk_per_frame":True,"intermediate_frames_fabricated":False,"task_reward_success_done_outcome_consumed":False,"factual_public_diagnostic_only":True}
        atomic_json(partial/"receipt.json",row)
        for item in (path,partial/"receipt.json"):
            with item.open("rb") as stream:os.fsync(stream.fileno())
        os.replace(partial,final);fd=os.open(str(out),os.O_RDONLY);os.fsync(fd);os.close(fd)
        return row,technical
    finally:
        signal.alarm(0)
        if env is not None: env.close(clear_cache=True)
