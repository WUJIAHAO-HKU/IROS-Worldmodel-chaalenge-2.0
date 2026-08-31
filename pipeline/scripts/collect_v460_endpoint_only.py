#!/usr/bin/env python3
"""HDF5-only simulator endpoint collector for v460.

This is the frozen v455 replay/intervention kernel with its public-window
diagnostic removed.  It consumes only the selected public HDF5 actions and
stores the six within-simulator endpoint branches; it never opens a derived
window, reward, success label, or policy target.
"""
from __future__ import annotations

import hashlib, importlib, os, signal, sys, types
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

def ensure_open3d_importable():
    existing=sys.modules.get("open3d")
    if existing is not None and getattr(existing,"__v460_rgb_only_stub__",False): return True
    try:
        __import__("open3d"); return False
    except ModuleNotFoundError as exc:
        if getattr(exc,"name",None)!="open3d": raise
        stub=types.ModuleType("open3d"); stub.__v460_rgb_only_stub__=True; stub.__all__=[]
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
    raise TimeoutError("v460 context exceeded 300 seconds")

def collect_one(spec,support_root,task_config,out_dir,lower,upper):
    signal.signal(signal.SIGALRM,alarm); signal.alarm(300)
    out=Path(out_dir); cfg=yaml.safe_load(Path(task_config).read_text())
    cfg.update({"task_name":"adjust_bottle","planner_backend":"mplib","step_lim":100000,"clear_cache_freq":1})
    os.environ["ASSETS_PATH"]=str(Path(support_root).resolve()); stub=ensure_open3d_importable()
    from robotwin.envs.vector_env import VectorEnv
    env=None; technical=[]
    try:
        env=VectorEnv(task_config=cfg,n_envs=1,env_seeds=[int(spec["dataset_seed"])]); start=int(spec["start"])
        with h5py.File(spec["source_hdf5"],"r") as h5: actions=np.asarray(h5["joint_action/vector"],dtype=np.float32)
        history=actions[start:start+4].copy(); future=actions[start+4:start+12].copy(); action_map=make_branches(history,future)
        contexts=[]; states=[]; poses=[]; bottles=[]; end_rgb=[]; end_state=[]; end_bottle=[]; executed=[]
        for name in VARIANTS:
            task=None; original=None
            try:
                env.reset(env_idx=[0],env_seeds=[int(spec["dataset_seed"])]); task=env.envs[0].task; task.eval_success=False
                original=task.check_success; task.check_success=types.MethodType(lambda self:False,task)
                if start: env.step(actions[:start][None])
                env.step(history[None]); obs=env.get_obs()[0]
                contexts.append(image(obs)); states.append(np.asarray(obs["state"],dtype=np.float32).copy()); poses.append(pose(task)); bottles.append(bottle(task))
                env.step(action_map[name][None]); obs=env.get_obs()[0]
                end_rgb.append(image(obs)); end_state.append(np.asarray(obs["state"],dtype=np.float32).copy()); end_bottle.append(bottle(task)); executed.append(digest(action_map[name]))
            except Exception as exc:
                technical.append({"episode":int(spec["episode"]),"variant":name,"error":repr(exc)}); break
            finally:
                if task is not None and original is not None: task.check_success=original
        if len(end_rgb)!=6: return None,technical
        contexts=np.stack(contexts); states=np.stack(states); poses=np.stack(poses); bottles=np.stack(bottles)
        end_rgb=np.stack(end_rgb); end_state=np.stack(end_state); end_bottle=np.stack(end_bottle)
        futures=np.stack([action_map[name] for name in VARIANTS]); lower=np.asarray(lower,np.float32); upper=np.asarray(upper,np.float32)
        context_exact=all(np.array_equal(contexts[0],contexts[i]) and np.array_equal(states[0],states[i]) and np.array_equal(poses[0],poses[i]) and np.array_equal(bottles[0],bottles[i]) for i in range(1,6))
        action_bounds=bool(np.all(futures>=lower) and np.all(futures<=upper))
        action_diff={name:float(np.linalg.norm(futures[i,-1,7:13]-futures[0,-1,7:13])) for i,name in enumerate(TRANSPORT,1)}
        duplicate=float(np.abs(end_rgb[0].astype(np.float32)-end_rgb[5].astype(np.float32)).mean()); effects={}
        for i,name in enumerate(TRANSPORT,1):
            rgb=float(np.abs(end_rgb[i].astype(np.float32)-end_rgb[0].astype(np.float32)).mean()); qpos=float(np.linalg.norm(end_state[i]-end_state[0])); obj=float(np.linalg.norm(end_bottle[i]-end_bottle[0]))
            effects[name]={"endpoint_rgb_mae":rgb,"endpoint_qpos_l2":qpos,"bottle_position_l2":obj,"passed":rgb>=1. and (qpos>=.01 or obj>=.005)}
        path=out/f"episode{int(spec['episode'])}_start{start:05d}.npz"
        np.savez_compressed(path,episode=np.int64(spec["episode"]),dataset_seed=np.int64(spec["dataset_seed"]),start=np.int64(start),variants=np.asarray(VARIANTS),instruction=np.asarray(spec["instruction"]),branch_context_rgb=contexts,branch_context_state=states,branch_context_pose=poses,branch_context_bottle_position=bottles,history_actions=history,future_actions=futures,endpoint_rgb=end_rgb,endpoint_state=end_state,endpoint_bottle_position=end_bottle,context_rgb_sha256=np.asarray([digest(x) for x in contexts]),context_state_sha256=np.asarray([digest(x) for x in states]),executed_action_sha256=np.asarray(executed))
        row={"episode":int(spec["episode"]),"start":start,"file":path.name,"context_bitexact":context_exact,"executed_actions_exact":executed==[digest(x) for x in futures],"action_bounds_passed":action_bounds,"action_endpoint_diff_l2":action_diff,"action_diff_passed":min(action_diff.values())>=.01,"factual_duplicate_rgb_mae":duplicate,"duplicate_passed":duplicate<=.5,"effects":effects,"effect_count":sum(x["passed"] for x in effects.values()),"effect_passed":sum(x["passed"] for x in effects.values())>=3,"open3d_import_mode":"rgb_only_stub" if stub else "native","endpoint_only":True,"public_window_or_endpoint_diagnostic_consumed":False}
        return row,technical
    finally:
        signal.alarm(0)
        if env is not None: env.close(clear_cache=True)

