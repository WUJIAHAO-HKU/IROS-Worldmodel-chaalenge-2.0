#!/usr/bin/env python3
"""Sharded, resumable v478 temporal-8 simulator collection.

This orchestrator never selects contexts.  It consumes the immutable v478
action-only selection, delegates each row to the frozen v477 collector, and
only promotes a batch after all twenty atomic rows pass the frozen technical
integrity gates.  Stale or malformed work is moved to immutable forensics.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, multiprocessing as mp, os, queue
import subprocess, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
import h5py

PRE_FORMAT = "strict-track2-v478-public-train-temporal200-preregistration-v1"
BATCH_FORMAT = "strict-track2-v478-public-train-temporal200-batch-report-v1"
FINAL_FORMAT = "strict-track2-v478-public-train-temporal200-generation-report-v1"
ROW_FORMAT = "strict-track2-v477-public-train-paired-temporal8-row-v1"
VARIANTS = ("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4","factual_duplicate")
TRANSPORT = VARIANTS[1:5]
GIB = 1 << 30
ROOT_PREFLIGHT_FREE = 6 * GIB
ROOT_POST_FREE = 4 * GIB
REG_FREE = 3 * GIB
ROW_WORST_CASE_BYTES = 11_000_000
ATOMIC_QUARANTINE_RESERVE = 256 * (1 << 20)
EXPECTED_NPZ = {
    "episode","dataset_seed","start","variants","prefix_lengths","instruction",
    "pre_future_context_rgb","pre_future_context_state","pre_future_context_pose",
    "pre_future_context_bottle_position","history_actions","future_actions",
    "temporal_rgb","temporal_state","temporal_pose","temporal_bottle_position",
    "context_rgb_sha256","context_state_sha256","executed_prefix_action_sha256",
    "public_factual_target_rgb","public_factual_target_sha256","factual_public_frame_rgb_mae",
}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def arrsha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()

def branches(history,future):
    anchor=history[-1,7:13];delta=future[:,7:13]-anchor;out={}
    for name,scale in (("no_transport",0.),("scale_0p4",.4),("scale_1p25",1.25),("reverse_direction_0p4",-.4)):
        value=future.copy();value[:,7:13]=anchor+scale*delta;out[name]=value
    return {"factual":future.copy(),**out,"factual_duplicate":future.copy()}

def support_tree(root: Path):
    root=Path(root).resolve();paths=sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in (".py",".yml",".yaml"));items=[[p.relative_to(root).as_posix(),sha(p)] for p in paths]
    return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),len(items)

def atomic_json(path: Path, value: dict) -> None:
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    if path.exists() or tmp.exists():raise FileExistsError(path)
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)

def fsync_dir(path: Path) -> None:
    fd=os.open(str(path),os.O_RDONLY);os.fsync(fd);os.close(fd)

def free_bytes(path: Path) -> int:
    value=os.statvfs(path)
    return int(value.f_bavail*value.f_frsize)

def storage_gate(root: Path, remaining_rows: int, initial: bool=False) -> dict:
    root_parent=root.parent.resolve();reg_parent=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810").resolve()
    if str(root.resolve()).startswith(("/tmp/","/dev/shm/")) or str(root.resolve()) in {"/tmp","/dev/shm"}:raise RuntimeError("v478 final dataset may not use tmpfs/tmp")
    root_free=free_bytes(root_parent);reg_free=free_bytes(reg_parent)
    required=ROOT_POST_FREE+remaining_rows*ROW_WORST_CASE_BYTES+ATOMIC_QUARANTINE_RESERVE
    if initial:required=max(required,ROOT_PREFLIGHT_FREE)
    result={"root_free_bytes":root_free,"registry_free_bytes":reg_free,"remaining_rows":remaining_rows,"remaining_worst_case_bytes":remaining_rows*ROW_WORST_CASE_BYTES,"atomic_quarantine_reserve_bytes":ATOMIC_QUARANTINE_RESERVE,"required_root_free_bytes":required,"required_registry_free_bytes":REG_FREE}
    if root_free<required or reg_free<REG_FREE:raise RuntimeError(f"v478 storage gate failed {result}")
    return result

def module(path: Path):
    spec=importlib.util.spec_from_file_location("v478_frozen_collector",path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def gpu_mib():
    try:
        r=subprocess.run(["nvidia-smi","--query-compute-apps=used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
        if r.returncode:return None
        values=[int(x.strip()) for x in r.stdout.splitlines() if x.strip()]
        return sum(values)
    except Exception:return None

def stem(spec):return f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"

def validate_row(rowdir: Path, spec: dict) -> dict:
    npz=rowdir/"temporal.npz";rp=rowdir/"receipt.json"
    if {p.name for p in rowdir.iterdir()}!={"temporal.npz","receipt.json"}:raise RuntimeError(f"row tree drift {rowdir}")
    rec=json.loads(rp.read_text())
    if rec.get("episode")!=int(spec["episode"]) or rec.get("start")!=int(spec["start"]) or rec.get("npz")!="temporal.npz" or rec.get("npz_sha256")!=sha(npz):raise RuntimeError(f"row receipt drift {rowdir}")
    if sha(Path(spec["source_hdf5"]))!=spec["source_hdf5_sha256"] or sha(Path(spec["source_window"]))!=spec["source_window_sha256"] or sha(Path(spec["v461_endpoint_npz"]))!=spec["v461_endpoint_npz_sha256"]:raise RuntimeError(f"row immutable source drift {rowdir}")
    with h5py.File(spec["source_hdf5"],"r") as h5:raw=np.asarray(h5["joint_action/vector"])
    if raw.dtype!=np.float64 or raw.ndim!=2 or raw.shape[1]!=14 or not np.isfinite(raw).all():raise RuntimeError(f"row raw action schema drift {rowdir}")
    action=raw.astype(np.float32);start=int(spec["start"]);history=action[start:start+4].copy();factual=action[start+4:start+12].copy();formula=branches(history,factual);future=np.stack([formula[n] for n in VARIANTS])
    if arrsha(history)!=spec["history_action_sha256"] or arrsha(factual)!=spec["future_action_sha256"] or any(arrsha(formula[n])!=spec["branch_action_sha256"][n] for n in VARIANTS):raise RuntimeError(f"row action SHA/formula drift {rowdir}")
    with np.load(spec["source_window"],allow_pickle=False) as wz:
        wh=np.asarray(wz["history_actions"]);wf=np.asarray(wz["future_actions"]);wt=np.asarray(wz["target_frames"])
    if wh.shape!=(4,14) or wh.dtype!=np.float32 or wf.shape!=(8,14) or wf.dtype!=np.float32 or wt.shape!=(8,256,256,3) or wt.dtype!=np.uint8 or not np.array_equal(wh,history) or not np.array_equal(wf,factual):raise RuntimeError(f"row source window drift {rowdir}")
    with np.load(npz,allow_pickle=False) as z:
        if set(z.files)!=EXPECTED_NPZ:raise RuntimeError(f"row NPZ key drift {rowdir}")
        if int(z["episode"])!=int(spec["episode"]) or int(z["dataset_seed"])!=int(spec["dataset_seed"]) or int(z["start"])!=int(spec["start"]):raise RuntimeError(f"row scalar drift {rowdir}")
        x={k:np.asarray(z[k]).copy() for k in z.files}
    exact=(("pre_future_context_rgb",(6,8,256,256,3),np.uint8),("pre_future_context_state",(6,8,14),np.float32),("pre_future_context_pose",(6,8,16),np.float64),("pre_future_context_bottle_position",(6,8,7),np.float64),("temporal_rgb",(6,8,256,256,3),np.uint8),("temporal_state",(6,8,14),np.float32),("temporal_pose",(6,8,16),np.float64),("temporal_bottle_position",(6,8,7),np.float64),("history_actions",(4,14),np.float32),("future_actions",(6,8,14),np.float32),("public_factual_target_rgb",(8,256,256,3),np.uint8),("factual_public_frame_rgb_mae",(8,),np.float64))
    if any(x[k].shape!=shape or x[k].dtype!=dtype or (dtype!=np.uint8 and not np.isfinite(x[k]).all()) for k,shape,dtype in exact):raise RuntimeError(f"row array schema drift {rowdir}")
    if not (x["episode"].shape==x["dataset_seed"].shape==x["start"].shape==()) or x["episode"].dtype!=np.int64 or x["dataset_seed"].dtype!=np.int64 or x["start"].dtype!=np.int64:raise RuntimeError(f"row scalar schema drift {rowdir}")
    if x["prefix_lengths"].shape!=(8,) or x["prefix_lengths"].dtype!=np.int64 or not np.array_equal(x["prefix_lengths"],np.arange(1,9)) or list(x["variants"].astype("U"))!=list(VARIANTS) or str(x["instruction"])!=str(spec["instruction"]):raise RuntimeError(f"row identity schema drift {rowdir}")
    if not np.array_equal(x["history_actions"],history) or not np.array_equal(x["future_actions"],future) or not np.array_equal(x["public_factual_target_rgb"],wt):raise RuntimeError(f"row stored action/public drift {rowdir}")
    if not all(str(x["executed_prefix_action_sha256"][i,k-1])==arrsha(future[i,:k]) for i in range(6) for k in range(1,9)):raise RuntimeError(f"row prefix hash drift {rowdir}")
    if not all(str(x["context_rgb_sha256"][i,k])==arrsha(x["pre_future_context_rgb"][i,k]) and str(x["context_state_sha256"][i,k])==arrsha(x["pre_future_context_state"][i,k]) for i in range(6) for k in range(8)):raise RuntimeError(f"row context hash drift {rowdir}")
    flat=lambda key:x[key].reshape(48,*x[key].shape[2:])
    if not all(np.array_equal(flat(key)[0],flat(key)[i]) for key in ("pre_future_context_rgb","pre_future_context_state","pre_future_context_pose","pre_future_context_bottle_position") for i in range(1,48)):raise RuntimeError(f"row same context drift {rowdir}")
    with np.load(spec["v461_endpoint_npz"],allow_pickle=False) as oz:oracle={k:np.asarray(oz[k]).copy() for k in ("branch_context_rgb","branch_context_state","branch_context_pose","branch_context_bottle_position","endpoint_rgb","endpoint_state","endpoint_bottle_position")}
    equal_context=lambda key,oracle_key:all(np.array_equal(x[key][i],np.repeat(oracle[oracle_key][i,None],8,axis=0)) for i in range(6))
    oracle_ok=equal_context("pre_future_context_rgb","branch_context_rgb") and equal_context("pre_future_context_state","branch_context_state") and equal_context("pre_future_context_pose","branch_context_pose") and equal_context("pre_future_context_bottle_position","branch_context_bottle_position") and np.array_equal(x["temporal_rgb"][:,-1],oracle["endpoint_rgb"]) and np.array_equal(x["temporal_state"][:,-1],oracle["endpoint_state"]) and np.array_equal(x["temporal_bottle_position"][:,-1],oracle["endpoint_bottle_position"])
    duplicate=all(np.array_equal(x[key][0],x[key][5]) for key in ("temporal_rgb","temporal_state","temporal_pose","temporal_bottle_position"))
    lower=np.asarray(spec.get("per_dim_action_lower",[-np.inf]*14),np.float32);upper=np.asarray(spec.get("per_dim_action_upper",[np.inf]*14),np.float32)
    effects={}
    for i,name in enumerate(TRANSPORT,1):
        rgb=[float(np.abs(x["temporal_rgb"][i,k].astype(np.float32)-x["temporal_rgb"][0,k].astype(np.float32)).mean()) for k in range(8)];state=[float(np.linalg.norm(x["temporal_state"][i,k]-x["temporal_state"][0,k])) for k in range(8)];bottle=[float(np.linalg.norm(x["temporal_bottle_position"][i,k]-x["temporal_bottle_position"][0,k])) for k in range(8)];effective=[rgb[k]>=1. and (state[k]>=.01 or bottle[k]>=.005) for k in range(8)];effects[name]={"rgb_mae":rgb,"state_l2":state,"bottle_position_l2":bottle,"effective_prefixes":effective,"effective_prefix_count":sum(effective),"passed":sum(effective)>=2}
    public=np.asarray([np.abs(x["temporal_rgb"][0,k].astype(np.float32)-wt[k].astype(np.float32)).mean() for k in range(8)],np.float64)
    integrity=oracle_ok and duplicate and np.all(future>=lower) and np.all(future<=upper) and np.array_equal(public,x["factual_public_frame_rgb_mae"]) and all(str(x["public_factual_target_sha256"][k])==arrsha(wt[k]) for k in range(8))
    required=("context_48way_bitexact","context_v461_oracle_bitexact","k8_v461_rgb_state_bottle_bitexact","executed_prefix_actions_exact","action_bounds_passed","factual_duplicate_temporal_rgb_state_pose_bottle_bitexact","all_fixed_rows_branches_retained","temporal_real_simulator_frames","cumulative_prefix_single_chunk_per_frame")
    if not integrity or not all(rec.get(k) is True for k in required) or rec.get("task_reward_success_done_outcome_consumed") is not False or set(rec.get("effects",{}))!=set(TRANSPORT):raise RuntimeError(f"row independent integrity drift {rowdir}")
    result=dict(rec);result["effects"]=effects
    return result

def forensic_root(root: Path,batch_id: int):return root/"_forensic_uncommitted"/f"batch_{batch_id:03d}"

def quarantine(path: Path,root: Path,batch_id: int,reason: str) -> Path:
    if path.is_symlink() or (path.is_dir() and any(p.is_symlink() for p in path.rglob("*"))):raise RuntimeError("refusing symlink in v478 quarantine")
    target_root=forensic_root(root,batch_id);target_root.mkdir(parents=True,exist_ok=True)
    # Recover receipt-preparation crash windows without deleting anything.
    for orphan in list(target_root.glob(".*.receipt-prep")):
        receipt_path=orphan/"quarantine_receipt.json"
        if receipt_path.is_file():
            value=json.loads(receipt_path.read_text());planned=Path(value["planned_destination"])
            if planned.parent.resolve()!=target_root.resolve() or planned.exists():raise RuntimeError(f"ambiguous quarantine receipt prep {orphan}")
            os.replace(orphan,planned);fsync_dir(target_root)
        else:
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ");planned=target_root/f"orphan-receipt-prep-{stamp}"
            (orphan/"payload").mkdir();atomic_json(orphan/"quarantine_receipt.json",{"format":"strict-track2-v478-uncommitted-work-quarantine-v1","source":str(orphan.resolve()),"planned_destination":str(planned.resolve()),"source_files_sha256":{},"payload_name":"payload","payload_move_pending_when_receipt_fsynced":False,"reason":"crash before quarantine receipt materialized","deleted_or_reused":False,"effect_or_outcome_conditioned":False});fsync_dir(orphan);os.replace(orphan,planned);fsync_dir(target_root)
    source=str(path.resolve());matches=[]
    for receipt_path in target_root.glob("*/quarantine_receipt.json"):
        value=json.loads(receipt_path.read_text())
        if value.get("source")==source and not (receipt_path.parent/"payload").exists():matches.append((receipt_path.parent,value))
    if len(matches)>1:raise RuntimeError(f"ambiguous v478 quarantine recovery {path}")
    if matches:
        target,receipt=matches[0]
    else:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ");target=target_root/f"{path.name}-{stamp}";prep=target_root/f".{target.name}.receipt-prep"
        if target.exists() or prep.exists():raise FileExistsError(target)
        source_files={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob("*")) if p.is_file() and not p.is_symlink()} if path.is_dir() else {path.name:sha(path)}
        receipt={"format":"strict-track2-v478-uncommitted-work-quarantine-v1","reason":reason,"source":source,"planned_destination":str(target.resolve()),"source_files_sha256":source_files,"payload_name":"payload","payload_move_pending_when_receipt_fsynced":True,"deleted_or_reused":False,"effect_or_outcome_conditioned":False}
        prep.mkdir();atomic_json(prep/"quarantine_receipt.json",receipt);fsync_dir(prep);os.replace(prep,target);fsync_dir(target_root)
    payload=target/"payload"
    if payload.exists():
        if path.exists():raise RuntimeError(f"quarantine has both source and payload {path}")
    else:
        if not path.exists():raise RuntimeError(f"quarantine lost source before payload move {path}")
        os.replace(path,payload);fsync_dir(path.parent);fsync_dir(target);fsync_dir(target_root)
    actual={str(p.relative_to(payload)):sha(p) for p in sorted(payload.rglob("*")) if p.is_file() and not p.is_symlink()} if payload.is_dir() else {path.name:sha(payload)}
    if actual!=receipt["source_files_sha256"]:raise RuntimeError(f"quarantine payload hash drift {path}")
    return target

def row_receipts(batch: Path,specs: list[dict]):
    result=[]
    for spec in specs:result.append(validate_row(batch/"rows"/stem(spec),spec))
    return result

def forensic_hashes(root: Path,batch_id: int):
    base=forensic_root(root,batch_id)
    if not base.exists():return {},{}
    if any(p.is_symlink() for p in base.rglob("*")):raise RuntimeError(f"v478 forensic symlink {base}")
    q={};f={};allowed=set()
    for attempt in sorted(p for p in base.iterdir() if p.name!="failures"):
        if not attempt.is_dir() or {p.name for p in attempt.iterdir()}!={"quarantine_receipt.json","payload"}:raise RuntimeError(f"v478 forensic quarantine tree {attempt}")
        receipt_path=attempt/"quarantine_receipt.json";receipt=json.loads(receipt_path.read_text());payload=attempt/"payload";source=Path(receipt["source"])
        actual={str(p.relative_to(payload)):sha(p) for p in sorted(payload.rglob("*")) if p.is_file()} if payload.is_dir() else {source.name:sha(payload)}
        if receipt.get("format")!="strict-track2-v478-uncommitted-work-quarantine-v1" or Path(receipt.get("planned_destination","")).resolve()!=attempt.resolve() or receipt.get("source_files_sha256")!=actual or receipt.get("payload_name")!="payload" or receipt.get("deleted_or_reused") is not False or receipt.get("effect_or_outcome_conditioned") is not False:raise RuntimeError(f"v478 forensic receipt drift {attempt}")
        q[str(receipt_path.relative_to(root))]=sha(receipt_path);allowed.add(attempt.name)
    failures=base/"failures"
    if failures.exists():
        if not failures.is_dir():raise RuntimeError("v478 forensic failures not directory")
        for path in sorted(failures.iterdir()):
            if not path.is_file() or path.suffix!=".json":raise RuntimeError(f"v478 forensic failure tree {path}")
            value=json.loads(path.read_text());cleanup=value.get("cleanup",{})
            if value.get("format")!="strict-track2-v478-batch-failure-receipt-v1" or value.get("passed") is not False or value.get("batch_id")!=batch_id or value.get("cleanup_passed") is not True or cleanup.get("alive_after_cleanup")!=[] or not np.isfinite(float(value.get("attempt_wall_seconds",-1))) or float(value.get("attempt_wall_seconds",-1))<0 or value.get("retry_based_on_effect_authorized") is not False or any(value.get(k) not in (False,0) for k in ("full200_collection_authorized","training_authorized","s1_authorized","policy_updates","rl_authorized")):raise RuntimeError(f"v478 forensic failure receipt drift {path}")
            f[str(path.relative_to(root))]=sha(path)
    return q,f

def batch_report(batch: Path,batch_id: int,specs: list[dict],rows: list[dict],wall: float,peak: int,root: Path,storage_pre: dict,committed_rows_pre: int,valid_partial_rows_pre: int):
    counts={name:sum(bool(row["effects"][name]["passed"]) for row in rows) for name in TRANSPORT}
    output_bytes=sum(p.stat().st_size for p in (batch/"rows").rglob("*") if p.is_file())
    q,f=forensic_hashes(root,batch_id)
    failure_wall=0.0
    for rel in f:
        failure_wall+=float(json.loads((root/rel).read_text()).get("attempt_wall_seconds",0.0))
    forensic_bytes=sum(p.stat().st_size for p in forensic_root(root,batch_id).rglob("*") if p.is_file()) if forensic_root(root,batch_id).exists() else 0
    cumulative_wall=wall+failure_wall
    remaining_rows=200-committed_rows_pre-20
    storage_post=storage_gate(root,max(0,remaining_rows),False)
    # Named-effect counts are diagnostics here.  They are evaluated exactly
    # once at finalization after all 200 immutable rows have been retained.
    checks={"exact20":len(specs)==len(rows)==20,"all_rows_integrity":all(r["all_fixed_rows_branches_retained"] for r in rows),"cumulative_attempt_wall_le_5400":cumulative_wall<=5400,"gpu_le_24576":0<=peak<=24576,"output_plus_forensic_within_contract":output_bytes+forensic_bytes<=400000000,"storage_post_and_remaining_gate":storage_post["root_free_bytes"]>=storage_post["required_root_free_bytes"] and storage_post["registry_free_bytes"]>=storage_post["required_registry_free_bytes"]}
    contexts_at_least3=sum(sum(bool(r["effects"][name]["passed"]) for name in TRANSPORT)>=3 for r in rows)
    return {"format":BATCH_FORMAT,"passed":all(checks.values()),"batch_id":batch_id,"checks":checks,"rows":[{"episode":r["episode"],"start":r["start"],"npz_sha256":r["npz_sha256"]} for r in rows],"transport_effect_context_counts_diagnostic":counts,"contexts_at_least_three_of_four_effects_diagnostic":contexts_at_least3,"effect_used_for_batch_pass_retry_filter":False,"committed_rows_pre":committed_rows_pre,"valid_partial_rows_pre":valid_partial_rows_pre,"remaining_rows_pre":200-committed_rows_pre-valid_partial_rows_pre,"remaining_rows_post":remaining_rows,"attempt_wall_seconds":wall,"prior_failed_attempt_wall_seconds":failure_wall,"cumulative_attempt_wall_seconds":cumulative_wall,"gpu_peak_mib":peak,"row_output_bytes":output_bytes,"forensic_output_bytes":forensic_bytes,"total_attempt_output_bytes":output_bytes+forensic_bytes,"storage_pre":storage_pre,"storage_post":storage_post,"quarantine_receipts_sha256":q,"failure_receipts_sha256":f,"guards":{"all_selected_rows_branches_frames_retained":True,"technical_effect_filters_rows":False,"technical_effect_gates_batch":False,"task_reward_success_outcome_consumed":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}}

def validate_batch(path: Path,batch_id: int,specs: list[dict]):
    if {p.name for p in path.iterdir()}!={"rows","batch_report.json"}:raise RuntimeError(f"complete batch tree drift {batch_id}")
    if {p.name for p in (path/"rows").iterdir()}!={stem(s) for s in specs}:raise RuntimeError(f"complete row identity drift {batch_id}")
    rows=row_receipts(path,specs);report=json.loads((path/"batch_report.json").read_text())
    actual=[(r["episode"],r["start"],r["npz_sha256"]) for r in rows];declared=[(r["episode"],r["start"],r["npz_sha256"]) for r in report.get("rows",[])];counts={name:sum(bool(row["effects"][name]["passed"]) for row in rows) for name in TRANSPORT};contexts3=sum(sum(bool(row["effects"][name]["passed"]) for name in TRANSPORT)>=3 for row in rows)
    guards={"all_selected_rows_branches_frames_retained":True,"technical_effect_filters_rows":False,"technical_effect_gates_batch":False,"task_reward_success_outcome_consumed":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False}
    storage_ok=True
    for key in ("storage_pre","storage_post"):
        value=report.get(key,{});remaining=int(value.get("remaining_rows",-1));required=max(ROOT_PREFLIGHT_FREE if key=="storage_pre" and remaining==200 else 0,ROOT_POST_FREE+remaining*ROW_WORST_CASE_BYTES+ATOMIC_QUARANTINE_RESERVE);storage_ok &= remaining>=0 and value.get("remaining_worst_case_bytes")==remaining*ROW_WORST_CASE_BYTES and value.get("required_root_free_bytes")==required and int(value.get("root_free_bytes",-1))>=required and int(value.get("registry_free_bytes",-1))>=REG_FREE
    root=path.parent if path.name.endswith(".partial") else path.parent;row_bytes=sum(p.stat().st_size for p in (path/"rows").rglob("*") if p.is_file());q,f=forensic_hashes(root,batch_id);forensic_bytes=sum(p.stat().st_size for p in forensic_root(root,batch_id).rglob("*") if p.is_file()) if forensic_root(root,batch_id).exists() else 0;failure_wall=sum(float(json.loads((root/rel).read_text()).get("attempt_wall_seconds",0.0)) for rel in f);attempt=float(report.get("attempt_wall_seconds",-1));cumulative=float(report.get("cumulative_attempt_wall_seconds",-1));peak=int(report.get("gpu_peak_mib",-1))
    expected_checks={"exact20":len(specs)==len(rows)==20,"all_rows_integrity":all(r["all_fixed_rows_branches_retained"] for r in rows),"cumulative_attempt_wall_le_5400":cumulative<=5400,"gpu_le_24576":0<=peak<=24576,"output_plus_forensic_within_contract":row_bytes+forensic_bytes<=400000000,"storage_post_and_remaining_gate":bool(storage_ok)}
    report_keys={"format","passed","batch_id","checks","rows","transport_effect_context_counts_diagnostic","contexts_at_least_three_of_four_effects_diagnostic","effect_used_for_batch_pass_retry_filter","committed_rows_pre","valid_partial_rows_pre","remaining_rows_pre","remaining_rows_post","attempt_wall_seconds","prior_failed_attempt_wall_seconds","cumulative_attempt_wall_seconds","gpu_peak_mib","row_output_bytes","forensic_output_bytes","total_attempt_output_bytes","storage_pre","storage_post","quarantine_receipts_sha256","failure_receipts_sha256","guards"}
    counts_exact=report.get("committed_rows_pre")==batch_id*20 and 0<=int(report.get("valid_partial_rows_pre",-1))<=20 and report.get("remaining_rows_pre")==200-report.get("committed_rows_pre")-report.get("valid_partial_rows_pre") and report.get("remaining_rows_post")==200-(batch_id+1)*20 and report.get("storage_pre",{}).get("remaining_rows")==report.get("remaining_rows_pre") and report.get("storage_post",{}).get("remaining_rows")==report.get("remaining_rows_post")
    report_exact=set(report)==report_keys and counts_exact and report.get("attempt_wall_seconds")==attempt and report.get("prior_failed_attempt_wall_seconds")==failure_wall and cumulative==attempt+failure_wall and report.get("row_output_bytes")==row_bytes and report.get("forensic_output_bytes")==forensic_bytes and report.get("total_attempt_output_bytes")==row_bytes+forensic_bytes and report.get("quarantine_receipts_sha256")==q and report.get("failure_receipts_sha256")==f and report.get("checks")==expected_checks
    if report.get("format")!=BATCH_FORMAT or report.get("passed") is not True or report.get("batch_id")!=batch_id or actual!=declared or counts!=report.get("transport_effect_context_counts_diagnostic") or contexts3!=report.get("contexts_at_least_three_of_four_effects_diagnostic") or report.get("effect_used_for_batch_pass_retry_filter") is not False or report.get("guards")!=guards or not report_exact or not storage_ok:raise RuntimeError(f"complete batch report drift {batch_id}")
    return report

def worker(lane,specs,collector_path,support,task,rows_dir,lower,upper,outq):
    os.sched_setaffinity(0,set(range(lane*6,(lane+1)*6)));os.environ.update({"OMP_NUM_THREADS":"3","MKL_NUM_THREADS":"3","OPENBLAS_NUM_THREADS":"3","NUMEXPR_NUM_THREADS":"3"})
    try:
        collector=module(Path(collector_path))
        for spec in specs:
            row,technical=collector.collect_one(spec,Path(support),Path(task),Path(rows_dir),lower,upper,timeout_seconds=900)
            outq.put({"kind":"row","lane":lane,"row":row,"technical":technical})
            if row is None or technical:return
        outq.put({"kind":"done","lane":lane})
    except BaseException as exc:outq.put({"kind":"error","lane":lane,"error":repr(exc),"traceback":traceback.format_exc()})

def stop(processes):
    initial=[p.pid for p in processes if p.pid]
    for p in processes:
        if p.is_alive():p.terminate()
    deadline=time.monotonic()+15
    for p in processes:p.join(max(0,deadline-time.monotonic()))
    for p in processes:
        if p.is_alive():p.kill()
    for p in processes:p.join(5)
    return {"initial_pids":initial,"exitcodes":[p.exitcode for p in processes],"alive_after_cleanup":[p.pid for p in processes if p.is_alive()]}

def failure(root,batch_id,exc,cleanup,peak,wall):
    base=forensic_root(root,batch_id)/"failures";base.mkdir(parents=True,exist_ok=True)
    path=base/("batch_failure_receipt-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+".json")
    atomic_json(path,{"format":"strict-track2-v478-batch-failure-receipt-v1","passed":False,"batch_id":batch_id,"error":repr(exc),"cleanup":cleanup,"cleanup_passed":not cleanup["alive_after_cleanup"],"gpu_peak_mib":peak,"attempt_wall_seconds":wall,"deleted_or_overwritten":False,"retry_based_on_effect_authorized":False,"full200_collection_authorized":False,"training_authorized":False,"s1_authorized":False,"policy_updates":0,"rl_authorized":False})

def batch_specs(pre,batch_id):
    specs=[]
    for raw in pre["contexts"]:
        if int(raw["batch_id"])!=batch_id:continue
        value=dict(raw);value["per_dim_action_lower"]=pre["per_dim_action_lower"];value["per_dim_action_upper"]=pre["per_dim_action_upper"];specs.append(value)
    return specs

def preflight(a,pre):
    if pre.get("format")!=PRE_FORMAT or pre.get("status")!="preregistered_public_train_temporal_collection_authorized":raise RuntimeError("v478 preregistration not authorized")
    closure=pre["execution_closure"]
    contract_path=Path(closure["phase_b_contract"]["path"]).resolve();contract=json.loads(contract_path.read_text())
    if sha(contract_path)!="a478b6f8e24d4a2312fcd4f62d294ad681b35ea4c19c1162f7dae7f9748047d1" or contract_path.name!="v478_temporal200_phase_b_frozen_contract_r4.json" or closure["phase_b_contract"]["sha256"]!=sha(contract_path) or contract.get("format")!="strict-track2-v478-public-train-temporal200-phase-b-contract-v1" or contract.get("status")!="phase_b_collection_semantics_authorized_execution_preregistration_pending":raise RuntimeError("v478 Phase-B contract drift")
    for key,path in (("collector",a.collector),("generator",Path(__file__).resolve())):
        if Path(closure[key]["path"]).resolve()!=path.resolve() or closure[key]["sha256"]!=sha(path):raise RuntimeError(f"v478 {key} drift")
    if Path(pre["dataset_root"]).resolve()!=a.output.resolve() or str(a.output.resolve())!="/root/v478_temporal8_dataset_seed1622_20260824":raise RuntimeError("v478 dataset root drift")
    support_digest,support_count=support_tree(a.support_root);resize_path=Path(closure["resize_source"]["path"]).resolve()
    if Path(closure["support_root"]["path"]).resolve()!=a.support_root.resolve() or closure["support_root"]["source_config_tree_sha256"]!=support_digest or closure["support_root"]["source_config_file_count"]!=support_count or Path(closure["task_config"]["path"]).resolve()!=a.task_config.resolve() or sha(a.task_config)!=closure["task_config"]["sha256"] or sha(resize_path)!=closure["resize_source"]["sha256"]:raise RuntimeError("v478 simulator closure drift")
    selection_path=Path(closure["selection"]["path"]).resolve();selection=json.loads(selection_path.read_text());selection_receipt_path=Path(closure["selection_receipt"]["path"]).resolve();selection_receipt=json.loads(selection_receipt_path.read_text())
    if sha(selection_path)!=closure["selection"]["sha256"] or sha(selection_receipt_path)!=closure["selection_receipt"]["sha256"] or selection_receipt.get("passed") is not True or selection_receipt.get("selection",{}).get("sha256")!=sha(selection_path):raise RuntimeError("v478 selection closure drift")
    union_path=Path(closure["endpoint_oracle_union"]["path"]).resolve();union=json.loads(union_path.read_text())
    v479_path=Path(closure["v479_reconciliation_receipt"]["path"]).resolve();v479=json.loads(v479_path.read_text())
    if sha(union_path)!=closure["endpoint_oracle_union"]["sha256"] or union.get("format")!="strict-track2-v478-temporal200-endpoint-oracle-union-v1" or union.get("rows_digest_sha256")!="3a2aceff3a81dfaa05d8122dbab83181c60071a18570a25a5ae4108463ec04ca" or union.get("counts")!={"batches":10,"contexts_per_batch":20,"total":200,"v461_reused":188,"v478_v479_reconciled":12}:raise RuntimeError("v478 endpoint oracle union drift")
    if sha(v479_path)!=closure["v479_reconciliation_receipt"]["sha256"] or v479.get("format")!="strict-track2-v479-v478-endpoint-oracle12-immutable-reconciliation-v1" or v479.get("passed") is not True or v479.get("authorization")!={"phase_b_temporal_collection_authorized":True,"training_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False}:raise RuntimeError("v479 authorization drift")
    contexts=pre.get("contexts",[])
    if len(contexts)!=200 or any(len(batch_specs(pre,b))!=20 for b in range(10)) or [(int(x["selection_order"]),int(x["episode"]),int(x["start"])) for x in contexts]!=[(i,int(x["episode"]),int(x["start"])) for i,x in enumerate(contexts)]:raise RuntimeError("v478 10x20/order drift")
    if pre.get("endpoint_oracle_legacy_alias_semantics")!="generic_bound_endpoint_oracle" or sum(x["endpoint_oracle"]["source"]=="v461" for x in contexts)!=188 or sum(x["endpoint_oracle"]["source"]=="v478_v479_reconciled" for x in contexts)!=12:raise RuntimeError("v478 oracle lineage drift")
    for spec in contexts:
        oracle=spec["endpoint_oracle"]
        if Path(spec["v461_endpoint_npz"]).resolve()!=Path(oracle["path"]).resolve() or spec["v461_endpoint_npz_sha256"]!=oracle["sha256"]:raise RuntimeError("v478 generic oracle alias drift")
    if len(selection.get("contexts",[]))!=200 or len(union.get("rows",[]))!=200:raise RuntimeError("v478 source rows drift")
    for i,(spec,selected,bound) in enumerate(zip(contexts,selection["contexts"],union["rows"])):
        common=("episode","start","batch_id","fold","phase_bin","motion_bin","history_action_sha256","future_action_sha256","branch_action_sha256","source_hdf5","source_hdf5_sha256","source_window","source_window_sha256")
        if any(spec[k]!=selected[k] for k in common) or int(bound["selection_order"])!=i or any(bound[k]!=selected[k] for k in ("episode","start","batch_id","fold","phase_bin","motion_bin","history_action_sha256","future_action_sha256","branch_action_sha256")) or Path(spec["endpoint_oracle"]["path"]).resolve()!=Path(bound["endpoint_npz"]["path"]).resolve() or spec["endpoint_oracle"]["sha256"]!=bound["endpoint_npz"]["sha256"] or Path(spec["endpoint_oracle"]["receipt_path"]).resolve()!=Path(bound["endpoint_receipt"]["path"]).resolve() or spec["endpoint_oracle"]["receipt_sha256"]!=bound["endpoint_receipt"]["sha256"]:raise RuntimeError(f"v478 context source mapping drift {i}")
    immutable_files={}
    for spec in contexts:
        for path_key,sha_key in (("source_hdf5","source_hdf5_sha256"),("source_window","source_window_sha256")):
            path=Path(spec[path_key]).resolve();expected=spec[sha_key]
            if path in immutable_files and immutable_files[path]!=expected:raise RuntimeError(f"v478 conflicting source SHA {path}")
            immutable_files[path]=expected
        oracle=spec["endpoint_oracle"]
        for path_key,sha_key in (("path","sha256"),("receipt_path","receipt_sha256")):
            path=Path(oracle[path_key]).resolve();expected=oracle[sha_key]
            if path in immutable_files and immutable_files[path]!=expected:raise RuntimeError(f"v478 conflicting oracle SHA {path}")
            immutable_files[path]=expected
    for path,expected in immutable_files.items():
        if not path.is_file() or sha(path)!=expected:raise RuntimeError(f"v478 full200 source prehash drift {path}")
    identities=[(int(x["episode"]),int(x["start"])) for x in contexts]
    quotas=len(set(identities))==200 and Counter(int(x["fold"]) for x in contexts)==Counter({i:40 for i in range(5)}) and Counter(int(x["phase_bin"]) for x in contexts)==Counter({i:40 for i in range(5)}) and Counter(int(x["motion_bin"]) for x in contexts)==Counter({i:50 for i in range(4)})
    quotas &= all(Counter(int(x["fold"]) for x in batch_specs(pre,b))==Counter({i:4 for i in range(5)}) and Counter(int(x["phase_bin"]) for x in batch_specs(pre,b))==Counter({i:4 for i in range(5)}) and Counter(int(x["motion_bin"]) for x in batch_specs(pre,b))==Counter({i:5 for i in range(4)}) for b in range(10))
    quotas &= all(Counter(int(x["phase_bin"]) for x in contexts if int(x["fold"])==fold)==Counter({i:8 for i in range(5)}) and Counter(int(x["motion_bin"]) for x in contexts if int(x["fold"])==fold)==Counter({i:10 for i in range(4)}) for fold in range(5))
    if not quotas:raise RuntimeError("v478 quota/identity drift")
    guards=pre.get("guards",{})
    if guards.get("effect_used_for_retry_filter_batch_gate") is not False or guards.get("training_authorized") is not False or guards.get("s1_authorized") is not False or guards.get("zero_update_authorized") is not False or guards.get("policy_updates")!=0 or guards.get("rl_authorized") is not False:raise RuntimeError("v478 unsafe authorization")

def main():
    invocation_start=time.monotonic()
    p=argparse.ArgumentParser()
    for n in ("preregistration","collector","support-root","task-config","output"):p.add_argument("--"+n,type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--batch-id",type=int);mode.add_argument("--finalize",action="store_true")
    a=p.parse_args();pre=json.loads(a.preregistration.read_text());preflight(a,pre);root=a.output
    if a.finalize:
        reports=[validate_batch(root/f"batch_{b:03d}",b,batch_specs(pre,b)) for b in range(10)]
        counts={n:sum(r["transport_effect_context_counts_diagnostic"][n] for r in reports) for n in TRANSPORT};contexts_at_least3=sum(int(r["contexts_at_least_three_of_four_effects_diagnostic"]) for r in reports);wall=sum(float(r["cumulative_attempt_wall_seconds"]) for r in reports)
        output_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        integrity_checks={"ten_batches":len(reports)==10,"exact200":sum(len(r["rows"]) for r in reports)==200,"all_batch_integrity_passed":all(r["passed"] is True for r in reports),"whole_cumulative_attempt_wall_le_43200":wall<=43200,"total_output_including_forensics_le_3221225472":output_bytes<=3221225472,"no_partial_batches":not any(root.glob("batch_*.partial")),"no_effect_used_for_batch_pass_retry_filter":all(r["effect_used_for_batch_pass_retry_filter"] is False for r in reports)}
        collection_integrity_passed=all(integrity_checks.values())
        effect_checks={"all_200_contexts_have_at_least_three_of_four_transport_effects":contexts_at_least3==200,"each_named_transport_effect_contexts_at_least_150":all(counts[n]>=150 for n in TRANSPORT),"computed_after_all200_integrity_complete":collection_integrity_passed,"never_used_for_batch_stop_resume_retry_filter":all(r["effect_used_for_batch_pass_retry_filter"] is False for r in reports)}
        effect_passed=all(effect_checks.values());passed=collection_integrity_passed and effect_passed
        atomic_json(root/"generation_report.json",{"format":FINAL_FORMAT,"passed":passed,"collection_integrity_passed":collection_integrity_passed,"technical_effect_diagnostic_qualification_passed":effect_passed,"integrity_checks":integrity_checks,"technical_effect_diagnostic_checks":effect_checks,"batch_reports_sha256":{f"batch_{b:03d}":sha(root/f"batch_{b:03d}"/"batch_report.json") for b in range(10)},"transport_effect_context_counts_diagnostic":counts,"contexts_at_least_three_of_four_effects_diagnostic":contexts_at_least3,"cumulative_attempt_wall_seconds":wall,"output_bytes_including_forensics":output_bytes,"preregistration_sha256":sha(a.preregistration),"generator_sha256":sha(Path(__file__).resolve()),"collector_sha256":sha(a.collector),"guards":{"complete_dataset_retained_on_diagnostic_failure":True,"retry_authorized_on_diagnostic_failure":False,"temporal_model_design_authorized_if_pass":passed,"training_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False}})
        return 0 if passed else 2
    if a.batch_id is None or not 0<=a.batch_id<10:raise RuntimeError("batch-id 0..9 required")
    initial=not root.exists()
    if initial:storage_gate(root,200,True)
    root.mkdir(parents=True,exist_ok=True);bid=a.batch_id;specs=batch_specs(pre,bid);complete=root/f"batch_{bid:03d}";partial=root/f"batch_{bid:03d}.partial"
    committed_rows=0
    complete_ids=set()
    for prior in range(10):
        prior_path=root/f"batch_{prior:03d}"
        if prior_path.exists():validate_batch(prior_path,prior,batch_specs(pre,prior));committed_rows+=20;complete_ids.add(prior)
    if complete.exists():return 0
    if complete_ids!=set(range(bid)):raise RuntimeError(f"v478 batches must run/resume in frozen order before {bid}: {sorted(complete_ids)}")
    if not partial.exists():partial.mkdir();(partial/"rows").mkdir();fsync_dir(root)
    for stale_top in list(partial.iterdir()):
        if stale_top.name not in {"rows","batch_report.json"}:quarantine(stale_top,root,bid,"unexpected stale partial-batch entry")
    if not (partial/"rows").is_dir():raise RuntimeError("v478 partial rows directory missing")
    if (partial/"batch_report.json").exists():
        stale=json.loads((partial/"batch_report.json").read_text())
        if stale.get("passed") is not True:
            raise RuntimeError(f"v478 batch {bid} has an immutable failed report; retry is not authorized")
        # A crash may occur after the report fsync and before the directory
        # rename.  Rebuild every row and the report closure, then promote.
        validate_batch(partial,bid,specs)
        os.replace(partial,complete);fsync_dir(root);validate_batch(complete,bid,specs);return 0
    rows_dir=partial/"rows"
    for pth in list(rows_dir.iterdir()):
        matching=next((s for s in specs if stem(s)==pth.name),None)
        try:
            if matching is None or not pth.is_dir():raise RuntimeError("unexpected row")
            validate_row(pth,matching)
        except Exception as exc:quarantine(pth,root,bid,"invalid stale row: "+repr(exc))
    valid_partial=sum((rows_dir/stem(s)).is_dir() for s in specs);storage_pre=storage_gate(root,200-committed_rows-valid_partial,False)
    remaining=[s for s in specs if not (rows_dir/stem(s)).exists()];start=invocation_start;peak=gpu_mib()
    if peak is None:raise RuntimeError("v478 GPU query failed")
    ctx=mp.get_context("spawn");q=ctx.Queue();lanes=[remaining[0::2],remaining[1::2]];procs=[ctx.Process(target=worker,args=(i,lanes[i],str(a.collector.resolve()),str(a.support_root.resolve()),str(a.task_config.resolve()),str(rows_dir),pre["per_dim_action_lower"],pre["per_dim_action_upper"],q)) for i in range(2)];done=set();fatal=None
    try:
        for proc in procs:proc.start()
        while len(done)<2 and fatal is None:
            if time.monotonic()-start>5400:fatal=TimeoutError("v478 batch timeout");break
            used=gpu_mib()
            if used is None or used>24576:fatal=RuntimeError(f"v478 GPU gate {used}");break
            peak=max(peak,used)
            try:item=q.get(timeout=1)
            except queue.Empty:
                bad=[p.exitcode for p in procs if not p.is_alive() and p.exitcode not in (None,0)]
                if bad:fatal=RuntimeError(f"v478 worker exit {bad}")
                elif all(not p.is_alive() for p in procs) and len(done)<2:fatal=RuntimeError(f"v478 all workers exited without complete done messages: done={sorted(done)} exitcodes={[p.exitcode for p in procs]}")
                continue
            if item["kind"]=="done":done.add(item["lane"])
            elif item["kind"]=="error":fatal=RuntimeError(item["error"]+"\n"+item["traceback"])
            elif item["row"] is None or item["technical"]:fatal=RuntimeError(f"v478 technical row failure {item}")
    finally:cleanup=stop(procs)
    if fatal is not None:failure(root,bid,fatal,cleanup,peak,time.monotonic()-start);raise fatal
    rows=row_receipts(partial,specs);report=batch_report(partial,bid,specs,rows,time.monotonic()-start,peak,root,storage_pre,committed_rows,valid_partial);atomic_json(partial/"batch_report.json",report)
    if not report["passed"]:raise RuntimeError(f"v478 batch {bid} technical short gate failed")
    os.replace(partial,complete);fsync_dir(root);validate_batch(complete,bid,specs);return 0

if __name__=="__main__":raise SystemExit(main())
