#!/usr/bin/env python3
"""Action/index-only v478 temporal-200 selection materializer.

The only NPZ arrays read are ``history_actions`` and ``future_actions`` from
the authoritative public windows.  RGB, reward, success and outcome are never
read.  Window availability and action identity are technical eligibility
conditions applied before bins or MILPs are constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SEED = 1622
EXPECTED_V461_SELECTION_SHA256 = "edf2ea5e91c7b6a464c62a5d3c042c11bc69d4ca851cc03c4879cad3dcf55919"
EXPECTED_SCIPY_VERSION = "1.15.3"
EXPECTED_HIGHS_VERSION = "1.8.0"
BRANCHES = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4", "factual_duplicate")
FOLDS = (
    (28, 37, 49),
    (15, 20, 42),
    (25, 40, 46),
    (12, 33, 47),
    (30, 32, 44),
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def stable_key(episode: int, start: int) -> tuple[str, int]:
    value = hashlib.sha256(f"{SEED}:{episode}:{start}".encode()).hexdigest()
    return value, int(value, 16)


def branch_actions(history: np.ndarray, future: np.ndarray) -> dict[str, np.ndarray]:
    anchor = history[-1, 7:13]
    delta = future[:, 7:13] - anchor
    result = {"factual": future.copy()}
    for name, scale in (("no_transport", 0.0), ("scale_0p4", 0.4), ("scale_1p25", 1.25), ("reverse_direction_0p4", -0.4)):
        value = future.copy()
        value[:, 7:13] = anchor + scale * delta
        result[name] = value
    result["factual_duplicate"] = future.copy()
    return result


def add_constraint(rows, cols, values, lower, upper, indices, lo, hi):
    row = len(lower)
    for index in indices:
        rows.append(row); cols.append(index); values.append(1.0)
    lower.append(float(lo)); upper.append(float(hi))


def milp_exact(cost, rows, cols, values, lower, upper, label, allow_failure=False):
    import scipy
    import scipy.optimize._highspy._core as highspy_core
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    hv = f"{highspy_core.HIGHS_VERSION_MAJOR}.{highspy_core.HIGHS_VERSION_MINOR}.{highspy_core.HIGHS_VERSION_PATCH}"
    if scipy.__version__ != EXPECTED_SCIPY_VERSION or hv != EXPECTED_HIGHS_VERSION:
        raise RuntimeError(f"solver drift scipy={scipy.__version__} highs={hv}")
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), len(cost))).tocsr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = milp(c=np.asarray(cost, dtype=np.float64), integrality=np.ones(len(cost), dtype=np.int8),
                      bounds=Bounds(np.zeros(len(cost)), np.ones(len(cost))),
                      constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                      options={"time_limit": 300.0, "mip_rel_gap": 0.0, "presolve": True,
                               "threads": 1, "parallel": False, "random_seed": SEED})
    canonical = []
    for item in caught:
        raw = str(item.message)
        if raw.startswith("Unrecognized options detected:") and "passed to HiGHS verbatim" in raw:
            canonical.append({"code": "unrecognized_options_forwarded_to_highs", "options": ["parallel", "random_seed", "threads"]})
        else:
            raise RuntimeError(f"unexpected solver warning: {raw}")
    meta={"label": label, "scipy_version": scipy.__version__, "highs_version": hv,
                      "threads": 1, "parallel": False, "random_seed": SEED, "objective": None if result.fun is None else float(result.fun),
                      "solver_warnings": canonical, "status": int(result.status), "message": str(result.message),
                      "success":bool(result.success and result.x is not None)}
    if not meta["success"]:
        if allow_failure: return None,meta
        raise RuntimeError(f"{label} infeasible: {result.message}")
    return result.x,meta


def select200(candidates: list[dict], reusable: set[tuple[int, int]]):
    rows=[]; cols=[]; values=[]; lower=[]; upper=[]
    for fold in range(5):
        add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["fold"]==fold], 40, 40)
        # Preserve the authoritative v461 fold-local marginal allocation.
        for b in range(5):
            add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["fold"]==fold and r["phase_bin"]==b], 8, 8)
        for b in range(4):
            add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["fold"]==fold and r["motion_bin"]==b], 10, 10)
    for b in range(5):
        add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["phase_bin"]==b], 40, 40)
    for b in range(4):
        add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["motion_bin"]==b], 50, 50)
    episodes = sorted(e for fold in FOLDS for e in fold)
    for episode in episodes:
        add_constraint(rows, cols, values, lower, upper, [i for i,r in enumerate(candidates) if r["episode"]==episode], 6, 17)
        by_start = sorted([i for i,r in enumerate(candidates) if r["episode"]==episode], key=lambda i:candidates[i]["start"])
        for left,right in zip(by_start, by_start[1:]):
            if candidates[right]["start"]-candidates[left]["start"] < 2:
                add_constraint(rows, cols, values, lower, upper, [left,right], -np.inf, 1)
    # Prove that an episode-count range <=10 is impossible under the frozen
    # eligibility/quotas.  Since the 6..17 bounds imply range<=11, the later
    # feasible solution then has the exact pre-registered minimum range 11.
    r10,c10,v10,l10,u10=list(rows),list(cols),list(values),list(lower),list(upper)
    for ai,a in enumerate(episodes):
        ia=[i for i,r in enumerate(candidates) if r["episode"]==a]
        for b in episodes[ai+1:]:
            ib=[i for i,r in enumerate(candidates) if r["episode"]==b]
            row=len(l10)
            for i in ia: r10.append(row); c10.append(i); v10.append(1.0)
            for i in ib: r10.append(row); c10.append(i); v10.append(-1.0)
            l10.append(-10.0); u10.append(10.0)
    range10_solution,range10_meta=milp_exact(np.zeros(len(candidates)),r10,c10,v10,l10,u10,"episode_range_le_10_infeasibility_proof",allow_failure=True)
    if range10_solution is not None: raise RuntimeError("episode range<=10 unexpectedly feasible")
    # Frozen lexicographic objective: first maximize reuse of already-valid
    # v461 endpoint-oracle rows, then minimize the full-SHA rank sum.  The
    # first MILP is retained as an explicit proof of the attainable maximum.
    new_cost = [0 if (r["episode"], r["start"]) in reusable else 1 for r in candidates]
    reuse_solution, reuse_meta = milp_exact(new_cost, rows, cols, values, lower, upper, "maximize_valid_v461_oracle_reuse")
    minimum_new = int(round(sum(c * x for c, x in zip(new_cost, reuse_solution))))
    new_indices = [i for i,c in enumerate(new_cost) if c]
    add_constraint(rows, cols, values, lower, upper, new_indices, minimum_new, minimum_new)
    solution, rank_meta = milp_exact([r["cost_rank"]+1 for r in candidates], rows, cols, values, lower, upper, "rank_min_at_maximum_oracle_reuse")
    selected=[dict(r) for r,take in zip(candidates,solution) if take>.5]
    if len(selected)!=200: raise RuntimeError(f"selected {len(selected)} != 200")
    counts=[sum(r["episode"]==e for r in selected) for e in episodes]
    if max(counts)-min(counts)!=11: raise RuntimeError("selected episode range is not proven exact11")
    return selected,{"episode_range_le_10_infeasibility_proof":range10_meta,"minimum_episode_range":11,"maximum_reused_valid_v461_oracles":200-minimum_new,"minimum_new_endpoint_oracles":minimum_new,"reuse_proof_solver":reuse_meta,"rank_solver":rank_meta}


def assign_batches(selected: list[dict]):
    variables=[(i,b) for i in range(200) for b in range(10)]
    pos={pair:i for i,pair in enumerate(variables)}
    keys=[]
    for i,b in variables:
        r=selected[i]; keys.append((int(hashlib.sha256(f"{SEED}:batch:{b}:{r['episode']}:{r['start']}".encode()).hexdigest(),16),r["episode"],r["start"],b))
    ranks={idx:rank for rank,idx in enumerate(sorted(range(len(variables)),key=lambda i:keys[i]))}
    rows=[]; cols=[]; values=[]; lower=[]; upper=[]
    for i in range(200): add_constraint(rows,cols,values,lower,upper,[pos[(i,b)] for b in range(10)],1,1)
    for b in range(10):
        add_constraint(rows,cols,values,lower,upper,[pos[(i,b)] for i in range(200)],20,20)
        for f in range(5): add_constraint(rows,cols,values,lower,upper,[pos[(i,b)] for i,r in enumerate(selected) if r["fold"]==f],4,4)
        for p in range(5): add_constraint(rows,cols,values,lower,upper,[pos[(i,b)] for i,r in enumerate(selected) if r["phase_bin"]==p],4,4)
        for m in range(4): add_constraint(rows,cols,values,lower,upper,[pos[(i,b)] for i,r in enumerate(selected) if r["motion_bin"]==m],5,5)
    solution,meta=milp_exact([ranks[i]+1 for i in range(len(variables))],rows,cols,values,lower,upper,"exact_batch10x20")
    for v,take in enumerate(solution):
        if take>.5:
            i,b=variables[v]
            if "batch_id" in selected[i]: raise RuntimeError("duplicate batch assignment")
            selected[i]["batch_id"]=b
    if any("batch_id" not in r for r in selected): raise RuntimeError("missing batch assignment")
    return meta


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--v461-selection",type=Path,required=True)
    ap.add_argument("--windows",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if file_sha256(args.v461_selection)!=EXPECTED_V461_SELECTION_SHA256: raise RuntimeError("v461 selection SHA drift")
    old=json.loads(args.v461_selection.read_text())
    if old.get("format")!="strict-track2-v461-endpoint200-preregistration-v1": raise RuntimeError("v461 format drift")
    episode_meta={}
    for r in old["contexts"]:
        episode_meta.setdefault(int(r["episode"]), {"dataset_seed":r["dataset_seed"],"instruction":r["instruction"],"source_hdf5":r["source_hdf5"],"source_hdf5_sha256":r["source_hdf5_sha256"]})
    fold_by_episode={e:f for f,eps in enumerate(FOLDS) for e in eps}
    train_hdf5=[Path(old["contexts"][0]["source_hdf5"]).parent/f"episode{e}.hdf5" for e in range(50) if (Path(old["contexts"][0]["source_hdf5"]).parent/f"episode{e}.hdf5").is_file()]
    actions={}
    for episode in sorted(fold_by_episode):
        path=Path(episode_meta[episode]["source_hdf5"])
        if path.resolve()!=path.parent.resolve()/f"episode{episode}.hdf5" or file_sha256(path)!=episode_meta[episode]["source_hdf5_sha256"]: raise RuntimeError(f"HDF5 closure drift ep{episode}")
        with h5py.File(path,"r") as h:
            raw=np.asarray(h["joint_action/vector"])
        if raw.dtype!=np.float64 or raw.ndim!=2 or raw.shape[1]!=14 or not np.isfinite(raw).all():
            raise RuntimeError(f"raw HDF5 action dtype/shape/finite drift ep{episode}: {raw.dtype} {raw.shape}")
        actions[episode]=raw.astype(np.float32)
    # Bounds are frozen from all public train HDF5 actions, as in v461.  The
    # exact vectors are inherited and verified by the old receipt.
    lower=np.asarray(old["per_dim_action_lower"],dtype=np.float32); upper=np.asarray(old["per_dim_action_upper"],dtype=np.float32)
    candidates=[]; rejected_missing=0; rejected_action=0
    for episode,source in actions.items():
        closed=np.flatnonzero(source[:,13]<.5)
        if not len(closed): raise RuntimeError(f"no close ep{episode}")
        first_close=int(closed[0])
        for start in range(len(source)-11):
            history=source[start:start+4]; future=source[start+4:start+12]
            if not(history[-1,13]<.5 and np.all(future[:,13]<.5)) or not(np.isfinite(history).all() and np.isfinite(future).all()): continue
            branches=branch_actions(history,future)
            if any(not np.isfinite(v).all() or np.any(v<lower) or np.any(v>upper) for v in branches.values()): continue
            ed={n:float(np.linalg.norm(v[-1,7:13]-future[-1,7:13])) for n,v in branches.items() if n not in("factual","factual_duplicate")}
            pd={n:float(np.linalg.norm(v[:,7:13]-future[:,7:13])) for n,v in branches.items() if n not in("factual","factual_duplicate")}
            if min(ed.values())<.01 or min(pd.values())<.01: continue
            window=args.windows/f"episode{episode}_{start:05d}.npz"
            if not window.is_file(): rejected_missing+=1; continue
            with np.load(window,allow_pickle=False) as z:
                if "history_actions" not in z.files or "future_actions" not in z.files: rejected_action+=1; continue
                wh=np.asarray(z["history_actions"]); wf=np.asarray(z["future_actions"])
            if wh.dtype!=np.float32 or wf.dtype!=np.float32 or wh.shape!=history.shape or wf.shape!=(8,14) or not np.array_equal(wh,history) or not np.array_equal(wf,future): rejected_action+=1; continue
            anchor=history[-1,7:13]; right_path=np.vstack((anchor,future[:,7:13]))
            motion=float(np.linalg.norm(np.diff(right_path,axis=0),axis=1).sum())
            phase=float((start+4-first_close)/max(1,len(source)-1-first_close))
            sha,uint=stable_key(episode,start)
            candidates.append({"episode":episode,"start":start,"fold":fold_by_episode[episode],"phase":phase,"motion":motion,"endpoint_diffs":ed,"path_diffs":pd,"history_action_sha256":array_sha256(history),"future_action_sha256":array_sha256(future),"branch_action_sha256":{n:array_sha256(v) for n,v in branches.items()},"source_window":str(window.resolve()),"source_window_sha256":file_sha256(window),"cost_sha256":sha,"cost_uint256":uint})
    coverage={}
    for fold in range(5):
        fr=[r for r in candidates if r["fold"]==fold]
        ps=sorted(fr,key=lambda r:(r["phase"],r["cost_uint256"],r["episode"],r["start"]))
        for rank,r in enumerate(ps): r["phase_bin"]=min(4,5*rank//len(ps))
        for p in range(5):
            mr=sorted([r for r in fr if r["phase_bin"]==p],key=lambda r:(r["motion"],r["cost_uint256"],r["episode"],r["start"]))
            for rank,r in enumerate(mr): r["motion_bin"]=min(3,4*rank//len(mr))
        coverage[str(fold)]={"eligible":len(fr),"phase":[sum(r["phase_bin"]==p for r in fr) for p in range(5)],"conditional_motion":[sum(r["motion_bin"]==m for r in fr) for m in range(4)]}
    for rank,r in enumerate(sorted(candidates,key=lambda r:(r["cost_uint256"],r["episode"],r["start"]))): r["cost_rank"]=rank
    old_context_by_id={(int(r["episode"]),int(r["start"])):r for r in old["contexts"]}
    old_selected=set(old_context_by_id)
    # Every candidate has already passed source-window availability/action
    # identity; intersection therefore means a directly reusable v461 oracle.
    reusable=old_selected & {(r["episode"],r["start"]) for r in candidates}
    selected,sel_meta=select200(candidates,reusable); batch_meta=assign_batches(selected)
    selected.sort(key=lambda r:(r["batch_id"],r["fold"],r["phase_bin"],r["motion_bin"],r["episode"],r["start"]))
    for r in selected:
        r.pop("cost_uint256"); r.update(episode_meta[r["episode"]])
        identity=(r["episode"],r["start"])
        if identity in reusable:
            old_row=old_context_by_id[identity]
            row_dir=args.v461_selection.parent/"dataset"/f"batch_{int(old_row['batch_id']):03d}"/"rows"/f"episode{r['episode']}_start{r['start']:05d}"
            endpoint=row_dir/"endpoint.npz"; receipt=row_dir/"receipt.json"
            if not endpoint.is_file() or not receipt.is_file(): raise RuntimeError(f"missing reusable oracle row {identity}")
            receipt_payload=json.loads(receipt.read_text())
            if receipt_payload.get("npz_sha256")!=file_sha256(endpoint): raise RuntimeError(f"reusable oracle receipt mismatch {identity}")
            r["endpoint_oracle"]={"source":"reused_v461","npz_path":str(endpoint.resolve()),"npz_sha256":file_sha256(endpoint),"receipt_path":str(receipt.resolve()),"receipt_sha256":file_sha256(receipt)}
        else:
            r["endpoint_oracle"]={"source":"v478_phase_a_required","npz_path":None,"npz_sha256":None,"receipt_path":None,"receipt_sha256":None}
    def counts(key,n): return [sum(r[key]==i for r in selected) for i in range(n)]
    checks={"exact200":len(selected)==200,"fold40":counts("fold",5)==[40]*5,"phase40":counts("phase_bin",5)==[40]*5,"motion50":counts("motion_bin",4)==[50]*4,"fold_phase_exact8":all(sum(r["fold"]==f and r["phase_bin"]==p for r in selected)==8 for f in range(5) for p in range(5)),"fold_motion_exact10":all(sum(r["fold"]==f and r["motion_bin"]==m for r in selected)==10 for f in range(5) for m in range(4)),"all_window_eligible":all(Path(r["source_window"]).is_file() for r in selected)}
    for b in range(10):
        br=[r for r in selected if r["batch_id"]==b]
        checks[f"batch{b:02d}_20_fold4_phase4_motion5"]=len(br)==20 and [sum(r["fold"]==i for r in br) for i in range(5)]==[4]*5 and [sum(r["phase_bin"]==i for r in br) for i in range(5)]==[4]*5 and [sum(r["motion_bin"]==i for r in br) for i in range(4)]==[5]*4
    if not all(checks.values()): raise RuntimeError(f"post audit failed {checks}")
    selected_ids={(r["episode"],r["start"]) for r in selected}
    hdf5_closure=[{"episode":e,"path":episode_meta[e]["source_hdf5"],"sha256":episode_meta[e]["source_hdf5_sha256"],"raw_dtype":"float64","raw_shape":list(actions[e].shape),"canonical_dtype":"float32","finite":True} for e in sorted(actions)]
    payload={"format":"strict-track2-v478-window-eligible-action-only-selection-v1","status":"selected_action_only_not_collected","classification":"public train technical selection; no RGB/reward/success/outcome/model/policy/RL/submit authority","created_at":datetime.now(timezone.utc).isoformat(),"seed":SEED,"source":{"v461_selection_path":str(args.v461_selection.resolve()),"v461_selection_sha256":EXPECTED_V461_SELECTION_SHA256,"windows_root":str(args.windows.resolve()),"right15_hdf5_closure":hdf5_closure},"technical_eligibility":{"applied_before_binning_and_milp":True,"window_file_exists":True,"source_hdf5_raw_action_dtype":"float64","canonical_action_dtype":"float32","window_raw_action_dtype":"float32","history_action_shape":[4,14],"future_action_shape":[8,14],"history_actions_bitexact_canonical_hdf5":True,"future8_actions_bitexact_canonical_hdf5":True,"window_full_file_sha_bound":True,"rgb_reward_success_outcome_read":False,"rejected_missing_window":rejected_missing,"rejected_action_identity":rejected_action,"eligible_count":len(candidates)},"binning":"fold-local phase quintiles then conditional motion quartiles over the window-eligible pool","quotas":{"selected":200,"fold":[40]*5,"phase":[40]*5,"motion":[50]*4,"per_fold_phase":[8]*5,"per_fold_motion":[10]*4,"batches":10,"batch_size":20,"per_batch_fold":[4]*5,"per_batch_phase":[4]*5,"per_batch_motion":[5]*4,"episode_bounds":[6,17],"minimum_episode_range":11,"nonadjacent_starts":True},"oracle_reuse":{"valid_v461_candidates":len(reusable),"selected_reused":len(selected_ids & reusable),"selected_new":len(selected_ids-reusable),"new_contexts":[{"episode":e,"start":s} for e,s in sorted(selected_ids-reusable)]},"eligible_coverage":coverage,"checks":checks,"selection_solver":sel_meta,"batch_solver":batch_meta,"selected_coverage":{"fold":counts("fold",5),"phase":counts("phase_bin",5),"motion":counts("motion_bin",4),"episode":{str(e):sum(r["episode"]==e for r in selected) for e in sorted(fold_by_episode)}},"contexts":selected,"guards":{"simulator_started":False,"training_started":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False,"hidden_final_submission_used":False}}
    if args.output.exists(): raise RuntimeError("refusing overwrite")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"sha256":file_sha256(args.output),"eligible":len(candidates),"coverage":payload["selected_coverage"]},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
