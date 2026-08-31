"""Frozen CPU-only outer-five-fold v471 group-context prototype KNN4 S0."""
from __future__ import annotations
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import probe_v469_closed_form_residual_knn_s0 as base
BRANCHES=base.BRANCHES;FORMAT="strict-track2-v471-group-context-prototype-knn-s0-v1"
def sha(p):return base.sha(p)
def atomic(p,v):return base.atomic_json(p,v)
def cache_digests(rows,pre):
 manifest=[{"episode":x["episode"],"start":x["start"],"seed":int(base.stable_seed(x["episode"],x["start"])),"action_sha":[x["keys"][b][0] for b in range(5)]} for x in rows]
 order=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 closure=hashlib.sha256(json.dumps(pre["v469"]["preregistration"]["v468"]["v169"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
 return order,closure
def select4(distance,keys):
 order=np.lexsort((np.asarray([x[1] for x in keys]),np.asarray([x[0] for x in keys]),np.asarray(distance,np.float64)));return order[:4]
def predict_group(rows,requested,canonical,pre,fold):
 fit=[i for i,x in enumerate(rows) if x["fold"]!=fold];hold=[i for i,x in enumerate(rows) if x["fold"]==fold];Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=Y[:,1]-canonical.astype(np.float32);E=(Y-Y[:,1:2])-(requested.astype(np.float32)-canonical[:,None].astype(np.float32));E[:,1]=0
 V=np.stack([rows[i]["context_feature"] for i in fit]);vm=V.mean(0);vs=np.maximum(V.std(0),1e-4);ckeys=[rows[i]["keys"][1] for i in fit];X=np.concatenate([rows[i]["action_feature"] for i in fit]);xm=X.mean(0);xs=np.maximum(X.std(0),1e-4);mapping=pre["v469"]["preregistration"]["v468"]["phase_shuffle"]["mappings"][fold];lookup={(x["episode"],x["start"]):i for i,x in enumerate(rows)}
 if int(mapping["fold"])!=fold:raise RuntimeError("v471 phase mapping fold")
 out={m:np.empty((len(hold),5,256,256,3),np.float32) for m in ("action","context_only","phase_shuffle")}
 for q,i in enumerate(hold):
  dc=np.square((V-rows[i]["context_feature"])/vs).mean(1);chosen=[fit[j] for j in select4(dc,ckeys)];chat=C[chosen].mean(0);donor=mapping["holdout"][f"{rows[i]['episode']}:{rows[i]['start']}"];di=lookup[(int(donor["donor_episode"]),int(donor["donor_start"]))]
  for b in range(5):
   nt=np.array_equal(rows[i]["future"][b,:,7:13],np.broadcast_to(rows[i]["history"][-1,7:13],(8,6)))
   estimates={}
   if nt:estimates={m:0. for m in out}
   else:
    for m,query in (("action",rows[i]["action_feature"][b]),("phase_shuffle",rows[di]["action_feature"][b])):
     maps=[]
     for ci in chosen:
      dist=np.square((rows[ci]["action_feature"]-query)/xs).mean(1);idx=np.lexsort((np.asarray([x[1] for x in rows[ci]["keys"]]),np.asarray([x[0] for x in rows[ci]["keys"]]),dist.astype(np.float64)))[0];maps.append(E[ci,idx])
     estimates[m]=np.mean(maps,axis=0)
    estimates["context_only"]=np.mean([E[ci].mean(0) for ci in chosen],axis=0)
   for m in out:out[m][q,b]=np.clip(requested[i,b].astype(np.float32)+chat+estimates[m],0,255)
 return hold,out,Y[hold],requested[hold]
def aggregate(values,folds):
 ae=np.concatenate([x[0] for x in values]);be=np.concatenate([x[1] for x in values]);ce=np.concatenate([x[2] for x in values]);pe=np.concatenate([x[3] for x in values]);ue=np.concatenate([x[4] for x in values]);branch={BRANCHES[b]:float(np.concatenate([x[0][b::5] for x in values]).mean()/np.concatenate([x[1][b::5] for x in values]).mean()) for b in range(5)}
 return {"passing_folds":sum(x["passed"] for x in folds),"continuous_ratio":float(ae.mean()/be.mean()),"uint8_ratio":float(ue.mean()/be.mean()),"branch_ratio":branch,"action_over_context_only":float(ae.mean()/ce.mean()),"action_over_phase_shuffle":float(ae.mean()/pe.mean()),"positive_delta_cosine_fraction":float(np.mean([x["positive_delta_cosine_fraction"] for x in folds])),"nonzero_prediction_delta_fraction":float(np.mean([x["nonzero_prediction_delta_fraction"] for x in folds])),"improved_episodes":int(sum(x["episode_wins"] for x in folds))}
def aggregate_pass(a):
 b=a["branch_ratio"];return a["passing_folds"]>=4 and a["continuous_ratio"]<=.95 and a["uint8_ratio"]<=1 and b["factual"]<=1 and all(b[x]<=.98 for x in BRANCHES[1:]) and a["action_over_context_only"]<=.95 and a["action_over_phase_shuffle"]<=.97 and a["positive_delta_cosine_fraction"]>=.6 and a["nonzero_prediction_delta_fraction"]>=.95 and a["improved_episodes"]>=12
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","contract","selection","dataset","output-dir"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());out=a.output_dir;old=pre["v469"]["preregistration"]
 if pre.get("format")!="strict-track2-v471-group-context-prototype-knn-preregistration-v1" or sha(a.contract)!=pre["contract"]["sha256"] or sha(Path(__file__))!=pre["source"]["probe_sha256"] or Path(base.__file__).resolve()!=Path(pre["source"]["v469_probe_path"]).resolve() or sha(base.__file__)!=pre["source"]["v469_probe_sha256"] or out.exists():raise RuntimeError("v471 closure/output")
 if sha(a.selection)!=old["v461"]["selection_sha256"] or a.dataset.resolve()!=Path(old["v461"]["dataset"]).resolve() or any(sha(a.dataset/x["relative"])!=x["sha256"] for x in old["v461"]["files"]):raise RuntimeError("v471 data drift")
 rows=base.load_rows(old,a.dataset,json.loads(a.selection.read_text()));cache=Path(pre["v469"]["root"])/"result/v169_endpoint_cache.npz"
 if sha(cache)!=pre["v469"]["immutable_sha256"]["result/v169_endpoint_cache.npz"]:raise RuntimeError("v471 cache file drift")
 with np.load(cache,allow_pickle=False) as z:requested=z["requested"].copy();canonical=z["canonical"].copy();seeds=z["seeds"].copy()
 cr=pre["v469"]["cache_receipt"];order_digest,closure_digest=cache_digests(rows,pre)
 if requested.shape!=(200,5,256,256,3) or canonical.shape!=(200,256,256,3) or requested.dtype!=np.uint8 or canonical.dtype!=np.uint8 or seeds.shape!=(200,) or base.arrsha(requested)!=cr["requested_array_sha256"] or base.arrsha(canonical)!=cr["canonical_array_sha256"] or base.arrsha(seeds)!=cr["seed_array_sha256"] or order_digest!=cr["ordered_request_manifest_sha256"] or closure_digest!=cr["v169_closure_digest"] or not np.array_equal(seeds,np.asarray([base.stable_seed(x["episode"],x["start"]) for x in rows],np.int64)):raise RuntimeError("v471 cache arrays drift")
 out.mkdir(parents=True);folds=[];values=[];failed=0
 for fold in range(5):
  hold,pred,target,requested_hold=predict_group(rows,requested,canonical,pre,fold);receipt,*v=base.fold_metrics(fold,hold,pred,target,requested_hold,rows);folds.append(receipt);values.append(v);atomic(out/f"fold{fold}_receipt.json",receipt);failed+=not receipt["passed"]
  if failed>=2:break
 early=failed>=2;agg=None;passed=False;library=None
 if not early and len(folds)==5:
  agg=aggregate(values,folds);passed=aggregate_pass(agg)
  if passed:
   Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=(Y[:,1]-canonical.astype(np.float32)).astype(np.int16);E=((Y-Y[:,1:2])-(requested.astype(np.float32)-canonical[:,None].astype(np.float32))).astype(np.int16);E[:,1]=0;path=out/"all200_group_library.npz";base.atomic_npz(path,context=np.stack([x["context_feature"] for x in rows]).astype(np.float32),action=np.stack([x["action_feature"] for x in rows]).astype(np.float32),appearance=C,transport=E,action_sha=np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows]),context_sha=np.asarray([x["keys"][0][1] for x in rows]));library={"path":str(path.resolve()),"sha256":sha(path)}
 report={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"passed":bool(passed),"early_stop_mathematically_unreachable":early,"completed_folds":len(folds),"failed_folds":failed,"folds":folds,"aggregate":agg,"all200_library_built":bool(passed),"all200_library":library,"reused_cache_sha256":sha(cache),"ordered_request_manifest_sha256":order_digest,"v169_closure_digest":closure_digest,"guards":{"reward_loaded":False,"gpu_used":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False,"hyperparameter_sweep":False}}
 atomic(out/"s0_report.json",report);print(json.dumps(report,sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
