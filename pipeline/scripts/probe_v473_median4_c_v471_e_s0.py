"""Frozen CPU-only v473: median4 appearance with bitexact v471 transport."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import probe_v471_group_context_prototype_knn_s0 as ref
BRANCHES=ref.BRANCHES;FORMAT="strict-track2-v473-median4-c-v471-e-s0-v1"
def sha(p):return ref.sha(p)
def jsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def median4(maps):
 s=np.sort(np.asarray(maps,dtype=np.float32),axis=0);out=np.float32(.5)*(s[1]+s[2])
 if out.dtype!=np.float32 or out.shape!=(256,256,3):raise RuntimeError("v473 median dtype/shape")
 return out
def predict_both(rows,requested,canonical,pre,fold):
 hold,reference,target,requested_hold=ref.predict_group(rows,requested,canonical,pre["v471"]["preregistration"],fold);fit=[i for i,x in enumerate(rows) if x["fold"]!=fold];Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=Y[:,1]-canonical.astype(np.float32);E=(Y-Y[:,1:2])-(requested.astype(np.float32)-canonical[:,None].astype(np.float32));E[:,1]=0
 V=np.stack([rows[i]["context_feature"] for i in fit]);vm=V.mean(0);vs=np.maximum(V.std(0),1e-4);ckeys=[rows[i]["keys"][1] for i in fit];X=np.concatenate([rows[i]["action_feature"] for i in fit]);xm=X.mean(0);xs=np.maximum(X.std(0),1e-4);mapping=pre["v471"]["preregistration"]["v469"]["preregistration"]["v468"]["phase_shuffle"]["mappings"][fold];lookup={(x["episode"],x["start"]):i for i,x in enumerate(rows)}
 mean={m:np.empty((len(hold),5,256,256,3),np.float32) for m in ("action","context_only","phase_shuffle")};candidate={m:np.empty_like(mean[m]) for m in mean};ehat={m:np.empty_like(mean[m]) for m in mean};manifest=[]
 for q,i in enumerate(hold):
  dc=np.square((V-rows[i]["context_feature"])/vs).mean(1);chosen=[fit[j] for j in ref.select4(dc,ckeys)];cmean=C[chosen].mean(0);cmedian=median4(C[chosen]);donor=mapping["holdout"][f"{rows[i]['episode']}:{rows[i]['start']}"];di=lookup[(int(donor["donor_episode"]),int(donor["donor_start"]))]
  for b in range(5):
   nt=np.array_equal(rows[i]["future"][b,:,7:13],np.broadcast_to(rows[i]["history"][-1,7:13],(8,6)));est={};idx={}
   if nt:
    est={m:np.zeros((256,256,3),np.float32) for m in mean};idx={m:[-1]*4 for m in mean}
   else:
    for mode,query in (("action",rows[i]["action_feature"][b]),("phase_shuffle",rows[di]["action_feature"][b])):
     maps=[];ids=[]
     for ci in chosen:
      dist=np.square((rows[ci]["action_feature"]-query)/xs).mean(1);k=int(np.lexsort((np.asarray([x[1] for x in rows[ci]["keys"]]),np.asarray([x[0] for x in rows[ci]["keys"]]),dist.astype(np.float64)))[0]);ids.append(k);maps.append(E[ci,k])
     est[mode]=np.mean(maps,axis=0);idx[mode]=ids
    est["context_only"]=np.mean([E[ci].mean(0) for ci in chosen],axis=0);idx["context_only"]=[[0,1,2,3,4] for _ in chosen]
   for mode in mean:
    ehat[mode][q,b]=est[mode];mean[mode][q,b]=np.clip(requested[i,b].astype(np.float32)+cmean+est[mode],0,255);candidate[mode][q,b]=np.clip(requested[i,b].astype(np.float32)+cmedian+est[mode],0,255)
   manifest.append({"query":q,"branch_action_sha256":rows[i]["keys"][b][0],"contexts":[rows[x]["keys"][0][1] for x in chosen],"indices":idx})
 if any(not np.array_equal(mean[m],reference[m]) for m in mean):raise RuntimeError(f"v473 v471 mean reference mismatch fold{fold}")
 evidence={"reference_index_manifest_sha256":jsha(manifest),"reference_ehat_sha256":{m:ref.base.arrsha(ehat[m]) for m in ehat},"candidate_index_manifest_sha256":jsha(manifest),"candidate_ehat_sha256":{m:ref.base.arrsha(ehat[m]) for m in ehat},"candidate_e_equals_reference":True}
 return hold,reference,candidate,target,requested_hold,evidence
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","contract","selection","dataset","output-dir"):p.add_argument(f"--{n}",type=Path,required=True)
 a=p.parse_args();pre=json.loads(a.preregistration.read_text());out=a.output_dir;old=pre["v471"]["preregistration"];v469=old["v469"]["preregistration"]
 if pre.get("format")!="strict-track2-v473-median4-c-v471-e-preregistration-v1" or sha(a.contract)!=pre["contract"]["sha256"] or sha(Path(__file__))!=pre["source"]["probe_sha256"] or Path(ref.__file__).resolve()!=Path(pre["source"]["v471_probe_path"]).resolve() or sha(ref.__file__)!=pre["source"]["v471_probe_sha256"] or Path(ref.base.__file__).resolve()!=Path(pre["source"]["v469_probe_path"]).resolve() or sha(ref.base.__file__)!=pre["source"]["v469_probe_sha256"] or out.exists():raise RuntimeError("v473 closure")
 if sha(a.selection)!=v469["v461"]["selection_sha256"] or a.dataset.resolve()!=Path(v469["v461"]["dataset"]).resolve() or any(not (a.dataset/x["relative"]).is_file() or sha(a.dataset/x["relative"])!=x["sha256"] for x in v469["v461"]["files"]):raise RuntimeError("v473 data")
 rows=ref.base.load_rows(v469,a.dataset,json.loads(a.selection.read_text()));cache=Path(old["v469"]["root"])/"result/v169_endpoint_cache.npz"
 if sha(cache)!=old["v469"]["immutable_sha256"]["result/v169_endpoint_cache.npz"]:raise RuntimeError("v473 cache")
 with np.load(cache,allow_pickle=False) as z:requested=z["requested"].copy();canonical=z["canonical"].copy();seeds=z["seeds"].copy()
 cr=old["v469"]["cache_receipt"];order,closure=ref.cache_digests(rows,old)
 if requested.shape!=(200,5,256,256,3) or canonical.shape!=(200,256,256,3) or requested.dtype!=np.uint8 or canonical.dtype!=np.uint8 or ref.base.arrsha(requested)!=cr["requested_array_sha256"] or ref.base.arrsha(canonical)!=cr["canonical_array_sha256"] or ref.base.arrsha(seeds)!=cr["seed_array_sha256"] or order!=cr["ordered_request_manifest_sha256"] or closure!=cr["v169_closure_digest"] or not np.array_equal(seeds,np.asarray([ref.base.stable_seed(x["episode"],x["start"]) for x in rows],np.int64)):raise RuntimeError("v473 cache closure")
 out.mkdir(parents=True);folds=[];values=[];evidence=[];failed=0
 for fold in range(5):
  hold,reference,candidate,target,b,ev=predict_both(rows,requested,canonical,pre,fold);rr,*_=ref.base.fold_metrics(fold,hold,reference,target,b,rows);immutable=json.loads((Path(pre["v471"]["root"])/f"result/fold{fold}_receipt.json").read_text())
  if rr!=immutable:raise RuntimeError(f"v473 immutable v471 receipt mismatch fold{fold}")
  receipt,*v=ref.base.fold_metrics(fold,hold,candidate,target,b,rows);receipt["v471_reference_receipt_sha256"]=sha(Path(pre["v471"]["root"])/f"result/fold{fold}_receipt.json");receipt["bitexact_e_evidence"]=ev;folds.append(receipt);values.append(v);evidence.append(ev);ref.atomic(out/f"fold{fold}_receipt.json",receipt);failed+=not receipt["passed"]
  if failed>=2:break
 early=failed>=2;agg=None;passed=False;library=None
 if not early and len(folds)==5:
  plain=[{k:v for k,v in x.items() if k not in ("v471_reference_receipt_sha256","bitexact_e_evidence")} for x in folds];agg=ref.aggregate(values,plain);passed=ref.aggregate_pass(agg)
  if passed:
   Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=(Y[:,1]-canonical.astype(np.float32)).astype(np.int16);E=((Y-Y[:,1:2])-(requested.astype(np.float32)-canonical[:,None].astype(np.float32))).astype(np.int16);E[:,1]=0;path=out/"all200_median4_library.npz";ref.base.atomic_npz(path,context=np.stack([x["context_feature"] for x in rows]).astype(np.float32),action=np.stack([x["action_feature"] for x in rows]).astype(np.float32),appearance=C,transport=E,action_sha=np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows]),context_sha=np.asarray([x["keys"][0][1] for x in rows]));library={"path":str(path.resolve()),"sha256":sha(path),"median_definition":"sort-float32-middle-two-times-0.5"}
 report={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"passed":bool(passed),"early_stop_mathematically_unreachable":early,"completed_folds":len(folds),"failed_folds":failed,"folds":folds,"aggregate":agg,"all200_library_built":bool(passed),"all200_library":library,"reused_cache_sha256":sha(cache),"ordered_request_manifest_sha256":order,"v169_closure_digest":closure,"guards":{"reward_loaded":False,"dev_final_loaded":False,"outcome_loaded":False,"gpu_used":False,"policy_updates":0,"endpoint_parent_data_authorized":False,"s1_authorized":False,"rl_authorized":False}}
 ref.atomic(out/"s0_report.json",report);print(json.dumps(report,sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
