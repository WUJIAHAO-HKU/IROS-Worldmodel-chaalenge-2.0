"""Single-shot outer-five-fold v469 C/E KNN4 probe on immutable public train data."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,random
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

BRANCHES=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
NON_NT=(0,2,3,4);K=4
FORMAT="strict-track2-v469-closed-form-residual-knn-s0-v1"

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise RuntimeError(f"v469 refuses overwrite {path}")
 with tmp.open("w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def atomic_npz(path,**values):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if path.exists() or tmp.exists():raise RuntimeError(f"v469 refuses overwrite {path}")
 with tmp.open("wb") as f:np.savez(f,**values);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def stable_seed(ep,start):return int.from_bytes(hashlib.sha256(f"v466/1616/{ep}/{start}".encode()).digest()[:8],"little")%(2**31)
def context_feature(rgb):
 x=np.asarray(rgb,np.float32)
 if x.shape!=(256,256,3):raise RuntimeError("v469 context shape")
 return x.reshape(8,32,8,32,3).mean((1,3)).reshape(192)
def action_feature(history,future):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32);anchor=h[-1,7:13]
 return np.concatenate((f[-1,7:13]-anchor,(f[:,7:13]-anchor).reshape(48))).astype(np.float32)
def canonical(history,future):
 out=np.asarray(future,np.float32).copy();out[:,7:13]=np.asarray(history,np.float32)[-1,7:13];return out
def key(history,future,context):return (arrsha(np.concatenate((np.asarray(history,np.float32),np.asarray(future,np.float32)),0)),arrsha(np.asarray(context,np.uint8)))
def nearest(distance,keys):
 order=np.lexsort((np.asarray([x[1] for x in keys]),np.asarray([x[0] for x in keys]),np.asarray(distance,np.float64)))
 return order[:K]
def mae(pred,target):return np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean((1,2,3))
def causal(candidate,target):
 pos=[];nz=[]
 for i in range(len(candidate)):
  for b in range(1,5):
   p=(candidate[i,b]-candidate[i,0]).reshape(-1).astype(np.float64);t=(target[i,b]-target[i,0]).reshape(-1).astype(np.float64)
   pos.append(float(np.dot(p,t)/(np.linalg.norm(p)*np.linalg.norm(t)+1e-9))>0);nz.append(float(np.abs(p).mean())>1e-6)
 return float(np.mean(pos)),float(np.mean(nz))
def load_rows(pre,dataset,selection):
 ordered_specs=selection["contexts"]
 if len(ordered_specs)!=200 or len({(int(x["episode"]),int(x["start"])) for x in ordered_specs})!=200:raise RuntimeError("v469 selection order/uniqueness")
 paths={}
 for d in Path(dataset).glob("batch_*/rows/*"):
  receipt=json.loads((d/"receipt.json").read_text());paths[(int(receipt["episode"]),int(receipt["start"]))]=d
 rows=[]
 for spec in ordered_specs:
  ep,start=int(spec["episode"]),int(spec["start"]);d=paths[(ep,start)]
  with np.load(d/"endpoint.npz",allow_pickle=False) as z:
   history=z["history_actions"].astype(np.float32);future=z["future_actions"][:5].astype(np.float32);target=z["endpoint_rgb"][:5].astype(np.uint8);contexts=z["branch_context_rgb"].astype(np.uint8);variants=tuple(map(str,z["variants"][:5]))
  if variants!=BRANCHES or history.shape!=(4,14) or future.shape!=(5,8,14) or target.shape!=(5,256,256,3) or contexts.shape!=(6,256,256,3):raise RuntimeError("v469 row schema")
  if not all(np.array_equal(contexts[0],x) for x in contexts) or not np.array_equal(future[1],canonical(history,future[0])):raise RuntimeError("v469 same-context/canonical drift")
  if arrsha(history)!=spec["history_action_sha256"] or [arrsha(x) for x in future]!=[spec["branch_action_sha256"][x] for x in BRANCHES]:raise RuntimeError("v469 action closure drift")
  rows.append({"episode":ep,"start":start,"fold":int(spec["fold"]),"prompt":spec["instruction"],"history":history,"future":future,"target":target,"context":contexts[0],"context_feature":context_feature(contexts[0]),"action_feature":np.stack([action_feature(history,x) for x in future]),"keys":[key(history,x,contexts[0]) for x in future]})
 if len(rows)!=200:return (_ for _ in ()).throw(RuntimeError("v469 exact 200 rows"))
 return rows
def cache_v169(rows,pre,device):
 from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
 v=Track2V169ArmRoutedRuntime(Path(pre["v468"]["v169"]["release"]),Path(pre["v468"]["v169"]["library"]),device);requested=np.empty((200,5,256,256,3),np.uint8);canonical_out=np.empty((200,256,256,3),np.uint8)
 samples=[(i,b) for i in range(200) for b in range(5)]
 for begin in range(0,1000,4):
  part=samples[begin:begin+4];context=np.stack([np.repeat(rows[i]["context"][None],5,0) for i,b in part]);history=np.stack([rows[i]["history"] for i,b in part]);future=np.stack([rows[i]["future"][b] for i,b in part]);seeds=np.asarray([stable_seed(rows[i]["episode"],rows[i]["start"]) for i,b in part]);pred=v.predict_batch(context,history,future,seeds,[rows[i]["prompt"] for i,b in part])
  if pred.shape!=(len(part),8,256,256,3) or pred.dtype!=np.uint8:raise RuntimeError("v469 requested v169 output")
  for j,(i,b) in enumerate(part):requested[i,b]=pred[j,7]
 for begin in range(0,200,4):
  ids=list(range(begin,min(begin+4,200)));context=np.stack([np.repeat(rows[i]["context"][None],5,0) for i in ids]);history=np.stack([rows[i]["history"] for i in ids]);future=np.stack([canonical(rows[i]["history"],rows[i]["future"][0]) for i in ids]);seeds=np.asarray([stable_seed(rows[i]["episode"],rows[i]["start"]) for i in ids]);pred=v.predict_batch(context,history,future,seeds,[rows[i]["prompt"] for i in ids])
  if pred.shape!=(len(ids),8,256,256,3) or pred.dtype!=np.uint8:raise RuntimeError("v469 canonical v169 output")
  for j,i in enumerate(ids):canonical_out[i]=pred[j,7]
 del v;gc.collect()
 try:
  import torch;torch.cuda.empty_cache()
 except Exception:pass
 return requested,canonical_out
def predict_knn(rows,requested,canonical_out,pre,fold):
 fit=[i for i,x in enumerate(rows) if x["fold"]!=fold];hold=[i for i,x in enumerate(rows) if x["fold"]==fold];Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=Y[:,1]-canonical_out.astype(np.float32);E=(Y-Y[:,1:2])-(requested.astype(np.float32)-canonical_out[:,None].astype(np.float32));E[:,1]=0
 V=np.stack([rows[i]["context_feature"] for i in fit]);vm=V.mean(0);vs=np.maximum(V.std(0),1e-4);ckeys=[rows[i]["keys"][1] for i in fit];Clib=C[fit]
 lib=[(i,b) for i in fit for b in NON_NT];X=np.stack([rows[i]["action_feature"][b] for i,b in lib]);VL=np.stack([rows[i]["context_feature"] for i,b in lib]);xm=X.mean(0);xs=np.maximum(X.std(0),1e-4);vlm=VL.mean(0);vls=np.maximum(VL.std(0),1e-4);Elib=np.stack([E[i,b] for i,b in lib]);ekeys=[rows[i]["keys"][b] for i,b in lib];meta=pre["v468"]["phase_shuffle"]["mappings"][fold]
 if int(meta["fold"])!=fold:raise RuntimeError("v469 phase mapping fold drift")
 mapping=meta["holdout"];lookup={(x["episode"],x["start"]):i for i,x in enumerate(rows)}
 out={m:np.empty((len(hold),5,256,256,3),np.float32) for m in ("action","context_only","phase_shuffle")}
 for q,i in enumerate(hold):
  dc_c=np.square((V-rows[i]["context_feature"])/vs).mean(1);chat=Clib[nearest(dc_c,ckeys)].mean(0);dc=np.square((VL-rows[i]["context_feature"])/vls).mean(1);donor=mapping[f"{rows[i]['episode']}:{rows[i]['start']}"];di=lookup[(int(donor["donor_episode"]),int(donor["donor_start"]))]
  for b in range(5):
   if b==1:eh={m:0. for m in out}
   else:
    eh={};
    for m,query in (("action",rows[i]["action_feature"][b]),("phase_shuffle",rows[di]["action_feature"][b])):
     da=np.square((X-query)/xs).mean(1);eh[m]=Elib[nearest(da+dc,ekeys)].mean(0)
    eh["context_only"]=Elib[nearest(dc,ekeys)].mean(0)
   for m in out:out[m][q,b]=np.clip(requested[i,b].astype(np.float32)+chat+eh[m],0,255)
 return hold,out,Y[hold],requested[hold],{"context_mean":vm,"context_std":vs,"action_mean":xm,"action_std":xs}
def fold_metrics(fold,hold,out,target,baseline,rows):
 ae=mae(out["action"].reshape(-1,256,256,3),target.reshape(-1,256,256,3));be=mae(baseline.reshape(-1,256,256,3),target.reshape(-1,256,256,3));ce=mae(out["context_only"].reshape(-1,256,256,3),target.reshape(-1,256,256,3));pe=mae(out["phase_shuffle"].reshape(-1,256,256,3),target.reshape(-1,256,256,3));ue=mae(np.clip(np.rint(out["action"]),0,255).astype(np.uint8).reshape(-1,256,256,3),target.reshape(-1,256,256,3));branch={BRANCHES[b]:float(ae[b::5].mean()/be[b::5].mean()) for b in range(5)};cos,nz=causal(out["action"],target);eps=sorted({rows[i]["episode"] for i in hold});wins=sum(ae[np.repeat([rows[i]["episode"]==e for i in hold],5)].mean()<be[np.repeat([rows[i]["episode"]==e for i in hold],5)].mean() for e in eps);ratios={"overall":float(ae.mean()/be.mean()),"uint8":float(ue.mean()/be.mean()),"context_only":float(ae.mean()/ce.mean()),"phase_shuffle":float(ae.mean()/pe.mean())};passed=ratios["overall"]<=.95 and ratios["uint8"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and ratios["context_only"]<=.95 and ratios["phase_shuffle"]<=.97 and cos>=.6 and nz>=.95
 sums={"action":float(ae.sum()),"baseline":float(be.sum()),"uint8":float(ue.sum()),"context_only":float(ce.sum()),"phase_shuffle":float(pe.sum()),"count":int(len(ae)),"branch_action":[float(ae[b::5].sum()) for b in range(5)],"branch_baseline":[float(be[b::5].sum()) for b in range(5)],"branch_count":[int(len(ae[b::5])) for b in range(5)]}
 return {"fold":fold,"ratios":ratios,"branch_ratio":branch,"positive_delta_cosine_fraction":cos,"nonzero_prediction_delta_fraction":nz,"episode_wins":int(wins),"metric_sums":sums,"passed":bool(passed)},ae,be,ce,pe,ue
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","contract","selection","dataset","output-dir"):p.add_argument(f"--{n}",type=Path,required=True)
 p.add_argument("--device",default="cuda");a=p.parse_args();pre=json.loads(a.preregistration.read_text());contract=json.loads(a.contract.read_text());out=a.output_dir
 if pre.get("format")!="strict-track2-v469-closed-form-residual-knn-preregistration-v1" or contract.get("status")!="preregistered_public_train_s0_authorized" or sha(a.contract)!=pre["contract"]["sha256"] or sha(Path(__file__))!=pre["source"]["probe_sha256"] or out.exists():raise RuntimeError("v469 prereg/source/output drift")
 if sha(a.selection)!=pre["v461"]["selection_sha256"] or a.dataset.resolve()!=Path(pre["v461"]["dataset"]).resolve():raise RuntimeError("v469 selection/dataset path drift")
 for row in pre["v461"]["files"]:
  path=a.dataset/row["relative"]
  if not path.is_file() or sha(path)!=row["sha256"]:raise RuntimeError(f"v469 runtime dataset drift: {path}")
 out.mkdir(parents=True);rows=load_rows(pre,a.dataset,json.loads(a.selection.read_text()));requested,canonical_out=cache_v169(rows,pre,a.device);seeds=np.asarray([stable_seed(x["episode"],x["start"]) for x in rows],np.int64);cache=out/"v169_endpoint_cache.npz";atomic_npz(cache,requested=requested,canonical=canonical_out,seeds=seeds);folds=[];failed=0;all_values=[]
 for fold in range(5):
  hold,pred,target,base,stats=predict_knn(rows,requested,canonical_out,pre,fold);receipt,*values=fold_metrics(fold,hold,pred,target,base,rows);folds.append(receipt);all_values.append(values);atomic_json(out/f"fold{fold}_receipt.json",receipt);failed+=not receipt["passed"]
  if failed>=2:break
 early=failed>=2;aggregate=None;passed=False;library_receipt=None
 if not early and len(folds)==5:
  ae=np.concatenate([x[0] for x in all_values]);be=np.concatenate([x[1] for x in all_values]);ce=np.concatenate([x[2] for x in all_values]);pe=np.concatenate([x[3] for x in all_values]);ue=np.concatenate([x[4] for x in all_values]);branch={BRANCHES[b]:float(np.concatenate([x[0][b::5] for x in all_values]).mean()/np.concatenate([x[1][b::5] for x in all_values]).mean()) for b in range(5)};aggregate={"passing_folds":sum(x["passed"] for x in folds),"continuous_ratio":float(ae.mean()/be.mean()),"uint8_ratio":float(ue.mean()/be.mean()),"branch_ratio":branch,"action_over_context_only":float(ae.mean()/ce.mean()),"action_over_phase_shuffle":float(ae.mean()/pe.mean()),"positive_delta_cosine_fraction":float(np.mean([x["positive_delta_cosine_fraction"] for x in folds])),"nonzero_prediction_delta_fraction":float(np.mean([x["nonzero_prediction_delta_fraction"] for x in folds])),"improved_episodes":int(sum(x["episode_wins"] for x in folds))};passed=aggregate["passing_folds"]>=4 and aggregate["continuous_ratio"]<=.95 and aggregate["uint8_ratio"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and aggregate["action_over_context_only"]<=.95 and aggregate["action_over_phase_shuffle"]<=.97 and aggregate["positive_delta_cosine_fraction"]>=.6 and aggregate["nonzero_prediction_delta_fraction"]>=.95 and aggregate["improved_episodes"]>=12
  if passed:
   Y=np.stack([x["target"] for x in rows]).astype(np.float32);C=(Y[:,1]-canonical_out.astype(np.float32)).astype(np.int16);E=((Y-Y[:,1:2])-(requested.astype(np.float32)-canonical_out[:,None].astype(np.float32))).astype(np.int16);E[:,1]=0;library=out/"all200_library.npz";atomic_npz(library,context=np.stack([x["context_feature"] for x in rows]).astype(np.float32),action=np.stack([x["action_feature"] for x in rows]).astype(np.float32),appearance=C,transport=E,action_sha=np.asarray([[x["keys"][b][0] for b in range(5)] for x in rows]),context_sha=np.asarray([x["keys"][0][1] for x in rows]));library_receipt={"path":str(library.resolve()),"sha256":sha(library),"keys":["action","action_sha","appearance","context","context_sha","transport"]}
 Y=np.stack([x["target"] for x in rows]).astype(np.float32);Cstar=Y[:,1]-canonical_out.astype(np.float32);Estar=(Y-Y[:,1:2])-(requested.astype(np.float32)-canonical_out[:,None].astype(np.float32));Estar[:,1]=0;order_manifest=[{"episode":x["episode"],"start":x["start"],"seed":int(seeds[i]),"action_sha":[x["keys"][b][0] for b in range(5)]} for i,x in enumerate(rows)];order_digest=hashlib.sha256(json.dumps(order_manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest();closure_digest=hashlib.sha256(json.dumps(pre["v468"]["v169"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
 report={"format":FORMAT,"created_at":datetime.now(timezone.utc).isoformat(),"passed":bool(passed),"early_stop_mathematically_unreachable":bool(early),"completed_folds":len(folds),"failed_folds":int(failed),"folds":folds,"aggregate":aggregate,"all200_library_built":bool(passed),"all200_library":library_receipt,"cache":{"path":str(cache.resolve()),"sha256":sha(cache),"requested_shape":list(requested.shape),"canonical_shape":list(canonical_out.shape),"requested_array_sha256":arrsha(requested),"canonical_array_sha256":arrsha(canonical_out),"seed_array_sha256":arrsha(seeds),"ordered_request_manifest_sha256":order_digest,"v169_closure_digest":closure_digest},"targets":{"appearance_sha256":arrsha(Cstar),"transport_sha256":arrsha(Estar),"no_transport_exact_zero":bool(np.array_equal(Estar[:,1],np.zeros_like(Estar[:,1])))},"guards":{"reward_loaded":False,"policy_updates":0,"rl_authorized":False,"hyperparameter_sweep":False}}
 atomic_json(out/"s0_report.json",report);print(json.dumps(report,sort_keys=True));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
