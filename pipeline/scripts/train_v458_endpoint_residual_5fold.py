#!/usr/bin/env python3
"""Preregistered v458 three-head 5-fold endpoint kill gate; no reward or RL."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,random
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v458_v169_endpoint_residual_runtime import CHECKPOINT_FORMAT,EndpointResidualUNet128FiLM,action_features,instruction_tokens
SEED=1611;STEPS=50;BATCH=8;BRANCHES=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_json(path,payload):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if tmp.exists():raise RuntimeError(f"refusing stale v458 temporary receipt: {tmp}")
 with tmp.open("x") as handle:
  json.dump(payload,handle,indent=2);handle.write("\n");handle.flush();os.fsync(handle.fileno())
 os.replace(tmp,path);descriptor=os.open(path.parent,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
def stable_seed(episode,start):return int.from_bytes(hashlib.sha256(f"v458/{SEED}/{episode}/{start}".encode()).digest()[:8],"little")%(2**31)
def arrsha(value):return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def verify_tree(root,records,exact):
 root=Path(root).resolve();actual={row["relative"]:row["sha256"] for row in records}
 if any(not (root/name).is_file() or sha(root/name)!=digest for name,digest in actual.items()):raise RuntimeError(f"v458 dependency tree drift: {root}")
 if exact and {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}!=set(actual):raise RuntimeError(f"v458 dependency extra/missing files: {root}")
def validate_and_load(args,pre):
 if sha(Path(__file__))!=pre["source"]["trainer_sha256"]:raise RuntimeError("v458 trainer source SHA drift")
 runtime=Path(pre["source"]["runtime_path"])
 if not runtime.is_file() or sha(runtime)!=pre["source"]["runtime_sha256"]:raise RuntimeError("v458 runtime source SHA drift")
 if str(args.v169_release.resolve())!=pre["v169"]["release"] or str(args.v169_library.resolve())!=pre["v169"]["library"]:raise RuntimeError("v458 v169 path drift")
 verify_tree(args.v169_release,pre["v169"]["release_files"],True);verify_tree(args.v169_library,pre["v169"]["library_files"],False)
 if sha(Path(pre["v169"]["runtime_source"]["path"]))!=pre["v169"]["runtime_source"]["sha256"]:raise RuntimeError("v169 runtime source drift")
 receipt_path=Path(pre["v461"]["final_receipt"]["path"]);receipt=json.loads(receipt_path.read_text())
 if sha(receipt_path)!=pre["v461"]["final_receipt"]["sha256"] or receipt.get("format")!="strict-track2-v461-endpoint200-batch-audit-v1" or receipt.get("mode")!="final" or receipt.get("passed") is not True or not all(receipt.get("checks",{}).values()):raise RuntimeError("v458 hard blocked: bound v461 final receipt is not PASS")
 for key,arg in (("selection",args.v456_selection),("generation_report",args.v456_generation_report)):
  if str(arg.resolve())!=pre["v461"][key]["path"] or sha(arg)!=pre["v461"][key]["sha256"]:raise RuntimeError(f"v458 {key} closure failed")
 if str(args.v456_dataset.resolve())!=pre["v461"]["dataset"]:raise RuntimeError("v458 dataset path drift")
 for row in pre["v461"]["files"]:
  path=args.v456_dataset/row["relative"]
  if not path.is_file() or sha(path)!=row["sha256"]:raise RuntimeError(f"v458 frozen file drift {path}")
 selection=json.loads(args.v456_selection.read_text());specs={(int(x["episode"]),int(x["start"])):x for x in selection["contexts"]};samples=[];contexts=[]
 for batch in range(10):
  for rowdir in sorted((args.v456_dataset/f"batch_{batch:03d}"/"rows").iterdir()):
   row_receipt=json.loads((rowdir/"receipt.json").read_text())
   with np.load(rowdir/"endpoint.npz",allow_pickle=False) as z:
    episode=int(z["episode"]);start=int(z["start"]);spec=specs[(episode,start)];history=z["history_actions"].astype(np.float32);futures=z["future_actions"][:5].astype(np.float32);all_context=z["branch_context_rgb"];all_state=z["branch_context_state"];all_pose=z["branch_context_pose"];all_bottle=z["branch_context_bottle_position"];context=all_context[0].astype(np.uint8);target=z["endpoint_rgb"][:5].astype(np.uint8);variants=list(map(str,z["variants"][:5]));executed=list(map(str,z["executed_action_sha256"][:5]))
   exact_context=all(np.array_equal(all_context[0],all_context[i]) and np.array_equal(all_state[0],all_state[i]) and np.array_equal(all_pose[0],all_pose[i]) and np.array_equal(all_bottle[0],all_bottle[i]) for i in range(1,6))
   if variants!=list(BRANCHES) or not exact_context or arrsha(history)!=spec["history_action_sha256"] or [arrsha(x) for x in futures]!=[spec["branch_action_sha256"][x] for x in BRANCHES] or executed!=[arrsha(x) for x in futures] or row_receipt.get("selection_action_sha256")!=spec["branch_action_sha256"]:raise RuntimeError("v458 row/context/action closure drift")
   cid=len(contexts);contexts.append({"episode":episode,"start":start,"fold":int(spec["fold"]),"prompt":spec["instruction"],"context":context,"history":history})
   for branch in range(5):samples.append({"context_id":cid,"episode":episode,"fold":int(spec["fold"]),"branch":branch,"future":futures[branch],"target":target[branch]})
 if len(contexts)!=200 or len(samples)!=1000:raise RuntimeError("v458 exact 200/1000 contract")
 return contexts,samples
def cache_v169(contexts,samples,v169,device_batch=4):
 for begin in range(0,len(samples),device_batch):
  part=samples[begin:begin+device_batch];ctx=np.stack([np.repeat(contexts[x["context_id"]]["context"][None],5,0) for x in part]);history=np.stack([contexts[x["context_id"]]["history"] for x in part]);future=np.stack([x["future"] for x in part]);prompts=[contexts[x["context_id"]]["prompt"] for x in part];seeds=np.asarray([stable_seed(x["episode"],contexts[x["context_id"]]["start"]) for x in part]);pred=v169.predict_batch(ctx,history,future,seeds,prompts)
  for i,row in enumerate(part):row["baseline"]=pred[i,7].copy()
def stats(contexts,samples,fit_ids):
 values=np.concatenate([np.concatenate((contexts[samples[i]["context_id"]]["history"],samples[i]["future"]),0) for i in fit_ids],0).astype(np.float64);return values.mean(0).astype(np.float32),np.maximum(values.std(0),1e-4).astype(np.float32)
def donor_ids(pre,fold,contexts,samples,partition):
 meta=pre["phase_shuffle"]["mappings"][fold];mapping=meta[partition];lookup={(x["episode"],x["start"]):i for i,x in enumerate(contexts)};expected=set(meta[f"{partition}_episodes"]);result={}
 for i,row in enumerate(samples):
  key=f"{contexts[row['context_id']]['episode']}:{contexts[row['context_id']]['start']}"
  if key in mapping:
   donor=mapping[key];dcid=lookup[(donor["donor_episode"],donor["donor_start"])]
   if row["episode"] not in expected or dcid==row["context_id"] or contexts[dcid]["episode"]!=row["episode"] or contexts[dcid]["episode"] not in expected:raise RuntimeError("v458 phase donor leakage/self-map")
   result[i]=5*dcid+row["branch"]
 return result
def features_for(contexts,samples,feature_ids,mean,std,zero=False):
 if zero:
  value=np.zeros((len(feature_ids),12*14+6+48+1),np.float32);value[:,-1]=1.;return value
 h=np.stack([contexts[samples[int(i)]["context_id"]]["history"] for i in feature_ids]);f=np.stack([samples[int(i)]["future"] for i in feature_ids]);return action_features(h,f,mean,std)
def tensors(contexts,samples,ids,mean,std,device,feature_ids=None,zero=False):
 ids=np.asarray(ids,int);feature_ids=ids if feature_ids is None else np.asarray(feature_ids,int);rows=[samples[i] for i in ids];base=torch.as_tensor(np.stack([x["baseline"] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;last=torch.as_tensor(np.stack([contexts[x["context_id"]]["context"] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;feat=torch.as_tensor(features_for(contexts,samples,feature_ids,mean,std,zero),device=device);tokens=torch.as_tensor(instruction_tokens([contexts[x["context_id"]]["prompt"] for x in rows]),device=device);target=torch.as_tensor(np.stack([x["target"] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2);return base,last,feat,tokens,target
def grad_l1(pred,target):return .5*(F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])+F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],target[:,:,1:]-target[:,:,:-1]))
def schedules(fit_ids,fit_contexts,seed):
 rng=np.random.default_rng(seed);return [(rng.choice(fit_ids,size=BATCH,replace=False),int(sorted(fit_contexts)[step%len(fit_contexts)])) for step in range(STEPS)]
def schedule_sha(schedule):
 payload=[{"sample_ids":list(map(int,ids)),"context_id":int(cid)} for ids,cid in schedule];return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def train_head(initial,contexts,samples,fit_ids,fit_contexts,mean,std,device,schedule,mode,donors):
 model=EndpointResidualUNet128FiLM(16).to(device);model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[];model.train()
 for ids,cid in schedule:
  fids=[donors[int(i)] for i in ids] if mode=="phase_shuffle" else ids;base,last,feat,tok,target=tensors(contexts,samples,ids,mean,std,device,fids,mode=="context_only")
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
   pred=(255*base+model(base,last,feat,tok)).clamp(0,255);loss=F.l1_loss(pred.float(),target)+.25*grad_l1(pred.float(),target);group=np.asarray([i for i,x in enumerate(samples) if x["context_id"]==cid]);gf=[donors[int(i)] for i in group] if mode=="phase_shuffle" else group;gb,gl,gn,gt,gta=tensors(contexts,samples,group,mean,std,device,gf,mode=="context_only");gp=(255*gb+model(gb,gl,gn,gt)).clamp(0,255);loss=loss+.5*F.l1_loss((gp[1:]-gp[0:1]).float(),gta[1:]-gta[0:1])
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
 model.eval();return model,losses
@torch.inference_mode()
def predict(model,contexts,samples,ids,mean,std,device,mode,donors):
 values=[]
 for begin in range(0,len(ids),BATCH):
  part=np.asarray(ids[begin:begin+BATCH]);fids=[donors[int(i)] for i in part] if mode=="phase_shuffle" else part;base,last,feat,tok,_=tensors(contexts,samples,part,mean,std,device,fids,mode=="context_only")
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):out=(255*base+model(base,last,feat,tok)).clamp(0,255)
  values.append(out.permute(0,2,3,1).float().cpu().numpy())
 continuous=np.concatenate(values);return continuous,np.clip(np.rint(continuous),0,255).astype(np.uint8)
def mae(pred,target):return np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean((1,2,3))
def causal(ids,samples,pred,target):
 pos={int(idx):i for i,idx in enumerate(ids)};cos=[];nonzero=[]
 for cid in sorted({samples[int(i)]["context_id"] for i in ids}):
  group=sorted((int(i) for i in ids if samples[int(i)]["context_id"]==cid),key=lambda i:samples[i]["branch"]);p=np.stack([pred[pos[i]] for i in group]);t=np.stack([target[pos[i]] for i in group])
  for j in range(1,5):
   td=(t[j]-t[0]).reshape(-1);pd=(p[j]-p[0]).reshape(-1);cos.append(float(np.dot(pd,td)/(np.linalg.norm(pd)*np.linalg.norm(td)+1e-9))>0);nonzero.append(float(np.abs(pd).mean())>1e-6)
 return float(np.mean(cos)),float(np.mean(nonzero))
def main():
 p=argparse.ArgumentParser()
 for n in ("preregistration","v456-selection","v456-generation-report","v456-dataset","v169-release","v169-library","output-dir"):p.add_argument(f"--{n}",type=Path,required=True)
 p.add_argument("--device",default="cuda");a=p.parse_args();pre=json.loads(a.preregistration.read_text())
 if pre.get("format")!="strict-track2-v458-endpoint-residual-5fold-preregistration-v2" or a.output_dir.exists():raise RuntimeError("v458 prereg/output contract")
 expected_training={"heads_per_fold":["action","context_only","phase_shuffle"],"steps_per_head":50,"batch_size":8,"working_resolution":128,"channels":16,"optimizer":"AdamW","learning_rate":.0003,"weight_decay":.0001,"gradient_clip":1.,"endpoint_l1":1.,"gradient_l1":.25,"same_context_action_delta_l1":.5,"precision":"bf16","all200_action_head_steps_after_oof_pass":50}
 expected_features={"normalized_actions":168,"right6_endpoint_delta":6,"right6_anchor_relative_path":48,"request_only_postclose":1,"arm_tokens":2,"postclose_formula":"shape/finite and history[-1,13]<.5 and all future[:,13]<.5"}
 expected_gate={"continuous_over_v169_max":.95,"uint8_over_v169_max":1.,"factual_branch_max":1.,"each_counterfactual_branch_max":.98,"action_over_context_only_max":.95,"action_over_phase_shuffle_max":.97,"positive_target_delta_cosine_fraction_min":.6,"distinct_action_prediction_nonzero_fraction_min":.95,"passing_folds_min":4,"improved_episodes_min":12}
 if pre.get("training")!=expected_training or pre.get("features")!=expected_features or pre.get("kill_gate")!=expected_gate or pre.get("common_v169_seed")!="sha256('v458/1611/{episode}/{start}')[:8] little-endian mod 2**31; branch-independent":raise RuntimeError("v458 prereg constants drift")
 final_dir=a.output_dir;work_dir=final_dir.with_name(final_dir.name+".partial")
 if final_dir.exists() or work_dir.exists():raise RuntimeError("v458 output/partial already exists")
 a.output_dir=work_dir;contexts,samples=validate_and_load(a,pre);a.output_dir.mkdir(parents=True);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);device=torch.device(a.device);v169=Track2V169ArmRoutedRuntime(a.v169_release,a.v169_library,a.device);cache_v169(contexts,samples,v169);del v169;gc.collect();torch.cuda.empty_cache() if device.type=="cuda" else None
 closure_digest=hashlib.sha256(json.dumps({"v169":pre["v169"],"source":pre["source"],"v461_files":pre["v461"]["files"]},sort_keys=True,separators=(",",":")).encode()).hexdigest();feature_schema={"dim":223,"normalized_actions":168,"right6_endpoint_delta":6,"right6_anchor_relative_path":48,"postclose":1}
 fold_reports=[];outputs={mode:[None]*1000 for mode in ("action","context_only","phase_shuffle")};uints=[None]*1000
 for fold in range(5):
  fit_contexts={i for i,x in enumerate(contexts) if x["fold"]!=fold};fit_ids=np.asarray([i for i,x in enumerate(samples) if x["context_id"] in fit_contexts]);hold=np.asarray([i for i,x in enumerate(samples) if x["fold"]==fold]);mean,std=stats(contexts,samples,fit_ids);torch.manual_seed(SEED+fold);initial=copy.deepcopy(EndpointResidualUNet128FiLM(16).state_dict());schedule=schedules(fit_ids,fit_contexts,SEED+fold)
  if schedule_sha(schedule)!=pre["training_schedules"][fold]["sha256"]:raise RuntimeError("v458 training schedule drift")
  fit_donors=donor_ids(pre,fold,contexts,samples,"fit");hold_donors=donor_ids(pre,fold,contexts,samples,"holdout");models={};losses={}
  for mode in ("action","context_only","phase_shuffle"):models[mode],losses[mode]=train_head(initial,contexts,samples,fit_ids,fit_contexts,mean,std,device,schedule,mode,fit_donors)
  preds={};uint=None
  for mode in models:preds[mode],candidate_uint=predict(models[mode],contexts,samples,hold,mean,std,device,mode,hold_donors);uint=candidate_uint if mode=="action" else uint
  target=np.stack([samples[int(i)]["target"] for i in hold]);base=np.stack([samples[int(i)]["baseline"] for i in hold]);am=mae(preds["action"],target);bm=mae(base,target);branch={BRANCHES[b]:float(am[[samples[int(i)]["branch"]==b for i in hold]].mean()/bm[[samples[int(i)]["branch"]==b for i in hold]].mean()) for b in range(5)};cos,nz=causal(hold,samples,preds["action"],target.astype(np.float32));episode_wins=sum(am[[samples[int(i)]["episode"]==e for i in hold]].mean()<bm[[samples[int(i)]["episode"]==e for i in hold]].mean() for e in sorted({samples[int(i)]["episode"] for i in hold}));ratios={"overall":float(am.mean()/bm.mean()),"uint8":float(mae(uint,target).mean()/bm.mean()),"context_only":float(am.mean()/mae(preds["context_only"],target).mean()),"phase_shuffle":float(am.mean()/mae(preds["phase_shuffle"],target).mean())};passed=ratios["overall"]<=.95 and ratios["uint8"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and ratios["context_only"]<=.95 and ratios["phase_shuffle"]<=.97 and cos>=.6 and nz>=.95
  fold_reports.append({"fold":fold,"ratios":ratios,"branch_ratio":branch,"positive_delta_cosine_fraction":cos,"nonzero_prediction_delta_fraction":nz,"episode_wins":episode_wins,"loss_first_last":{m:[losses[m][0],losses[m][-1]] for m in losses},"passed":passed})
  for mode,model in models.items():torch.save({"format":CHECKPOINT_FORMAT,"step":50,"training_scope":f"fold{fold}-{mode}","channels":16,"model":model.state_dict(),"action_mean":mean,"action_std":std,"precision":"bf16","feature_schema":feature_schema,"closure_digest":closure_digest,"trainer_sha256":pre["source"]["trainer_sha256"],"runtime_sha256":pre["source"]["runtime_sha256"],"schedule_sha256":pre["training_schedules"][fold]["sha256"],"mapping_sha256":hashlib.sha256(json.dumps(pre["phase_shuffle"]["mappings"][fold],sort_keys=True,separators=(",",":")).encode()).hexdigest(),"preregistration_sha256":sha(a.preregistration)},a.output_dir/f"fold{fold}_{mode}_step50.pt")
  for k,i in enumerate(hold):
   for mode in outputs:outputs[mode][int(i)]=preds[mode][k]
   uints[int(i)]=uint[k]
  del models;torch.cuda.empty_cache() if device.type=="cuda" else None
 target=np.stack([x["target"] for x in samples]);base=np.stack([x["baseline"] for x in samples]);action=np.stack(outputs["action"]);am=mae(action,target);bm=mae(base,target);branch={BRANCHES[b]:float(am[[x["branch"]==b for x in samples]].mean()/bm[[x["branch"]==b for x in samples]].mean()) for b in range(5)};cos,nz=causal(np.arange(1000),samples,action,target.astype(np.float32));episode_wins=sum(am[[x["episode"]==e for x in samples]].mean()<bm[[x["episode"]==e for x in samples]].mean() for e in sorted({x["episode"] for x in samples}));aggregate={"passing_folds":sum(x["passed"] for x in fold_reports),"continuous_ratio":float(am.mean()/bm.mean()),"uint8_ratio":float(mae(np.stack(uints),target).mean()/bm.mean()),"branch_ratio":branch,"action_over_context_only":float(am.mean()/mae(np.stack(outputs["context_only"]),target).mean()),"action_over_phase_shuffle":float(am.mean()/mae(np.stack(outputs["phase_shuffle"]),target).mean()),"positive_delta_cosine_fraction":cos,"nonzero_prediction_delta_fraction":nz,"improved_episodes":episode_wins}
 passed=aggregate["passing_folds"]>=4 and aggregate["continuous_ratio"]<=.95 and aggregate["uint8_ratio"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and aggregate["action_over_context_only"]<=.95 and aggregate["action_over_phase_shuffle"]<=.97 and cos>=.6 and nz>=.95 and episode_wins>=12;all200=None
 s0_report={"format":"strict-track2-v458-step50-three-head-5fold-s0-receipt-v1","created_at":datetime.now(timezone.utc).isoformat(),"passed":passed,"folds":fold_reports,"aggregate":aggregate,"all200_training_started":False,"guards":{"v461_final_receipt_passed_before_data_read":True,"reward_loaded":False,"policy_updates":0,"rl_authorized":False}}
 atomic_json(a.output_dir/"s0_report.json",s0_report)
 if passed:
  ids=np.arange(1000);fit=set(range(200));mean,std=stats(contexts,samples,ids);torch.manual_seed(SEED+100);initial=EndpointResidualUNet128FiLM(16).state_dict();schedule=schedules(ids,fit,SEED+100)
  if schedule_sha(schedule)!=pre["all200_schedule_sha256"]:raise RuntimeError("v458 all200 schedule drift")
  model,loss=train_head(initial,contexts,samples,ids,fit,mean,std,device,schedule,"action",{});all200=a.output_dir/"all200_action_step50.pt";torch.save({"format":CHECKPOINT_FORMAT,"step":50,"training_scope":"all200","channels":16,"model":model.state_dict(),"action_mean":mean,"action_std":std,"precision":"bf16","feature_schema":feature_schema,"closure_digest":closure_digest,"trainer_sha256":pre["source"]["trainer_sha256"],"runtime_sha256":pre["source"]["runtime_sha256"],"schedule_sha256":schedule_sha(schedule),"mapping_sha256":None,"preregistration_sha256":sha(a.preregistration)},all200)
 report={"format":"strict-track2-v458-step50-three-head-5fold-killgate-v2","created_at":datetime.now(timezone.utc).isoformat(),"passed":passed,"folds":fold_reports,"aggregate":aggregate,"s0_report_sha256":sha(a.output_dir/"s0_report.json"),"all200_training_performed":all200 is not None,"all200_checkpoint_sha256":sha(all200) if all200 else None,"all200_training_authorized":passed,"guards":{"v461_final_receipt_passed_before_data_read":True,"reward_loaded":False,"policy_updates":0,"rl_authorized":False}};atomic_json(a.output_dir/"training_report.json",report)
 if not passed:atomic_json(a.output_dir/"failure_receipt.json",{"passed":False,"reason":"preregistered S0 gate failed","all200_training_performed":False})
 for path in a.output_dir.iterdir():
  if path.is_file():
   with path.open("rb") as handle:os.fsync(handle.fileno())
 descriptor=os.open(a.output_dir,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
 if passed:
  os.replace(a.output_dir,final_dir);descriptor=os.open(final_dir.parent,os.O_RDONLY);os.fsync(descriptor);os.close(descriptor)
 print(json.dumps(report,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
