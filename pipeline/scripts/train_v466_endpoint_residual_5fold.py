#!/usr/bin/env python3
"""V466 5-fold canonical-no-transport appearance + paired-dynamics S0."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,random
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v466_canonical_notransport_paired_delta_runtime import CHECKPOINT_FORMAT,FEATURE_SCHEMA,EndpointResidualUNet128FiLM,action_features,appearance_features,canonical_no_transport,instruction_tokens,no_transport_mask
SEED=1616;STEPS=50;BATCH=8;BRANCHES=("factual","no_transport","scale_0p4","scale_1p25","reverse_direction_0p4")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8<<20),b""):h.update(block)
 return h.hexdigest()
def json_default(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,np.ndarray):return value.tolist()
 raise TypeError(f"not JSON serializable: {type(value)!r}")
def atomic_json(path,payload):
 path=Path(path);tmp=path.with_name(path.name+".tmp")
 if tmp.exists():raise RuntimeError(f"stale v466 tmp: {tmp}")
 with tmp.open("x") as handle:json.dump(payload,handle,indent=2,default=json_default);handle.write("\n");handle.flush();os.fsync(handle.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def stable_seed(episode,start):return int.from_bytes(hashlib.sha256(f"v466/{SEED}/{episode}/{start}".encode()).digest()[:8],"little")%(2**31)
def arrsha(value):return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
def inside(path,root):path=Path(path).resolve();root=Path(root).resolve();return path==root or root in path.parents
def directory_target_sha(path):
 path=Path(path).resolve();items=[]
 for parent,dirs,files in os.walk(path,followlinks=False):
  for name in sorted(dirs+files):
   item=Path(parent)/name;rel=str(item.relative_to(path));items.append(["link",rel,os.readlink(item)] if item.is_symlink() else ["file",rel,sha(item)] if item.is_file() else ["dir",rel,None])
 return hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def release_inventory(release,joint_root):
 release=Path(release).absolute();joint_root=Path(joint_root).resolve();items=[];seen_dirs=set();seen_files=set();active=set()
 def visit(logical,relative):
  target=logical.resolve();key=(target.stat().st_dev,target.stat().st_ino)
  if not inside(target,joint_root) or not target.is_dir() or key in active or key in seen_dirs:raise RuntimeError("v466 release inventory cycle/duplicate/escape")
  active.add(key);seen_dirs.add(key)
  for child in sorted(logical.iterdir(),key=lambda x:x.name):
   rel=relative/child.name;resolved=child.resolve()
   if not inside(resolved,joint_root):raise RuntimeError("v466 release target escape")
   if child.is_symlink():items.append(("directory_symlink" if resolved.is_dir() else "file_symlink","v169_release_link",str(rel),str(child.absolute()),os.readlink(child),str(resolved)))
   if resolved.is_dir():visit(child,rel)
   elif resolved.is_file():
    key=(resolved.stat().st_dev,resolved.stat().st_ino)
    if key in seen_files:raise RuntimeError("v466 duplicate release file target")
    seen_files.add(key)
    if not child.is_symlink():items.append(("file","v169_release_target" if not inside(resolved,release.resolve()) else "v169_release",str(rel),str(child.absolute()),None,str(resolved)))
   else:raise RuntimeError("v466 unsupported release target")
  active.remove(key)
 visit(release,Path("."));return sorted(items)
def verify_closure(pre,release,library):
 release=Path(release).absolute();library=Path(library).resolve();joint=release.resolve().parent;roots={"v169_release":release,"v169_release_target":release,"v169_release_link":release,"v169_base_release":(release/"v168_release/base_release").resolve(),"v169_library":library};seen=set()
 for records in (pre["v169"]["release_files"],pre["v169"]["library_files"]):
  for row in records:
   base=roots[row["base_kind"]];lexical=(base/row["relative"]).absolute();resolved=lexical.resolve();key=(row["record_type"],row["base_kind"],row["relative"],row["resolved_path"])
   if key in seen or str(lexical)!=row["lexical_path"] or str(resolved)!=row["resolved_path"] or not inside(resolved,library if row["base_kind"]=="v169_library" else joint):raise RuntimeError("v466 closure drift")
   seen.add(key);link=os.readlink(lexical) if lexical.is_symlink() else None
   if link!=row["link_target"]:raise RuntimeError("v466 link drift")
   digest=directory_target_sha(resolved) if row["record_type"]=="directory_symlink" else sha(resolved)
   if digest!=row["target_sha"]:raise RuntimeError("v466 closure SHA drift")
 declared=sorted((x["record_type"],x["base_kind"],x["relative"],x["lexical_path"],x["link_target"],x["resolved_path"]) for x in pre["v169"]["release_files"])
 if declared!=release_inventory(release,joint):raise RuntimeError("v466 exact release inventory drift")
def validate_and_load(args,pre):
 if sha(Path(__file__))!=pre["source"]["trainer_sha256"] or sha(Path(pre["source"]["runtime_path"]))!=pre["source"]["runtime_sha256"]:raise RuntimeError("v466 source drift")
 if sha(args.contract)!=pre["source"]["contract_sha256"]:raise RuntimeError("v466 contract drift")
 verify_closure(pre,args.v169_release,args.v169_library)
 receipt_path=Path(pre["v461"]["final_receipt"]["path"]);receipt=json.loads(receipt_path.read_text())
 if sha(receipt_path)!=pre["v461"]["final_receipt"]["sha256"] or receipt.get("passed") is not True or receipt.get("mode")!="final":raise RuntimeError("v466 blocked: v461 final")
 for key,arg in (("selection",args.v456_selection),("generation_report",args.v456_generation_report)):
  if sha(arg)!=pre["v461"][key]["sha256"]:raise RuntimeError("v466 v461 closure")
 for row in pre["v461"]["files"]:
  path=args.v456_dataset/row["relative"]
  if sha(path)!=row["sha256"]:raise RuntimeError("v466 dataset drift")
 specs={(int(x["episode"]),int(x["start"])):x for x in json.loads(args.v456_selection.read_text())["contexts"]};contexts=[];samples=[]
 for batch in range(10):
  for rowdir in sorted((args.v456_dataset/f"batch_{batch:03d}"/"rows").iterdir()):
   receipt=json.loads((rowdir/"receipt.json").read_text())
   with np.load(rowdir/"endpoint.npz",allow_pickle=False) as z:
    episode=int(z["episode"]);start=int(z["start"]);spec=specs[(episode,start)];history=z["history_actions"].astype(np.float32);futures=z["future_actions"][:5].astype(np.float32);context=z["branch_context_rgb"][0].astype(np.uint8);target=z["endpoint_rgb"][:5].astype(np.uint8);variants=list(map(str,z["variants"][:5]))
    if variants!=list(BRANCHES) or arrsha(history)!=spec["history_action_sha256"] or [arrsha(x) for x in futures]!=[spec["branch_action_sha256"][x] for x in BRANCHES] or receipt.get("selection_action_sha256")!=spec["branch_action_sha256"]:raise RuntimeError("v466 row action drift")
   cid=len(contexts);contexts.append({"episode":episode,"start":start,"fold":int(spec["fold"]),"prompt":spec["instruction"],"context":context,"history":history,"targets":target})
   for branch in range(5):samples.append({"context_id":cid,"episode":episode,"fold":int(spec["fold"]),"branch":branch,"future":futures[branch],"target":target[branch]})
 if len(contexts)!=200 or len(samples)!=1000:raise RuntimeError("v466 exact data contract")
 return contexts,samples
def v169_micro(v169,contexts,samples):
 for begin in range(0,len(samples),4):
  part=samples[begin:begin+4];ctx=np.stack([np.repeat(contexts[x["context_id"]]["context"][None],5,0) for x in part]);history=np.stack([contexts[x["context_id"]]["history"] for x in part]);future=np.stack([x["future"] for x in part]);prompts=[contexts[x["context_id"]]["prompt"] for x in part];seeds=np.asarray([stable_seed(x["episode"],contexts[x["context_id"]]["start"]) for x in part]);pred=v169.predict_batch(ctx,history,future,seeds,prompts)
  for i,row in enumerate(part):row["comparator"]=pred[i,7].copy()
 for begin in range(0,len(contexts),4):
  part=contexts[begin:begin+4];ctx=np.stack([np.repeat(x["context"][None],5,0) for x in part]);history=np.stack([x["history"] for x in part]);requested=np.stack([samples[5*(begin+i)]["future"] for i in range(len(part))]);future=canonical_no_transport(history,requested);prompts=[x["prompt"] for x in part];seeds=np.asarray([stable_seed(x["episode"],x["start"]) for x in part]);pred=v169.predict_batch(ctx,history,future,seeds,prompts)
  for i,row in enumerate(part):row["b0"]=pred[i,7].copy()
def stats(contexts,samples,ids):
 values=np.concatenate([np.concatenate((contexts[samples[i]["context_id"]]["history"],samples[i]["future"]),0) for i in ids],0).astype(np.float64);return values.mean(0).astype(np.float32),np.maximum(values.std(0),1e-4).astype(np.float32)
def schedule_sha(schedule):return hashlib.sha256(json.dumps([{"ids":list(map(int,x[0])),"context_id":int(x[1])} for x in schedule],sort_keys=True,separators=(",",":")).encode()).hexdigest()
def make_schedule(ids,contexts,seed):
 rng=np.random.default_rng(seed);ordered=sorted(contexts);return [(rng.choice(ids,size=BATCH,replace=False),ordered[i%len(ordered)]) for i in range(STEPS)]
def donor_ids(pre,fold,contexts,samples,partition):
 mapping=pre["phase_shuffle"]["mappings"][fold][partition];lookup={(x["episode"],x["start"]):i for i,x in enumerate(contexts)};result={}
 for i,row in enumerate(samples):
  context=contexts[row["context_id"]];key=f"{context['episode']}:{context['start']}"
  if key in mapping:result[i]=5*lookup[(mapping[key]["donor_episode"],mapping[key]["donor_start"])]+row["branch"]
 return result
def d_features(contexts,samples,feature_ids,mean,std,mode):
 if mode=="context_only":value=np.zeros((len(feature_ids),223),np.float32);value[:,-1]=1.;return value
 h=np.stack([contexts[samples[int(i)]["context_id"]]["history"] for i in feature_ids]);f=np.stack([samples[int(i)]["future"] for i in feature_ids]);return action_features(h,f,mean,std)
def common_tensors(contexts,samples,ids,device):
 rows=[samples[int(i)] for i in ids];base=torch.as_tensor(np.stack([contexts[x["context_id"]]["b0"] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;last=torch.as_tensor(np.stack([contexts[x["context_id"]]["context"][-1] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;tokens=torch.as_tensor(instruction_tokens([contexts[x["context_id"]]["prompt"] for x in rows]),device=device);return rows,base,last,tokens
def grad_l1(pred,target):return .5*(F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])+F.l1_loss(pred[:,:,1:]-pred[:,:,:-1],target[:,:,1:]-target[:,:,:-1]))
def train_appearance(initial,contexts,fit_contexts,device,schedule):
 model=EndpointResidualUNet128FiLM(16).to(device);model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[]
 for ids,_ in schedule:
  rows=[contexts[int(i)] for i in ids];base=torch.as_tensor(np.stack([x["b0"] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;last=torch.as_tensor(np.stack([x["context"][-1] for x in rows]),dtype=torch.float32,device=device).permute(0,3,1,2)/255.;feat=torch.as_tensor(appearance_features(len(rows)),device=device);tok=torch.as_tensor(instruction_tokens([x["prompt"] for x in rows]),device=device);target=torch.as_tensor(np.stack([x["targets"][1].astype(np.float32)-x["b0"].astype(np.float32) for x in rows]),device=device).permute(0,3,1,2)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):pred=model(base,last,feat,tok);loss=F.l1_loss(pred.float(),target)+.25*grad_l1(pred.float(),target)
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
 return model.eval(),losses
def train_dynamics(initial,contexts,samples,fit_ids,device,schedule,mean,std,mode,donors):
 model=EndpointResidualUNet128FiLM(16).to(device);model.load_state_dict(copy.deepcopy(initial));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[]
 for ids,cid in schedule:
  fids=[donors[int(i)] for i in ids] if mode=="phase_shuffle" else ids;rows,base,last,tok=common_tensors(contexts,samples,ids,device);feat=torch.as_tensor(d_features(contexts,samples,fids,mean,std,mode),device=device);target=torch.as_tensor(np.stack([x["target"].astype(np.float32)-contexts[x["context_id"]]["targets"][1].astype(np.float32) for x in rows]),device=device).permute(0,3,1,2);mask=torch.as_tensor(no_transport_mask(np.stack([contexts[x["context_id"]]["history"] for x in rows]),np.stack([x["future"] for x in rows])),device=device)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
   pred=model(base,last,feat,tok);pred=torch.where(mask[:,None,None,None],torch.zeros_like(pred),pred);loss=F.l1_loss(pred.float(),target)+.25*grad_l1(pred.float(),target);group=np.asarray([5*cid+b for b in range(5)]);gf=[donors[int(i)] for i in group] if mode=="phase_shuffle" else group;grows,gb,gl,gt=common_tensors(contexts,samples,group,device);gfeat=torch.as_tensor(d_features(contexts,samples,gf,mean,std,mode),device=device);gpred=model(gb,gl,gfeat,gt);gmask=torch.as_tensor(no_transport_mask(np.stack([contexts[x["context_id"]]["history"] for x in grows]),np.stack([x["future"] for x in grows])),device=device);gpred=torch.where(gmask[:,None,None,None],torch.zeros_like(gpred),gpred);gta=torch.as_tensor(np.stack([x["target"].astype(np.float32)-contexts[x["context_id"]]["targets"][1].astype(np.float32) for x in grows]),device=device).permute(0,3,1,2);loss=loss+.5*F.l1_loss((gpred-gpred[1:2]).float(),gta-gta[1:2])
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
 return model.eval(),losses
@torch.inference_mode()
def predict_models(appearance,dynamics,contexts,samples,ids,device,mean,std,mode,donors):
 out=[]
 for begin in range(0,len(ids),BATCH):
  part=np.asarray(ids[begin:begin+BATCH]);fids=[donors[int(i)] for i in part] if mode=="phase_shuffle" else part;rows,base,last,tok=common_tensors(contexts,samples,part,device);afeat=torch.as_tensor(appearance_features(len(part)),device=device);dfeat=torch.as_tensor(d_features(contexts,samples,fids,mean,std,mode),device=device);mask=torch.as_tensor(no_transport_mask(np.stack([contexts[x["context_id"]]["history"] for x in rows]),np.stack([x["future"] for x in rows])),device=device)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):a=appearance(base,last,afeat,tok);d=dynamics(base,last,dfeat,tok);d=torch.where(mask[:,None,None,None],torch.zeros_like(d),d);candidate=(255*base+a+d).clamp(0,255)
  out.append(candidate.permute(0,2,3,1).float().cpu().numpy())
 continuous=np.concatenate(out);return continuous,np.clip(np.rint(continuous),0,255).astype(np.uint8)
def mae(pred,target):return np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean((1,2,3))
def causal(ids,samples,pred,target):
 pos={int(idx):i for i,idx in enumerate(ids)};cos=[];nz=[]
 for cid in sorted({samples[int(i)]["context_id"] for i in ids}):
  group=[5*cid+b for b in range(5)];p=np.stack([pred[pos[i]] for i in group]);t=np.stack([target[pos[i]] for i in group])
  for j in range(1,5):
   pd=(p[j]-p[0]).reshape(-1);td=(t[j]-t[0]).reshape(-1);cos.append(float(np.dot(pd,td)/(np.linalg.norm(pd)*np.linalg.norm(td)+1e-9))>0);nz.append(float(np.abs(pd).mean())>1e-6)
 return float(np.mean(cos)),float(np.mean(nz))
def main():
 p=argparse.ArgumentParser()
 for name in ("preregistration","contract","v456-selection","v456-generation-report","v456-dataset","v169-release","v169-library","output-dir"):p.add_argument(f"--{name}",type=Path,required=True)
 p.add_argument("--device",default="cuda");a=p.parse_args();pre=json.loads(a.preregistration.read_text());final=a.output_dir;work=final.with_name(final.name+".partial")
 if pre.get("format")!="strict-track2-v466-canonical-notransport-paired-delta-preregistration-v1" or final.exists() or work.exists():raise RuntimeError("v466 prereg/output contract")
 a.output_dir=work;contexts,samples=validate_and_load(a,pre);work.mkdir(parents=True);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);device=torch.device(a.device);v169=Track2V169ArmRoutedRuntime(a.v169_release,a.v169_library,a.device);v169_micro(v169,contexts,samples);del v169;gc.collect();torch.cuda.empty_cache() if device.type=="cuda" else None
 closure=hashlib.sha256(json.dumps({"v169":pre["v169"],"source":pre["source"],"v461_files":pre["v461"]["files"]},sort_keys=True,separators=(",",":")).encode()).hexdigest();folds=[];failed=0;outputs={m:[None]*1000 for m in ("action","context_only","phase_shuffle")};uints=[None]*1000
 for fold in range(5):
  fit_contexts={i for i,x in enumerate(contexts) if x["fold"]!=fold};fit_ids=np.asarray([i for i,x in enumerate(samples) if x["context_id"] in fit_contexts]);hold=np.asarray([i for i,x in enumerate(samples) if x["fold"]==fold]);mean,std=stats(contexts,samples,fit_ids);torch.manual_seed(SEED+fold);initial=copy.deepcopy(EndpointResidualUNet128FiLM(16).state_dict());asched=make_schedule(np.asarray(sorted(fit_contexts)),fit_contexts,1816+fold);dsched=make_schedule(fit_ids,fit_contexts,SEED+fold)
  if schedule_sha(asched)!=pre["appearance_schedules"][fold]["sha256"] or schedule_sha(dsched)!=pre["dynamics_schedules"][fold]["sha256"]:raise RuntimeError("v466 schedule drift")
  appearance,aloss=train_appearance(initial,contexts,fit_contexts,device,asched);fitdon=donor_ids(pre,fold,contexts,samples,"fit");holddon=donor_ids(pre,fold,contexts,samples,"holdout");models={};losses={}
  for mode in ("action","context_only","phase_shuffle"):models[mode],losses[mode]=train_dynamics(initial,contexts,samples,fit_ids,device,dsched,mean,std,mode,fitdon)
  predictions={};uint=None
  for mode in models:predictions[mode],candidate=predict_models(appearance,models[mode],contexts,samples,hold,device,mean,std,mode,holddon);uint=candidate if mode=="action" else uint
  target=np.stack([samples[int(i)]["target"] for i in hold]);base=np.stack([samples[int(i)]["comparator"] for i in hold]);am=mae(predictions["action"],target);bm=mae(base,target);branch={BRANCHES[b]:float(am[[samples[int(i)]["branch"]==b for i in hold]].mean()/bm[[samples[int(i)]["branch"]==b for i in hold]].mean()) for b in range(5)};cos,nz=causal(hold,samples,predictions["action"],target.astype(np.float32));episode_wins=int(sum(am[[samples[int(i)]["episode"]==e for i in hold]].mean()<bm[[samples[int(i)]["episode"]==e for i in hold]].mean() for e in sorted({samples[int(i)]["episode"] for i in hold})));ratios={"overall":float(am.mean()/bm.mean()),"uint8":float(mae(uint,target).mean()/bm.mean()),"context_only":float(am.mean()/mae(predictions["context_only"],target).mean()),"phase_shuffle":float(am.mean()/mae(predictions["phase_shuffle"],target).mean())};passed=ratios["overall"]<=.95 and ratios["uint8"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and ratios["context_only"]<=.95 and ratios["phase_shuffle"]<=.97 and cos>=.6 and nz>=.95
  folds.append({"fold":fold,"ratios":ratios,"branch_ratio":branch,"positive_delta_cosine_fraction":cos,"nonzero_prediction_delta_fraction":nz,"episode_wins":episode_wins,"appearance_loss_first_last":[aloss[0],aloss[-1]],"dynamics_loss_first_last":{m:[losses[m][0],losses[m][-1]] for m in losses},"passed":passed})
  common={"format":CHECKPOINT_FORMAT,"step":50,"channels":16,"precision":"bf16","feature_schema":FEATURE_SCHEMA,"closure_digest":closure,"trainer_sha256":pre["source"]["trainer_sha256"],"runtime_sha256":pre["source"]["runtime_sha256"],"preregistration_sha256":sha(a.preregistration)}
  torch.save({**common,"training_scope":f"fold{fold}-appearance","model":appearance.state_dict(),"action_mean":mean,"action_std":std,"schedule_sha256":pre["appearance_schedules"][fold]["sha256"]},work/f"fold{fold}_appearance_step50.pt")
  for mode,model in models.items():torch.save({**common,"training_scope":f"fold{fold}-dynamics-{mode}","model":model.state_dict(),"action_mean":mean,"action_std":std,"schedule_sha256":pre["dynamics_schedules"][fold]["sha256"]},work/f"fold{fold}_dynamics_{mode}_step50.pt")
  for k,index in enumerate(hold):
   for mode in outputs:outputs[mode][int(index)]=predictions[mode][k]
   uints[int(index)]=uint[k]
  if not passed:failed+=1
  atomic_json(work/f"fold{fold}_receipt.json",folds[-1])
  del appearance,models;torch.cuda.empty_cache() if device.type=="cuda" else None
  if failed>=2:break
 early=failed>=2;passed=False;aggregate=None;all200_appearance=None;all200_dynamics=None
 if not early and len(folds)==5:
  target=np.stack([x["target"] for x in samples]);base=np.stack([x["comparator"] for x in samples]);action=np.stack(outputs["action"]);am=mae(action,target);bm=mae(base,target);branch={BRANCHES[b]:float(am[[x["branch"]==b for x in samples]].mean()/bm[[x["branch"]==b for x in samples]].mean()) for b in range(5)};cos,nz=causal(np.arange(1000),samples,action,target.astype(np.float32));episode_wins=int(sum(am[[x["episode"]==e for x in samples]].mean()<bm[[x["episode"]==e for x in samples]].mean() for e in sorted({x["episode"] for x in samples})));aggregate={"passing_folds":sum(x["passed"] for x in folds),"continuous_ratio":float(am.mean()/bm.mean()),"uint8_ratio":float(mae(np.stack(uints),target).mean()/bm.mean()),"branch_ratio":branch,"action_over_context_only":float(am.mean()/mae(np.stack(outputs["context_only"]),target).mean()),"action_over_phase_shuffle":float(am.mean()/mae(np.stack(outputs["phase_shuffle"]),target).mean()),"positive_delta_cosine_fraction":cos,"nonzero_prediction_delta_fraction":nz,"improved_episodes":episode_wins};passed=aggregate["passing_folds"]>=4 and aggregate["continuous_ratio"]<=.95 and aggregate["uint8_ratio"]<=1 and branch["factual"]<=1 and all(branch[x]<=.98 for x in BRANCHES[1:]) and aggregate["action_over_context_only"]<=.95 and aggregate["action_over_phase_shuffle"]<=.97 and cos>=.6 and nz>=.95 and episode_wins>=12
  if passed:
   all_ids=np.arange(1000);all_contexts=set(range(200));mean,std=stats(contexts,samples,all_ids);asched=make_schedule(np.arange(200),all_contexts,1916);dsched=make_schedule(all_ids,all_contexts,1716)
   if schedule_sha(asched)!=pre["all200_appearance_schedule_sha256"] or schedule_sha(dsched)!=pre["all200_dynamics_schedule_sha256"]:raise RuntimeError("v466 all200 schedule drift")
   torch.manual_seed(1916);ainitial=copy.deepcopy(EndpointResidualUNet128FiLM(16).state_dict());all200_appearance,_=train_appearance(ainitial,contexts,all_contexts,device,asched);torch.manual_seed(1716);dinitial=copy.deepcopy(EndpointResidualUNet128FiLM(16).state_dict());all200_dynamics,_=train_dynamics(dinitial,contexts,samples,all_ids,device,dsched,mean,std,"action",{})
   common={"format":CHECKPOINT_FORMAT,"step":50,"channels":16,"precision":"bf16","feature_schema":FEATURE_SCHEMA,"closure_digest":closure,"trainer_sha256":pre["source"]["trainer_sha256"],"runtime_sha256":pre["source"]["runtime_sha256"],"preregistration_sha256":sha(a.preregistration)}
   torch.save({**common,"training_scope":"all200-appearance","model":all200_appearance.state_dict(),"action_mean":mean,"action_std":std,"schedule_sha256":pre["all200_appearance_schedule_sha256"]},work/"all200_appearance_step50.pt");torch.save({**common,"training_scope":"all200-dynamics-action","model":all200_dynamics.state_dict(),"action_mean":mean,"action_std":std,"schedule_sha256":pre["all200_dynamics_schedule_sha256"]},work/"all200_dynamics_action_step50.pt")
 s0={"format":"strict-track2-v466-canonical-notransport-paired-delta-s0-v1","created_at":datetime.now(timezone.utc).isoformat(),"passed":passed,"early_stop_mathematically_unreachable":early,"completed_folds":len(folds),"failed_folds":failed,"folds":folds,"aggregate":aggregate,"all200_training_started":bool(passed),"guards":{"reward_loaded":False,"policy_updates":0,"rl_authorized":False}}
 atomic_json(work/"s0_report.json",s0);atomic_json(work/"training_report.json",{**s0,"format":"strict-track2-v466-canonical-notransport-paired-delta-training-report-v1","all200_training_performed":bool(passed),"appearance_checkpoint_sha256":sha(work/"all200_appearance_step50.pt") if passed else None,"dynamics_checkpoint_sha256":sha(work/"all200_dynamics_action_step50.pt") if passed else None})
 if not passed:atomic_json(work/"failure_receipt.json",{"passed":False,"reason":"second failed fold; passing_folds_min=4 mathematically unreachable" if early else "preregistered S0 aggregate failed","all200_training_performed":False})
 for path in work.iterdir():
  if path.is_file():
   with path.open("rb") as handle:os.fsync(handle.fileno())
 if passed:os.replace(work,final);fd=os.open(final.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(s0,indent=2,default=json_default));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
