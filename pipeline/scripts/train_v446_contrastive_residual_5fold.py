#!/usr/bin/env python3
"""Run all five frozen v446 train-only folds; train all15 only after every gate passes."""

from __future__ import annotations

import argparse, hashlib, json, random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision
from wam_pipeline.v446_v169_contrastive_residual_unet_runtime import (
    CHECKPOINT_FORMAT, ResidualUNet128FiLM, apply_residual, instruction_tokens,
)


SEED=1597; STEPS=50; BATCH=4; VARIANTS=("true","shuffle","open","static","reverse"); NEG=VARIANTS[1:]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_inputs(args, prereg: dict) -> dict[str, bool]:
    inputs = prereg.get("inputs", {})
    actual = {
        "split": args.split,
        "v169_release_manifest": args.v169_release / "v169_arm_routed_manifest.json",
        "v169_library_manifest": args.v169_release / "v168_release/base_release/release_manifest.json",
        "v169_library_split_manifest": args.v169_library / "splits/adjust_bottle_50episodes_full.json",
        "reward_checkpoint": args.reward_checkpoint,
        "t5_config": args.t5_model / "config.json",
    }
    checks: dict[str, bool] = {}
    for key, path in actual.items():
        row = inputs.get(key, {})
        checks[f"{key}_path"] = str(path.resolve()) == row.get("resolved_path")
        checks[f"{key}_sha256"] = path.is_file() and sha256(path) == row.get("sha256")
    checks["windows_path"] = str(args.windows.resolve()) == inputs.get("windows", {}).get("resolved_path")
    if not all(checks.values()): raise RuntimeError(f"v446 preregistered input closure mismatch: {checks}")
    return checks


def stable_seed(episode:int,start:int,variant:str,chunk:int=0)->int:
    del variant  # same-request seed is invariant across true/counterfactual actions
    d=hashlib.sha256(f"v446/{SEED}/{episode}/{start}/chunk{chunk}".encode()).digest()
    return int.from_bytes(d[:8],"little")%(2**31)


def action_variants(history:np.ndarray,future:np.ndarray)->dict[str,np.ndarray]:
    true=np.asarray(future,dtype=np.float32)
    shuffled=np.roll(true,4,axis=0).copy()
    opened=true.copy(); opened[:,13]=1.0
    static=np.repeat(np.asarray(history[-1:],dtype=np.float32),8,axis=0)
    reverse=true[::-1].copy()
    return {"true":true,"shuffle":shuffled,"open":opened,"static":static,"reverse":reverse}


def select_rows(windows:Path,episodes:list[int],prompts:dict[int,str])->list[dict]:
    ds=WindowDataset(windows,episodes,rollout_horizon=8); rows=[]; counts={e:0 for e in episodes}
    for index,path in enumerate(ds.paths):
        episode=int(path.name.split("_")[0][7:]); history,future=ds.load_actions(index)
        decision=gate_decision(history,future,prompts[episode])
        if decision.get("gate") and decision.get("phase")=="close":
            arrays=ds.load_arrays(index); start=int(path.stem.split("_")[1])
            rows.append({"episode":episode,"start":start,"source_path":path.name,"prompt":prompts[episode],"variant_actions":action_variants(history,future),**arrays}); counts[episode]+=1
    if len(rows)!=120 or any(value!=8 for value in counts.values()): raise RuntimeError(f"v446 exact close120 drift: {counts}")
    return rows


def cache_baselines(rows:list[dict],v169,device_batch:int=4)->None:
    for variant in VARIANTS:
        for begin in range(0,len(rows),device_batch):
            batch=rows[begin:begin+device_batch]
            context=np.stack([r["context_frames"] for r in batch]); history=np.stack([r["history_actions"] for r in batch])
            future=np.stack([r["variant_actions"][variant] for r in batch]); prompts=[r["prompt"] for r in batch]
            seeds=np.asarray([stable_seed(r["episode"],r["start"],variant) for r in batch],dtype=np.int64)
            baseline=v169.predict_batch(context,history,future,seeds,prompts)
            for local,row in enumerate(batch): row.setdefault("baseline",{})[variant]=baseline[local].copy()


def stats(rows:list[dict])->tuple[np.ndarray,np.ndarray]:
    values=np.concatenate([np.concatenate((r["history_actions"],r["variant_actions"]["true"]),axis=0) for r in rows],axis=0).astype(np.float64)
    mean=values.mean(0).astype(np.float32); std=np.maximum(values.std(0).astype(np.float32),1e-4); return mean,std


def batch_tensors(rows:list[dict],indices:np.ndarray,mean:np.ndarray,std:np.ndarray,device):
    selected=[rows[int(i)] for i in indices]; contexts=np.stack([r["context_frames"][-1] for r in selected])
    target=np.stack([r["target_frames"] for r in selected]).astype(np.float32)
    bases=[]; actions=[]; tokens=[]
    for variant in VARIANTS:
        bases.append(np.stack([r["baseline"][variant] for r in selected]))
        actions.append(np.stack([np.concatenate((r["history_actions"],r["variant_actions"][variant]),axis=0) for r in selected]))
        tokens.extend([r["prompt"] for r in selected])
    base_np=np.concatenate(bases,axis=0); action_np=np.concatenate(actions,axis=0)
    base=torch.as_tensor(base_np,dtype=torch.float32,device=device).permute(0,1,4,2,3)/255.0
    context=torch.as_tensor(np.tile(contexts,(len(VARIANTS),1,1,1)),dtype=torch.float32,device=device).permute(0,3,1,2)/255.0
    action=torch.as_tensor((action_np-mean)/std,dtype=torch.float32,device=device)
    token=torch.as_tensor(instruction_tokens(tokens),device=device)
    target=torch.as_tensor(target,dtype=torch.float32,device=device).permute(0,1,4,2,3)
    return base,context,action,token,target


def train_model(rows:list[dict],device,seed_offset:int):
    mean,std=stats(rows); torch.manual_seed(SEED+seed_offset); torch.cuda.manual_seed_all(SEED+seed_offset)
    model=ResidualUNet128FiLM(16).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(SEED+seed_offset); order=np.asarray([],dtype=np.int64); losses=[]; model.train()
    for _step in range(STEPS):
        if len(order)<BATCH: order=np.concatenate((order,rng.permutation(len(rows))))
        index,order=order[:BATCH],order[BATCH:]
        base,context,action,token,target=batch_tensors(rows,index,mean,std,device)
        prediction=model(base,context,action,token); parts=prediction.split(BATCH,dim=0); base_parts=base.split(BATCH,dim=0)
        target128=F.interpolate(target.flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).reshape(BATCH,8,3,128,128)
        true_base128=F.interpolate((255.0*base_parts[0]).flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).reshape(BATCH,8,3,128,128)
        true_target=(target128-true_base128).clamp(-4,4)
        true_pred=F.interpolate(parts[0].flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).reshape_as(true_target)
        true_loss=F.l1_loss(true_pred,true_target)
        neg_loss=torch.stack([part.abs().mean() for part in parts[1:]]).mean()
        true_error=(true_base128+true_pred-target128).abs().mean((1,2,3,4)); hinges=[]
        for part,negbase in zip(parts[1:],base_parts[1:]):
            neg128=F.interpolate(part.flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).reshape_as(true_target)
            negbase128=F.interpolate((255.0*negbase).flatten(0,1),size=(128,128),mode="bilinear",align_corners=False).reshape_as(true_target)
            neg_error=(negbase128+neg128-target128).abs().mean((1,2,3,4)); hinges.append(F.relu(true_error-neg_error+.10).mean())
        hinge=torch.stack(hinges).mean(); loss=true_loss+.25*neg_loss+.25*hinge
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    model.eval(); return model,mean,std,losses


@torch.inference_mode()
def predict_variant(model,rows:list[dict],variant:str,mean,std,device)->np.ndarray:
    outputs=[]; gates=[]
    for begin in range(0,len(rows),BATCH):
        batch=rows[begin:begin+BATCH]; base_np=np.stack([r["baseline"][variant] for r in batch]); context_np=np.stack([r["context_frames"][-1] for r in batch])
        actions=np.stack([np.concatenate((r["history_actions"],r["variant_actions"][variant]),axis=0) for r in batch])
        base=torch.as_tensor(base_np,dtype=torch.float32,device=device).permute(0,1,4,2,3)/255.; context=torch.as_tensor(context_np,dtype=torch.float32,device=device).permute(0,3,1,2)/255.
        action=torch.as_tensor((actions-mean)/std,dtype=torch.float32,device=device); token=torch.as_tensor(instruction_tokens([r["prompt"] for r in batch]),device=device)
        residual=model(base,context,action,token).permute(0,1,3,4,2).cpu().numpy()
        for i,row in enumerate(batch):
            deployed_gate=gate_decision(row["history_actions"],row["variant_actions"][variant],row["prompt"])["gate"]
            gates.append(bool(deployed_gate))
            outputs.append(apply_residual(base_np[i],residual[i]) if deployed_gate else base_np[i].copy())
    return np.stack(outputs),np.asarray(gates,dtype=np.bool_)


@torch.inference_mode()
def score_reward(model,frames:np.ndarray,prompts:list[str],device,batch_size:int=32)->np.ndarray:
    count,horizon=frames.shape[:2]; flat=torch.from_numpy(frames).permute(0,1,4,2,3).reshape(-1,3,frames.shape[2],frames.shape[3]).float().div_(255.0)
    expanded=[prompt for prompt in prompts for _ in range(horizon)]; values=[]
    for begin in range(0,len(flat),batch_size):
        values.append(model.compute_reward(flat[begin:begin+batch_size].to(device),expanded[begin:begin+batch_size]).float().cpu().numpy())
    return np.concatenate(values).reshape(count,horizon)


def rgb_mae(output:np.ndarray,rows:list[dict])->np.ndarray:
    target=np.stack([r["target_frames"] for r in rows]).astype(np.float64); return np.abs(output.astype(np.float64)-target).mean((1,2,3,4))


def recursive_rows(windows:Path,episodes:list[int],prompts:dict[int,str])->list[dict]:
    short=WindowDataset(windows,episodes,8); candidates={e:[] for e in episodes}
    for i,p in enumerate(short.paths):
        e=int(p.name.split("_")[0][7:]); h,f=short.load_actions(i); d=gate_decision(h,f,prompts[e])
        if d.get("gate"): candidates[e].append((abs(int(d["first_close_index"])-4),int(p.stem.split("_")[1])))
    long=WindowDataset(windows,episodes,32); keys={(int(p.name.split("_")[0][7:]),int(p.stem.split("_")[1])):i for i,p in enumerate(long.paths)}; result=[]
    for e in episodes:
        usable=sorted((dist,start) for dist,start in candidates[e] if (e,start) in keys)
        if not usable: raise RuntimeError(f"v446 episode {e} no close recursive32 row")
        _,start=usable[0]; arrays=long.load_arrays(keys[(e,start)]); result.append({"episode":e,"start":start,"prompt":prompts[e],**arrays})
    return result


@torch.inference_mode()
def recursive_eval(rows,fold_for_episode,models,v169,device):
    base_abs=[]; true_abs=[]; chunk_base=[[],[]]; chunk_true=[[],[]]
    for row in rows:
        fold=fold_for_episode[row["episode"]]; model,mean,std=models[fold]; target=row["target_frames"].astype(np.float64)
        contexts={"base":row["context_frames"][None].copy(),"true":row["context_frames"][None].copy()}; histories={"base":row["history_actions"][None].copy(),"true":row["history_actions"][None].copy()}; preds={"base":[],"true":[]}
        for chunk in range(4):
            future=row["future_actions"][chunk*8:(chunk+1)*8][None]; seed=np.asarray([stable_seed(row["episode"],row["start"],"recursive",chunk)])
            for name in ("base","true"):
                b=v169.predict_batch(contexts[name],histories[name],future,seed,[row["prompt"]]); out=b
                if name=="true" and gate_decision(histories[name][0],future[0],row["prompt"])["gate"]:
                    actions=np.concatenate((histories[name],future),axis=1); norm=torch.as_tensor((actions-mean)/std,dtype=torch.float32,device=device)
                    bt=torch.as_tensor(b,dtype=torch.float32,device=device).permute(0,1,4,2,3)/255.; ct=torch.as_tensor(contexts[name][:,-1],dtype=torch.float32,device=device).permute(0,3,1,2)/255.; tok=torch.as_tensor(instruction_tokens([row["prompt"]]),device=device)
                    res=model(bt,ct,norm,tok).permute(0,1,3,4,2).cpu().numpy(); out=np.stack([apply_residual(b[0],res[0])])
                preds[name].append(out[0]); contexts[name]=np.concatenate((contexts[name],out),axis=1)[:,-5:]; histories[name]=np.concatenate((histories[name],future),axis=1)[:,-4:]
        bp=np.concatenate(preds["base"]); tp=np.concatenate(preds["true"]); ba=np.abs(bp.astype(np.float64)-target); ta=np.abs(tp.astype(np.float64)-target)
        base_abs.append(ba.mean()); true_abs.append(ta.mean()); chunk_base[0].append(ba[16:24].mean()); chunk_base[1].append(ba[24:32].mean()); chunk_true[0].append(ta[16:24].mean()); chunk_true[1].append(ta[24:32].mean())
    return {"overall":float(np.mean(true_abs)/np.mean(base_abs)),"chunk3":float(np.mean(chunk_true[0])/np.mean(chunk_base[0])),"chunk4":float(np.mean(chunk_true[1])/np.mean(chunk_base[1]))}


def save_checkpoint(path,model,mean,std,scope,episodes,prereg_sha):
    torch.save({"format":CHECKPOINT_FORMAT,"step":50,"channels":16,"training_scope":scope,"episodes":episodes,"model":model.state_dict(),"action_mean":mean,"action_std":std,"preregistration_sha256":prereg_sha},path)


def main()->int:
    parser=argparse.ArgumentParser()
    for name in ("windows","split","preregistration","v169-release","v169-library","reward-checkpoint","t5-model","output-dir"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    prereg=json.loads(args.preregistration.read_text());
    if prereg.get("format")!="strict-track2-v446-contrastive-residual-5fold-preregistration-v1": raise RuntimeError("wrong v446 preregistration")
    input_checks=validate_inputs(args,prereg)
    split=json.loads(args.split.read_text()); prompts={int(k):str(v) for k,v in split["episode_to_instruction"].items()}; right=prereg["data"]["right_train_episodes"]; folds=prereg["data"]["fold_holdout_episodes"]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; device=torch.device(args.device)
    rows=select_rows(args.windows,right,prompts)
    expected_source=prereg["data"]["selected_window_source_manifest"]; source_body={"format":expected_source["format"],"files":{row["source_path"]:sha256(args.windows/row["source_path"]) for row in rows}}
    source_ok=source_body["files"]==expected_source["files"] and canonical_sha(source_body)==expected_source["canonical_sha256"] and expected_source["canonical_sha256"]==prereg["inputs"]["windows"]["selected_source_manifest_sha256"]
    if not source_ok: raise RuntimeError("v446 selected close120 source manifest mismatch")
    input_checks["selected_window_source_manifest"]=True
    args.output_dir.mkdir(parents=True)
    v169=Track2V169ArmRoutedRuntime(args.v169_release,args.v169_library,args.device); cache_baselines(rows,v169)
    models={}; fold_reports=[]; all_outputs={name:[None]*len(rows) for name in VARIANTS}; row_index={id(row):i for i,row in enumerate(rows)}
    for fold,hold in enumerate(folds):
        fit=[r for r in rows if r["episode"] not in hold]; held=[r for r in rows if r["episode"] in hold]; model,mean,std,losses=train_model(fit,device,fold); models[fold]=(model,mean,std)
        save_checkpoint(args.output_dir/f"fold{fold}_step50.pt",model,mean,std,f"fold{fold}",fit and sorted(set(r["episode"] for r in fit)),sha256(args.preregistration))
        predicted={name:predict_variant(model,held,name,mean,std,device) for name in VARIANTS}; outputs={name:value[0] for name,value in predicted.items()}; gates={name:value[1] for name,value in predicted.items()}; maes={name:rgb_mae(value,held) for name,value in outputs.items()}; baseline=rgb_mae(np.stack([r["baseline"]["true"] for r in held]),held)
        episode_improved=sum(float(maes["true"][[r["episode"]==e for r in held]].mean())<float(baseline[[r["episode"]==e for r in held]].mean()) for e in hold)
        negative_order={}
        for name in NEG:
            episode_wins=sum(maes["true"][[r["episode"]==e for r in held]].mean()<maes[name][[r["episode"]==e for r in held]].mean() for e in hold)
            negative_order[name]={"mean":float(maes["true"].mean())<float(maes[name].mean()),"pairwise":float(np.mean(maes["true"]<maes[name])),"episode_wins":int(episode_wins)}
        passed=float(maes["true"].mean()/baseline.mean())<=1.0 and all(v["mean"] and v["pairwise"]>=.75 and v["episode_wins"]==3 for v in negative_order.values()) and episode_improved==3
        gate_count={name:int(value.sum()) for name,value in gates.items()}; recomputed={name:sum(bool(gate_decision(r["history_actions"],r["variant_actions"][name],r["prompt"])["gate"]) for r in held) for name in VARIANTS}
        gate_match=gate_count==recomputed
        passed=bool(passed and gate_match and gate_count["true"]==24 and gate_count["open"]==0 and gate_count["static"]==0)
        fold_reports.append({"fold":fold,"fit_episodes":sorted(set(r["episode"] for r in fit)),"holdout_episodes":hold,"true_rgb_ratio":float(maes["true"].mean()/baseline.mean()),"negative_order":negative_order,"variant_gate_count":gate_count,"variant_gate_runtime_recomputed_match":gate_match,"episode_improved":episode_improved,"passed":passed,"loss_first":losses[0],"loss_last":losses[-1]})
        for local,row in enumerate(held):
            index=row_index[id(row)]
            for name in VARIANTS: all_outputs[name][index]=outputs[name][local]
    all_outputs={name:np.stack(value) for name,value in all_outputs.items()}; target=np.stack([r["target_frames"] for r in rows]); baseline_true=np.stack([r["baseline"]["true"] for r in rows]); true_mae=rgb_mae(all_outputs["true"],rows); base_mae=rgb_mae(baseline_true,rows)
    rgb_negative={}
    for name in NEG:
        neg=rgb_mae(all_outputs[name],rows); rgb_negative[name]={"pairwise":float(np.mean(true_mae<neg)),"margin":float(np.mean(neg-true_mae))}
    episode_improved=sum(true_mae[[r["episode"]==e for r in rows]].mean()<base_mae[[r["episode"]==e for r in rows]].mean() for e in right)
    # Reward is imported only after all five fold outputs are immutable.
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint),config={"t5_model_name":str(args.t5_model)}).to(device).eval().requires_grad_(False); prompt_list=[r["prompt"] for r in rows]
    reward_arrays={name:score_reward(reward,all_outputs[name],prompt_list,device,32) for name in VARIANTS}; reward_negative={}
    true_final=reward_arrays["true"][:,-1].astype(np.float64)
    for name in NEG:
        neg_final=reward_arrays[name][:,-1].astype(np.float64); diff=true_final-neg_final; norm=diff/(np.abs(true_final)+np.abs(neg_final)+1e-6); reward_negative[name]={"pairwise":float(np.mean(diff>0)),"normalized_margin":float(np.mean(norm))}
    del reward; torch.cuda.empty_cache()
    fold_for_episode={episode:fold for fold,held in enumerate(folds) for episode in held}; recursive=recursive_eval(recursive_rows(args.windows,right,prompts),fold_for_episode,models,v169,device)
    fold_pass=all(row["passed"] for row in fold_reports); global_pass=(float(true_mae.mean()/base_mae.mean())<=.992 and episode_improved>=12 and all(v["pairwise"]>=.75 and v["margin"]>0 for v in rgb_negative.values()) and all(v["pairwise"]>=.75 and v["normalized_margin"]>=.05 for v in reward_negative.values()) and recursive["overall"]<=1 and recursive["chunk3"]<=1 and recursive["chunk4"]<=1)
    passed=bool(fold_pass and global_pass); final_path=args.output_dir/"final_all15_step50.pt"
    if passed:
        final_model,final_mean,final_std,final_losses=train_model(rows,device,99); save_checkpoint(final_path,final_model,final_mean,final_std,"all15",right,sha256(args.preregistration)); final_training={"performed":True,"loss_first":final_losses[0],"loss_last":final_losses[-1]}
    else: final_training={"performed":False}
    report={"format":"strict-track2-v446-5fold-s0-training-report-v1","created_at":datetime.now(timezone.utc).isoformat(),"passed":passed,"fold_pass":fold_pass,"global_pass":global_pass,"input_validation":input_checks,"folds":fold_reports,"global":{"true_rgb_ratio":float(true_mae.mean()/base_mae.mean()),"episode_improved":int(episode_improved),"rgb_negative":rgb_negative,"reward_negative":reward_negative,"recursive32":recursive},"final_training":final_training,"sha256":{"preregistration":sha256(args.preregistration),**({"final_checkpoint":sha256(final_path)} if final_path.exists() else {})},"guards":{"all_five_folds_completed":True,"development_or_final_used":False,"reward_used_after_frozen_outputs":True,"policy_updates":0,"s1_authorized":False,"rl_authorized":False}}
    (args.output_dir/"training_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); return 0 if passed else 2


if __name__=="__main__": raise SystemExit(main())
