#!/usr/bin/env python3
"""Eight-trajectory parent-only signal gate for passed v432 step25."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np, torch

def score(model,frames,prompts,device,batch):
    out=[]
    for i in range(0,len(frames),batch):
        x=torch.from_numpy(frames[i:i+batch]).permute(0,3,1,2).float().div(255).to(device)
        with torch.inference_mode():out.append(model.compute_reward(x,prompts[i:i+batch]).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def main():
    p=argparse.ArgumentParser()
    for n in ("trace-dir","preregistration","reward-checkpoint","t5-model","output"):p.add_argument(f"--{n}",required=True,type=Path)
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=32);a=p.parse_args()
    pre=json.loads(a.preregistration.read_text()); assert pre["classification"]=="parent world-model diagnostic only"
    batches=[json.loads(x)["batch"] for x in (a.trace_dir/"route_trace.jsonl").read_text().splitlines()]
    pf=[];cf=[];prompts=[]
    for i,rows in enumerate(batches):
        s=f"batch_{i:05d}";x=np.load(a.trace_dir/f"{s}_parent.npy");y=np.load(a.trace_dir/f"{s}_candidate.npy");t=json.loads((a.trace_dir/f"{s}_prompts.json").read_text())
        if not(len(rows)==len(x)==len(y)==len(t)):raise RuntimeError("trace alignment")
        pf.append(x);cf.append(y);prompts+=t
    rows=[r for b in batches for r in b];pf=np.concatenate(pf);cf=np.concatenate(cf)
    if pf.shape!=cf.shape or pf.shape!=(200,256,256,3):raise RuntimeError("expected 200 requests")
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    rm=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(a.reward_checkpoint),config={"t5_model_name":str(a.t5_model)}).to(a.device).eval().requires_grad_(False)
    ps=score(rm,pf,prompts,torch.device(a.device),a.batch_size);cs=score(rm,cf,prompts,torch.device(a.device),a.batch_size);delta=cs-ps
    learned=np.asarray([i for i,r in enumerate(rows) if r["learned_request"]],dtype=np.int64);ld=delta[learned]
    ptime=ps.reshape(25,8);ctime=cs.reshape(25,8);dtime=ctime-ptime
    inc=np.diff(np.concatenate([np.zeros((1,8)),dtime]),axis=0);gamma=.99;ret=(5*np.power(gamma,np.arange(25))[:,None]*inc).sum(0)
    spreads=[float(np.ptp(ret[i:i+4])) for i in (0,4)]
    checks={"exact_requests":len(rows)==200,"exact_trajectories":len(ret)==8,"learned_requests_ge40":len(learned)>=40,"learned_request_mean_delta_positive":float(ld.mean())>0,"learned_request_positive_fraction_ge0p55":float((ld>0).mean())>=.55,"trajectory_positive_fraction_ge0p5":float((ret>0).mean())>=.5,"at_least_one_group_spread_ge0p02":sum(x>=.02 for x in spreads)>=1,"candidate_final_mean_not_lower":float(ctime[-1].mean())>=float(ptime[-1].mean())}
    report={"format":"strict-track2-v436-v432-step25-trace8-result-v1","created_at":datetime.now(timezone.utc).isoformat(),"classification":"parent world-model diagnostic only","candidate_stage":"passed v432 step25 diagnostic candidate","passed":all(checks.values()),"formal_candidate_authorized":False,"checks":checks,"learned_requests":len(learned),"reward":{"learned_mean_delta":float(ld.mean()),"learned_positive_fraction":float((ld>0).mean())},"trajectory":{"discount":gamma,"return_delta":ret.tolist(),"positive_fraction":float((ret>0).mean()),"group_spread":spreads},"guards":{"policy_updates":0,"checkpoint_writes":0,"hidden_or_final_data":False,"real_submission":False}}
    a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return 0 if report["passed"] else 3
if __name__=="__main__":raise SystemExit(main())
