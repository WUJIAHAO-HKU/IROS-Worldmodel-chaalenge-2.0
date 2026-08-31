#!/usr/bin/env python3
"""Map official public clean-demo successor frames to the local reward model."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def corr(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    if a.size<2 or a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])

def stats(x):
    x=np.asarray(x,float); return {'count':int(x.size),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}

def nchw(x): return torch.from_numpy(x).permute(0,1,4,2,3).reshape(-1,3,256,256).float().div_(255)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--library',type=Path,required=True); p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--reward-checkpoint',type=Path,required=True);p.add_argument('--t5-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index-output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=48);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.output.exists() or a.index_output.exists():raise SystemExit('refusing to overwrite')
    with np.load(a.library,allow_pickle=False) as x: paths=x['path'].astype(str); episodes=x['episode_id'].astype(np.int64)
    instructions=set()
    for f in sorted(a.audit_dir.glob('rollout_*.npz')):
        with np.load(f,allow_pickle=False) as x: instructions.update(map(str,json.loads(str(x['instructions_json']))))
    instructions=sorted(instructions)
    if len(instructions)!=4:raise ValueError(f'expected four public instructions, got {instructions}')
    clean=[]; starts=np.full(len(paths),-1,dtype=np.int64); is_clean=np.zeros(len(paths),bool)
    for i,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as x:
            starts[i]=int(x['start'])
            clean_demo='capture_success' not in x.files and 'aloha-agilex_clean_50' in str(x['source'])
        if clean_demo:is_clean[i]=True;clean.append(i)
    if len(clean)!=1926:raise ValueError(f'expected 1926 official clean rows, got {len(clean)}')
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    model=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(a.reward_checkpoint),config={'t5_model_name':str(a.t5_model)}).eval().to(a.device)
    score=np.full((len(paths),len(instructions),8),np.nan,dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0,len(clean),a.batch_size):
            rows=clean[begin:begin+a.batch_size]; frames=[]
            for row in rows:
                with np.load(paths[row],allow_pickle=False) as x:frames.append(x['target_frames'].copy())
            frames=np.stack(frames)
            expanded_frames=np.repeat(frames[:,None],len(instructions),axis=1).reshape(-1,8,256,256,3)
            texts=[text for _ in rows for text in instructions for _ in range(8)]
            values=model.compute_reward(nchw(expanded_frames).to(a.device),texts).float().cpu().numpy().reshape(len(rows),len(instructions),8)
            score[rows]=values
            print(json.dumps({'completed':begin+len(rows),'total':len(clean)}),flush=True)
    clean_rows=np.flatnonzero(is_clean); terminal=np.full(len(paths),np.nan,dtype=np.float32)
    terminal[clean_rows]=score[clean_rows,:,-1].mean(1); ct=terminal[clean_rows]
    by_episode=defaultdict(list)
    for row in clean_rows:by_episode[int(episodes[row])].append(int(row))
    episode_corr=[]
    for rows in by_episode.values():
        rows=np.asarray(rows); c=corr(starts[rows],terminal[rows])
        if c is not None:episode_corr.append(c)
    offsets=[]
    for offset in (4,8,12,16,24,32):
        before=[];after=[]
        for rows in by_episode.values():
            lookup={int(starts[r]):r for r in rows}
            for start,row in lookup.items():
                later=lookup.get(start+offset)
                if later is not None:before.append(terminal[row]);after.append(terminal[later])
        before,after=np.asarray(before),np.asarray(after); delta=after-before
        offsets.append({'offset':offset,'pairs':int(len(delta)),'before':stats(before),'after':stats(after),'delta':stats(delta),
                        'improved_fraction':float((delta>0).mean()),'reward_correlation':corr(before,after)})
    bins=[]
    for low,high in ((0,32),(32,64),(64,80),(80,96),(96,112),(112,128),(128,10000)):
        rows=clean_rows[(starts[clean_rows]>=low)&(starts[clean_rows]<high)]
        if len(rows):bins.append({'start_min':low,'start_max_exclusive':high,'reward':stats(terminal[rows]),'gt_0p1':int((terminal[rows]>=.1).sum()),'gt_0p5':int((terminal[rows]>=.5).sum())})
    top=clean_rows[np.argsort(ct)[-30:][::-1]]
    per_instruction={text:stats(score[clean_rows,i,-1]) for i,text in enumerate(instructions)}
    report={'format':'strict-track2-v243-public-clean-successor-reward-map-v1','instructions':instructions,'aggregation':'mean over four public capture instructions',
      'clean_rows':len(clean_rows),'clean_episodes':len(by_episode),'terminal_reward':stats(ct),'per_instruction_terminal_reward':per_instruction,
      'reward_gt_0p1':int((ct>=.1).sum()),'reward_gt_0p5':int((ct>=.5).sum()),
      'global_start_reward_correlation':corr(starts[clean_rows],ct),'mean_episode_start_reward_correlation':float(np.mean(episode_corr)),
      'start_bins':bins,'temporal_offsets':offsets,
      'top_rows':[{'row':int(r),'episode':int(episodes[r]),'start':int(starts[r]),'terminal_reward':float(terminal[r]),'path':paths[r]} for r in top],
      'guards':{'official_public_clean_demonstrations_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.index_output,path=paths,episode_id=episodes,start=starts,is_clean=is_clean,reward=score)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
