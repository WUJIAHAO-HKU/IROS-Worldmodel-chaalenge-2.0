#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet

def corr(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok]
 if a.size<2 or a.std()==0 or b.std()==0:return None
 return float(np.corrcoef(a,b)[0,1])
def visual(x):return (x.astype(np.float32).reshape(16,16,16,16,3).mean((1,3))/255).reshape(-1)
def pathlen(h,f):return float(np.linalg.norm(np.diff(np.concatenate((h[-1:,7:13],f[:,7:13]),0),axis=0),axis=-1).sum())
def cosine(a,b):
 d=float(np.linalg.norm(a)*np.linalg.norm(b));return float(np.dot(a,b)/d) if d>1e-8 else 0.
def pair_concordance(values,target,groups):
 agree=ties=total=0
 for idx in groups.values():
  if len(idx)<2:continue
  for p in range(len(idx)):
   for q in range(p+1,len(idx)):
    dv=values[idx[p]]-values[idx[q]];dt=target[idx[p]]-target[idx[q]];total+=1
    if dv==0 or dt==0:ties+=1
    elif dv*dt>0:agree+=1
 return float(agree/max(total-ties,1))
def stats(x):
 x=np.asarray(x,float);return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--reward-map',type=Path,required=True);p.add_argument('--details',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('refusing overwrite')
 with np.load(a.library,allow_pickle=False) as x:
  paths=x['path'].astype(str);episodes=x['episode_id'].astype(int);libvis=x['visual'].astype(np.float32);libact=x['action'].astype(np.float32);mean=x['normalization_mean'].astype(np.float32);std=x['normalization_std'].astype(np.float32)
 with np.load(a.reward_map,allow_pickle=False) as x:
  starts=x['start'].astype(int);clean=x['is_clean'].astype(bool);target_reward=np.nanmean(x['reward'][:,:,-1],axis=1)
 with np.load(a.details,allow_pickle=False) as x:parent_reward=x['rewards'][:,0,-1]
 cleanrows=np.flatnonzero(clean);episode_lookup={}
 for ep in set(episodes[cleanrows].tolist()):
  rows=cleanrows[episodes[cleanrows]==ep];episode_lookup[ep]={int(starts[r]):int(r) for r in rows}
 records=[]
 for fi,fpath in enumerate(sorted(a.audit_dir.glob('rollout_*.npz'))):
  with np.load(fpath,allow_pickle=False) as x:
   cs=x['context_frames'];hs=x['history_actions'].astype(np.float32);fs=x['future_actions'].astype(np.float32);texts=map(str,json.loads(str(x['instructions_json'])))
  for ai,(c,h,f,text) in enumerate(zip(cs,hs,fs,texts)):
   route=Track2ArmRoutedAutoregressiveUNet.active_arm(h,f,text)
   if route=='right' and h[-1,13]<.5 and (f[:,13]<.5).mean()>=.75:records.append((fi,ai,c,h,f))
 assert len(records)==101 and len(parent_reward)==101
 groups={}
 for i,(fi,ai,*_) in enumerate(records):groups.setdefault((fi,ai//4),[]).append(i)
 qvis=np.stack([visual(x[2][-1]) for x in records]);qact=np.stack([((np.concatenate((x[3],x[4]),0)-mean)/std).reshape(-1) for x in records]);motion=np.asarray([pathlen(x[3],x[4]) for x in records]);motion_scale=float(np.median(motion))
 candidates=[]
 for action_weight in (.5,1.,2.5,4.):
  selected=[];distance=[];alignment=[]
  for i,(_,_,_,h,f) in enumerate(records):
   vd=((libvis[cleanrows]-qvis[i])**2).mean(1);ad=((libact[cleanrows]-qact[i])**2).mean(1)
   score=vd/max(float(np.median(vd)),1e-9)+action_weight*ad/max(float(np.median(ad)),1e-9);loc=int(np.argmin(score));row=int(cleanrows[loc]);selected.append(row);distance.append(float(ad[loc]))
   raw=libact[row].reshape(-1,14)*std+mean;lh,lf=raw[:-8],raw[-8:];alignment.append(cosine(f[-1,7:13]-h[-1,7:13],lf[-1,7:13]-lh[-1,7:13]))
  selected=np.asarray(selected);distance=np.asarray(distance);alignment=np.asarray(alignment);dscale=float(np.median(distance));expert=-distance
  for offset in (8,12,16,24):
   advanced=[]
   for row in selected:
    lookup=episode_lookup[int(episodes[row])];goal=int(starts[row])+offset
    choices=np.asarray(sorted(lookup));later=choices[choices>=goal];chosen=int(later[0] if len(later) else choices[-1]);advanced.append(lookup[chosen])
   advanced=np.asarray(advanced);tr=target_reward[advanced]
   for temperature in (1.,3.,5.):
    confidence=np.exp(-distance/(temperature*max(dscale,1e-9)));align_gate=np.clip((alignment+1.)/2.,0.,1.);motion_gate=np.clip(motion/max(motion_scale,1e-9),0.,1.)
    base=confidence*align_gate*motion_gate
    for alpha_scale in (1.,2.,4.):
     alpha=np.clip(alpha_scale*base,0.,1.);proxy=parent_reward+(tr-parent_reward)*(alpha**8)
     vals={'proxy_expert_corr':corr(proxy,expert),'proxy_motion_corr':corr(proxy,motion),'proxy_alignment_corr':corr(proxy,alignment),'alpha_expert_corr':corr(alpha,expert),
      'proxy_expert_group_concordance':pair_concordance(proxy,expert,groups),'proxy_motion_group_concordance':pair_concordance(proxy,motion,groups),'proxy_alignment_group_concordance':pair_concordance(proxy,alignment,groups)}
     objective=min(vals['proxy_expert_group_concordance'],vals['proxy_alignment_group_concordance'],(vals['proxy_expert_corr']+1)/2,(vals['proxy_alignment_corr']+1)/2)
     candidates.append({'action_weight':action_weight,'offset':offset,'temperature':temperature,'alpha_scale':alpha_scale,'distance_scale':dscale,'motion_scale':motion_scale,'selected_unique':int(len(set(selected.tolist()))),'advanced_unique':int(len(set(advanced.tolist()))),'target_reward':stats(tr),'target_gt_0p5':float((tr>=.5).mean()),'alpha':stats(alpha),'proxy_reward':stats(proxy),**vals,'selection_score':float(objective)})
 valid=[x for x in candidates if x['proxy_expert_corr']>0 and x['proxy_alignment_corr']>0 and x['proxy_expert_group_concordance']>=.55 and x['proxy_alignment_group_concordance']>=.55 and x['target_gt_0p5']>=.5]
 selected=max(valid,key=lambda x:x['selection_score']) if valid else None
 failure_counts={
  'expert_global_nonpositive':sum(x['proxy_expert_corr']<=0 for x in candidates),
  'alignment_global_nonpositive':sum(x['proxy_alignment_corr']<=0 for x in candidates),
  'expert_group_below_0p55':sum(x['proxy_expert_group_concordance']<.55 for x in candidates),
  'alignment_group_below_0p55':sum(x['proxy_alignment_group_concordance']<.55 for x in candidates),
  'target_coverage_below_0p5':sum(x['target_gt_0p5']<.5 for x in candidates)}
 best_by_offset={str(offset):sorted([x for x in candidates if x['offset']==offset],key=lambda x:x['selection_score'],reverse=True)[:5] for offset in (8,12,16,24)}
 report={'format':'strict-track2-v244-public-clean-progressive-successor-proxy-v1','post_queries':len(records),'candidate_count':len(candidates),'selection_rule':'positive global expert/alignment correlations, >=.55 group concordance for both, >=.5 high-reward public successor coverage; maximize conservative score','selected':selected,'failure_counts':failure_counts,'top_candidates':sorted(valid,key=lambda x:x['selection_score'],reverse=True)[:15],'top_unfiltered':sorted(candidates,key=lambda x:x['selection_score'],reverse=True)[:20],'best_by_offset':best_by_offset,'guards':{'runtime_uses_reward':False,'official_public_clean_demonstrations_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
