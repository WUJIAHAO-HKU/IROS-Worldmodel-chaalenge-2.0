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
def concord(v,t,groups):
 ag=ti=to=0
 for idx in groups.values():
  if len(idx)<2:continue
  for i in range(len(idx)):
   for j in range(i+1,len(idx)):
    dv=v[idx[i]]-v[idx[j]];dt=t[idx[i]]-t[idx[j]];to+=1
    if dv==0 or dt==0:ti+=1
    elif dv*dt>0:ag+=1
 return float(ag/max(to-ti,1))
def stats(x):
 x=np.asarray(x,float);return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--reward-map',type=Path,required=True);p.add_argument('--details',type=Path,required=True);p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('refusing overwrite')
 with np.load(a.library,allow_pickle=False) as x:paths=x['path'].astype(str);episodes=x['episode_id'].astype(int);lv=x['visual'].astype(np.float32);la=x['action'].astype(np.float32);mean=x['normalization_mean'].astype(np.float32);std=x['normalization_std'].astype(np.float32)
 with np.load(a.reward_map,allow_pickle=False) as x:starts=x['start'].astype(int);clean=x['is_clean'].astype(bool);rr=x['reward']
 reward=np.full(len(paths),np.nan);cr=np.flatnonzero(clean);reward[cr]=rr[cr,:,-1].mean(1)
 with np.load(a.details,allow_pickle=False) as x:parent=x['rewards'][:,0,-1]
 terminal={}
 for ep in set(episodes[cr].tolist()):
  rows=cr[episodes[cr]==ep];late=rows[starts[rows]>=112];chosen=int(late[np.argmax(starts[late])]) if len(late) else int(rows[np.argmax(starts[rows])]);terminal[ep]=chosen
 rec=[]
 for fi,path in enumerate(sorted(a.audit_dir.glob('rollout_*.npz'))):
  with np.load(path,allow_pickle=False) as x:cs=x['context_frames'];hs=x['history_actions'].astype(np.float32);fs=x['future_actions'].astype(np.float32);texts=map(str,json.loads(str(x['instructions_json'])))
  for ai,(c,h,f,t) in enumerate(zip(cs,hs,fs,texts)):
   if Track2ArmRoutedAutoregressiveUNet.active_arm(h,f,t)=='right' and h[-1,13]<.5 and (f[:,13]<.5).mean()>=.75:rec.append((fi,ai,c,h,f))
 assert len(rec)==101
 groups={}
 for i,(fi,ai,*_) in enumerate(rec):groups.setdefault((fi,ai//4),[]).append(i)
 distance=[];alignment=[];motion=[];target=[];ref=[]
 for _,_,c,h,f in rec:
  qv=visual(c[-1]);qa=((np.concatenate((h,f),0)-mean)/std).reshape(-1);vd=((lv[cr]-qv)**2).mean(1);ad=((la[cr]-qa)**2).mean(1);score=vd/max(float(np.median(vd)),1e-9)+4*ad/max(float(np.median(ad)),1e-9);row=int(cr[int(np.argmin(score))]);d=float(((la[row]-qa)**2).mean());raw=la[row].reshape(-1,14)*std+mean;al=cosine(f[-1,7:13]-h[-1,7:13],raw[-1,7:13]-raw[-9,7:13]);vrow=int(cr[int(np.argmin(vd))]);trow=terminal[int(episodes[vrow])];distance.append(d);alignment.append(al);motion.append(pathlen(h,f));target.append(trow);ref.append(row)
 distance,alignment,motion,target=map(np.asarray,(distance,alignment,motion,target));expert=-distance;tr=reward[target];dscale=float(np.median(distance));mscale=float(np.median(motion));cands=[]
 gates={'positive':lambda x:np.clip(x,0,1),'positive_squared':lambda x:np.clip(x,0,1)**2,'smooth':lambda x:np.clip((x+1)/2,0,1),'floor025':lambda x:np.clip((x-.25)/.75,0,1),'floor050':lambda x:np.clip((x-.50)/.50,0,1),'floor075':lambda x:np.clip((x-.75)/.25,0,1)}
 for temp in (.25,.5,1.,3.,5.):
  conf=np.exp(-distance/(temp*dscale));mg=np.clip(motion/mscale,0,1)
  for gate_name,gate_fn in gates.items():
   ag=gate_fn(alignment)
   for scale in (2.,4.,8.,16.,32.,64.):
    alpha=np.clip(scale*conf*ag*mg,0,1);proxy=parent+(tr-parent)*(alpha**8)
    vals={'expert_corr':corr(proxy,expert),'alignment_corr':corr(proxy,alignment),'motion_corr':corr(proxy,motion),'expert_group':concord(proxy,expert,groups),'alignment_group':concord(proxy,alignment,groups),'motion_group':concord(proxy,motion,groups)}
    passed=(int((proxy>=.1).sum())>=8 and vals['expert_corr']>0 and vals['alignment_corr']>0 and vals['expert_group']>=.55 and vals['alignment_group']>=.55)
    score=min(vals['expert_group'],vals['alignment_group'],(vals['expert_corr']+1)/2,(vals['alignment_corr']+1)/2)
    cands.append({'temperature':temp,'alignment_gate':gate_name,'alpha_scale':scale,'distance_scale':dscale,'motion_scale':mscale,'target_unique':int(len(set(target.tolist()))),'target_reward':stats(tr),'target_gt_0p5':int((tr>=.5).sum()),'alpha':stats(alpha),'proxy':stats(proxy),'success_like':int((proxy>=.1).sum()),**vals,'passes':passed,'selection_score':float(score)})
 valid=[x for x in cands if x['passes']];selected=max(valid,key=lambda x:x['selection_score']) if valid else None
 best_by_gate={name:sorted([x for x in cands if x['alignment_gate']==name],key=lambda x:x['selection_score'],reverse=True)[:5] for name in gates}
 report={'format':'strict-track2-v246-public-clean-terminal-successor-proxy-v1','post_queries':len(rec),'candidate_count':len(cands),'selection_rule':'success-like >=8, positive expert/alignment global correlation, expert/alignment group concordance >=.55; maximize conservative score','selected':selected,'top':sorted(cands,key=lambda x:x['selection_score'],reverse=True)[:15],'best_by_gate':best_by_gate,'guards':{'runtime_uses_reward':False,'terminal_target_rule':'visually nearest official public clean episode, final available row at start>=112','policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
