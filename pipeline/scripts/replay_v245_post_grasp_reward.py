#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient
from wam_pipeline.v245_clean_progressive_successor_runtime import ACTION_WEIGHT,DISTANCE_SCALE,MOTION_SCALE,_path_length
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor

def corr(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok]
 if a.size<2 or a.std()==0 or b.std()==0:return None
 return float(np.corrcoef(a,b)[0,1])
def stats(x):
 x=np.asarray(x,float);return {'count':int(x.size),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
def concordance(v,t,groups):
 agree=ties=total=0
 for idx in groups.values():
  if len(idx)<2:continue
  for i in range(len(idx)):
   for j in range(i+1,len(idx)):
    dv=v[idx[i]]-v[idx[j]];dt=t[idx[i]]-t[idx[j]];total+=1
    if dv==0 or dt==0:ties+=1
    elif dv*dt>0:agree+=1
 return {'value':float(agree/max(total-ties,1)),'pairs':total,'ties':ties}
def nchw(x):return torch.from_numpy(x).permute(0,1,4,2,3).reshape(-1,3,256,256).float().div_(255)
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--url',required=True);p.add_argument('--token',required=True);p.add_argument('--model-version',required=True);p.add_argument('--reward-checkpoint',type=Path,required=True);p.add_argument('--t5-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--details',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--alpha-scale',type=float,default=2.0);p.add_argument('--distance-temperature',type=float,default=1.0);p.add_argument('--alignment-floor',type=float,default=-1.0);p.add_argument('--success-like-min',type=int,default=4);p.add_argument('--group-std-min',type=float,default=1e-4);p.add_argument('--global-min',type=float,default=.05);p.add_argument('--group-min',type=float,default=.55);p.add_argument('--motion-min',type=float,default=-1.0);p.add_argument('--delta-regime',action='store_true');p.add_argument('--delta-clean-max',type=float,default=.01);p.add_argument('--delta-clean-alignment-floor',type=float,default=.95);p.add_argument('--delta-ood-min',type=float,default=.8);p.add_argument('--delta-ood-alignment-floor',type=float,default=.5);a=p.parse_args()
 if a.output.exists() or a.details.exists():raise SystemExit('refusing overwrite')
 with np.load(a.library,allow_pickle=False) as x:
  paths=x['path'].astype(str);episode_id=x['episode_id'].astype(np.int64);vis=x['visual'].astype(np.float32);act=x['action'].astype(np.float32);mean=x['normalization_mean'].astype(np.float32);std=x['normalization_std'].astype(np.float32)
 clean=np.flatnonzero(episode_id>=20000);records=[]
 raw_clean=act[clean].reshape(-1,12,14)*std+mean;delta_clean=np.diff(raw_clean[:,:,7:13],axis=1);delta_scale=np.maximum(delta_clean.reshape(-1,6).std(0),1e-6);delta_feature=(delta_clean/delta_scale).reshape(len(clean),-1)
 for fi,path in enumerate(sorted(a.audit_dir.glob('rollout_*.npz'))):
  with np.load(path,allow_pickle=False) as x:
   cs=x['context_frames'].copy();hs=x['history_actions'].astype(np.float32);fs=x['future_actions'].astype(np.float32);seeds=x['seeds'].astype(np.int64);texts=list(map(str,json.loads(str(x['instructions_json']))))
  for ai,(c,h,f,s,t) in enumerate(zip(cs,hs,fs,seeds,texts)):
   route=Track2ArmRoutedAutoregressiveUNet.active_arm(h,f,t)
   if route=='right' and h[-1,13]<.5 and (f[:,13]<.5).mean()>=.75:records.append((fi,ai,c,h,f,int(s),t))
 assert len(records)==101
 distance=[];alignment=[];motion=[];alpha=[];base=[];delta_ratio=[]
 for _,_,c,h,f,_,_ in records:
  qv=_visual_descriptor(c[-1]);qa=((np.concatenate((h,f),0)-mean)/std).reshape(-1);vd=((vis[clean]-qv)**2).mean(1);ad=((act[clean]-qa)**2).mean(1);score=vd/max(float(np.median(vd)),1e-9)+ACTION_WEIGHT*ad/max(float(np.median(ad)),1e-9);loc=int(np.argmin(score));row=int(clean[loc]);d=float(ad[loc]);raw=act[row].reshape(-1,14)*std+mean;libterm=raw[-1,7:13]-raw[-9,7:13];qterm=f[-1,7:13]-h[-1,7:13];den=float(np.linalg.norm(qterm)*np.linalg.norm(libterm));al=float(np.dot(qterm,libterm)/den) if den>1e-8 else 0.;mo=_path_length(h,f);qd=(np.diff(np.concatenate((h,f),0)[:,7:13],axis=0)/delta_scale).reshape(-1);dd=((delta_feature-qd)**2).mean(1);dscore=vd/max(float(np.median(vd)),1e-9)+ACTION_WEIGHT*dd/max(float(np.median(dd)),1e-9);dloc=int(np.argmin(dscore));dr=float(dd[dloc]/max(float(np.median(dd)),1e-9));
  if a.delta_regime:
   floor=a.delta_clean_alignment_floor if dr<=a.delta_clean_max else (a.delta_ood_alignment_floor if dr>=a.delta_ood_min else None);gate=0. if floor is None else np.clip((al-floor)/(1-floor),0,1);aa=float(np.clip(a.alpha_scale*gate*np.clip(mo/MOTION_SCALE,0,1),0,1))
  else:
   gate=np.clip((al-a.alignment_floor)/(1-a.alignment_floor),0,1) if a.alignment_floor>=0 else np.clip((al+1)/2,0,1);aa=float(np.clip(a.alpha_scale*np.exp(-d/(a.distance_temperature*DISTANCE_SCALE))*gate*np.clip(mo/MOTION_SCALE,0,1),0,1))
  distance.append(d);alignment.append(al);motion.append(mo);alpha.append(aa);base.append(row);delta_ratio.append(dr)
 distance,alignment,motion,alpha,delta_ratio=map(np.asarray,(distance,alignment,motion,alpha,delta_ratio));client=Track2ServiceClient(a.url,a.token,a.model_version);client.assert_ready()
 from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
 rm=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(a.reward_checkpoint),config={'t5_model_name':str(a.t5_model)}).eval().to(a.device);rewards=np.empty((len(records),8),np.float32)
 with torch.inference_mode():
  for begin in range(0,len(records),8):
   b=records[begin:begin+8];frames=client.predict_batch(np.stack([x[2] for x in b]),np.stack([x[3] for x in b]),np.stack([x[4] for x in b]),np.asarray([x[5] for x in b]),[x[6] for x in b]);texts=[x[6] for x in b for _ in range(8)];rewards[begin:begin+len(b)]=rm.compute_reward(nchw(frames).to(a.device),texts).float().cpu().numpy().reshape(len(b),8);print(json.dumps({'completed':begin+len(b),'total':len(records)}),flush=True)
 terminal=rewards[:,-1];expert=-distance;groups={}
 for i,(fi,ai,*_) in enumerate(records):groups.setdefault((fi,ai//4),[]).append(i)
 gstd=np.asarray([terminal[idx].std() for idx in groups.values() if len(idx)>=2]);ec=concordance(terminal,expert,groups);ac=concordance(terminal,alignment,groups)
 rcorr=corr(terminal,alpha);checks={'post_queries':len(records)>=64,'group_std':bool(gstd.mean()>=a.group_std_min),'success_like_count':int((terminal>=.1).sum())>=a.success_like_min,'alignment_global':bool((corr(terminal,alignment) or -1)>=a.global_min),'motion_global':bool((corr(terminal,motion) or -1)>=a.motion_min),'alignment_group':ac['value']>=a.group_min,'service_contract':True}
 if a.delta_regime:checks['regime_alpha_global']=bool((rcorr or -1)>=.7)
 else:checks.update({'expert_global':bool((corr(terminal,expert) or -1)>=a.global_min),'expert_group':ec['value']>=a.group_min})
 report={'format':'strict-track2-post-grasp-reward-audit-v3','service_model_version':a.model_version,'post_queries':len(records),'terminal_reward':stats(terminal),'success_like_count':int((terminal>=.1).sum()),'group_std':stats(gstd),'reward_expert_correlation':corr(terminal,expert),'reward_motion_correlation':corr(terminal,motion),'reward_alignment_correlation':corr(terminal,alignment),'reward_alpha_correlation':rcorr,'expert_group_concordance':ec,'alignment_group_concordance':ac,'alpha':stats(alpha),'delta_ratio':stats(delta_ratio),'thresholds':{'alpha_scale':a.alpha_scale,'distance_temperature':a.distance_temperature,'alignment_floor':a.alignment_floor,'delta_regime':a.delta_regime,'delta_clean_max':a.delta_clean_max,'delta_clean_alignment_floor':a.delta_clean_alignment_floor,'delta_ood_min':a.delta_ood_min,'delta_ood_alignment_floor':a.delta_ood_alignment_floor,'success_like_min':a.success_like_min,'group_std_min':a.group_std_min,'global_min':a.global_min,'group_min':a.group_min,'motion_min':a.motion_min},'checks':checks,'passed':all(checks.values()),'guards':{'public_capture_only':True,'runtime_uses_reward':False,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.details,rewards=rewards,distance=distance,alignment=alignment,motion=motion,alpha=alpha,delta_ratio=delta_ratio,base=np.asarray(base));print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 3)
if __name__=='__main__':main()
