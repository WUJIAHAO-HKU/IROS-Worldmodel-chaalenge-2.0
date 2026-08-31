#!/usr/bin/env python3
"""Fit a grouped real-vs-generated RGB-context classifier without outcomes."""
from __future__ import annotations
import argparse,gc,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v337_public_recursive_ood_gate import FEATURE_VERSION,context_quality_features
from wam_pipeline.v378_source_routed_blended_cartesian_runtime import Track2V378SourceRoutedBlendedCartesian
CS=(.01,.1,1.0);THRESHOLDS=(.5,.7,.8,.9,.95,.975,.99)
def seed(p):return int.from_bytes(hashlib.sha256(p.name.encode()).digest()[:8],'little')%(2**31)
def rates(y,prob,kinds,t):
 pred=prob>=t;teacher=kinds=='teacher';gen=y==1;v355=kinds=='v355_generated';v378=kinds=='v378_generated'
 return {'threshold':t,'teacher_specificity':float((~pred[teacher]).mean()),'generated_recall':float(pred[gen].mean()),'v355_recall':float(pred[v355].mean()),'v378_recall':float(pred[v378].mean())}
def load_v355(path,device):
 m=json.loads((path/'arm_routed_autoregressive_manifest.json').read_text());return Track2ArmRoutedAutoregressiveUNet(path/m['left_expert'],path/m['right_expert'],device)
def main():
 p=argparse.ArgumentParser()
 for n in ('v355-release','v378-release','library','windows','split','preregistration','output','report'):p.add_argument(f'--{n}',required=True,type=Path)
 p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
 if args.output.exists() or args.report.exists():raise FileExistsError('refusing overwrite')
 prereg=json.loads(args.preregistration.read_text());split=json.loads(args.split.read_text());train=set(map(int,split['train_episodes']));arm={int(k):v for k,v in split['arm_by_episode'].items()};episodes=sorted(e for e in train if arm[e]=='right')
 if len(episodes)!=15:raise RuntimeError(episodes)
 v355=load_v355(args.v355_release,args.device);v378=Track2V378SourceRoutedBlendedCartesian(args.v378_release,args.library,args.device)
 states=[]
 for ep in episodes:
  avail={int(q.stem.split('_')[1]):q for q in args.windows.glob(f'episode{ep}_*.npz')}
  for a in range(8):
   if a in avail:
    with np.load(avail[a],allow_pickle=False) as x:initial=x['context_frames'].astype(np.uint8)
    states.append({'episode':ep,'alignment':a,'start':a,'available':avail,'v355':initial.copy(),'v378':initial.copy()})
 features=[];labels=[];groups=[];kinds=[];rows=[]
 while True:
  active=[s for s in states if s['start'] in s['available']]
  if not active:break
  histories=[];futures=[];seeds=[];teacher=[]
  for s in active:
   q=s['available'][s['start']]
   with np.load(q,allow_pickle=False) as x:real=x['context_frames'].astype(np.uint8);histories.append(x['history_actions'].astype(np.float32));futures.append(x['future_actions'].astype(np.float32))
   seeds.append(seed(q));teacher.append(real);post=s['start']>=8
   for value,label,kind in ((real,0,'teacher'),(s['v355'],int(post),'v355_generated' if post else 'initial'),(s['v378'],int(post),'v378_generated' if post else 'initial')):
    features.append(context_quality_features(value));labels.append(label);groups.append(s['episode']);kinds.append(kind)
   rows.append({'episode':s['episode'],'alignment':s['alignment'],'start':s['start'],'post_first_prediction':post})
  h=np.stack(histories);f=np.stack(futures);sd=np.asarray(seeds);prompts=['Adjust bottle']*len(active);out355=[];out378=[]
  for begin in range(0,len(active),args.batch_size):
   end=begin+args.batch_size;out355.extend(v355.predict_batch(np.stack([s['v355'] for s in active[begin:end]]),h[begin:end],f[begin:end],sd[begin:end],prompts[begin:end]));out378.extend(v378.predict_batch(np.stack([s['v378'] for s in active[begin:end]]),h[begin:end],f[begin:end],sd[begin:end],prompts[begin:end]))
  for s,a,b in zip(active,out355,out378,strict=True):s['v355']=a[-5:].copy();s['v378']=b[-5:].copy();s['start']+=8
 x=np.asarray(features,np.float64);y=np.asarray(labels,np.int64);group=np.asarray(groups);kind=np.asarray(kinds);cv=[]
 for c in CS:
  oof=np.full(len(y),np.nan)
  for held in episodes:
   tr=group!=held;te=~tr;mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),1e-6);model=LogisticRegression(C=c,max_iter=2000,class_weight='balanced',solver='lbfgs',random_state=1544).fit((x[tr]-mean)/scale,y[tr]);oof[te]=model.predict_proba((x[te]-mean)/scale)[:,1]
  for t in THRESHOLDS:
   r=rates(y,oof,kind,t);r['c']=c;r['eligible']=bool(r['teacher_specificity']==1.0 and r['generated_recall']>=.95 and r['v355_recall']>=.90 and r['v378_recall']>=.90);cv.append(r)
 eligible=[r for r in cv if r['eligible']];selected=max(eligible,key=lambda r:(r['generated_recall'],min(r['v355_recall'],r['v378_recall']),r['threshold'],-r['c'])) if eligible else None;checks={'fifteen_train_episodes':len(episodes)==15,'all_alignments':len(states)==120,'post_first_generated_examples_ge_3000':int(y.sum())>=3000,'eligible_oof_operating_point':selected is not None};passed=all(checks.values());final=None;artifact=None
 if passed:
  mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);model=LogisticRegression(C=selected['c'],max_iter=2000,class_weight='balanced',solver='lbfgs',random_state=1544).fit((x-mean)/scale,y);prob=model.predict_proba((x-mean)/scale)[:,1];final=rates(y,prob,kind,selected['threshold']);np.savez_compressed(args.output,feature_mean=mean.astype(np.float32),feature_scale=scale.astype(np.float32),coefficient=model.coef_[0].astype(np.float32),intercept=np.asarray(model.intercept_[0],np.float32),threshold=np.asarray(selected['threshold'],np.float32),c=np.asarray(selected['c'],np.float32),corruption_mae_floor=np.asarray(0.0,np.float32),feature_version=np.asarray(FEATURE_VERSION));artifact=hashlib.sha256(args.output.read_bytes()).hexdigest()
 report={'format':'strict-track2-v381-public-recursive-source-gate-training-v1','created_at':datetime.now(timezone.utc).isoformat(),'episodes':episodes,'replay_rows':len(rows),'examples':len(y),'generated_examples':int(y.sum()),'feature_dimension':x.shape[1],'selection':selected,'final_train_rates':final,'cv_rows':cv,'checks':checks,'passed':passed,'artifact_sha256':artifact,'guards':{'public_train_only':True,'labels_use_only_real_vs_generated_provenance':True,'reward_or_outcomes_used':False,'public_holdout_read':False,'hidden_or_final_data':False,'real_submission':False}}
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('replay_rows','examples','generated_examples','selection','checks','passed')},indent=2));return 0 if passed else 3
if __name__=='__main__':raise SystemExit(main())
