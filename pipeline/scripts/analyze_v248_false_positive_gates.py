#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v216_public_knn_blend_runtime import _visual_descriptor
from wam_pipeline.v245_clean_progressive_successor_runtime import ACTION_WEIGHT,DISTANCE_SCALE,MOTION_SCALE,_path_length

def parse(name):
 e,s=Path(name).stem.split('_');return int(e[7:]),int(s)
def stats(x):
 x=np.asarray(x,float);return {'count':len(x),'mean':float(x.mean()),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--success-windows',type=Path,required=True);p.add_argument('--failure-windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('refusing overwrite')
 with np.load(a.library,allow_pickle=False) as x:episode=x['episode_id'].astype(int);vis=x['visual'].astype(np.float32);act=x['action'].astype(np.float32);mean=x['normalization_mean'].astype(np.float32);std=x['normalization_std'].astype(np.float32)
 clean=np.flatnonzero(episode>=20000);allrows=[]
 for split,windows in [('success',a.success_windows),('failure',a.failure_windows)]:
  base=a.run/'audit'/f'public_{split}_baseline.npz';cand=a.run/'audit'/f'public_{split}_candidate.npz';reward_path=a.run/'audit'/f'public_{split}_reward.json'
  with np.load(base,allow_pickle=False) as x:paths=x['path'].astype(str);arms=x['arm_right'].astype(bool);texts=x['instruction'].astype(str)
  with np.load(cand,allow_pickle=False) as x:frames=x['candidate']
  scores=np.asarray(json.loads(reward_path.read_text())['raw_scores']['candidate'],float)
  qindex={parse(x.name):x for x in windows.glob('episode*_*.npz')}
  for seq in np.flatnonzero(arms):
   ep,start=parse(paths[seq]);first=qindex[(ep,start)]
   with np.load(first,allow_pickle=False) as x:context=x['context_frames'].copy();history=x['history_actions'].astype(np.float32).copy()
   chunks=[]
   for chunk in range(16):
    with np.load(qindex[(ep,start+8*chunk)],allow_pickle=False) as x:future=x['future_actions'].astype(np.float32).copy()
    route=Track2ArmRoutedAutoregressiveUNet.active_arm(history,future,texts[seq]);post=route=='right' and history[-1,13]<.5 and (future[:,13]<.5).mean()>=.75
    qv=_visual_descriptor(context[-1]);qa=((np.concatenate((history,future),0)-mean)/std).reshape(-1);vd=((vis[clean]-qv)**2).mean(1);ad=((act[clean]-qa)**2).mean(1);score=vd/max(float(np.median(vd)),1e-9)+ACTION_WEIGHT*ad/max(float(np.median(ad)),1e-9);loc=int(np.argmin(score));row=int(clean[loc]);distance=float(ad[loc]);raw=act[row].reshape(-1,14)*std+mean;qterm=future[-1,7:13]-history[-1,7:13];lterm=raw[-1,7:13]-raw[-9,7:13];den=float(np.linalg.norm(qterm)*np.linalg.norm(lterm));alignment=float(np.dot(qterm,lterm)/den) if den>1e-8 else 0.;motion=_path_length(history,future);alpha=float(np.clip(4*np.exp(-distance/DISTANCE_SCALE)*np.clip((alignment+1)/2,0,1)*np.clip(motion/MOTION_SCALE,0,1),0,1)) if post else 0.
    chunks.append({'post':post,'distance':distance,'alignment':alignment,'motion':motion,'alpha':alpha,'reward_peak':float(scores[seq,chunk*8:(chunk+1)*8].max())})
    pred=frames[seq,chunk*8:(chunk+1)*8];context=np.concatenate((context,pred),0)[-5:];history=np.concatenate((history,future),0)[-4:]
   peak_chunk=int(np.argmax([x['reward_peak'] for x in chunks]));peak=max(x['reward_peak'] for x in chunks);row={'split':split,'sequence':int(seq),'path':paths[seq],'hit':bool(peak>=.9),'peak':float(peak),'peak_chunk':peak_chunk,'peak_post':bool(chunks[peak_chunk]['post']),**{f'peak_{k}':float(chunks[peak_chunk][k]) for k in ('distance','alignment','motion','alpha')}};allrows.append(row)
 success=[x for x in allrows if x['split']=='success'];failure=[x for x in allrows if x['split']=='failure'];features={}
 for feature,direction in [('peak_distance','le'),('peak_alignment','ge'),('peak_alpha','ge'),('peak_motion','ge')]:
  values=sorted(set(x[feature] for x in allrows));best=[]
  for threshold in values:
   keep=lambda x:(x[feature]<=threshold if direction=='le' else x[feature]>=threshold)
   sh=sum(x['hit'] and keep(x) for x in success)/len(success);fh=sum(x['hit'] and keep(x) for x in failure)/len(failure)
   if sh>=.75 and fh<=.2:best.append({'threshold':float(threshold),'success_rate':sh,'failure_rate':fh,'margin':sh-fh})
  features[feature]={'direction':direction,'solutions':sorted(best,key=lambda x:(x['margin'],-x['failure_rate']),reverse=True)[:10]}
 report={'format':'strict-track2-v248-false-positive-gate-analysis-v1','rows':allrows,'features':features,'summary':{name:{'success_hit':stats([x[name] for x in success if x['hit']]),'failure_hit':stats([x[name] for x in failure if x['hit']]),'failure_nonhit':stats([x[name] for x in failure if not x['hit']])} for name in ('peak_distance','peak_alignment','peak_alpha','peak_motion')},'guards':{'public_data_only':True,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'features':features,'summary':report['summary']},indent=2))
if __name__=='__main__':main()
