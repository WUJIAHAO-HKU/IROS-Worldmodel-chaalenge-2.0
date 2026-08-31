#!/usr/bin/env python3
import argparse, glob
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument('--windows',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
X=[]; y=[]; labels=[]
for p in sorted(a.windows.glob('episode*_*.npz')):
    with np.load(p,allow_pickle=False) as z:
        actions=np.concatenate((z['history_actions'],z['future_actions']),axis=0).astype(np.float32)
        X.append(np.abs(np.diff(actions,axis=0)).mean(0))
        y.append(np.abs(z['target_frames'].astype(np.float32)[0]-z['context_frames'].astype(np.float32)[-1]).mean())
        labels.append(bool(z['arm_right']))
X=np.asarray(X); y=np.asarray(y); labels=np.asarray(labels)
out={'count':len(y),'groups':{}}
for name,mask in [('left',~labels),('right',labels)]:
    out['groups'][name]={'count':int(mask.sum()),'corr':[float(np.corrcoef(X[mask,i],y[mask])[0,1]) if mask.sum()>2 else None for i in range(X.shape[1])], 'mean_motion':X[mask].mean(0).tolist()}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(__import__('json').dumps(out,indent=2)+'\n'); print(__import__('json').dumps(out,indent=2))
