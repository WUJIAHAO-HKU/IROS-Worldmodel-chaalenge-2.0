#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--cache',required=True,type=Path)
ap.add_argument('--output',required=True,type=Path)
ap.add_argument('--strength',required=True,type=float)
a=ap.parse_args()
with np.load(a.cache,allow_pickle=False) as z:
    data={k:z[k].copy() for k in z.files}
data['candidate'][:,-1] = ((1-a.strength)*data['candidate'][:,-1].astype(np.float32) + a.strength*data['baseline'][:,-1].astype(np.float32)).clip(0,255).astype(np.uint8)
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,**data)
print(a.output)
