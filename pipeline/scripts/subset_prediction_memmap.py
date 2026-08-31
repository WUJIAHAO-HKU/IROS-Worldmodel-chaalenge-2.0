#!/usr/bin/env python3
"""Subset an uncompressed prediction memmap directory by reference names."""
import argparse
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument("--source-dir",required=True);p.add_argument("--reference",required=True);p.add_argument("--output",required=True);a=p.parse_args();root=Path(a.source_dir);source=np.load(root/"prediction.npy",mmap_mode="r");names=np.load(root/"windows.npy",allow_pickle=False).astype(str)
 with np.load(a.reference,allow_pickle=False) as z:wanted=z["windows"].astype(str)
 lookup={n:i for i,n in enumerate(names)};prediction=np.stack([source[lookup[n]] for n in wanted]);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("wb") as h:np.savez_compressed(h,prediction=prediction,windows=wanted)
 print({"samples":len(wanted),"output":str(out)})
if __name__=="__main__":main()
