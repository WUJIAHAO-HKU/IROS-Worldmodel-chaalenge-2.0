#!/usr/bin/env python3
"""Real-v169 n=9 interface audit for v475; no reward/dev/policy/RL."""
from __future__ import annotations
import argparse,hashlib,importlib,json,os
from pathlib import Path
import numpy as np
FORMAT="strict-track2-v475-real-v169-interface-s0-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def tree(root):
 root=Path(root).resolve();return hashlib.sha256("".join(f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(x for x in root.rglob("*") if x.is_file())).encode()).hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+".tmp")
 if p.exists() or tmp.exists():raise FileExistsError(p)
 with tmp.open("x") as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument("--release",required=True,type=Path);p.add_argument("--runtime",required=True,type=Path);p.add_argument("--packager",required=True,type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args();m=json.loads((a.release/"v475_serial_v169_manifest.json").read_text());checks={}
 checks["release_tree_exact"]={x.name for x in a.release.iterdir()}=={"v475_serial_v169_manifest.json","v475_v474_serial_v169_runtime.py","v475_serial_interface_s1_migration_contract.json"} and sha(a.release/"v475_v474_serial_v169_runtime.py")==sha(a.runtime)==m["runtime_source_sha256"] and sha(a.release/"v475_serial_interface_s1_migration_contract.json")==m["contract_sha256"]=="e4562d466bf9c82234eeabad2397e5531653a23d6fd393246bca27af6cd5eaa9" and m["parent_tree_sha256"]=="5a41ab6b8e1617dc1f32b1d095dac742d87ed0822db1fdb1d40690b19c1fe371"
 checks["source_closure"]=sha(a.packager)=="4e01342a0d5b4ebf459c7fb5677f463bcdf47dfc951a01d10e085a8d1fe4a56e" and m["algorithm_or_gate_modified"] is False and m["library_or_model_modified"] is False and m["s1_authorized"] is False and m["rl_authorized"] is False
 mod=importlib.import_module("wam_pipeline.v475_v474_serial_v169_runtime");checks["module_exact"]=Path(mod.__file__).resolve()==a.runtime.resolve()
 obj=mod.Track2V475V474SerialV169(a.release,"cuda")
 n=9;context=np.stack([np.full((5,256,256,3),i*17,np.uint8) for i in range(n)]);history=np.zeros((n,4,14),np.float32);future=np.zeros((n,8,14),np.float32);future[2:8,:,7]=np.linspace(0,.2,8);history[1,-1,13]=1.;future[1,:,13]=1.;seeds=np.arange(11,20,dtype=np.int64);instructions=["use left arm","use right arm"]+["use right arm"]*7
 baseline,batch,decisions=obj.predict_batch_with_baseline(context,history,future,seeds,instructions)
 scalar_base=[];scalar=[]
 for i in range(n):
  b,o,_=obj.predict_with_baseline(context[i],history[i],future[i],int(seeds[i]),instructions[i]);scalar_base.append(b);scalar.append(o)
 scalar_base=np.stack(scalar_base);scalar=np.stack(scalar)
 direct=np.stack([obj.v169.predict(context[i],history[i],future[i],int(seeds[i]),instructions[i]) for i in range(n)])
 perm=np.asarray([8,3,0,7,1,6,2,5,4]);pb,po,_=obj.predict_batch_with_baseline(context[perm],history[perm],future[perm],seeds[perm],[instructions[i] for i in perm]);rb=np.empty_like(pb);ro=np.empty_like(po);rb[perm]=pb;ro[perm]=po
 base=importlib.import_module("wam_pipeline.v474_v473_median4_parent_runtime");original=obj.v169.predict;calls=[0]
 def counted(*args,**kwargs):calls[0]+=1;return original(*args,**kwargs)
 obj.v169.predict=counted
 try:pair_b,pair_t,pair_o,pair_d,pair_od=obj.predict_one_s1_pair(context[2],history[2],future[2],int(seeds[2]),instructions[2],base.action_feature(history[3],future[3]))
 finally:obj.v169.predict=original
 checks.update({"real_v169_scalar_reference":np.array_equal(baseline,direct),"baseline_scalar_batch_bitexact":np.array_equal(baseline,scalar_base),"baseline_permutation_bitexact":np.array_equal(baseline,rb),"candidate_scalar_batch_bitexact":np.array_equal(batch,scalar),"candidate_permutation_bitexact":np.array_equal(batch,ro),"pair_single_Bb_call":calls[0]==1 and np.array_equal(pair_b,baseline[2]) and np.array_equal(pair_t,batch[2]) and pair_d["gate"] is True and pair_od["gate"] is True,"left_and_g0_all8_exact":np.array_equal(batch[:2],baseline[:2]),"enabled_frames0_to6_exact":np.array_equal(batch[2:,0:7],baseline[2:,0:7]),"gate_formula_unchanged":[d["gate"] for d in decisions]==[False,False]+[True]*7,"no_transport_empty_E":decisions[8]["no_transport"] is True and decisions[8]["action_prototype_sha256"]==[],"shape_dtype":batch.shape==baseline.shape==(9,8,256,256,3) and batch.dtype==baseline.dtype==np.uint8})
 passed=all(checks.values());receipt={"format":FORMAT,"passed":passed,"checks":checks,"manifest_sha256":sha(a.release/"v475_serial_v169_manifest.json"),"release_tree_sha256":tree(a.release),"n":9,"real_v169_loaded":True,"rgb_source":"synthetic deterministic uint8 interface fixture; no public dev/final RGB","reward_loaded":False,"policy_updates":0,"s1_authorized":False,"rl_authorized":False};atomic(a.output,receipt);print(json.dumps(receipt,sort_keys=True));return 0 if passed else 3
if __name__=="__main__":raise SystemExit(main())
