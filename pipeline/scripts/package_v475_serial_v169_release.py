#!/usr/bin/env python3
"""Package the mechanical v475 serial-v169 runtime without copying parent assets."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path
PARENT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v474_v473_median4_parent_release_seed1618_20260824")
PARENT_MANIFEST="4eecc242de272787e5aeac9df11739bf0d62cb4ba42d9224d8616a5e779ef2b9";PARENT_TREE="5a41ab6b8e1617dc1f32b1d095dac742d87ed0822db1fdb1d40690b19c1fe371";FORMAT="track2-v475-v474-serial-v169-parent-release-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def tree(root):
 root=Path(root).resolve();return hashlib.sha256("".join(f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(x for x in root.rglob("*") if x.is_file())).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--runtime",required=True,type=Path);p.add_argument("--contract",required=True,type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args();tmp=a.output.with_name(a.output.name+".partial")
 if a.output.exists() or tmp.exists():raise FileExistsError(a.output)
 if sha(PARENT/"v474_median4_c_group_e_manifest.json")!=PARENT_MANIFEST or tree(PARENT)!=PARENT_TREE:raise RuntimeError("v475 parent drift")
 if sha(a.contract)!="e4562d466bf9c82234eeabad2397e5531653a23d6fd393246bca27af6cd5eaa9":raise RuntimeError("v475 contract drift")
 tmp.mkdir(parents=True);shutil.copy2(a.runtime,tmp/"v475_v474_serial_v169_runtime.py");shutil.copy2(a.contract,tmp/"v475_serial_interface_s1_migration_contract.json")
 m={"format":FORMAT,"runtime":"v475_v474_serial_v169_runtime.py","runtime_source_sha256":sha(a.runtime),"contract_sha256":sha(a.contract),"parent_release":str(PARENT.resolve()),"parent_manifest_sha256":PARENT_MANIFEST,"parent_tree_sha256":PARENT_TREE,"v474_runtime_source_sha256":"139a178a91be1b8d987f7c708e82c33922afb920f8eace98df89636b1d0142d4","all200_library_sha256":"009274cadb1b98a05b4f1297e6c1b6d666428f69086e69b7c139988f30937478","serial_v169_scalar":True,"algorithm_or_gate_modified":False,"library_or_model_modified":False,"official_reward_runtime_used":False,"policy_modified":False,"s1_authorized":False,"rl_authorized":False}
 with (tmp/"v475_serial_v169_manifest.json").open("x") as f:json.dump(m,f,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,a.output);fd=os.open(a.output.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps({"passed":True,"release":str(a.output),"manifest_sha256":sha(a.output/"v475_serial_v169_manifest.json")},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
