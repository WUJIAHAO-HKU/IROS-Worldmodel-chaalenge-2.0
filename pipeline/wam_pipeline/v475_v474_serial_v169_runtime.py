"""v475: v474 formula with composition-independent scalar v169 calls."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from wam_pipeline.v474_v473_median4_parent_runtime import Track2V474V473Median4Parent

FORMAT="track2-v475-v474-serial-v169-parent-release-v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def tree(root):
 root=Path(root).resolve();return hashlib.sha256("".join(f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(x for x in root.rglob("*") if x.is_file())).encode()).hexdigest()
class Track2V475V474SerialV169(Track2V474V473Median4Parent):
 def __init__(self,release_dir,device="cuda"):
  root=Path(release_dir).resolve();mp=root/"v475_serial_v169_manifest.json";m=json.loads(mp.read_text());parent=Path(m["parent_release"]).resolve()
  if m.get("format")!=FORMAT or not parent.is_dir() or sha(Path(__file__))!=m.get("runtime_source_sha256") or sha(parent/"v474_median4_c_group_e_manifest.json")!=m.get("parent_manifest_sha256") or not (tree(parent)==m.get("parent_tree_sha256")=="5a41ab6b8e1617dc1f32b1d095dac742d87ed0822db1fdb1d40690b19c1fe371") or not (sha(root/"v475_serial_interface_s1_migration_contract.json")==m.get("contract_sha256")=="e4562d466bf9c82234eeabad2397e5531653a23d6fd393246bca27af6cd5eaa9") or m.get("serial_v169_scalar") is not True or m.get("algorithm_or_gate_modified") is not False or m.get("s1_authorized") is not False or m.get("rl_authorized") is not False:raise RuntimeError("bad v475 release")
  super().__init__(parent,device)
 def _v169(self,context,history,future,seeds,instructions):
  if len(context)==0:return np.empty((0,8,256,256,3),np.uint8)
  out=[self.v169.predict(context[i],history[i],future[i],int(seeds[i]),instructions[i]) for i in range(len(context))]
  value=np.stack(out)
  if value.shape!=(len(context),8,256,256,3) or value.dtype!=np.uint8:raise RuntimeError("v475 scalar v169 output")
  return value
 def predict_one_with_baseline(self,context,history,future,seed,instruction):
  return self.predict_with_baseline(context,history,future,seed,instruction)
 def predict_one_s1_pair(self,context,history,true_future,seed,instruction,override_action_feature):
  baseline,true,true_decision=self.predict_one_with_baseline(context,history,true_future,seed,instruction);override=baseline.copy();d=dict(true_decision);d["context_neighbor_sha256"]=list(true_decision["context_neighbor_sha256"]);d["action_prototype_sha256"]=list(true_decision["action_prototype_sha256"]);d["frame7_changed"]=False
  if d["gate"]:
   residual,contexts,actions=self._residual(np.asarray(context)[-1],history,true_future,override_action_feature);override[7]=np.clip(np.rint(baseline[7].astype(np.float32)+residual),0,255).astype(np.uint8);d["context_neighbor_sha256"]=contexts;d["action_prototype_sha256"]=actions;d["frame7_changed"]=bool(not np.array_equal(override[7],baseline[7]))
  return baseline,true,override,true_decision,d
