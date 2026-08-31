"""Frozen v476 S1 interface; inference semantics are byte-identical to v475."""
from __future__ import annotations
import numpy as np
from wam_pipeline.v474_v473_median4_parent_runtime import action_feature,gate_decision
from wam_pipeline.v475_v474_serial_v169_runtime import Track2V475V474SerialV169

LINEAGE="v476"
INPUT_FORMAT="strict-track2-v476-s1-offline-inputs-v1"
RUNTIME_CLASS=Track2V475V474SerialV169

def action_only_postclose_mask(history,future,instructions):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32)
 if h.ndim!=3 or h.shape[1:]!=(4,14) or f.shape!=(len(h),8,14) or len(instructions)!=len(h):raise ValueError("v475 S1 action mask")
 return np.asarray([gate_decision(x,y,t)["gate"] for x,y,t in zip(h,f,instructions)],np.bool_)

def build_override_action_features(donor_history,donor_future):
 h=np.asarray(donor_history,np.float32);f=np.asarray(donor_future,np.float32)
 if h.ndim!=3 or h.shape[1:]!=(4,14) or f.shape!=(len(h),8,14):raise ValueError("v475 S1 donor actions")
 return np.stack([action_feature(x,y) for x,y in zip(h,f)]).astype(np.float32)

def predict_true_and_action_override(runtime,context,history,true_future,seeds,instructions,override_action_features):
 c=np.asarray(context);h=np.asarray(history,np.float32);f=np.asarray(true_future,np.float32);s=np.asarray(seeds);o=np.asarray(override_action_features,np.float32)
 n=len(c)
 if c.shape!=(n,5,256,256,3) or c.dtype!=np.uint8 or h.shape!=(n,4,14) or f.shape!=(n,8,14) or s.shape!=(n,) or o.shape!=(n,54) or len(instructions)!=n:raise ValueError("v475 S1 pair inputs")
 pairs=[runtime.predict_one_s1_pair(c[i],h[i],f[i],int(s[i]),instructions[i],o[i]) for i in range(n)]
 baseline=np.stack([x[0] for x in pairs]);true=np.stack([x[1] for x in pairs]);override=np.stack([x[2] for x in pairs]);decision=[x[3] for x in pairs];override_decision=[x[4] for x in pairs]
 if any(x.shape!=(n,8,256,256,3) or x.dtype!=np.uint8 for x in (baseline,true,override)):raise RuntimeError("v475 S1 pair output")
 for a,b in zip(decision,override_decision):
  for key in ("gate","explicit_right","postclose","no_transport","phase","context_neighbor_sha256"):
   if a[key]!=b[key]:raise RuntimeError(f"v475 S1 changed structural decision {key}")
 return {"same_request_v169":baseline,"v476_true":true,"v476_action_override":override,"true_decisions":decision,"override_decisions":override_decision}
