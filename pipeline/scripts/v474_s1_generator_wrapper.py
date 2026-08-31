"""Frozen v474 S1 generator interface only; performs no selection, RGB loading, reward, or inference by itself."""
from __future__ import annotations
import numpy as np
from wam_pipeline.v474_v473_median4_parent_runtime import Track2V474V473Median4Parent,action_feature,gate_decision
LINEAGE="v474";INPUT_FORMAT="strict-track2-v474-v473-parent-s1-offline-inputs-v1";RUNTIME_CLASS=Track2V474V473Median4Parent
def action_only_postclose_mask(history,future,instructions):
 h=np.asarray(history,np.float32);f=np.asarray(future,np.float32)
 if h.ndim!=3 or h.shape[1:]!=(4,14) or f.shape!=(len(h),8,14) or len(instructions)!=len(h):raise ValueError("v474 S1 action mask")
 return np.asarray([gate_decision(x,y,t)["gate"] for x,y,t in zip(h,f,instructions)],np.bool_)
def build_override_action_features(donor_history,donor_future):
 h=np.asarray(donor_history,np.float32);f=np.asarray(donor_future,np.float32)
 if h.ndim!=3 or h.shape[1:]!=(4,14) or f.shape!=(len(h),8,14):raise ValueError("v474 S1 donor actions")
 return np.stack([action_feature(x,y) for x,y in zip(h,f)]).astype(np.float32)
def predict_true_and_action_override(runtime,context,history,true_future,seeds,instructions,override_action_features):
 baseline,true,decision=runtime.predict_batch_with_baseline(context,history,true_future,seeds,instructions);override_baseline,override,override_decision=runtime.predict_batch_s1_action_override(context,history,true_future,seeds,instructions,override_action_features)
 if not np.array_equal(baseline,override_baseline):raise RuntimeError("v474 S1 changed same-request v169")
 for a,b in zip(decision,override_decision):
  for key in ("gate","explicit_right","postclose","no_transport","phase","context_neighbor_sha256"):
   if a[key]!=b[key]:raise RuntimeError(f"v474 S1 changed structural decision {key}")
 return {"same_request_v169":baseline,"v474_true":true,"v474_action_override":override,"true_decisions":decision,"override_decisions":override_decision}
