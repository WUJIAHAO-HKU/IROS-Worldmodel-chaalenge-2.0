"""Generated-context routed Cartesian terminal-hold world model."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from .v337_public_recursive_ood_gate import PublicRecursiveOODGate
from .v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase
FORMAT='strict-track2-v382-source-gate-terminal-hold-release-v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class Track2V382SourceGateTerminalHold(Track2V375BoundedCartesianPhase):
 def __init__(self,checkpoint_dir,library_index,device='cuda'):
  super().__init__(checkpoint_dir,library_index,device);m=json.loads((self.root/'source_terminal_hold_manifest.json').read_text())
  if m.get('format')!=FORMAT:raise RuntimeError('unsupported v382 manifest')
  p=self.root/m['source_gate']
  if not p.is_file() or sha(p)!=m['source_gate_sha256']:raise RuntimeError('v382 source gate hash mismatch')
  if m.get('reward_or_outcomes_used') is not False:raise RuntimeError('invalid v382 data boundary')
  self.source_gate=PublicRecursiveOODGate(p);self.source_threshold=float(m['source_threshold']);self.cartesian_alpha=float(m['cartesian_alpha']);self.last_source_probability=None;self.last_cartesian_route=False
 def _route(self,c,a):
  p=float(self.source_gate.probability(c));return bool(a=='right' and p>=self.source_threshold),p
 def _correct(self,base,c,h,f):
  cart=self._right_prediction(base,c,h,f);terminal=np.clip(np.rint(base[-1].astype(np.float32)+self.cartesian_alpha*(cart[-1].astype(np.float32)-base[-1].astype(np.float32))),0,255).astype(np.uint8);return np.repeat(terminal[None],8,axis=0)
 def predict(self,c,h,f,seed,instruction):
  base=self.parent.predict(c,h,f,seed,instruction);arm=self.parent.active_arm(h,f,instruction or '');route,p=self._route(c,arm);self.last_arm_route=arm;self.last_source_probability=p;self.last_cartesian_route=route
  return self._correct(base,c,h,f) if route else base
 def predict_batch(self,contexts,histories,futures,seeds,instructions):
  base=self.parent.predict_batch(contexts,histories,futures,seeds,instructions);arms=[self.parent.active_arm(h,f,t or '') for h,f,t in zip(histories,futures,instructions,strict=True)];decisions=[self._route(c,a) for c,a in zip(contexts,arms,strict=True)];out=base.copy()
  for i,(route,_) in enumerate(decisions):
   if route:out[i]=self._correct(base[i],contexts[i],histories[i],futures[i])
  self.last_arm_route='mixed' if len(set(arms))>1 else arms[0];self.last_source_probability=decisions[0][1] if len(decisions)==1 else None;self.last_cartesian_route=decisions[0][0] if len(decisions)==1 else any(x[0] for x in decisions);return out
