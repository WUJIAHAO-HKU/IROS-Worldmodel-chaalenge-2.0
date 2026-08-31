"""Output-equivalent v400 runtime with route-stage telemetry only."""
from __future__ import annotations
import json,os
from pathlib import Path
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v400_supported_posterior_blend_runtime import Track2V400SupportedPosteriorBlend
class Track2V401V400RouteTrace(Track2V400SupportedPosteriorBlend):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);value=os.environ.get('WAM_V401_TRACE_PATH','')
  if not value:raise RuntimeError('WAM_V401_TRACE_PATH required')
  self.trace_path=Path(value);self.trace_path.parent.mkdir(parents=True,exist_ok=True);self._trace_records=[]
 def _route(self,context,history,future,arm):
  source_probability=float(self.source_gate.probability(context));right=arm=='right';source_ready=source_probability>=self.source_threshold;post_grasp=bool(self._post_grasp(history,future));action_probability=float(self._probability(history,future));probability_ready=action_probability>=ACTION_PROBABILITY_MIN;signature=self._signature(history,future,action_probability);base=None;hard=False;p=None;continuous=False;supported=False
  eligible=bool(right and source_ready and post_grasp and probability_ready and signature is None)
  if eligible:
   base=self._action_phase_base(history,future);hard,_,_,_=self._phase(base);p=float(self.continuous_phase_gate.probability(context,history,future));continuous=p>=self.continuous_phase_gate.threshold;supported=p>=self.support_probability_min
  self.last_continuous_phase_probability=p;self.last_hard_phase_ready=bool(hard)
  self._trace_records.append({'arm':arm,'right':right,'source_probability':source_probability,'source_ready':source_ready,'post_grasp':post_grasp,'action_probability':action_probability,'probability_ready':probability_ready,'failure_signature':signature,'eligible':eligible,'phase_base_row':base,'hard_phase_ready':bool(hard),'continuous_phase_probability':p,'continuous_phase_ready':bool(continuous),'supported_phase_ready':bool(supported),'route':bool(eligible and supported)})
  return bool(eligible and continuous),source_probability,base
 def _flush(self):
  if self._trace_records:
   with self.trace_path.open('a',encoding='utf-8') as f:f.write(json.dumps({'batch':self._trace_records},separators=(',',':'))+'\n')
   self._trace_records=[]
 def predict(self,*args,**kwargs):self._trace_records=[];out=super().predict(*args,**kwargs);self._flush();return out
 def predict_batch(self,*args,**kwargs):self._trace_records=[];out=super().predict_batch(*args,**kwargs);self._flush();return out
