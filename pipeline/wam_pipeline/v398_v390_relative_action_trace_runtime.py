"""Output-equivalent v390 runtime with relative-action rescue telemetry."""
from __future__ import annotations
import json, os
from pathlib import Path
from .v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN
from .v390_continuous_phase_clean_reanchor_runtime import Track2V390ContinuousPhaseCleanReanchor
from .v397_relative_action_phase_gate import PublicRelativeActionPhaseGate

class Track2V398V390RelativeActionTrace(Track2V390ContinuousPhaseCleanReanchor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        trace=os.environ.get('WAM_V398_TRACE_PATH','');gate=os.environ.get('WAM_V398_ACTION_PHASE_GATE','')
        if not trace or not Path(gate).is_file():raise RuntimeError('v398 trace path/action phase gate required')
        self.trace_path=Path(trace);self.trace_path.parent.mkdir(parents=True,exist_ok=True);self.relative_action_phase_gate=PublicRelativeActionPhaseGate(gate);self._trace_records=[]
    def _route(self,context,history,future,arm):
        source_probability=float(self.source_gate.probability(context));right=arm=='right';source_ready=source_probability>=self.source_threshold;post_grasp=bool(self._post_grasp(history,future));action_probability=float(self._probability(history,future));probability_ready=action_probability>=ACTION_PROBABILITY_MIN;signature=self._signature(history,future,action_probability)
        base=None;continuous_probability=None;continuous_ready=False;relative_probability=None;relative_ready=False
        eligible=bool(right and source_ready and post_grasp and probability_ready and signature is None)
        if eligible:
            base=self._action_phase_base(history,future);continuous_probability=float(self.continuous_phase_gate.probability(context,history,future));continuous_ready=continuous_probability>=self.continuous_phase_gate.threshold;relative_probability=float(self.relative_action_phase_gate.probability(history,future));relative_ready=relative_probability>=self.relative_action_phase_gate.threshold
        route=bool(eligible and continuous_ready)
        self._trace_records.append({'arm':arm,'right':right,'source_probability':source_probability,'source_ready':source_ready,'post_grasp':post_grasp,'action_probability':action_probability,'probability_ready':probability_ready,'failure_signature':signature,'eligible':eligible,'phase_base_row':base,'continuous_phase_probability':continuous_probability,'continuous_phase_ready':bool(continuous_ready),'relative_action_phase_probability':relative_probability,'relative_action_phase_ready':bool(relative_ready),'union_ready':bool(continuous_ready or relative_ready),'route':route})
        return route,source_probability,base
    def _flush(self):
        if self._trace_records:
            with self.trace_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps({'batch':self._trace_records},separators=(',',':'))+'\n')
            self._trace_records=[]
    def predict(self,*args,**kwargs):
        self._trace_records=[];output=super().predict(*args,**kwargs);self._flush();return output
    def predict_batch(self,*args,**kwargs):
        self._trace_records=[];output=super().predict_batch(*args,**kwargs);self._flush();return output
