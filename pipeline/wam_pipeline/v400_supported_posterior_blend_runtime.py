"""Posterior terminal blend only inside the classifier's declared support."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .v390_continuous_phase_clean_reanchor_runtime import Track2V390ContinuousPhaseCleanReanchor
FORMAT='strict-track2-v400-supported-posterior-blend-release-v1'
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
class Track2V400SupportedPosteriorBlend(Track2V390ContinuousPhaseCleanReanchor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);manifest=json.loads((self.root/'supported_posterior_blend_manifest.json').read_text())
        if manifest.get('format')!=FORMAT or float(manifest.get('support_probability_min',-1))!=0.5:raise RuntimeError('invalid v400 manifest')
        gate=self.root/manifest['phase_gate']
        if not gate.is_file() or sha(gate)!=manifest['phase_gate_sha256'] or manifest.get('reward_or_outcomes_used') is not False:raise RuntimeError('v400 data boundary/hash violation')
        self.support_probability_min=0.5;self.last_posterior_blend_alpha=0.0
    def _apply(self,baseline,context,history,future,arm):
        _,source_probability,phase_row=self._route(context,history,future,arm);probability=self.last_continuous_phase_probability
        if phase_row is None or probability is None or probability<self.support_probability_min:
            self.last_posterior_blend_alpha=0.0;return baseline,False,source_probability,phase_row,None
        alpha=float(probability);target,episode=self._endpoint_terminal(future);self.last_posterior_blend_alpha=alpha
        return self._blend(baseline,self._target(target),future,alpha),True,source_probability,phase_row,episode
