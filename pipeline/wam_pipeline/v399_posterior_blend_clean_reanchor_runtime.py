"""Use the frozen v389 phase posterior as a continuous terminal blend weight."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from .v390_continuous_phase_clean_reanchor_runtime import Track2V390ContinuousPhaseCleanReanchor

FORMAT='strict-track2-v399-posterior-blend-clean-reanchor-release-v1'
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
class Track2V399PosteriorBlendCleanReanchor(Track2V390ContinuousPhaseCleanReanchor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);path=self.root/'posterior_blend_clean_reanchor_manifest.json';manifest=json.loads(path.read_text())
        if manifest.get('format')!=FORMAT or manifest.get('blend_formula')!='alpha = frozen phase posterior probability':raise RuntimeError('invalid v399 manifest')
        gate=self.root/manifest['phase_gate']
        if not gate.is_file() or sha(gate)!=manifest['phase_gate_sha256']:raise RuntimeError('v399 phase gate hash mismatch')
        if manifest.get('reward_or_outcomes_used') is not False:raise RuntimeError('v399 data boundary violation')
        self.last_posterior_blend_alpha=0.0
    def _apply(self,baseline,context,history,future,arm):
        _,source_probability,phase_row=self._route(context,history,future,arm)
        probability=self.last_continuous_phase_probability
        if phase_row is None or probability is None:
            self.last_posterior_blend_alpha=0.0;return baseline,False,source_probability,phase_row,None
        alpha=float(probability);target,episode=self._endpoint_terminal(future);self.last_posterior_blend_alpha=alpha
        return self._blend(baseline,self._target(target),future,alpha),bool(alpha>0.0),source_probability,phase_row,episode
