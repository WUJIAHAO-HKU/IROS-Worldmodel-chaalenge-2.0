"""Action-conditioned terminal successor from official public clean demos."""
from __future__ import annotations
import numpy as np
from .v216_public_knn_blend_runtime import _visual_descriptor
from .v245_clean_progressive_successor_runtime import (
    Track2V245CleanProgressiveSuccessor,DISTANCE_SCALE,MOTION_SCALE,_path_length,PREGRASP_ALPHA
)

ALPHA_SCALE=4.0
TERMINAL_START_MIN=112

class Track2V247TerminalSuccessor(Track2V245CleanProgressiveSuccessor):
    """Use one shared terminal successor per observed expert episode."""
    def __init__(self,checkpoint_dir,library_index,device='cuda'):
        super().__init__(checkpoint_dir,library_index,device)
        self.terminal_row={}
        for episode,lookup in self.episode_starts.items():
            starts=np.asarray(sorted(lookup),dtype=np.int64)
            late=starts[starts>=TERMINAL_START_MIN]
            chosen=int(late[-1] if late.size else starts[-1])
            self.terminal_row[int(episode)]=int(lookup[chosen])

    def _terminal_for_context(self,context)->int:
        q=_visual_descriptor(context[-1]);rows=self.clean_rows
        distance=((self.visual[rows]-q)**2).mean(1)
        visual_row=int(rows[int(np.argmin(distance))])
        return self.terminal_row[int(self.row_episode[visual_row])]

    def _alpha(self,history,future,base,distance)->float:
        raw=self.action[base].reshape(-1,14)*self.action_std+self.action_mean
        query_terminal=future[-1,7:13]-history[-1,7:13]
        library_terminal=raw[-1,7:13]-raw[-9,7:13]
        denominator=float(np.linalg.norm(query_terminal)*np.linalg.norm(library_terminal))
        alignment=float(np.dot(query_terminal,library_terminal)/denominator) if denominator>1e-8 else 0.0
        confidence=float(np.exp(-distance/DISTANCE_SCALE))
        alignment_gate=float(np.clip((alignment+1.0)/2.0,0.0,1.0))
        motion_gate=float(np.clip(_path_length(history,future)/MOTION_SCALE,0.0,1.0))
        return float(np.clip(ALPHA_SCALE*confidence*alignment_gate*motion_gate,0.0,1.0))

    def _right_prediction(self,parent,context,history,future):
        base,distance=self._nearest_clean(context,history,future);self.last_base_index=base
        if not self._post_grasp(history,future):
            self.last_progressive_index=None;self.last_progressive_alpha=0.0;self.last_retrieval_index=base
            return self._blend(parent,self._target(base),future,PREGRASP_ALPHA)
        target=self._terminal_for_context(context);alpha=self._alpha(history,future,base,distance)
        self.last_progressive_index=target;self.last_progressive_alpha=alpha;self.last_retrieval_index=target
        return self._blend(parent,self._target(target),future,alpha)
