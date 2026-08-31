"""Specific terminal successor with a public expert direction floor."""
from __future__ import annotations
import numpy as np
from .v245_clean_progressive_successor_runtime import DISTANCE_SCALE,MOTION_SCALE,_path_length
from .v247_terminal_successor_runtime import Track2V247TerminalSuccessor

ALIGNMENT_FLOOR=.50
DISTANCE_TEMPERATURE=.50
ALPHA_SCALE=16.0

class Track2V250SpecificTerminalSuccessor(Track2V247TerminalSuccessor):
    def _alpha(self,history,future,base,distance)->float:
        raw=self.action[base].reshape(-1,14)*self.action_std+self.action_mean
        q=future[-1,7:13]-history[-1,7:13];r=raw[-1,7:13]-raw[-9,7:13]
        den=float(np.linalg.norm(q)*np.linalg.norm(r));alignment=float(np.dot(q,r)/den) if den>1e-8 else 0.0
        confidence=float(np.exp(-distance/(DISTANCE_TEMPERATURE*DISTANCE_SCALE)))
        gate=float(np.clip((alignment-ALIGNMENT_FLOOR)/(1-ALIGNMENT_FLOOR),0,1))
        motion=float(np.clip(_path_length(history,future)/MOTION_SCALE,0,1))
        return float(np.clip(ALPHA_SCALE*confidence*gate*motion,0,1))
