"""Public-clean, action-conditioned progressive successor world model.

The runtime reads only request RGB/actions/instruction and a frozen public
expert-demonstration library.  It never reads reward, outcome, seed, request
identity, or evaluation metadata, and returns only eight RGB frames.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .v216_public_knn_blend_runtime import Track2V216PublicKNNBlend, _visual_descriptor

ACTION_WEIGHT=4.0
PROGRESS_OFFSET=24
DISTANCE_SCALE=2.9357216358184814
MOTION_SCALE=0.16446852684020996
ALPHA_SCALE=2.0
PREGRASP_ALPHA=0.70

def _path_length(history:np.ndarray,future:np.ndarray)->float:
    path=np.concatenate((history[-1:,7:13],future[:,7:13]),axis=0)
    return float(np.linalg.norm(np.diff(path,axis=0),axis=-1).sum())

class Track2V245CleanProgressiveSuccessor(Track2V216PublicKNNBlend):
    """Route right post-grasp actions through progressive public successors."""
    def __init__(self,checkpoint_dir,library_index,device='cuda'):
        super().__init__(checkpoint_dir,library_index,device)
        with np.load(self.library_index,allow_pickle=False) as values:
            episode_id=values['episode_id'].astype(np.int64)
        clean=episode_id>=20000
        self.clean_rows=np.flatnonzero(clean)
        if self.clean_rows.size!=1926:
            raise RuntimeError(f'v245 expected 1926 official public clean rows, got {self.clean_rows.size}')
        self.row_episode=np.full(len(self.paths),-1,dtype=np.int64)
        self.row_start=np.full(len(self.paths),-1,dtype=np.int64)
        self.episode_starts={}
        for row in self.clean_rows:
            stem=Path(self.paths[row]).stem
            eptext,starttext=stem.split('_')
            episode,start=int(episode_id[row]),int(starttext)
            self.row_episode[row]=episode;self.row_start[row]=start
            self.episode_starts.setdefault(episode,{})[start]=int(row)
        self.last_base_index=None;self.last_progressive_index=None;self.last_progressive_alpha=0.0

    @staticmethod
    def _post_grasp(history,future)->bool:
        return bool(history[-1,13]<.5 and (future[:,13]<.5).mean()>=.75)

    def _nearest_clean(self,context,history,future):
        query_visual=_visual_descriptor(context[-1])
        actions=np.concatenate((history,future),axis=0).astype(np.float32)
        query_action=((actions-self.action_mean)/self.action_std).reshape(-1)
        rows=self.clean_rows
        visual_distance=((self.visual[rows]-query_visual)**2).mean(1)
        action_distance=((self.action[rows]-query_action)**2).mean(1)
        score=visual_distance/max(float(np.median(visual_distance)),1e-9)
        score+=ACTION_WEIGHT*action_distance/max(float(np.median(action_distance)),1e-9)
        local=int(np.argmin(score))
        return int(rows[local]),float(action_distance[local])

    def _progressive_row(self,base:int)->int:
        lookup=self.episode_starts[int(self.row_episode[base])]
        starts=np.asarray(sorted(lookup),dtype=np.int64)
        goal=int(self.row_start[base])+PROGRESS_OFFSET
        later=starts[starts>=goal]
        chosen=int(later[0] if later.size else starts[-1])
        return int(lookup[chosen])

    def _alpha(self,history,future,base,distance)->float:
        raw=self.action[base].reshape(-1,14)*self.action_std+self.action_mean
        lib_history,lib_future=raw[:-8],raw[-8:]
        query_terminal=future[-1,7:13]-history[-1,7:13]
        lib_terminal=lib_future[-1,7:13]-lib_history[-1,7:13]
        denominator=float(np.linalg.norm(query_terminal)*np.linalg.norm(lib_terminal))
        alignment=float(np.dot(query_terminal,lib_terminal)/denominator) if denominator>1e-8 else 0.0
        confidence=float(np.exp(-distance/DISTANCE_SCALE))
        alignment_gate=float(np.clip((alignment+1.0)/2.0,0.0,1.0))
        motion_gate=float(np.clip(_path_length(history,future)/MOTION_SCALE,0.0,1.0))
        return float(np.clip(ALPHA_SCALE*confidence*alignment_gate*motion_gate,0.0,1.0))

    @staticmethod
    def _blend(parent,target,future,alpha):
        per_frame=(future[:,13]<.5).astype(np.float32)*float(alpha)
        per_frame=per_frame.reshape(8,1,1,1)
        return np.clip(np.rint((1-per_frame)*parent.astype(np.float32)+per_frame*target.astype(np.float32)),0,255).astype(np.uint8)

    def _right_prediction(self,parent,context,history,future):
        base,distance=self._nearest_clean(context,history,future)
        self.last_base_index=base
        if not self._post_grasp(history,future):
            self.last_progressive_index=None;self.last_progressive_alpha=0.0;self.last_retrieval_index=base
            return self._blend(parent,self._target(base),future,PREGRASP_ALPHA)
        progressive=self._progressive_row(base);alpha=self._alpha(history,future,base,distance)
        self.last_progressive_index=progressive;self.last_progressive_alpha=alpha;self.last_retrieval_index=progressive
        return self._blend(parent,self._target(progressive),future,alpha)

    def predict(self,context_frames,history_actions,future_actions,seed,instruction):
        parent=self.parent.predict(context_frames,history_actions,future_actions,seed,instruction)
        arm=self.parent.active_arm(history_actions,future_actions,instruction or '')
        self.last_arm_route=arm
        if arm!='right':
            self.last_base_index=None;self.last_progressive_index=None;self.last_progressive_alpha=0.0;self.last_retrieval_index=None
            return parent
        return self._right_prediction(parent,context_frames,history_actions,future_actions)

    def predict_batch(self,context_frames,history_actions,future_actions,seeds,instructions):
        parent=self.parent.predict_batch(context_frames,history_actions,future_actions,seeds,instructions)
        routes=[self.parent.active_arm(h,f,t or '') for h,f,t in zip(history_actions,future_actions,instructions)]
        output=parent.copy()
        for i,arm in enumerate(routes):
            if arm=='right':output[i]=self._right_prediction(parent[i],context_frames[i],history_actions[i],future_actions[i])
        self.last_arm_route='mixed' if len(set(routes))>1 else routes[0]
        return output
