#!/usr/bin/env python3
"""Outcome-free relative-action rescue coverage on frozen v211 requests."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from diagnose_v396_v211_actions import causal_action_features
from wam_pipeline.v397_relative_action_phase_gate import PublicRelativeActionPhaseGate

ROOT=Path(__file__).resolve().parents[2];J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'
CAPTURE=ROOT/'artifacts/strict_track2_official_20260810/run_registry/v211_v209_train_action_capture_h200_r2_step1_seed1410_20260818/bridge_audit'
def main()->None:
    gate=PublicRelativeActionPhaseGate(J/'v397_relative_action_terminal_rescue_seed1557_20260823/relative_action_phase_gate.npz')
    with np.load(J/'v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz',allow_pickle=False) as p:
        mean,scale,coef=(p[k].astype(np.float32) for k in ('feature_mean','feature_scale','coefficient'));intercept=float(p['intercept'].item())
    probability=[]
    for path in sorted(CAPTURE.glob('rollout_*.npz')):
        with np.load(path,allow_pickle=False) as p:
            for history,future in zip(p['history_actions'],p['future_actions'],strict=True):
                feature=causal_action_features(history,future);logit=float(((feature-mean)/scale)@coef+intercept);action_p=float(1/(1+np.exp(-np.clip(logit,-40,40))))
                post=bool(history[-1,13]<.5 and (future[:,13]<.5).mean()>=.75);seq=np.concatenate((history[-1:,7:13],future[:,7:13]),axis=0);length=float(np.linalg.norm(np.diff(seq,axis=0),axis=1).sum());failure=bool((history[-1,13]<=.5 and float(future[:,13].mean())>.5) or length<=1e-6 or (float(future[:,13].mean())<=.5 and action_p<.01))
                if post and action_p>=.99 and not failure:probability.append(gate.probability(history,future))
    a=np.asarray(probability)
    print(json.dumps({'contract':'outcome-free relative-actions-only coverage diagnostic','rows':400,'runtime_action_eligible':int(len(a)),'relative_action_phase_ready':int((a>=gate.threshold).sum()),'threshold':gate.threshold,'eligible_probability_quantiles':{str(q):float(np.quantile(a,q)) for q in (0,.1,.25,.5,.75,.9,.99,1)},'guards':{'outcomes_read':False,'rgb_read':False,'absolute_pose_read':False,'hidden_or_final_data':False}},indent=2))
if __name__=='__main__':main()
