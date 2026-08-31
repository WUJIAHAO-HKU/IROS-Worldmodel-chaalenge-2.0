#!/usr/bin/env python3
"""Preregister one fixed terminal-hold renderer after v379 rejection."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v380_terminal_hold_diagnostic_seed1543_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 rejected=J/'v379_temporal_renderer_sweep_seed1542_20260823/audit/temporal_renderer_sweep.json';script=ROOT/'pipeline/scripts/audit_v380_terminal_hold_diagnostic.py'
 for p in (rejected,script):
  if not p.exists():raise FileNotFoundError(p)
 if json.loads(rejected.read_text()).get('passed') is not False:raise RuntimeError('v379 rejection missing')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True)
 x={'format':'strict-track2-v380-terminal-hold-diagnostic-preregistration-v1','created_at':datetime.now(timezone.utc).isoformat(),'fixed_model':{'source_threshold':0.05,'cartesian_alpha':0.75,'renderer':'on routed requests repeat the unchanged raw candidate terminal frame for all eight outputs'},'calibration_episodes':[30,42],'episode_disjoint_test_episodes':[32,44],'acceptance':{'teacher_routes_zero':True,'recursive_rgb_ratio_le_0p80':True,'recursive_temporal_ratio_le_0p90':True,'generated_route_rate_ge_0p50':True},'reward_or_outcome_used':False,'evidence_sha256':{'v379_rejection':sha(rejected),'script':sha(script)},'guards':{'public_train_only':True,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(x,indent=2)+'\n';(RUN/'preregistration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN)},indent=2))
if __name__=='__main__':main()
