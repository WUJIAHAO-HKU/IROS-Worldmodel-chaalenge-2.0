#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge');J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810';O=ROOT/'artifacts/strict_track2_official_20260810';NAME='v382_source_gate_terminal_hold_seed1545_20260823';RUN=J/NAME;REG=O/'run_registry'/NAME
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
 source=J/'v375_bounded_cartesian_phase_pilot_seed1538_20260823/release';training=J/'v381_public_recursive_source_gate_seed1544_20260823/training_report.json';gate=J/'v381_public_recursive_source_gate_seed1544_20260823/recursive_source_gate.npz';runtime=ROOT/'pipeline/wam_pipeline/v382_source_gate_terminal_hold_runtime.py';backends=ROOT/'pipeline/wam_pipeline/backends.py';library=J/'v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz'
 for p in (source,training,gate,runtime,backends,library):
  if not p.exists():raise FileNotFoundError(p)
 tr=json.loads(training.read_text())
 if not tr.get('passed') or tr['selection']['teacher_specificity']!=1 or tr['selection']['generated_recall']!=1:raise RuntimeError('v381 source gate not accepted')
 RUN.mkdir(parents=True);(RUN/'audit').mkdir();REG.mkdir(parents=True);release=RUN/'release';release.mkdir()
 for n in ('left_expert','right_expert','cartesian_pose.pt','arm_routed_autoregressive_manifest.json','cartesian_phase_manifest.json'):
  p=source/n;os.symlink(p.resolve(),release/n,target_is_directory=p.is_dir())
 os.symlink(gate.resolve(),release/'source_gate.npz');m={'format':'strict-track2-v382-source-gate-terminal-hold-release-v1','model_version':'track2-v382-v355-real-sourcegate-cartesian-terminal-hold','source_gate':'source_gate.npz','source_gate_sha256':sha(gate),'source_threshold':float(tr['selection']['threshold']),'cartesian_alpha':0.75,'renderer':'repeat blended Cartesian terminal frame for all 8 frames only on generated contexts','reward_or_outcomes_used':False,'hidden_or_final_data':False};mp=release/'source_terminal_hold_manifest.json';mp.write_text(json.dumps(m,indent=2)+'\n')
 pre={'format':'strict-track2-v382-source-gate-terminal-hold-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'model_version':m['model_version'],'release':str(release),'selection':{'source_gate_training':tr['selection'],'cartesian_alpha':0.75,'renderer':'fixed terminal hold after v379/v380 rejection'},'fixed_recursive_gate':{'exact_rows':512,'teacher_frames_and_reward_bit_exact_v355':True,'recursive_rgb_ratio_le_0p85_both':True,'recursive_temporal_ratio_le_0p98_both':True,'recursive_reward_error_improves_both':True,'recursive_reward_mean_nonnegative_gain_both':True,'all_required_before_service_or_rl':True},'evidence_sha256':{'source_training':sha(training),'source_gate':sha(gate),'runtime':sha(runtime),'backends':sha(backends),'library':sha(library)},'guards':{'public_world_model_data_only':True,'policy_modified':False,'official_reward_modified':False,'hidden_or_final_data':False,'real_submission':False}}
 text=json.dumps(pre,indent=2)+'\n';(RUN/'release_registration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'release':str(release)},indent=2))
if __name__=='__main__':main()
