#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/root/autodl-tmp/IROS_WAM_2.0 challenge'); J=ROOT/'artifacts/strict_track2_joint_augmentation_20260810'; O=ROOT/'artifacts/strict_track2_official_20260810'
NAME='v397_relative_action_terminal_rescue_seed1557_20260823'; RUN=J/NAME; REG=O/'run_registry'/NAME
def sha(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->None:
    if RUN.exists() or REG.exists():raise FileExistsError('refusing overwrite')
    sources={'trainer':ROOT/'pipeline/scripts/train_v397_relative_action_terminal_rescue.py','feature_contract':ROOT/'pipeline/wam_pipeline/v397_relative_action_phase_gate.py','public_success_split':J/'v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json','public_failure_split':J/'onpolicy_windows_full128_stride4/split_manifest.json','phase_labels':J/'v323_public_terminal_phase_gate_seed1493_20260822/terminal_phase_gate.npz','action_gate':J/'v311_public_action_causal_gate_seed1484_20260822/action_causal_gate.npz','v396_report':J/'v396_action_domain_terminal_rescue_seed1556_20260823/training_report.json','v396_domain_rejection':J/'v396_action_domain_terminal_rescue_seed1556_20260823/outcome_free_v211_diagnosis.json'}
    missing=[str(p) for p in sources.values() if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    diagnosis=json.loads(sources['v396_domain_rejection'].read_text())
    if diagnosis['runtime_action_eligible']!=63 or diagnosis['action_phase_ready']!=0:raise RuntimeError('v396 domain evidence drift')
    RUN.mkdir(parents=True);REG.mkdir(parents=True)
    payload={'format':'strict-track2-v397-relative-action-terminal-rescue-preregistration-v1','registered_at':datetime.now(timezone.utc).isoformat(),'purpose':'remove the diagnosed absolute-pose shift from v396; train only on v311-compatible relative action geometry','fixed_training':{'positive_and_early_negative_source':'15 declared public right success demos with frozen onset labels','failure_negative_source':'47 public official-Pi0.5 right failures','independent_test':'7 disjoint public right failures','features':'right-arm future-minus-current relative motion, per-step deltas, path shape, and gripper only','absolute_pose':False,'rgb':False,'c_values':[.001,.01,.1,1.0],'thresholds':[.5,.6,.7,.8,.9,.95,.975,.99,.995,.999],'grouping':'15-fold episode-grouped OOF'},'fixed_selection':{'success_all_early_specificity_min':.98,'success_route_early_specificity_min':.99,'success_route_terminal_recall_min':.50,'failure_all_specificity_min':.99,'failure_route_specificity_min':.96,'tie_break':'maximum route terminal recall, failure-route specificity, early specificity, higher threshold, smaller C'},'post_selection_gate':{'independent_failure_rows':336,'route_rows':3,'all_specificity_min':.99,'route_false_positives':0},'next_authority':'passing training authorizes exactly one outcome-free v211 coverage check; only nonzero robust coverage authorizes integration','evidence_sha256':{k:sha(p) for k,p in sources.items()},'guards':{'public_world_model_data_only':True,'runtime_reads_reward_or_outcomes':False,'policy_modified':False,'official_batch16_or_hidden_data':False,'real_submission':False}}
    text=json.dumps(payload,indent=2)+'\n';(RUN/'preregistration.json').write_text(text);(REG/'preregistration.json').write_text(text);print(json.dumps({'run':str(RUN),'registry':str(REG)},indent=2))
if __name__=='__main__':main()
