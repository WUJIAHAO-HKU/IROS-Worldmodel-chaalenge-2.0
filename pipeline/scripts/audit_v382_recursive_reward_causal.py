#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES,sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v382_source_gate_terminal_hold_runtime import Track2V382SourceGateTerminalHold
def instantiate(n,a):
 if n=='baseline':
  m=json.loads((a.baseline_checkpoint_dir/'arm_routed_autoregressive_manifest.json').read_text());return Track2ArmRoutedAutoregressiveUNet(a.baseline_checkpoint_dir/m['left_expert'],a.baseline_checkpoint_dir/m['right_expert'],a.device)
 return Track2V382SourceGateTerminalHold(a.candidate_checkpoint_dir,a.library_index,a.device)
def metric(a,m,s,k):return float(a[m][s]['all'][k]['mean'])
def ratio(a,b):return a/max(b,1e-9)
def main():
 p=argparse.ArgumentParser()
 for n in ('baseline-checkpoint-dir','candidate-checkpoint-dir','library-index','windows','instruction-map','reward-checkpoint','t5-model','preregistration','output'):p.add_argument(f'--{n}',required=True,type=Path)
 p.add_argument('--device',default='cuda');p.add_argument('--inference-batch-size',type=int,default=8);p.add_argument('--reward-batch-size',type=int,default=32);a=p.parse_args()
 pre=json.loads(a.preregistration.read_text());mapping=json.loads(a.instruction_map.read_text());from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel;reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(a.reward_checkpoint),config={'t5_model_name':str(a.t5_model)}).to(a.device).eval().requires_grad_(False);original=shared.instantiate;shared.instantiate=instantiate
 try:rows={n:shared.run_model(n,a,mapping['episode_to_instruction'],reward) for n in ('baseline','candidate')}
 finally:shared.instantiate=original
 if len(rows['candidate'])!=512 or [x['key'] for x in rows['baseline']]!=[x['key'] for x in rows['candidate']]:raise RuntimeError('row mismatch')
 agg={n:shared.aggregate(v,set()) for n,v in rows.items()};keys=('teacher_next_context_rgb_mae','recursive_next_context_rgb_mae','teacher_temporal_delta_error','recursive_temporal_delta_error','teacher_reward_absolute_error','recursive_reward_absolute_error');comp={s:{k:ratio(metric(agg,'candidate',s,k),metric(agg,'baseline',s,k)) for k in keys} for s in RIGHT_EPISODES};gain={s:metric(agg,'candidate',s,'recursive_terminal_reward')-metric(agg,'baseline',s,'recursive_terminal_reward') for s in RIGHT_EPISODES};exact={s:all(b['teacher_next_context_sha256']==c['teacher_next_context_sha256'] for b,c in zip(rows['baseline'],rows['candidate'],strict=True) if c['split']==s) for s in RIGHT_EPISODES};checks={'exact_512':len(rows['candidate'])==512}
 for s in RIGHT_EPISODES:
  q='validation' if s=='validation' else 'local';checks.update({f'{q}_teacher_exact':exact[s],f'{q}_teacher_rgb_eq1':abs(comp[s]['teacher_next_context_rgb_mae']-1)<1e-12,f'{q}_teacher_temporal_eq1':abs(comp[s]['teacher_temporal_delta_error']-1)<1e-12,f'{q}_teacher_reward_eq1':abs(comp[s]['teacher_reward_absolute_error']-1)<1e-9,f'{q}_recursive_rgb_le0p85':comp[s]['recursive_next_context_rgb_mae']<=.85,f'{q}_recursive_temporal_le0p98':comp[s]['recursive_temporal_delta_error']<=.98,f'{q}_recursive_reward_error_improves':comp[s]['recursive_reward_absolute_error']<1,f'{q}_reward_gain_nonnegative':gain[s]>=0})
 passed=all(checks.values());report={'format':'strict-track2-v382-recursive-reward-causal-gate-v1','created_at':datetime.now(timezone.utc).isoformat(),'candidate':pre['model_version'],'aggregates':agg,'v382_over_v355':comp,'reward_gain':gain,'teacher_exact':exact,'checks':checks,'passed':passed,'authorizes_service_and_fixed_official_rl':passed,'guards':{'public_holdout_only':True,'runtime_reads_reward_or_outcomes':False,'policy_modified':False,'hidden_or_final_data':False,'real_submission':False},'rows':rows};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'comparisons':comp,'gain':gain,'exact':exact,'checks':checks,'passed':passed},indent=2));return 0 if passed else 3
if __name__=='__main__':raise SystemExit(main())
