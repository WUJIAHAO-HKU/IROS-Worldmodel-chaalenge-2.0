#!/usr/bin/env python3
"""Final paired public holdout RGB/temporal/reward gate for v378."""

from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES,sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v378_source_routed_blended_cartesian_runtime import Track2V378SourceRoutedBlendedCartesian


def instantiate(name,args):
    if name=="baseline":
        m=json.loads((args.baseline_checkpoint_dir/"arm_routed_autoregressive_manifest.json").read_text())
        return Track2ArmRoutedAutoregressiveUNet(args.baseline_checkpoint_dir/m["left_expert"],args.baseline_checkpoint_dir/m["right_expert"],args.device)
    return Track2V378SourceRoutedBlendedCartesian(args.candidate_checkpoint_dir,args.library_index,args.device)


def metric(a,m,s,k): return float(a[m][s]["all"][k]["mean"])
def ratio(a,b): return float(a/max(b,1e-9))


def main():
    p=argparse.ArgumentParser()
    for n in ("baseline-checkpoint-dir","candidate-checkpoint-dir","library-index","windows","instruction-map","reward-checkpoint","t5-model","preregistration","output"):p.add_argument(f"--{n}",required=True,type=Path)
    p.add_argument("--device",default="cuda");p.add_argument("--inference-batch-size",type=int,default=8);p.add_argument("--reward-batch-size",type=int,default=32);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    prereg=json.loads(args.preregistration.read_text())
    if prereg.get("format")!="strict-track2-v378-source-routed-blended-cartesian-preregistration-v1":raise RuntimeError("wrong preregistration")
    mapping=json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint),config={"t5_model_name":str(args.t5_model)}).to(args.device).eval().requires_grad_(False)
    original=shared.instantiate;shared.instantiate=instantiate
    try:rows={n:shared.run_model(n,args,mapping["episode_to_instruction"],reward) for n in ("baseline","candidate")}
    finally:shared.instantiate=original
    if [r["key"] for r in rows["baseline"]]!=[r["key"] for r in rows["candidate"]] or len(rows["candidate"])!=512:raise RuntimeError("row mismatch")
    aggregates={n:shared.aggregate(v,set()) for n,v in rows.items()}
    keys=("teacher_next_context_rgb_mae","recursive_next_context_rgb_mae","teacher_temporal_delta_error","recursive_temporal_delta_error","teacher_reward_absolute_error","recursive_reward_absolute_error")
    comparisons={s:{k:ratio(metric(aggregates,"candidate",s,k),metric(aggregates,"baseline",s,k)) for k in keys} for s in RIGHT_EPISODES}
    reward_gain={s:metric(aggregates,"candidate",s,"recursive_terminal_reward")-metric(aggregates,"baseline",s,"recursive_terminal_reward") for s in RIGHT_EPISODES}
    hit_gain={s:float(aggregates["candidate"][s]["all"]["recursive_reward_hit_rate_at_0p9"]-aggregates["baseline"][s]["all"]["recursive_reward_hit_rate_at_0p9"]) for s in RIGHT_EPISODES}
    teacher_exact={s:all(b["teacher_next_context_sha256"]==c["teacher_next_context_sha256"] for b,c in zip(rows["baseline"],rows["candidate"],strict=True) if c["split"]==s) for s in RIGHT_EPISODES}
    checks={"exact_512_rows":len(rows["candidate"])==512}
    for s in RIGHT_EPISODES:
        q="validation" if s=="validation" else "local"
        checks.update({f"{q}_teacher_frames_bit_exact":teacher_exact[s],f"{q}_teacher_rgb_ratio_eq_1":abs(comparisons[s]["teacher_next_context_rgb_mae"]-1)<=1e-12,f"{q}_teacher_temporal_ratio_eq_1":abs(comparisons[s]["teacher_temporal_delta_error"]-1)<=1e-12,f"{q}_teacher_reward_ratio_eq_1":abs(comparisons[s]["teacher_reward_absolute_error"]-1)<=1e-9,f"{q}_recursive_rgb_ratio_le_0p75":comparisons[s]["recursive_next_context_rgb_mae"]<=.75,f"{q}_recursive_temporal_ratio_le_0p90":comparisons[s]["recursive_temporal_delta_error"]<=.90,f"{q}_recursive_reward_error_improves":comparisons[s]["recursive_reward_absolute_error"]<1.0,f"{q}_recursive_reward_mean_nonnegative_gain":reward_gain[s]>=0.0})
    passed=all(checks.values())
    report={"format":"strict-track2-v378-recursive-reward-causal-gate-v1","created_at":datetime.now(timezone.utc).isoformat(),"candidate":prereg["model_version"],"coverage":"512 public right holdout windows in eight recursive chains","teacher_frame_bit_exact":teacher_exact,"aggregates":aggregates,"v378_over_v355":comparisons,"recursive_reward_mean_gain":reward_gain,"recursive_reward_hit_rate_gain":hit_gain,"checks":checks,"passed":passed,"authorizes_service_and_fixed_official_rl":passed,"evidence_sha256":{"preregistration":sha256(args.preregistration),"routing_manifest":sha256(args.candidate_checkpoint_dir/"source_routed_blend_manifest.json"),"gate":sha256(args.candidate_checkpoint_dir/"source_gate.npz"),"library":sha256(args.library_index),"reward_checkpoint":sha256(args.reward_checkpoint)},"guards":{"public_world_model_holdout_only":True,"runtime_reads_reward_or_outcomes":False,"policy_modified":False,"official_reward_modified":False,"hidden_or_final_data":False,"real_submission":False},"rows":rows}
    args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"comparisons":comparisons,"teacher_exact":teacher_exact,"reward_gain":reward_gain,"hit_gain":hit_gain,"checks":checks,"passed":passed},indent=2));return 0 if passed else 3


if __name__=="__main__":raise SystemExit(main())
