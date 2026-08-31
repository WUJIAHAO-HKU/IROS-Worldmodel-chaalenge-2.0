#!/usr/bin/env python3
"""Paired all-offset recursive diagnosis of v355 versus frozen v209."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import audit_v335_all_offset_recursive_stability as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES,sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet

def instantiate(name,args):
    root=args.baseline_checkpoint_dir if name=="baseline" else args.candidate_checkpoint_dir
    manifest=json.loads((root/"arm_routed_autoregressive_manifest.json").read_text())
    return Track2ArmRoutedAutoregressiveUNet(root/manifest["left_expert"],root/manifest["right_expert"],args.device)
def metric(agg,model,split,key): return float(agg[model][split]["all"][key]["mean"])
def ratio(a,b): return float(a/max(b,1e-9))
def main():
    parser=argparse.ArgumentParser()
    for name in ("baseline-checkpoint-dir","candidate-checkpoint-dir","windows","instruction-map","reward-checkpoint","t5-model","preregistration","output"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--inference-batch-size",type=int,default=8); parser.add_argument("--reward-batch-size",type=int,default=32)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if json.loads(args.preregistration.read_text()).get("format")!="strict-track2-v356-v355-parametric-recursive-preregistration-v1": raise RuntimeError("wrong v356 preregistration")
    mapping=json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint),config={"t5_model_name":str(args.t5_model)}).to(args.device).eval().requires_grad_(False)
    original=shared.instantiate; shared.instantiate=instantiate
    try: rows={name:shared.run_model(name,args,mapping["episode_to_instruction"],reward) for name in ("baseline","candidate")}
    finally: shared.instantiate=original
    if [r["key"] for r in rows["baseline"]]!=[r["key"] for r in rows["candidate"]] or len(rows["candidate"])!=512: raise RuntimeError("recursive rows misaligned")
    aggregates={name:shared.aggregate(value,set()) for name,value in rows.items()}
    metrics=("teacher_next_context_rgb_mae","recursive_next_context_rgb_mae","teacher_temporal_delta_error","recursive_temporal_delta_error","teacher_reward_absolute_error","recursive_reward_absolute_error")
    comparisons={split:{key:ratio(metric(aggregates,"candidate",split,key),metric(aggregates,"baseline",split,key)) for key in metrics} for split in RIGHT_EPISODES}
    checks={"exact_512_rows":len(rows["candidate"])==512}
    for split in RIGHT_EPISODES:
        prefix="validation" if split=="validation" else "local"
        checks.update({f"{prefix}_teacher_rgb_ratio_le_0p99":comparisons[split]["teacher_next_context_rgb_mae"]<=.99,
                       f"{prefix}_recursive_rgb_ratio_le_0p99":comparisons[split]["recursive_next_context_rgb_mae"]<=.99,
                       f"{prefix}_teacher_temporal_ratio_le_1p00":comparisons[split]["teacher_temporal_delta_error"]<=1.0,
                       f"{prefix}_recursive_temporal_ratio_le_1p00":comparisons[split]["recursive_temporal_delta_error"]<=1.0,
                       f"{prefix}_recursive_reward_error_ratio_le_1p02":comparisons[split]["recursive_reward_absolute_error"]<=1.02})
    passed=all(checks.values())
    report={"format":"strict-track2-v356-v355-parametric-recursive-gate-v1","created_at":datetime.now(timezone.utc).isoformat(),
            "candidate":"v355 v202-left/v354-right parametric parent","coverage":"512 public right holdout windows in 8 recursive alignment chains",
            "aggregates":aggregates,"v355_over_v209":comparisons,"checks":checks,"passed":passed,
            "authorizes_service_acceptance_only":passed,
            "evidence_sha256":{"preregistration":sha256(args.preregistration),
                               "baseline_manifest":sha256(args.baseline_checkpoint_dir/"arm_routed_autoregressive_manifest.json"),
                               "candidate_manifest":sha256(args.candidate_checkpoint_dir/"arm_routed_autoregressive_manifest.json")},
            "guards":{"public_world_model_holdout_only":True,"outcomes_read":False,"official_batch16_outcomes_read":False,"hidden_or_final_data":False,"real_submission":False},
            "rows":{"baseline":rows["baseline"],"candidate":rows["candidate"]}}
    args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"v355_over_v209":comparisons,"checks":checks,"passed":passed},indent=2),flush=True)
    return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
