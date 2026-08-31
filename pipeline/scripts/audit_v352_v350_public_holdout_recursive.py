#!/usr/bin/env python3
"""Final public-WM holdout for the frozen v350 residual bridge."""
from __future__ import annotations
import argparse, json, os
from datetime import datetime, timezone
from pathlib import Path
import torch
import audit_v343_v342_public_holdout_recursive as shared
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, sha256
from wam_pipeline.v350_learned_residual_bridge_runtime import Track2V350LearnedResidualBridge

def ratio(a,b): return float(a/max(b,1e-9))
def main():
    parser=argparse.ArgumentParser()
    for name in ("checkpoint-dir","library-index","action-gate","phase-gate","recursive-ood-gate",
                 "bridge-profile","windows","instruction-map","reward-checkpoint","t5-model",
                 "v335-baseline-report","v335-causal-report","v351-train-report","preregistration","output"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--inference-batch-size",type=int,default=8)
    parser.add_argument("--reward-batch-size",type=int,default=32); parser.add_argument("--feature-workers",type=int,default=8)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if json.loads(args.preregistration.read_text()).get("format")!="strict-track2-v352-v350-public-holdout-preregistration-v1": raise RuntimeError("wrong v352 preregistration")
    if json.loads(args.v351_train_report.read_text()).get("passed") is not True: raise RuntimeError("v351 train gate failed")
    if json.loads(args.v335_causal_report.read_text()).get("passed") is not True: raise RuntimeError("v334 causal gate failed")
    baseline=json.loads(args.v335_baseline_report.read_text())["rows"]["v335"]
    mapping=json.loads(args.instruction_map.read_text())
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint),config={"t5_model_name":str(args.t5_model)}).to(args.device).eval().requires_grad_(False)
    os.environ["WAM_V350_BRIDGE_PROFILE"]=str(args.bridge_profile)
    original=shared.Track2V342TemporalBlendedPublicReanchor
    shared.Track2V342TemporalBlendedPublicReanchor=Track2V350LearnedResidualBridge
    try: candidate=shared.run_candidate(args,mapping["episode_to_instruction"],reward)
    finally: shared.Track2V342TemporalBlendedPublicReanchor=original
    if [r["key"] for r in baseline]!=[r["key"] for r in candidate]: raise RuntimeError("holdout rows misaligned")
    direct={r["key"] for r in candidate if r["recursive_reanchor"]}
    aggregates={"v334":shared.aggregate(baseline,direct),"v350":shared.aggregate(candidate,direct)}
    comparisons={}
    for split in RIGHT_EPISODES:
        comparisons[split]={group:{metric:ratio(aggregates["v350"][split][group][metric]["mean"],aggregates["v334"][split][group][metric]["mean"])
                                    for metric in ("recursive_next_context_rgb_mae","recursive_temporal_delta_error","recursive_reward_absolute_error")}
                            for group in ("all","direct")}
    counts={split:sum(r["recursive_reanchor"] and r["split"]==split for r in candidate) for split in RIGHT_EPISODES}
    teacher_rgb=max(abs(a["teacher_next_context_rgb_mae"]-b["teacher_next_context_rgb_mae"]) for a,b in zip(candidate,baseline,strict=True))
    teacher_temp=max(abs(a["teacher_temporal_delta_error"]-b["teacher_temporal_delta_error"]) for a,b in zip(candidate,baseline,strict=True))
    checks={"v351_focused_train_gate":True,"v334_teacher_causal_gate":True,"exact_512_rows":len(candidate)==512,
            "teacher_reanchor_count_zero":not any(r["teacher_reanchor"] for r in candidate),
            "teacher_rgb_metric_max_delta_le_0p01":teacher_rgb<=.01,"teacher_temporal_metric_max_delta_le_0p01":teacher_temp<=.01,
            "validation_direct_reanchors_ge_10":counts["validation"]>=10,"local_direct_reanchors_ge_10":counts["local_test"]>=10}
    for split in RIGHT_EPISODES:
        prefix="validation" if split=="validation" else "local"
        checks.update({f"{prefix}_direct_rgb_ratio_le_0p90":comparisons[split]["direct"]["recursive_next_context_rgb_mae"]<=.90,
                       f"{prefix}_direct_temporal_ratio_le_0p90":comparisons[split]["direct"]["recursive_temporal_delta_error"]<=.90,
                       f"{prefix}_direct_reward_error_ratio_le_1p10":comparisons[split]["direct"]["recursive_reward_absolute_error"]<=1.10,
                       f"{prefix}_direct_reward_mean_ge_0p90":aggregates["v350"][split]["direct"]["recursive_terminal_reward"]["mean"]>=.90,
                       f"{prefix}_direct_reward_hit_rate_ge_0p85":aggregates["v350"][split]["direct"]["recursive_reward_hit_rate_at_0p9"]>=.85,
                       f"{prefix}_all_rgb_ratio_le_1p02":comparisons[split]["all"]["recursive_next_context_rgb_mae"]<=1.02,
                       f"{prefix}_all_temporal_ratio_le_1p02":comparisons[split]["all"]["recursive_temporal_delta_error"]<=1.02,
                       f"{prefix}_all_reward_error_ratio_le_1p02":comparisons[split]["all"]["recursive_reward_absolute_error"]<=1.02})
    passed=all(checks.values())
    report={"format":"strict-track2-v352-v350-public-holdout-gate-v1","created_at":datetime.now(timezone.utc).isoformat(),
            "candidate":"track2-v350-learned-residual-bridge-v339-v334","direct_reanchor_counts":counts,
            "teacher_metric_max_delta":{"rgb":teacher_rgb,"temporal":teacher_temp},"aggregates":aggregates,"v350_over_v334":comparisons,
            "checks":checks,"passed":passed,"authorizes_service_acceptance_only":passed,
            "evidence_sha256":{"preregistration":sha256(args.preregistration),"v351_train_report":sha256(args.v351_train_report),
                               "bridge_profile":sha256(args.bridge_profile),"v335_baseline_report":sha256(args.v335_baseline_report)},
            "guards":{"public_world_model_holdout_only":True,"official_batch16_outcomes_read":False,"hidden_or_final_data":False,"real_submission":False},
            "rows":{"v350":candidate}}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"direct_reanchor_counts":counts,"teacher_metric_max_delta":report["teacher_metric_max_delta"],
                      "v350_over_v334":comparisons,"checks":checks,"passed":passed},indent=2),flush=True)
    return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
