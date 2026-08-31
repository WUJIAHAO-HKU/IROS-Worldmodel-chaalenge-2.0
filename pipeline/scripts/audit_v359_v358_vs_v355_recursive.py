#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import torch
import audit_v335_all_offset_recursive_stability as replay
from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES,sha256
from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
def instantiate(name,args):
    root=args.baseline_checkpoint_dir if name=="baseline" else args.candidate_checkpoint_dir; m=json.loads((root/"arm_routed_autoregressive_manifest.json").read_text()); return Track2ArmRoutedAutoregressiveUNet(root/m["left_expert"],root/m["right_expert"],args.device)
def value(agg,model,split,key): return float(agg[model][split]["all"][key]["mean"])
def ratio(a,b): return float(a/max(b,1e-9))
def main():
    p=argparse.ArgumentParser()
    for n in ("baseline-checkpoint-dir","candidate-checkpoint-dir","windows","instruction-map","reward-checkpoint","t5-model","preregistration","output"): p.add_argument(f"--{n}",required=True,type=Path)
    p.add_argument("--device",default="cuda"); p.add_argument("--inference-batch-size",type=int,default=8); p.add_argument("--reward-batch-size",type=int,default=32); args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if json.loads(args.preregistration.read_text()).get("format")!="strict-track2-v359-v358-v355-recursive-preregistration-v1": raise RuntimeError("wrong preregistration")
    mapping=json.loads(args.instruction_map.read_text()); from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward=RoboTwinT5CrossAttnRewardModel.from_pretrained(str(args.reward_checkpoint),config={"t5_model_name":str(args.t5_model)}).to(args.device).eval().requires_grad_(False)
    original=replay.instantiate; replay.instantiate=instantiate
    try: rows={n:replay.run_model(n,args,mapping["episode_to_instruction"],reward) for n in ("baseline","candidate")}
    finally: replay.instantiate=original
    if len(rows["candidate"])!=512 or [r["key"] for r in rows["baseline"]]!=[r["key"] for r in rows["candidate"]]: raise RuntimeError("rows misaligned")
    agg={n:replay.aggregate(v,set()) for n,v in rows.items()}; keys=("teacher_next_context_rgb_mae","recursive_next_context_rgb_mae","teacher_temporal_delta_error","recursive_temporal_delta_error","recursive_reward_absolute_error")
    comp={s:{k:ratio(value(agg,"candidate",s,k),value(agg,"baseline",s,k)) for k in keys} for s in RIGHT_EPISODES}; checks={"exact_512_rows":True}
    for s in RIGHT_EPISODES:
        x="validation" if s=="validation" else "local"; checks.update({f"{x}_teacher_rgb_ratio_le_1p01":comp[s]["teacher_next_context_rgb_mae"]<=1.01,f"{x}_recursive_rgb_ratio_le_1p01":comp[s]["recursive_next_context_rgb_mae"]<=1.01,f"{x}_teacher_temporal_ratio_le_0p995":comp[s]["teacher_temporal_delta_error"]<=.995,f"{x}_recursive_temporal_ratio_le_0p995":comp[s]["recursive_temporal_delta_error"]<=.995,f"{x}_recursive_reward_ratio_le_1p02":comp[s]["recursive_reward_absolute_error"]<=1.02})
    passed=all(checks.values()); report={"format":"strict-track2-v359-v358-v355-recursive-gate-v1","created_at":datetime.now(timezone.utc).isoformat(),"candidate":"v358 temporal right parent","baseline":"v355 v354 right parent","aggregates":agg,"v358_over_v355":comp,"checks":checks,"passed":passed,"authorizes_service_acceptance_only":passed,"evidence_sha256":{"preregistration":sha256(args.preregistration),"baseline_manifest":sha256(args.baseline_checkpoint_dir/"arm_routed_autoregressive_manifest.json"),"candidate_manifest":sha256(args.candidate_checkpoint_dir/"arm_routed_autoregressive_manifest.json")},"guards":{"public_world_model_holdout_only":True,"outcomes_read":False,"official_batch16_outcomes_read":False,"hidden_or_final_data":False,"real_submission":False},"rows":{"baseline":rows["baseline"],"candidate":rows["candidate"]}}
    args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"v358_over_v355":comp,"checks":checks,"passed":passed},indent=2),flush=True); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
