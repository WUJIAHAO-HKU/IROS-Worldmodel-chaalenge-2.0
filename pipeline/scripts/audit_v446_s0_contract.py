#!/usr/bin/env python3
"""Audit v446 five-fold/global gates and final all15 checkpoint closure."""

from __future__ import annotations

import argparse, ast, hashlib, json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision
from wam_pipeline.v446_v169_contrastive_residual_unet_runtime import CHECKPOINT_FORMAT, RESIDUAL_CAP, ResidualUNet128FiLM, apply_residual, instruction_tokens


FORMAT="strict-track2-v446-s0-contract-v1"
def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->int:
    parser=argparse.ArgumentParser()
    for name in ("training-report","preregistration","runtime","trainer","v169-runtime","close-gate-runtime","final-checkpoint","output"):
        parser.add_argument(f"--{name}",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    prereg=json.loads(args.preregistration.read_text()); report=json.loads(args.training_report.read_text()); runtime_source=args.runtime.read_text(); trainer_source=args.trainer.read_text()
    folds=report.get("folds",[]); global_row=report.get("global",{}); rgb_neg=global_row.get("rgb_negative",{}); reward_neg=global_row.get("reward_negative",{}); recursive=global_row.get("recursive32",{})
    checkpoint_exists=args.final_checkpoint.is_file(); checkpoint=None; model_checks={"final_checkpoint_exists":checkpoint_exists}
    if checkpoint_exists:
        checkpoint=torch.load(args.final_checkpoint,map_location="cpu",weights_only=False)
        try:
            model=ResidualUNet128FiLM(int(checkpoint["channels"])); model.load_state_dict(checkpoint["model"]); model.eval()
            with torch.inference_mode(): residual=model(torch.full((1,8,3,32,32),.5),torch.full((1,3,32,32),.5),torch.zeros((1,12,14)),torch.tensor([[1.,0.]]))
            base=np.full((8,32,32,3),128,dtype=np.uint8); enabled=apply_residual(base,residual[0].permute(0,2,3,1).numpy()); disabled=apply_residual(base,None)
            model_checks.update({
                "checkpoint_format_scope_step":checkpoint.get("format")==CHECKPOINT_FORMAT and checkpoint.get("step")==50 and checkpoint.get("training_scope")=="all15",
                "checkpoint_prereg_binding":checkpoint.get("preregistration_sha256")==sha256(args.preregistration),
                "checkpoint_report_binding":report.get("sha256",{}).get("final_checkpoint")==sha256(args.final_checkpoint),
                "action_normalization":np.asarray(checkpoint.get("action_mean")).shape==(14,) and np.asarray(checkpoint.get("action_std")).shape==(14,) and bool(np.all(np.asarray(checkpoint.get("action_std"))>0)),
                "finite_cap4":bool(torch.isfinite(residual).all() and residual.abs().max()<=RESIDUAL_CAP+1e-5),
                "g0_apply_bitexact":np.array_equal(disabled,base),
                "enabled_pixel_cap4":int(np.abs(enabled.astype(np.int16)-base.astype(np.int16)).max(initial=0))<=4,
            })
        except Exception as exc:
            model_checks["checkpoint_load_and_synthetic"] = False; model_checks["checkpoint_error"] = str(exc)
    history=np.zeros((4,14),dtype=np.float32); history[:,13]=1.; future=np.zeros((8,14),dtype=np.float32); future[:4,13]=1.; future[4:,13]=.25
    close=gate_decision(history,future,"Use the right arm to lift the bottle."); held=history.copy(); held[:,13]=.25
    tree=ast.parse(runtime_source); imports=[alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]+[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    registered_folds=prereg.get("data",{}).get("fold_holdout_episodes",[]); right=prereg.get("data",{}).get("right_train_episodes",[])
    fold_membership=len(folds)==5 and len(registered_folds)==5 and all(row.get("fold")==index and sorted(row.get("holdout_episodes",[]))==sorted(registered_folds[index]) and sorted(row.get("fit_episodes",[]))==sorted(set(right)-set(registered_folds[index])) for index,row in enumerate(folds)) and sorted(sum((list(row) for row in registered_folds),[]))==sorted(right) and len(set(sum((list(row) for row in registered_folds),[])))==len(right)
    gate_counts=len(folds)==5 and all(row.get("variant_gate_runtime_recomputed_match") is True and row.get("variant_gate_count",{}).get("true")==24 and row.get("variant_gate_count",{}).get("open")==0 and row.get("variant_gate_count",{}).get("static")==0 and all(isinstance(row.get("variant_gate_count",{}).get(name),int) for name in ("shuffle","reverse")) for row in folds)
    all_fold_pass=fold_membership and gate_counts and all(row.get("passed") is True and len(row.get("fit_episodes",[]))==12 and len(row.get("holdout_episodes",[]))==3 and row.get("episode_improved")==3 and float(row.get("true_rgb_ratio",2))<=1 and all(row.get("negative_order",{}).get(name,{}).get("mean") is True and row.get("negative_order",{}).get(name,{}).get("pairwise",0)>=.75 and row.get("negative_order",{}).get(name,{}).get("episode_wins")==3 for name in ("shuffle","open","static","reverse")) for row in folds)
    evidence=prereg.get("evidence_sha256",{})
    checks={
        "preregistration_format":prereg.get("format")=="strict-track2-v446-contrastive-residual-5fold-preregistration-v1",
        "training_report_format":report.get("format")=="strict-track2-v446-5fold-s0-training-report-v1",
        "all_five_folds_pass":all_fold_pass and report.get("fold_pass") is True,
        "fold_membership_partition_exact":fold_membership,
        "variant_gate_counts_runtime_exact":gate_counts,
        "input_closure_validated_before_training":bool(report.get("input_validation")) and all(value is True for value in report.get("input_validation",{}).values()),
        "global_true_rgb_ratio":float(global_row.get("true_rgb_ratio",2))<=.992,
        "global_episode_improved":int(global_row.get("episode_improved",0))>=12,
        "global_rgb_negative":all(rgb_neg.get(name,{}).get("pairwise",0)>=.75 and rgb_neg.get(name,{}).get("margin",-1)>0 for name in ("shuffle","open","static","reverse")),
        "global_reward_negative":all(reward_neg.get(name,{}).get("pairwise",0)>=.75 and reward_neg.get(name,{}).get("normalized_margin",-1)>=.05 for name in ("shuffle","open","static","reverse")),
        "recursive32":float(recursive.get("overall",2))<=1 and float(recursive.get("chunk3",2))<=1 and float(recursive.get("chunk4",2))<=1,
        "trainer_and_global_declared_pass":report.get("global_pass") is True and report.get("passed") is True,
        "all15_only_after_pass":report.get("final_training",{}).get("performed") is True,
        "all_five_completed_guard":report.get("guards",{}).get("all_five_folds_completed") is True,
        "reward_after_frozen_outputs":trainer_source.find("from rlinf.models.embodiment.reward")>trainer_source.find("all_outputs={name:np.stack") and report.get("guards",{}).get("reward_used_after_frozen_outputs") is True,
        "same_seed_all_variants":"del variant  # same-request seed" in trainer_source,
        "five_independent_baselines":"for variant in VARIANTS" in trainer_source and "row.setdefault(\"baseline\",{})[variant]" in trainer_source,
        "oof_uses_runtime_gate":"deployed_gate=gate_decision" in trainer_source and "if deployed_gate else base_np[i].copy()" in trainer_source,
        "runtime_no_reward":not any("reward" in value.lower() for value in imports),
        "runtime_cap4":RESIDUAL_CAP==4.0,
        "close_gate_fixture":close.get("gate") is True,
        "nonclose_repeat_g0":gate_decision(held,future,"Use the right arm to lift the bottle.").get("gate") is False,
        "left_g0":gate_decision(history,future,"Use the left arm to lift the bottle.").get("gate") is False,
        "runtime_hash":evidence.get("runtime")==sha256(args.runtime), "trainer_hash":evidence.get("trainer")==sha256(args.trainer),
        "v169_runtime_closure":evidence.get("v169_runtime")==sha256(args.v169_runtime), "close_gate_closure":evidence.get("close_gate_runtime")==sha256(args.close_gate_runtime),
        "no_dev_or_outcome":report.get("guards",{}).get("development_or_final_used") is False,
        **model_checks,
    }
    passed=all(value is True for key,value in checks.items() if key!="checkpoint_error")
    audit={"format":FORMAT,"passed":passed,"checks":checks,"global":global_row,"sha256":{"training_report":sha256(args.training_report),"preregistration":sha256(args.preregistration),"runtime":sha256(args.runtime),**({"final_checkpoint":sha256(args.final_checkpoint)} if checkpoint_exists else {})},"guards":{"service_started":False,"s1_authorized":False,"rl_authorized":False,"policy_updates":0,"real_submission":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(audit,indent=2)+"\n"); print(json.dumps(audit,indent=2)); return 0 if passed else 2


if __name__=="__main__": raise SystemExit(main())
