#!/usr/bin/env python3
"""Pre-register a short v208-initialized right-arm parametric dynamics pilot."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822"
SOURCE_SPLIT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True)
    registration=RUN/"release_registration.json"; split_out=RUN/"public_right_only_split.json"
    if registration.exists() or split_out.exists(): raise FileExistsError("v353 preregistration already exists")
    source=json.loads(SOURCE_SPLIT.read_text())
    split={"format":"strict-track2-v353-public-right-only-split-v1","source":str(SOURCE_SPLIT),
           "train_episodes":[e for e in source["train_episodes"] if source["arm_by_episode"][str(e)]=="right"],
           "validation_episodes":[e for e in source["validation_episodes"] if source["arm_by_episode"][str(e)]=="right"],
           "selection":"arm metadata only; no outcomes or rewards","hidden_or_final_evaluation_data":False}
    split_out.write_text(json.dumps(split,indent=2)+"\n")
    sources={"trainer":ROOT/"pipeline/scripts/train_autoregressive_unet.py",
             "model":ROOT/"pipeline/wam_pipeline/autoregressive_unet.py",
             "init_model":ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v208_v205_mixed_right_gripper_contrast_long32_seed1407/selected_right_expert/model.pt",
             "source_split":SOURCE_SPLIT,"right_split":split_out,
             "v352_rejection":ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v352_v350_public_holdout_seed1521_20260822/holdout_recursive_gate_report.json"}
    payload={"format":"strict-track2-v353-parametric-right-dynamics-pilot-preregistration-v1",
             "created_at":datetime.now(timezone.utc).isoformat(),"initialization":"frozen v208 selected right expert",
             "training":{"steps":100,"batch_size":1,"rollout_horizon":16,"train_rollout_steps":8,
                         "learning_rate":2e-7,"motion_weight":3.0,"motion_threshold":.02,
                         "horizon_loss_power":.5,"temporal_delta_weight":2.0,"temporal_delta_pool":4,
                         "lowfreq_anchor_weight":1.0,"texture_laplacian_weight":.1,
                         "high_motion_oversample_factor":2.0,"validation_interval":20},
             "stop_rule":"stop/deny if validation selection lacks a credible improving trend by step 100; no RL authorization",
             "guards":{"public_train_and_episode_disjoint_public_validation_only":True,"outcomes_or_rewards_used":False,
                       "no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},
             "source_sha256":{name:sha(path) for name,path in sources.items()}}
    registration.write_text(json.dumps(payload,indent=2)+"\n"); print(registration); return 0
if __name__=="__main__": raise SystemExit(main())
