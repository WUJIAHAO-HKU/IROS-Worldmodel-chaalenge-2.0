#!/usr/bin/env python3
"""Pre-register one temporal/high-motion refinement from v354 step 150."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge"); JOINT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
RUN=JOINT/"v357_v354s150_temporal_highmotion_refine_seed1525_20260822"
PARENT=JOINT/"v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
SPLIT=JOINT/"v353_v208_parametric_right_dynamics_pilot_seed1522_20260822/public_right_only_split.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); out=RUN/"release_registration.json"
    if out.exists(): raise FileExistsError(out)
    v354=json.loads((PARENT/"training_manifest.json").read_text())
    if int(v354.get("checkpoint_step",-1))!=150: raise RuntimeError("wrong v354 parent step")
    sources={"trainer":ROOT/"pipeline/scripts/train_autoregressive_unet.py","parent_model":PARENT/"model.pt",
             "parent_manifest":PARENT/"training_manifest.json","right_split":SPLIT,
             "v356_report":JOINT/"v356_v355_parametric_recursive_seed1524_20260822/recursive_gate_report.json"}
    payload={"format":"strict-track2-v357-temporal-highmotion-refine-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),
             "parent_choice":"v354 step150 has the minimum public-validation high-motion rollout MAE among steps 50..300",
             "training":{"steps":200,"batch_size":1,"rollout_horizon":16,"train_rollout_steps":8,"learning_rate":2e-7,
                         "motion_weight":4.0,"motion_threshold":.02,"horizon_loss_power":.75,
                         "temporal_delta_weight":4.0,"temporal_delta_pool":4,"lowfreq_anchor_weight":2.0,
                         "texture_laplacian_weight":.05,"high_motion_oversample_factor":4.0,
                         "selection":"100% public-validation high-motion rollout MAE","validation_interval":40},
             "stop_rule":"retain minimum frozen public-validation high-motion checkpoint; deny if no improvement over v354 step150 0.04984978586435318",
             "guards":{"public_train_and_episode_disjoint_public_validation_only":True,"outcomes_or_rewards_used":False,
                       "no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},
             "source_sha256":{name:sha(path) for name,path in sources.items()}}
    out.write_text(json.dumps(payload,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
