#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v354_v353_parametric_right_dynamics_extension_seed1523_20260822"
PARENT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822/model/best"
SPLIT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v353_v208_parametric_right_dynamics_pilot_seed1522_20260822/public_right_only_split.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); out=RUN/"release_registration.json"
    if out.exists(): raise FileExistsError(out)
    manifest=json.loads((PARENT/"training_manifest.json").read_text()); history=manifest["validation"]
    selections=[float(x["selection_metric"]) for x in history]
    if len(selections)<5 or selections[-1]>=selections[0]: raise RuntimeError("v353 lacks authorized improving trend")
    sources={"trainer":ROOT/"pipeline/scripts/train_autoregressive_unet.py","parent_model":PARENT/"model.pt",
             "parent_manifest":PARENT/"training_manifest.json","right_split":SPLIT}
    payload={"format":"strict-track2-v354-parametric-right-dynamics-extension-preregistration-v1",
             "created_at":datetime.now(timezone.utc).isoformat(),"v353_selection_series":selections,
             "authorization_reason":"monotone 20..100 validation improvement",
             "training":{"steps":300,"batch_size":1,"rollout_horizon":16,"train_rollout_steps":8,"learning_rate":5e-7,
                         "loss":"same v353 motion/temporal/low-frequency/texture objective","validation_interval":50},
             "selection":"minimum frozen public right validation selection metric; retain best checkpoint only",
             "guards":{"public_train_and_episode_disjoint_public_validation_only":True,"outcomes_or_rewards_used":False,
                       "no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},
             "source_sha256":{name:sha(path) for name,path in sources.items()}}
    out.write_text(json.dumps(payload,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
