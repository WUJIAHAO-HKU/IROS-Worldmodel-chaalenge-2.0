#!/usr/bin/env python3
"""Package unchanged v202 left expert with the v354 best right expert."""
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
JOINT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"; OFF=ROOT/"artifacts/strict_track2_official_20260810"
TAG="v355_v202_v354_parametric_arm_routed_release"; RELEASE=JOINT/TAG; REGISTRY=OFF/"run_registry"/TAG
LEFT=JOINT/"v202_v201_public_terminal_reward_calibration_seed1402/selected_model"
RIGHT=JOINT/"v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/best"
V354=JOINT/"v354_v353_parametric_right_dynamics_extension_seed1523_20260822"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if RELEASE.exists() or REGISTRY.exists(): raise FileExistsError(TAG)
    manifest=json.loads((RIGHT/"training_manifest.json").read_text())
    if int(manifest.get("best_checkpoint_step",-1))!=300: raise RuntimeError("v354 best checkpoint is not step300")
    RELEASE.mkdir(parents=True); REGISTRY.mkdir(parents=True)
    os.symlink(LEFT.resolve(),RELEASE/"left_expert",target_is_directory=True)
    os.symlink(RIGHT.resolve(),RELEASE/"right_expert",target_is_directory=True)
    os.symlink((V354/"release_registration.json").resolve(),RELEASE/"right_training_registration.json")
    payload={"format":"track2-arm-routed-autoregressive-release-v1","model_version":"track2-v355-v202-left-v354-parametric-right",
             "left_expert":"left_expert","right_expert":"right_expert","selected_v354_step":300,
             "model_sha256":{"left_expert":sha(LEFT/"model.pt"),"right_expert":sha(RIGHT/"model.pt")},
             "routing":{"classification":"fixed world-model expert routing only",
                        "rule":"explicit arm instruction first; otherwise action-half temporal-delta magnitude",
                        "inputs":["instruction","history_actions","future_actions"],"does_not_score_or_modify_actions":True,
                        "prohibited_inputs":["seed","request_id","reward","success","evaluation result"]},
             "right_training_registration":"right_training_registration.json",
             "right_training_registration_sha256":sha(V354/"release_registration.json"),
             "hidden_or_final_evaluation_data":False,"real_submission_performed":False}
    manifest_path=RELEASE/"arm_routed_autoregressive_manifest.json"; manifest_path.write_text(json.dumps(payload,indent=2)+"\n")
    registration={"format":"strict-track2-v355-arm-routed-release-registration-v1","registered_at":datetime.now(timezone.utc).isoformat(),
                  "release":str(RELEASE),"manifest_sha256":sha(manifest_path),"model_version":payload["model_version"],
                  "authorization":"recursive public-WM diagnosis only; no RL or submission",
                  "guards":{"participant_action_selection":False,"real_submission":False,"final_128_used":False}}
    (REGISTRY/"registration.json").write_text(json.dumps(registration,indent=2)+"\n")
    print(json.dumps(registration,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
