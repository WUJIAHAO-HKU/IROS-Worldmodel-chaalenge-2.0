#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
RUN=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v352_v350_public_holdout_seed1521_20260822"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); output=RUN/"release_registration.json"
    if output.exists(): raise FileExistsError(output)
    fit=ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v350_public_residual_bridge_fit_seed1519_20260822"
    sources={"runtime":ROOT/"pipeline/wam_pipeline/v350_learned_residual_bridge_runtime.py",
             "audit":ROOT/"pipeline/scripts/audit_v352_v350_public_holdout_recursive.py",
             "profile":fit/"residual_bridge_profile.npz","fit_report":fit/"fit_report.json",
             "v351_train":ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v351_bridge_focused_train_seed1520_20260822/focused_train_gate_report.json",
             "v335_baseline":ROOT/"artifacts/strict_track2_joint_augmentation_20260810/v335_v334_all_offset_recursive_gate_seed1505_20260822/audit/recursive_stability_report.json"}
    payload={"format":"strict-track2-v352-v350-public-holdout-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),
             "candidate":"frozen v350 residual bridge; no post-fit adjustment","coverage":"all 512 public WM holdout windows",
             "checks":"same frozen holdout thresholds; all required","authorizes":"service acceptance only",
             "guards":{"public_world_model_holdout_only":True,"no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},
             "source_sha256":{name:sha(path) for name,path in sources.items()}}
    output.write_text(json.dumps(payload,indent=2)+"\n"); print(output); return 0
if __name__=="__main__": raise SystemExit(main())
