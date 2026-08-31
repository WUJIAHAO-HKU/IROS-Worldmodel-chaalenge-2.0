#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge"); JOINT=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
RUN=JOINT/"v356_v355_parametric_recursive_seed1524_20260822"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); out=RUN/"release_registration.json"
    if out.exists(): raise FileExistsError(out)
    sources={"audit":ROOT/"pipeline/scripts/audit_v356_v355_parametric_recursive.py",
             "baseline_manifest":JOINT/"v209_v202_v208_public_arm_routed_release/arm_routed_autoregressive_manifest.json",
             "candidate_manifest":JOINT/"v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json",
             "v354_registration":JOINT/"v354_v353_parametric_right_dynamics_extension_seed1523_20260822/release_registration.json"}
    payload={"format":"strict-track2-v356-v355-parametric-recursive-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),
             "coverage":"same frozen validation/local public right episodes; all 512 recursive windows",
             "checks":"both splits: teacher+recursive RGB <=0.99, temporal <=1.00, recursive reward error <=1.02; all required",
             "authorizes":"service acceptance only","guards":{"outcomes_or_success_labels_read":False,"no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},
             "source_sha256":{name:sha(path) for name,path in sources.items()}}
    out.write_text(json.dumps(payload,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
