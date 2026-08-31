#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge"); J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"; RUN=J/"v359_v358_vs_v355_recursive_seed1526_20260822"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); out=RUN/"release_registration.json"
    if out.exists(): raise FileExistsError(out)
    sources={"audit":ROOT/"pipeline/scripts/audit_v359_v358_vs_v355_recursive.py","baseline":J/"v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json","candidate":J/"v358_v202_v357_temporal_arm_routed_release/arm_routed_autoregressive_manifest.json","v357":J/"v357_v354s150_temporal_highmotion_refine_seed1525_20260822/release_registration.json"}
    payload={"format":"strict-track2-v359-v358-v355-recursive-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),"coverage":"512 public right recursive windows","checks":"each split temporal ratios<=0.995; RGB<=1.01; recursive reward error<=1.02; all required","authorizes":"service acceptance only","guards":{"outcomes_or_success_labels_read":False,"no_official_batch16_outcomes":True,"no_hidden_or_final_data":True,"no_real_submission":True},"source_sha256":{n:sha(p) for n,p in sources.items()}}
    out.write_text(json.dumps(payload,indent=2)+"\n"); print(out); return 0
if __name__=="__main__": raise SystemExit(main())
