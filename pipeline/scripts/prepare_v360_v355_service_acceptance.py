#!/usr/bin/env python3
"""Register local-only service acceptance for the v355 parametric parent."""
from __future__ import annotations
import hashlib,json,secrets
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge"); J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"; RUN=J/"v360_v355_service_acceptance_seed1527_20260822"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True); reg=RUN/"release_registration.json"; token=RUN/"bearer_token.txt"
    if reg.exists() or token.exists(): raise FileExistsError("v360 registration exists")
    token.write_text(secrets.token_urlsafe(32)+"\n"); token.chmod(0o600)
    sources={"service":ROOT/"pipeline/wam_pipeline/service.py","backend":ROOT/"pipeline/wam_pipeline/backends.py","acceptance":ROOT/"pipeline/scripts/strict_service_acceptance.py","candidate_manifest":J/"v355_v202_v354_parametric_arm_routed_release/arm_routed_autoregressive_manifest.json","recursive_report":J/"v356_v355_parametric_recursive_seed1524_20260822/recursive_gate_report.json"}
    payload={"format":"strict-track2-v360-v355-service-acceptance-preregistration-v1","created_at":datetime.now(timezone.utc).isoformat(),"candidate":"v355 v202-left/v354-right parametric parent","backend":"arm-routed-autoregressive-unet","model_version":"track2-v355-v202-left-v354-parametric-right","port":18055,"acceptance":"complete official HTTP contract matrix plus latency and native batch8","authorization":"local service acceptance only; no RL or submission","recursive_evidence":"strong RGB improvement; temporal chain-bootstrap 97.5% upper material regression below 1%","guards":{"local_only":True,"real_submission":False,"official_batch16_outcomes":False,"hidden_or_final_data":False},"source_sha256":{n:sha(p) for n,p in sources.items()}}
    reg.write_text(json.dumps(payload,indent=2)+"\n"); print(reg); return 0
if __name__=="__main__": raise SystemExit(main())
