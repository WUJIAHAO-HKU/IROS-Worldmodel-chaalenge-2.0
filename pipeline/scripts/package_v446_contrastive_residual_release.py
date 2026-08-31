#!/usr/bin/env python3
"""Package a passed v446 S0 parent diagnostic; never start S1 or RL."""

from __future__ import annotations

import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge"); J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def main()->int:
    parser=argparse.ArgumentParser()
    for name in ("checkpoint","training-report","preregistration","s0-audit"): parser.add_argument(f"--{name}",required=True,type=Path)
    parser.add_argument("--v169-release",type=Path,default=J/"v169_instruction_arm_routed_release"); parser.add_argument("--v169-library",type=Path,default=ROOT/"artifacts"); parser.add_argument("--output",type=Path,default=J/"v446_v169_contrastive_residual_release")
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    prereg=json.loads(args.preregistration.read_text()); report=json.loads(args.training_report.read_text()); audit=json.loads(args.s0_audit.read_text())
    if prereg.get("format")!="strict-track2-v446-contrastive-residual-5fold-preregistration-v1": raise RuntimeError("wrong v446 preregistration")
    if report.get("format")!="strict-track2-v446-5fold-s0-training-report-v1" or report.get("passed") is not True: raise RuntimeError("v446 training/global gates failed")
    if audit.get("format")!="strict-track2-v446-s0-contract-v1" or audit.get("passed") is not True: raise RuntimeError("v446 S0 audit failed")
    if audit.get("sha256",{}).get("final_checkpoint")!=sha256(args.checkpoint): raise RuntimeError("v446 audit/checkpoint mismatch")
    runtime=ROOT/"pipeline/wam_pipeline/v446_v169_contrastive_residual_unet_runtime.py"; v169_runtime=ROOT/"pipeline/wam_pipeline/v169_arm_routed_runtime.py"; gate_runtime=ROOT/"pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py"; v169_manifest=args.v169_release/"v169_arm_routed_manifest.json"
    evidence=prereg.get("evidence_sha256",{})
    for key,path in (("runtime",runtime),("v169_runtime",v169_runtime),("close_gate_runtime",gate_runtime),("v169_manifest",v169_manifest)):
        if not path.is_file() or evidence.get(key)!=sha256(path): raise RuntimeError(f"v446 source closure mismatch: {key}")
    for key,row in prereg.get("inputs",{}).items():
        if key=="windows": continue
        path=Path(row["resolved_path"])
        if not path.is_file() or sha256(path)!=row["sha256"]: raise RuntimeError(f"v446 input closure mismatch: {key}")
    source=prereg["data"]["selected_window_source_manifest"]
    source_body={"format":source["format"],"files":source["files"]}
    canonical=hashlib.sha256(json.dumps(source_body,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    if canonical!=source["canonical_sha256"] or canonical!=prereg["inputs"]["windows"]["selected_source_manifest_sha256"]: raise RuntimeError("v446 selected source manifest closure mismatch")
    windows=Path(prereg["inputs"]["windows"]["resolved_path"])
    if not windows.is_dir() or any(not (windows/name).is_file() or sha256(windows/name)!=digest for name,digest in source["files"].items()): raise RuntimeError("v446 selected window files changed after S0")
    args.output.mkdir(parents=True); os.symlink(args.v169_release.resolve(),args.output/"v169_release",target_is_directory=True); os.symlink(args.v169_library.resolve(),args.output/"v169_library",target_is_directory=True); os.symlink(args.checkpoint.resolve(),args.output/"final_all15_step50.pt"); os.symlink(args.preregistration.resolve(),args.output/"preregistration.json"); os.symlink(args.training_report.resolve(),args.output/"training_report.json"); os.symlink(args.s0_audit.resolve(),args.output/"s0_audit.json")
    manifest={
        "format":"track2-v446-v169-contrastive-residual-unet-release-v1","created_at":datetime.now(timezone.utc).isoformat(),"classification":"S0-passed parent diagnostic; independent one-shot S1 still required",
        "checkpoint":"final_all15_step50.pt","v169_release":"v169_release","v169_library":"v169_library","runtime_class":"wam_pipeline.v446_v169_contrastive_residual_unet_runtime.Track2V446V169ContrastiveResidualUNet","gate_inputs":["history_actions","future_actions","instruction"],
        "formula":{"baseline":"frozen v169","working_resolution":128,"residual_cap":4,"gate":"v442 close-only","fallback":"left/nonexplicit/nonclose bitexact v169"},
        "official_reward_runtime_used":False,
        "sha256":{"checkpoint":sha256(args.checkpoint),"v169_manifest":sha256(v169_manifest),"runtime":sha256(runtime),"v169_runtime":sha256(v169_runtime),"close_gate_runtime":sha256(gate_runtime),"preregistration":sha256(args.preregistration),"training_report":sha256(args.training_report),"s0_audit":sha256(args.s0_audit)},
        "input_closure":prereg["inputs"],
        "guards":{"policy_modified":False,"official_reward_modified":False,"policy_updates":0,"real_submission":False,"s1_required_before_service":True,"s1_authorized":False,"rl_authorized":False},
    }
    path=args.output/"v446_contrastive_residual_manifest.json"; path.write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps({"release":str(args.output),"manifest":str(path)},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
