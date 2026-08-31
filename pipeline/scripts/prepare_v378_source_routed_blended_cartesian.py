#!/usr/bin/env python3
"""Freeze v378 from the untouched v377 train/test selection."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
O=ROOT/"artifacts/strict_track2_official_20260810"
NAME="v378_source_routed_blended_cartesian_seed1541_20260823"
RUN=J/NAME; REGISTRY=O/"run_registry"/NAME


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def main():
    if RUN.exists() or REGISTRY.exists(): raise FileExistsError("refusing overwrite")
    source=J/"v375_bounded_cartesian_phase_pilot_seed1538_20260823/release"
    sweep=J/"v377_recursive_source_blend_seed1540_20260823/audit/source_blend_sweep.json"
    gate=J/"v339_high_specificity_recursive_ood_gate_seed1508_20260822/recursive_ood_gate.npz"
    runtime=ROOT/"pipeline/wam_pipeline/v378_source_routed_blended_cartesian_runtime.py"
    backends=ROOT/"pipeline/wam_pipeline/backends.py"
    library=J/"v214_public_right_knn_action_visual_diagnostic_seed1413/library/public_right_knn.npz"
    for p in (source,sweep,gate,runtime,backends,library):
        if not p.exists(): raise FileNotFoundError(p)
    result=json.loads(sweep.read_text())
    if not result.get("passed"): raise RuntimeError("v377 sweep did not pass")
    config=result["selected"]["config"]
    if config!={"threshold":0.05,"alpha":0.75}: raise RuntimeError(f"unexpected selection {config}")
    RUN.mkdir(parents=True);(RUN/"audit").mkdir();REGISTRY.mkdir(parents=True);release=RUN/"release";release.mkdir()
    for name in ("left_expert","right_expert","cartesian_pose.pt","arm_routed_autoregressive_manifest.json","cartesian_phase_manifest.json"):
        source_path=source/name
        os.symlink(source_path.resolve(),release/name,target_is_directory=source_path.is_dir())
    os.symlink(gate.resolve(),release/"source_gate.npz")
    manifest={"format":"strict-track2-v378-source-routed-blended-cartesian-release-v1","model_version":"track2-v378-v355-real-v375-cartesian-a075-generated","source_gate":"source_gate.npz","source_gate_sha256":sha256(gate),"source_threshold":0.05,"cartesian_alpha":0.75,"default_route":"bit-exact v355","generated_context_route":"0.25*v355 + 0.75*v375 Cartesian phase","gate_inputs":"five request RGB context frames only","reward_or_outcomes_used":False,"hidden_or_final_data":False}
    manifest_path=release/"source_routed_blend_manifest.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    prereg={"format":"strict-track2-v378-source-routed-blended-cartesian-preregistration-v1","registered_at":datetime.now(timezone.utc).isoformat(),"model_version":manifest["model_version"],"release":str(release),"selection":{"source":"v377 public-train episode-disjoint calibration/test","selected":config,"test":result["episode_disjoint_test"],"passed":result["passed"]},"fixed_recursive_gate":{"exact_rows":512,"teacher_frames_and_reward_metrics_bit_exact_v355_both_splits":True,"recursive_rgb_ratio_le_0p75_both_splits":True,"recursive_temporal_ratio_le_0p90_both_splits":True,"recursive_reward_error_ratio_lt_1p00_both_splits":True,"recursive_reward_mean_nonnegative_gain_both_splits":True,"all_checks_required_before_service_or_rl":True},"evidence_sha256":{"sweep":sha256(sweep),"gate":sha256(gate),"runtime":sha256(runtime),"backends":sha256(backends),"library":sha256(library)},"release_sha256":{"arm_manifest":sha256(release/"arm_routed_autoregressive_manifest.json"),"cartesian_manifest":sha256(release/"cartesian_phase_manifest.json"),"source_manifest":sha256(manifest_path),"pose":sha256(release/"cartesian_pose.pt"),"gate":sha256(release/"source_gate.npz")},"guards":{"participant_component":"world-model RGB predictor only","public_world_model_data_only":True,"policy_modified":False,"official_reward_modified":False,"official_rl_algorithm_or_budget_modified":False,"hidden_or_final_data":False,"real_submission":False}}
    text=json.dumps(prereg,indent=2)+"\n";(RUN/"release_registration.json").write_text(text);(REGISTRY/"preregistration.json").write_text(text)
    print(json.dumps({"run":str(RUN),"release":str(release),"model_version":manifest["model_version"]},indent=2))


if __name__=="__main__": main()
