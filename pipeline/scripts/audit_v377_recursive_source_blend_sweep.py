#!/usr/bin/env python3
"""Reward-free grouped calibration/test of OOD threshold and Cartesian blend."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.v337_public_recursive_ood_gate import PublicRecursiveOODGate
from wam_pipeline.v375_bounded_cartesian_phase_runtime import Track2V375BoundedCartesianPhase


def deterministic_seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def rgb_mae(a, b) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def temporal_error(a, b) -> float:
    return float(np.abs(np.diff(a.astype(np.float32), axis=0) - np.diff(b.astype(np.float32), axis=0)).mean())


class BlendedRouter:
    def __init__(self, release, library, gate, threshold, alpha, device):
        self.runtime = Track2V375BoundedCartesianPhase(release, library, device)
        self.gate = PublicRecursiveOODGate(gate)
        self.threshold = float(threshold)
        self.alpha = float(alpha)

    def predict_batch(self, contexts, histories, futures, seeds, instructions):
        baseline = self.runtime.parent.predict_batch(contexts, histories, futures, seeds, instructions)
        output = baseline.copy()
        probabilities = np.asarray([self.gate.probability(x) for x in contexts])
        routes = probabilities >= self.threshold
        for i in np.flatnonzero(routes):
            cartesian = self.runtime._right_prediction(baseline[i], contexts[i], histories[i], futures[i])
            output[i] = np.clip(np.rint(
                baseline[i].astype(np.float32)
                + self.alpha * (cartesian.astype(np.float32) - baseline[i].astype(np.float32))
            ), 0, 255).astype(np.uint8)
        return output, probabilities, routes


def make_states(windows: Path, episodes, instructions):
    states = []
    for episode in episodes:
        available = {int(p.stem.split("_")[1]): p for p in windows.glob(f"episode{episode}_*.npz")}
        for alignment in range(8):
            if alignment not in available:
                continue
            with np.load(available[alignment], allow_pickle=False) as x:
                initial = x["context_frames"].astype(np.uint8)
            states.append({
                "episode": episode, "alignment": alignment, "start": alignment,
                "available": available, "context": initial,
                "instruction": str(instructions[str(episode)]),
            })
    return states


def run_baseline(args, episodes, instructions):
    manifest = json.loads((args.baseline_release / "arm_routed_autoregressive_manifest.json").read_text())
    runtime = Track2ArmRoutedAutoregressiveUNet(
        args.baseline_release / manifest["left_expert"],
        args.baseline_release / manifest["right_expert"], args.device,
    )
    states = make_states(args.windows, episodes, instructions)
    rows = []
    while True:
        active = [s for s in states if s["start"] in s["available"]]
        if not active:
            break
        contexts=[]; histories=[]; futures=[]; seeds=[]; prompts=[]; targets=[]
        for s in active:
            p=s["available"][s["start"]]
            with np.load(p,allow_pickle=False) as x:
                histories.append(x["history_actions"].astype(np.float32)); futures.append(x["future_actions"].astype(np.float32)); targets.append(x["target_frames"].astype(np.uint8)[-5:])
            contexts.append(s["context"]); seeds.append(deterministic_seed(p)); prompts.append(s["instruction"])
        predictions=[]
        for begin in range(0,len(active),args.batch_size):
            end=begin+args.batch_size
            predictions.extend(runtime.predict_batch(np.stack(contexts[begin:end]),np.stack(histories[begin:end]),np.stack(futures[begin:end]),np.asarray(seeds[begin:end]),prompts[begin:end]))
        for s,pred,target in zip(active,predictions,targets,strict=True):
            nxt=pred[-5:]
            rows.append((rgb_mae(nxt,target),temporal_error(nxt,target)))
            s["context"]=nxt.copy(); s["start"]+=8
    del runtime; gc.collect(); torch.cuda.empty_cache()
    return {"rows":len(rows),"rgb_mae":float(np.mean([x[0] for x in rows])),"temporal_error":float(np.mean([x[1] for x in rows]))}


def run_candidate(args, episodes, instructions, config):
    runtime=BlendedRouter(args.v375_release,args.library,args.gate,config["threshold"],config["alpha"],args.device)
    states=make_states(args.windows,episodes,instructions)
    rows=[]; teacher_routes=0; teacher_max=0.0; routed=0; eligible_generated=0
    while True:
        active=[s for s in states if s["start"] in s["available"]]
        if not active: break
        contexts=[]; histories=[]; futures=[]; seeds=[]; prompts=[]; targets=[]; teacher_probs=[]
        for s in active:
            p=s["available"][s["start"]]
            with np.load(p,allow_pickle=False) as x:
                real=x["context_frames"].astype(np.uint8); histories.append(x["history_actions"].astype(np.float32)); futures.append(x["future_actions"].astype(np.float32)); targets.append(x["target_frames"].astype(np.uint8)[-5:])
            contexts.append(s["context"]); seeds.append(deterministic_seed(p)); prompts.append(s["instruction"]); teacher_probs.append(runtime.gate.probability(real))
        teacher_max=max(teacher_max,max(teacher_probs)); teacher_routes+=int(np.sum(np.asarray(teacher_probs)>=runtime.threshold))
        predictions=[]; probabilities=[]; routes=[]
        for begin in range(0,len(active),args.batch_size):
            end=begin+args.batch_size
            out,prob,route=runtime.predict_batch(np.stack(contexts[begin:end]),np.stack(histories[begin:end]),np.stack(futures[begin:end]),np.asarray(seeds[begin:end]),prompts[begin:end])
            predictions.extend(out); probabilities.extend(prob.tolist()); routes.extend(route.tolist())
        for s,pred,target,route in zip(active,predictions,targets,routes,strict=True):
            nxt=pred[-5:]; generated=s["start"]>=8
            rows.append((rgb_mae(nxt,target),temporal_error(nxt,target)))
            if generated: eligible_generated+=1; routed+=int(route)
            s["context"]=nxt.copy(); s["start"]+=8
    del runtime; gc.collect(); torch.cuda.empty_cache()
    return {"rows":len(rows),"rgb_mae":float(np.mean([x[0] for x in rows])),"temporal_error":float(np.mean([x[1] for x in rows])),"teacher_routes":teacher_routes,"teacher_probability_max":teacher_max,"generated_route_rate":float(routed/max(eligible_generated,1)),"generated_routes":routed,"generated_rows":eligible_generated}


def compare(candidate,baseline):
    value=dict(candidate)
    value["rgb_ratio"]=candidate["rgb_mae"]/baseline["rgb_mae"]
    value["temporal_ratio"]=candidate["temporal_error"]/baseline["temporal_error"]
    value["eligible"]=bool(candidate["teacher_routes"]==0 and value["rgb_ratio"]<=.90 and value["temporal_ratio"]<=1.03)
    value["selection_score"]=float(max(value["rgb_ratio"]/.80,value["temporal_ratio"]/1.02))
    return value


def main():
    p=argparse.ArgumentParser()
    for name in ("baseline-release","v375-release","library","gate","windows","instruction-map","preregistration","output"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=8)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    prereg=json.loads(args.preregistration.read_text())
    if prereg.get("format")!="strict-track2-v377-recursive-source-blend-sweep-preregistration-v1": raise RuntimeError("wrong preregistration")
    instructions=json.loads(args.instruction_map.read_text())["episode_to_instruction"]
    calibration_episodes=prereg["calibration_episodes"]
    baseline_cal=run_baseline(args,calibration_episodes,instructions)
    calibration=[]
    for config in prereg["configs"]:
        metrics=compare(run_candidate(args,calibration_episodes,instructions,config),baseline_cal)
        calibration.append({"config":config,"metrics":metrics})
        print("CALIBRATION",json.dumps(calibration[-1]),flush=True)
    eligible=[x for x in calibration if x["metrics"]["eligible"]]
    selected=min(eligible,key=lambda x:(x["metrics"]["selection_score"],x["metrics"]["rgb_ratio"],x["config"]["alpha"],-x["config"]["threshold"])) if eligible else None
    test=None; checks={"eligible_calibration_candidate_found":selected is not None}
    if selected is not None:
        test_episodes=prereg["episode_disjoint_test_episodes"]
        baseline_test=run_baseline(args,test_episodes,instructions)
        test_metrics=compare(run_candidate(args,test_episodes,instructions,selected["config"]),baseline_test)
        test={"baseline":baseline_test,"candidate":test_metrics}
        checks.update({"test_zero_teacher_routes":test_metrics["teacher_routes"]==0,"test_recursive_rgb_ratio_le_0p90":test_metrics["rgb_ratio"]<=.90,"test_recursive_temporal_ratio_le_1p03":test_metrics["temporal_ratio"]<=1.03,"test_generated_route_rate_ge_0p50":test_metrics["generated_route_rate"]>=.50})
    report={"format":"strict-track2-v377-recursive-source-blend-sweep-v1","created_at":datetime.now(timezone.utc).isoformat(),"calibration_episodes":calibration_episodes,"calibration_baseline":baseline_cal,"calibration":calibration,"selected":selected,"test_episodes":prereg["episode_disjoint_test_episodes"],"episode_disjoint_test":test,"checks":checks,"passed":all(checks.values()),"authorizes_frozen_candidate_packaging":all(checks.values()),"guards":{"public_train_only":True,"reward_or_outcomes_used":False,"hidden_or_final_data":False,"real_submission":False}}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("FINAL",json.dumps({"selected":selected,"test":test,"checks":checks,"passed":report["passed"]},indent=2),flush=True)
    return 0 if report["passed"] else 3


if __name__=="__main__": raise SystemExit(main())
