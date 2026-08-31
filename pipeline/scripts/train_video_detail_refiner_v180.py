#!/usr/bin/env python3
"""Train a full eight-frame high-frequency refiner using only supplied episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_logo_mask
from wam_pipeline.contact_structure_v135 import structure_semantic_mask
from wam_pipeline.geometry_visibility_router_v172 import build_candidates
from wam_pipeline.video_detail_refiner_v180 import VideoDetailRefinerV180, parameter_count
from train_geometry_visibility_router_v172 import load_pose, precompute_geometry
from train_contact_occlusion_head_v131 import active_arm


def episode(name: str) -> int:
    match = re.match(r"episode(\d+)_", name)
    if match is None: raise ValueError(name)
    return int(match.group(1))


def load_actions(windows: Path, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions, arms = [], []
    for name in names:
        with np.load(windows / name, allow_pickle=False) as value:
            arm, action = active_arm(value["history_actions"], value["future_actions"])
        actions.append(action); arms.append(arm)
    return np.stack(actions), np.asarray(arms, np.int64)


def action_stats(actions: np.ndarray, arms: np.ndarray, indices: list[int]):
    mean = np.stack([actions[indices][arms[indices] == arm].mean((0, 1)) for arm in (0, 1)])
    std = np.stack([actions[indices][arms[indices] == arm].std((0, 1)) for arm in (0, 1)])
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def visual_features(index: int, parent: np.ndarray, context: np.ndarray,
                    source_geometry: np.ndarray, future_geometry: np.ndarray,
                    arms: np.ndarray) -> np.ndarray:
    frames = []; previous = context[index, -1].astype(np.float32)
    for time in range(8):
        candidates, support, _ = build_candidates(
            context[index], parent[index, time], source_geometry[index],
            future_geometry[index, time], int(arms[index]))
        current = parent[index, time].astype(np.float32)
        moved, static = candidates[4].astype(np.float32), candidates[9].astype(np.float32)
        moved_support = support[4, :4].transpose(1, 2, 0).astype(np.float32) * 255
        static_bottle = support[9, 3:4].transpose(1, 2, 0).astype(np.float32) * 255
        cleanup = support[10, 4:5].transpose(1, 2, 0).astype(np.float32) * 255
        gray = cv2.cvtColor(moved.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.2))[..., None] * 4
        frame = np.concatenate((current, context[index, -1].astype(np.float32), moved, static,
                                current - previous + 127.5, moved_support, static_bottle,
                                cleanup, detail), -1)
        frames.append(np.clip(frame, 0, 255)); previous = current
    return np.stack(frames).astype(np.float32) / 255


def target_weights(target: np.ndarray, arm: int) -> np.ndarray:
    weight = np.ones(target.shape[:3], np.float32)
    label = structure_semantic_mask(target)
    weight += 3 * np.isin(label, (2, 3)).astype(np.float32)
    side = "left" if arm == 0 else "right"; y0, y1, x0, x1 = REGIONS[side]
    for time, frame in enumerate(target):
        logo = _observed_logo_mask(frame[y0:y1, x0:x1])
        if logo.any(): weight[time, y0:y1, x0:x1] += 5 * cv2.dilate(logo, np.ones((5, 5), np.uint8))
    return weight


def batch_tensors(indices: list[int], parent, target, context, source_geometry,
                  future_geometry, arms, actions, mean, std, device):
    visual = np.stack([visual_features(i, parent, context, source_geometry, future_geometry, arms)
                       for i in indices])
    weights = np.stack([target_weights(target[i], int(arms[i])) for i in indices])
    visual = torch.from_numpy(visual.transpose(0, 4, 1, 2, 3).copy()).to(device)
    truth = torch.from_numpy(target[indices].transpose(0, 4, 1, 2, 3).copy()).to(device).float() / 255
    weights = torch.from_numpy(weights[:, None].transpose(0, 1, 2, 3, 4)).to(device)
    normalized = np.stack([(actions[i] - mean[arms[i]]) / std[arms[i]] for i in indices])
    action = torch.from_numpy(normalized).to(device).float()
    arm = torch.from_numpy(arms[indices]).to(device).long()
    return visual, truth, weights, action, arm


@torch.inference_mode()
def evaluate(model, indices, parent, target, context, source_geometry, future_geometry,
             arms, actions, mean, std, device, maximum=8):
    model.eval(); chosen = indices
    if maximum and len(chosen) > maximum:
        chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, maximum, dtype=int)]
    ps = np.zeros(8); rs = np.zeros(8); count = 0; arm_sum = np.zeros((2, 2)); arm_count = np.zeros(2)
    for index in chosen:
        visual, truth, _, action, arm = batch_tensors([index], parent, target, context,
            source_geometry, future_geometry, arms, actions, mean, std, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output, alpha = model(visual, action, arm)
        p = torch.from_numpy(parent[index].transpose(0, 3, 1, 2).copy()).to(device).float().div(255).permute(1, 0, 2, 3)
        pe = (p - truth[0]).abs().mean((0, 2, 3)); re = (output[0] - truth[0]).abs().mean((0, 2, 3))
        ps += pe.cpu().numpy(); rs += re.cpu().numpy(); count += 1
        a = int(arms[index]); arm_sum[a] += (float(pe.mean()), float(re.mean())); arm_count[a] += 1
    parent_mae = float(ps.mean() / max(count, 1) * 255); routed_mae = float(rs.mean() / max(count, 1) * 255)
    return {"windows": len(chosen), "parent_rgb_mae": parent_mae, "refined_rgb_mae": routed_mae,
            "improvement_percent": 100 * (parent_mae - routed_mae) / parent_mae,
            "frame_parent_mae": (ps / max(count, 1) * 255).tolist(),
            "frame_refined_mae": (rs / max(count, 1) * 255).tolist(),
            "arms": {f"arm{a}": {"parent_rgb_mae": arm_sum[a, 0] / max(arm_count[a], 1) * 255,
                                   "refined_rgb_mae": arm_sum[a, 1] / max(arm_count[a], 1) * 255}
                     for a in (0, 1)}}


def atomic_save(value, path):
    temporary = path.with_suffix(f".tmp.{os.getpid()}"); torch.save(value, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True); parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--pose-checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--dev-episodes", default="36,47"); parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--base-channels", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=100); parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260808); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); cv2.setNumThreads(1)
    device = torch.device(args.device); windows = Path(args.windows)
    with np.load(args.parent_cache, allow_pickle=False) as cache:
        parent=cache["prediction"]; target=cache["target"]; context=cache["context"]; names=cache["windows"].astype(str)
    pose=load_pose(args.pose_checkpoint,device); source_geometry,future_geometry,arms=precompute_geometry(windows,names,pose,device)
    actions, action_arms=load_actions(windows,names); assert np.array_equal(arms,action_arms)
    dev_eps={int(v) for v in args.dev_episodes.split(",")}; train=[i for i,n in enumerate(names) if episode(n) not in dev_eps]
    dev=[i for i,n in enumerate(names) if episode(n) in dev_eps]; mean,std=action_stats(actions,arms,train)
    by_arm={a:[i for i in train if arms[i]==a] for a in (0,1)}; rng=np.random.default_rng(args.seed)
    model=VideoDetailRefinerV180(args.base_channels).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    best=-1e9; history=[]
    for step in range(1,args.steps+1):
        index=int(rng.choice(by_arm[step%2])); visual,truth,weights,action,arm=batch_tensors([index],parent,target,context,source_geometry,future_geometry,arms,actions,mean,std,device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            prediction,alpha=model(visual,action,arm)
            pixel=((prediction-truth).abs()*weights).sum()/weights.sum()/3
            hp=lambda x:x-F.avg_pool3d(x,(1,5,5),1,(0,2,2))
            high=(hp(prediction)-hp(truth)).abs().mean()
            temporal=((prediction[:,:,1:]-prediction[:,:,:-1])-(truth[:,:,1:]-truth[:,:,:-1])).abs().mean()
            parent_rgb=visual[:,:3]; accurate=(parent_rgb-truth).abs().mean(1,keepdim=True)<(2/255)
            protect=(prediction-parent_rgb).abs()[accurate.expand_as(prediction)].mean() if accurate.any() else prediction.new_zeros(())
            loss=pixel+.35*high+.20*temporal+.15*protect
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        scaler.step(optimizer); scaler.update()
        if step==1 or step%20==0: print(json.dumps({"step":step,"loss":float(loss),"pixel":float(pixel),"high":float(high),"temporal":float(temporal),"protect":float(protect),"alpha":float(alpha.mean())}),flush=True)
        if step%args.validation_interval==0 or step==args.steps:
            metrics=evaluate(model,dev,parent,target,context,source_geometry,future_geometry,arms,actions,mean,std,device,args.validation_windows)
            history.append({"step":step,"metrics":metrics}); print(json.dumps(history[-1]),flush=True)
            arm_gain=min(100*(v["parent_rgb_mae"]-v["refined_rgb_mae"])/v["parent_rgb_mae"] for v in metrics["arms"].values())
            payload={"format":"track2-video-detail-refiner-v18.0","step":step,"state_dict":model.state_dict(),"base_channels":args.base_channels,"action_mean":mean,"action_std":std,"metrics":metrics}
            atomic_save(payload,output/"latest.pt")
            if arm_gain>best: best=arm_gain; atomic_save(payload,output/"best.pt")
    manifest={"format":"track2-video-detail-refiner-v18.0-training","data_boundary":"supplied_50_episodes_only","train_windows":len(train),"dev_windows":len(dev),"steps":args.steps,"parameter_count":parameter_count(args.base_channels),"best_worst_arm_improvement_percent":best,"history":history}
    (output/"training_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main()
