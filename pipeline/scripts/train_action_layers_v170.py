#!/usr/bin/env python3
"""Train independent action-to-screen-geometry heads for v17 object layers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from wam_pipeline.canonical_arm_texture_v11 import REGIONS, _observed_beam_mask
from wam_pipeline.contact_structure_v135 import structure_masks
from wam_pipeline.object_geometry_v170 import ActionLayerProjector, mask_landmarks


def decode(value: np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)


def active_arm(scene: dict, episode: int) -> int:
    side = scene[f"episode_{episode}"]["info"]["{a}"]
    return 0 if side == "left" else 1


def extract(dataset: Path, episodes: list[int], scene: dict) -> tuple[np.ndarray, ...]:
    actions = []; targets = []; arms = []; layers = []; episode_ids = []; frame_ids = []
    for episode in episodes:
        with h5py.File(dataset / f"episode{episode}.hdf5", "r") as handle:
            joint = handle["joint_action/vector"][:-1].astype(np.float32)
            rgb = handle["observation/head_camera/rgb"][1:]
            episode_arm = active_arm(scene, episode)
            for index, encoded in enumerate(rgb):
                frame = decode(encoded)
                for arm, side in enumerate(("left", "right")):
                    y0, y1, x0, x1 = REGIONS[side]
                    local = _observed_beam_mask(frame[y0:y1, x0:x1])
                    landmark = mask_landmarks(local, minimum_pixels=80)
                    if landmark is not None:
                        landmark[[0, 2, 4]] += x0; landmark[[1, 3, 5]] += y0
                        actions.append(joint[index, arm * 7:(arm + 1) * 7])
                        targets.append(landmark); arms.append(arm); layers.append(0); episode_ids.append(episode); frame_ids.append(index + 1)
                bottle, black, grey = structure_masks(frame)
                gripper = black | grey
                for layer, mask in ((1, gripper), (2, bottle)):
                    landmark = mask_landmarks(mask, minimum_pixels=30)
                    if landmark is not None:
                        arm = episode_arm
                        actions.append(joint[index, arm * 7:(arm + 1) * 7])
                        targets.append(landmark); arms.append(arm); layers.append(layer); episode_ids.append(episode); frame_ids.append(index + 1)
    return (np.asarray(actions, np.float32), np.asarray(targets, np.float32),
            np.asarray(arms, np.int64), np.asarray(layers, np.int64), np.asarray(episode_ids, np.int64),
            np.asarray(frame_ids, np.int64))


def group_stats(value: np.ndarray, arms: np.ndarray, layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((2, 3, value.shape[1]), np.float32)
    stds = np.ones_like(means)
    for arm in (0, 1):
        for layer in range(3):
            selected = (arms == arm) & (layers == layer)
            if selected.any():
                means[arm, layer] = value[selected].mean(0)
                stds[arm, layer] = np.maximum(value[selected].std(0), 1e-4)
    return means, stds


@torch.inference_mode()
def evaluate(model, tensors, stats, batch_size: int) -> dict:
    action, target, arms, layers = tensors
    action_mean, action_std, target_mean, target_std = stats
    values = []
    for start in range(0, len(action), batch_size):
        selected = slice(start, start + batch_size); a=arms[selected]; l=layers[selected]
        normalized = (action[selected] - action_mean[a, l]) / action_std[a, l]
        prediction = model(normalized, a, l) * target_std[a, l] + target_mean[a, l]
        values.append(prediction)
    error = (torch.cat(values)[:, :6] - target[:, :6]).abs()
    result = {"sample_count": len(target), "landmark_mae_px": float(error.mean()), "groups": {}}
    for arm in (0, 1):
        for layer, name in enumerate(model.layer_names):
            selected = (arms == arm) & (layers == layer)
            result["groups"][f"arm{arm}_{name}"] = {
                "sample_count": int(selected.sum()),
                "landmark_mae_px": float(error[selected].mean()) if selected.any() else None,
            }
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",required=True)
    parser.add_argument("--scene-info",required=True); parser.add_argument("--split",required=True)
    parser.add_argument("--output",required=True); parser.add_argument("--dev-episodes",default="36,47")
    parser.add_argument("--steps",type=int,default=4000); parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--learning-rate",type=float,default=2e-3); parser.add_argument("--hidden",type=int,default=192)
    parser.add_argument("--seed",type=int,default=20260808); parser.add_argument("--device",default="cuda")
    parser.add_argument("--log-every",type=int,default=250); parser.add_argument("--labels-only",action="store_true"); args=parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    split=json.loads(Path(args.split).read_text()); scene=json.loads(Path(args.scene_info).read_text())
    dev_episodes={int(x) for x in args.dev_episodes.split(",") if x}; train_episodes=set(split["train_episodes"])-dev_episodes
    all_values=extract(Path(args.dataset), sorted(train_episodes|dev_episodes), scene)
    np.savez_compressed(output/"geometry_labels.npz", action=all_values[0], target=all_values[1],
                        arm_id=all_values[2], layer_id=all_values[3], episode=all_values[4], frame=all_values[5])
    if args.labels_only:
        print(json.dumps({"label_count": len(all_values[0]), "episodes": len(set(all_values[4].tolist()))}))
        return
    train_mask=np.isin(all_values[4],sorted(train_episodes)); dev_mask=np.isin(all_values[4],sorted(dev_episodes))
    train=tuple(value[train_mask] for value in all_values[:4]); dev=tuple(value[dev_mask] for value in all_values[:4])
    action_mean,action_std=group_stats(train[0],train[2],train[3]); target_mean,target_std=group_stats(train[1],train[2],train[3])
    device=torch.device(args.device); train_t=tuple(torch.from_numpy(x).to(device) for x in train); dev_t=tuple(torch.from_numpy(x).to(device) for x in dev)
    stats=tuple(torch.from_numpy(x).to(device) for x in (action_mean,action_std,target_mean,target_std))
    model=ActionLayerProjector(args.hidden).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    generator=torch.Generator(device=device).manual_seed(args.seed); best=float("inf"); history=[]
    for step in range(1,args.steps+1):
        index=torch.randint(len(train_t[0]),(args.batch_size,),generator=generator,device=device)
        action,target,arms,layers=(x[index] for x in train_t); normalized=(action-stats[0][arms,layers])/stats[1][arms,layers]
        target_n=(target-stats[2][arms,layers])/stats[3][arms,layers]; prediction=model(normalized,arms,layers)
        loss=torch.nn.functional.smooth_l1_loss(prediction[:,:6],target_n[:,:6])+0.1*torch.nn.functional.smooth_l1_loss(prediction[:,6],target_n[:,6])
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if step%args.log_every==0 or step==args.steps:
            model.eval(); metrics=evaluate(model,dev_t,stats,args.batch_size); model.train()
            record={"step":step,"loss":float(loss),"dev":metrics}; history.append(record); print(json.dumps(record),flush=True)
            score=max(v["landmark_mae_px"] for v in metrics["groups"].values() if v["landmark_mae_px"] is not None)
            payload={"format":"track2-action-layer-projector-v17.0","step":step,"hidden":args.hidden,
                     "model":model.state_dict(),"action_mean":action_mean,"action_std":action_std,
                     "target_mean":target_mean,"target_std":target_std,"dev_metrics":metrics}
            torch.save(payload,output/"latest.pt")
            if score<best: best=score; torch.save(payload,output/"best.pt")
    manifest={"format":"track2-action-layer-projector-v17.0-training","train_episodes":sorted(train_episodes),
              "dev_episodes":sorted(dev_episodes),"train_samples":len(train[0]),"dev_samples":len(dev[0]),
              "steps":args.steps,"parameter_count":sum(p.numel() for p in model.parameters()),
              "best_worst_group_landmark_mae_px":best,"history":history}
    temporary=output/f"training_manifest.json.tmp.{os.getpid()}"; temporary.write_text(json.dumps(manifest,indent=2)+"\n"); os.replace(temporary,output/"training_manifest.json")


if __name__ == "__main__": main()
