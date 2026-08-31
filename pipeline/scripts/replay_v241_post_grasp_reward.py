#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from wam_pipeline.rlinf_bridge.track2_client import Track2ServiceClient


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def nchw(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3).reshape(-1, 3, 256, 256).float().div_(255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    files = sorted(args.audit_dir.glob("rollout_*.npz"))
    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
    reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).eval().to(args.device)
    client = Track2ServiceClient(args.url, args.token, args.model_version)
    client.assert_ready()
    rewards, histories_all, actions_all, routes_all = [], [], [], []
    with torch.inference_mode():
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as item:
                contexts = item["context_frames"].copy()
                histories = item["history_actions"].astype(np.float32).copy()
                actions = item["future_actions"].astype(np.float32).copy()
                seeds = item["seeds"].astype(np.int64).copy()
                instructions = [str(x) for x in json.loads(str(item["instructions_json"]))]
            frames = client.predict_batch(contexts, histories, actions, seeds, instructions)
            expanded = [text for text in instructions for _ in range(8)]
            score = reward_model.compute_reward(nchw(frames).to(args.device), expanded)
            rewards.append(score.float().cpu().numpy().reshape(actions.shape[:2]))
            histories_all.append(histories)
            actions_all.append(actions)
            routes_all.append([
                Track2ArmRoutedAutoregressiveUNet.active_arm(h, a, text)
                for h, a, text in zip(histories, actions, instructions)
            ])
            print(json.dumps({"completed": index + 1, "total": len(files)}), flush=True)
    rewards = np.stack(rewards)
    histories = np.stack(histories_all)
    actions = np.stack(actions_all)
    routes = np.asarray(routes_all)
    post = (routes == "right") & (histories[..., -1, 13] < 0.5) & ((actions[..., 13] < 0.5).mean(-1) >= 0.75)
    terminal = rewards[..., -1]
    path = np.linalg.norm(
        np.diff(np.concatenate((histories[..., -1:, 7:13], actions[..., 7:13]), axis=2), axis=2), axis=-1
    ).sum(-1)
    group_stds = []
    for chunk in range(terminal.shape[0]):
        for begin in range(0, terminal.shape[1], args.group_size):
            stop = begin + args.group_size
            if stop <= terminal.shape[1] and post[chunk, begin:stop].all():
                group_stds.append(float(terminal[chunk, begin:stop].std()))
    group_stds = np.asarray(group_stds)
    report = {
        "format": "strict-track2-v241-post-grasp-reward-audit-v1",
        "service_model_version": args.model_version,
        "capture_files": len(files), "post_grasp_queries": int(post.sum()),
        "post_grasp_terminal_reward": stats(terminal[post]),
        "post_grasp_group_terminal_std": stats(group_stds),
        "post_grasp_nonzero_group_fraction": float((group_stds > 0).mean()) if group_stds.size else 0.0,
        "post_grasp_motion_reward_correlation": corr(path[post], terminal[post]),
        "checks": {
            "post_grasp_queries_min": int(post.sum()) >= 64,
            "post_grasp_group_std_mean_min": bool(group_stds.size and group_stds.mean() >= 1e-5),
            "post_grasp_nonzero_groups_min": bool(group_stds.size and (group_stds > 0).mean() >= 0.25),
            "post_grasp_motion_correlation_positive": bool((corr(path[post], terminal[post]) or -1) >= 0.05),
            "service_contract": True,
        },
        "guards": {"public_capture_only": True, "policy_modified": False,
                   "hidden_or_final_data": False, "real_submission": False},
    }
    report["passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
