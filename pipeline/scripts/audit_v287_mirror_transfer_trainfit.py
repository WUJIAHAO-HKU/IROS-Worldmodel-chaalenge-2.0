#!/usr/bin/env python3
"""Audit inference-time left-to-right mirror transfer on paired public demos."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from audit_v269_offline_policy_trainfit import collate, evenly_spaced, make_overlay, parse_checkpoint, to_device
from rlinf.models.embodiment.openpi import get_model


MIRROR_SIGN = torch.tensor([-1, 1, 1, 1, -1, -1, 1], dtype=torch.float32)


def mirror_physical(actions: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(actions)
    sign = MIRROR_SIGN.to(device=actions.device, dtype=actions.dtype)
    result[..., :7] = actions[..., 7:14] * sign
    result[..., 7:14] = actions[..., :7] * sign
    return result


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for phase in ("all", "0.2", "0.55", "0.85"):
        subset = rows if phase == "all" else [row for row in rows if str(row["phase_fraction"]) == phase]
        direct = np.asarray([row["direct_scaled_mse"] for row in subset], dtype=np.float64)
        transfer = np.asarray([row["mirror_transfer_scaled_mse"] for row in subset], dtype=np.float64)
        result[phase] = {
            "samples": len(subset),
            "direct_scaled_mse": float(direct.mean()),
            "mirror_transfer_scaled_mse": float(transfer.mean()),
            "relative_mse_change": float(transfer.mean() / direct.mean() - 1.0),
            "transfer_better_fraction": float(np.mean(transfer < direct)),
            "direct_right_active_scaled_mse": float(np.mean([row["direct_right_active_scaled_mse"] for row in subset])),
            "transfer_right_active_scaled_mse": float(np.mean([row["transfer_right_active_scaled_mse"] for row in subset])),
            "direct_inactive_left_scaled_mse": float(np.mean([row["direct_inactive_left_scaled_mse"] for row in subset])),
            "transfer_inactive_left_scaled_mse": float(np.mean([row["transfer_inactive_left_scaled_mse"] for row in subset])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append", type=parse_checkpoint)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode-pairs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    conversion = json.loads((args.dataset / "conversion_summary.json").read_text())
    records = conversion["records"]
    if conversion["total_episodes"] != 65 or conversion["total_frames"] != 9367:
        raise RuntimeError("unexpected mirror-balanced dataset cardinality")
    offsets, offset = [], 0
    for record in records:
        offsets.append(offset)
        offset += int(record["frames"])
    real_left = {
        int(record["source_episode"]): index
        for index, record in enumerate(records)
        if record["kind"] == "real" and record["arm"] == "left"
    }
    mirrored = {
        int(record["source_episode"]): index
        for index, record in enumerate(records)
        if record["kind"] == "mirrored_left_to_right"
    }
    sources = sorted(set(real_left) & set(mirrored))
    chosen_sources = evenly_spaced(sources, args.episode_pairs)

    from openpi.training import data_loader as openpi_data
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(
        "pi05_aloha_robotwin_head_adjust_bottle",
        model_path=str(args.official),
        batch_size=args.batch_size,
        num_workers=0,
        repo_id=str(args.dataset),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data.transform_dataset(
        openpi_data.create_torch_dataset(data_config, config.model.action_horizon, config.model), data_config
    )
    selections, left_samples, right_samples = [], [], []
    for source in chosen_sources:
        left_index, right_index = real_left[source], mirrored[source]
        if int(records[left_index]["frames"]) != int(records[right_index]["frames"]):
            raise RuntimeError(f"paired frame mismatch for source {source}")
        frames = int(records[left_index]["frames"])
        for fraction in (0.2, 0.55, 0.85):
            frame = min(int(round((frames - 1) * fraction)), frames - 1)
            selections.append({"source_episode": source, "frame": frame, "phase_fraction": fraction})
            left_samples.append(dataset[offsets[left_index] + frame])
            right_samples.append(dataset[offsets[right_index] + frame])

    norm = json.loads((args.official / "rlinf/robotwin_headcam_adjust_bottle/norm_stats.json").read_text())
    norm_actions = norm["norm_stats"]["actions"]
    action_range = torch.tensor(norm_actions["q99"], dtype=torch.float32) - torch.tensor(
        norm_actions["q01"], dtype=torch.float32
    )
    scale = action_range[:14].clamp_min(1e-6)

    report = {
        "format": "strict-track2-v287-public-paired-mirror-transfer-trainfit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "paired_source_episodes": chosen_sources,
        "selection": selections,
        "mirror_sign": MIRROR_SIGN.tolist(),
        "fixed_noise_seed_base": 28700,
        "checkpoints": {},
        "rules": {
            "public_training_data_only": True,
            "simulator_evaluation_accessed": False,
            "reserved_final128_access": False,
            "real_competition_submission": False,
        },
    }

    device = torch.device("cuda")
    with tempfile.TemporaryDirectory(prefix="v287_openpi_overlays_") as temporary:
        overlay_root = Path(temporary)
        for name, checkpoint in args.checkpoint:
            print(f"V287_LOAD_BEGIN {name}", flush=True)
            overlay = make_overlay(overlay_root, args.official, checkpoint, name)
            model_cfg = OmegaConf.create({
                "model_path": str(overlay),
                "openpi": {
                    "config_name": "pi05_aloha_robotwin_head_adjust_bottle",
                    "num_images_in_input": 1,
                    "noise_level": 0.3,
                    "action_chunk": 8,
                    "num_steps": 5,
                    "train_expert_only": True,
                    "action_env_dim": 14,
                    "noise_method": "flow_sde",
                    "add_value_head": True,
                    "value_after_vlm": False,
                    "value_vlm_mode": "mean_token",
                    "detach_critic_input": True,
                    "use_dsrl": False,
                },
            })
            model = get_model(model_cfg).to(device).eval()
            rows = []
            with torch.inference_mode():
                for start in range(0, len(selections), args.batch_size):
                    stop = min(start + args.batch_size, len(selections))
                    left_obs, _ = collate(left_samples[start:stop])
                    right_obs, right_target = collate(right_samples[start:stop])
                    left_obs = to_device(left_obs, device)
                    right_obs = to_device(right_obs, device)
                    right_target = right_target.to(device=device, dtype=torch.float32)
                    generator = torch.Generator(device=device).manual_seed(28700 + start)
                    noise = torch.randn(right_target.shape, generator=generator, device=device)
                    direct_model = model.sample_actions(right_obs, noise=noise, mode="eval", compute_values=False)["actions"]
                    left_model = model.sample_actions(left_obs, noise=noise, mode="eval", compute_values=False)["actions"]
                    direct = model.output_transform({"actions": direct_model, "state": right_obs.state})["actions"].to(device)
                    left = model.output_transform({"actions": left_model, "state": left_obs.state})["actions"].to(device)
                    target = model.output_transform({"actions": right_target, "state": right_obs.state})["actions"].to(device)
                    transfer = mirror_physical(left)
                    scale_device = scale.to(device=device, dtype=target.dtype)
                    direct_error = (direct[:, :8, :14] - target[:, :8, :14]) / scale_device
                    transfer_error = (transfer[:, :8, :14] - target[:, :8, :14]) / scale_device
                    for local in range(stop - start):
                        selection = selections[start + local]
                        rows.append({
                            **selection,
                            "direct_scaled_mse": float(direct_error[local].square().mean().cpu()),
                            "mirror_transfer_scaled_mse": float(transfer_error[local].square().mean().cpu()),
                            "direct_right_active_scaled_mse": float(direct_error[local, :, 7:14].square().mean().cpu()),
                            "transfer_right_active_scaled_mse": float(transfer_error[local, :, 7:14].square().mean().cpu()),
                            "direct_inactive_left_scaled_mse": float(direct_error[local, :, :7].square().mean().cpu()),
                            "transfer_inactive_left_scaled_mse": float(transfer_error[local, :, :7].square().mean().cpu()),
                        })
                    print(f"V287_BATCH {name} {stop}/{len(selections)}", flush=True)
            report["checkpoints"][name] = {"path": str(checkpoint), "metrics": aggregate(rows), "rows": rows}
            print(json.dumps({name: report["checkpoints"][name]["metrics"]}, indent=2), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"V287_AUDIT_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
