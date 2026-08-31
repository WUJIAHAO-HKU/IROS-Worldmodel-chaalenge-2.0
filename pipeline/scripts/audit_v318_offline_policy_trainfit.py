#!/usr/bin/env python3
"""Compare v318 checkpoints on public train40 with the legacy PT loader."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

import openpi.models.model as openpi_model
import openpi.training.data_loader as openpi_data
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.pytree import register_pytree_dataclasses


def parse_checkpoint(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=/absolute/path/full_weights.pt")
    return name, Path(path)


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"need {count} records, got {len(values)}")
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(position)] for position in positions]


def make_overlay(root: Path, official: Path, checkpoint: Path, name: str) -> Path:
    overlay = root / name
    (overlay / "model_state_dict").mkdir(parents=True, exist_ok=False)
    os.symlink(checkpoint, overlay / "model_state_dict" / "full_weights.pt")
    for relative in ("metadata.pt", "rlinf"):
        source = official / relative
        if source.exists():
            os.symlink(source, overlay / relative, target_is_directory=source.is_dir())
    return overlay


def to_device(observation, device: torch.device):
    register_pytree_dataclasses(observation)
    return tree_map(
        lambda value: torch.as_tensor(value, device=device).contiguous()
        if value is not None
        else value,
        observation,
    )


def collate(samples: list[dict]):
    batch = openpi_data._collate_fn(samples)  # noqa: SLF001 - exact official loader collation.
    return openpi_model.Observation.from_dict(tree_map(torch.as_tensor, batch)), torch.as_tensor(batch["actions"])


def route_scores(actions: torch.Tensor, normalized_zero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = actions[:, :8, :14].float()
    left_joint = (chunk[:, :, :6] - normalized_zero[:6]).abs().mean((1, 2))
    right_joint = (chunk[:, :, 7:13] - normalized_zero[7:13]).abs().mean((1, 2))
    left_close = (0.5 * (1.0 - chunk[:, :, 6].clamp(-1.0, 1.0))).mean(1)
    right_close = (0.5 * (1.0 - chunk[:, :, 13].clamp(-1.0, 1.0))).mean(1)
    return left_joint + 0.25 * left_close, right_joint + 0.25 * right_close


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for arm in ("all", "left", "right"):
        subset = rows if arm == "all" else [row for row in rows if row["expected_arm"] == arm]
        result[arm] = {
            "samples": len(subset),
            "route_accuracy": float(np.mean([row["route_correct"] for row in subset])),
            "predicted_right_fraction": float(np.mean([row["predicted_arm"] == "right" for row in subset])),
            "flow_loss_first8_physical14": float(np.mean([row["flow_loss_first8_physical14"] for row in subset])),
            "sample_mse_first8_physical14": float(np.mean([row["sample_mse_first8_physical14"] for row in subset])),
            "active_arm_mse": float(np.mean([row["active_arm_mse"] for row in subset])),
            "inactive_arm_mse": float(np.mean([row["inactive_arm_mse"] for row in subset])),
            "mean_route_margin": float(np.mean([row["route_margin"] for row in subset])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append", type=parse_checkpoint)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--records-per-arm", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OpenPI audit")
    for _, checkpoint in args.checkpoint:
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
            raise FileNotFoundError(checkpoint)

    conversion = json.loads((args.dataset / "conversion_summary.json").read_text())
    records = conversion["records"]
    if conversion["total_episodes"] != 65 or conversion["total_frames"] != 9367:
        raise RuntimeError("unexpected public mirror-balanced dataset cardinality")
    by_arm = defaultdict(list)
    offsets = []
    offset = 0
    for index, record in enumerate(records):
        offsets.append(offset)
        offset += int(record["frames"])
        by_arm[record["arm"]].append(index)
    if offset != 9367:
        raise RuntimeError("conversion frame offsets do not sum to 9367")

    config = get_openpi_config(
        "pi05_aloha_robotwin_head_adjust_bottle",
        model_path=str(args.official),
        batch_size=args.batch_size,
        repo_id=str(args.dataset),
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data.transform_dataset(
        openpi_data.create_torch_dataset(data_config, config.model.action_horizon, config.model),
        data_config,
    )
    selections = []
    for arm in ("left", "right"):
        chosen = evenly_spaced(by_arm[arm], args.records_per_arm)
        for episode_index, fraction in zip(chosen * 3, [0.2] * len(chosen) + [0.55] * len(chosen) + [0.85] * len(chosen)):
            record = records[episode_index]
            frame = min(int(round((int(record["frames"]) - 1) * fraction)), int(record["frames"]) - 1)
            selections.append(
                {
                    "global_index": offsets[episode_index] + frame,
                    "dataset_episode": episode_index,
                    "frame": frame,
                    "phase_fraction": fraction,
                    "expected_arm": arm,
                    "kind": record["kind"],
                    "source_episode": record["source_episode"],
                }
            )
    selections.sort(key=lambda row: (row["expected_arm"], row["dataset_episode"], row["frame"]))
    samples = [dataset[row["global_index"]] for row in selections]

    norm = json.loads(
        (args.official / "rlinf/robotwin_headcam_adjust_bottle/norm_stats.json").read_text()
    )["norm_stats"]["actions"]
    q01 = torch.tensor(norm["q01"], dtype=torch.float32)
    q99 = torch.tensor(norm["q99"], dtype=torch.float32)
    normalized_zero = (2.0 * (torch.zeros(14) - q01) / (q99 - q01) - 1.0).clamp(-1.0, 1.0)

    report = {
        "format": "strict-track2-v269-public-train40-policy-fit-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_cardinality": {"episodes": 65, "frames": 9367},
        "selection": selections,
        "fixed_noise_seed_base": 26900,
        "checkpoints": {},
        "rules": {
            "public_training_data_only": True,
            "evaluation_batches_accessed": [],
            "reserved_final128_access": False,
            "real_competition_submission": False,
        },
    }

    device = torch.device("cuda")
    with tempfile.TemporaryDirectory(prefix="v269_openpi_overlays_") as temporary:
        overlay_root = Path(temporary)
        for model_index, (name, checkpoint) in enumerate(args.checkpoint):
            print(f"V269_LOAD_BEGIN {name}", flush=True)
            overlay = make_overlay(overlay_root, args.official, checkpoint, name)
            model_cfg = OmegaConf.create(
                {
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
                }
            )
            model = get_model(model_cfg).to(device).eval()
            rows = []
            with torch.inference_mode():
                for start in range(0, len(samples), args.batch_size):
                    stop = min(start + args.batch_size, len(samples))
                    observation, target = collate(samples[start:stop])
                    observation = to_device(observation, device)
                    target = target.to(device=device, dtype=torch.float32)
                    generator = torch.Generator(device=device)
                    generator.manual_seed(26900 + start)
                    noise = torch.randn(target.shape, generator=generator, device=device, dtype=torch.float32)
                    time = torch.full((target.shape[0],), 0.5, device=device, dtype=torch.float32)
                    torch.manual_seed(26900 + start)
                    flow_loss = PI0Pytorch.forward(model, observation, target, noise=noise, time=time)
                    torch.manual_seed(26900 + start)
                    sampled = model.sample_actions(
                        observation,
                        noise=noise,
                        mode="eval",
                        compute_values=False,
                    )["actions"].float()
                    left_score, right_score = route_scores(sampled, normalized_zero.to(device))
                    for local_index in range(stop - start):
                        selection = selections[start + local_index]
                        expected = selection["expected_arm"]
                        predicted = "right" if right_score[local_index] > left_score[local_index] else "left"
                        if expected == "left":
                            active, inactive = slice(0, 7), slice(7, 14)
                        else:
                            active, inactive = slice(7, 14), slice(0, 7)
                        difference = sampled[local_index, :8, :14] - target[local_index, :8, :14]
                        rows.append(
                            {
                                **selection,
                                "predicted_arm": predicted,
                                "route_correct": predicted == expected,
                                "route_margin": float((right_score[local_index] - left_score[local_index]).cpu()),
                                "flow_loss_first8_physical14": float(flow_loss[local_index, :8, :14].mean().cpu()),
                                "sample_mse_first8_physical14": float(difference.square().mean().cpu()),
                                "active_arm_mse": float(difference[:, active].square().mean().cpu()),
                                "inactive_arm_mse": float(difference[:, inactive].square().mean().cpu()),
                            }
                        )
                    print(f"V269_BATCH {name} {stop}/{len(samples)}", flush=True)
            report["checkpoints"][name] = {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "metrics": aggregate(rows),
                "rows": rows,
            }
            print(json.dumps({name: report["checkpoints"][name]["metrics"]}, indent=2), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"V269_AUDIT_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
