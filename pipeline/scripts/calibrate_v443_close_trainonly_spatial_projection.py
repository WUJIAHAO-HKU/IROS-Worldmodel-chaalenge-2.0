#!/usr/bin/env python3
"""Calibrate v443 spatial beta map on the 15 public right-train episodes only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_v423_mirror_augmented_autoregressive_unet import WindowDataset
from wam_pipeline.autoregressive_unet_runtime import Track2AutoregressiveUNet
from wam_pipeline.v169_arm_routed_runtime import Track2V169ArmRoutedRuntime
from wam_pipeline.v442_v169_close_aligned_projection_runtime import clipped_rgb_delta, gate_decision


SAMPLES_PER_EPISODE = 1
EXPECTED_CLOSE_WINDOWS_PER_EPISODE = 8
TARGET_FIRST_CLOSE_INDEX = 4
LOEO_SIGN_AGREEMENT_MIN = 0.80
LOEO_IMPROVEMENT_FRACTION_MIN = 0.80
COEFFICIENT_ABS_MAX = 1.0
MIN_ACTIVE_PIXELS = 64
PROTECTED_FRAMES = (0, 1, 6, 7)
INFERENCE_BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(episode: int, start: int) -> int:
    payload = f"v443-trainonly/episode{episode}/start{start}/seed1586"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**31)


def select_samples(dataset: WindowDataset, episodes: list[int], prompts: dict[int, str]) -> list[dict]:
    grouped = {episode: [] for episode in episodes}
    for index, path in enumerate(dataset.paths):
        episode = int(path.name.split("_")[0][7:])
        if episode not in grouped:
            continue
        history, future = dataset.load_actions(index)
        decision = gate_decision(history, future, prompts[episode])
        if decision.get("gate") and decision.get("phase") == "close":
            start = int(path.stem.split("_")[1])
            grouped[episode].append((abs(int(decision["first_close_index"]) - TARGET_FIRST_CLOSE_INDEX), start, index))
    rows = []
    for episode in episodes:
        candidates = sorted(grouped[episode])
        if len(candidates) != EXPECTED_CLOSE_WINDOWS_PER_EPISODE:
            raise RuntimeError(f"v443 requires exact eight close windows in train episode {episode}, got {len(candidates)}")
        _, start, index = candidates[0]
        arrays = dataset.load_arrays(index)
        decision = gate_decision(arrays["history_actions"], arrays["future_actions"], prompts[episode])
        rows.append({
            "episode": episode, "phase": "close", "start": start, "dataset_index": index,
            "first_close_index": int(decision["first_close_index"]), "prompt": prompts[episode], **arrays,
        })
    if len(rows) != 15:
        raise RuntimeError(f"v443 expected 15 calibration rows, got {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("windows", "split", "v169-release", "v169-library", "v436-release", "preregistration", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix != ".npz":
        raise ValueError("v443 alignment index must use .npz")
    if args.inference_batch_size != INFERENCE_BATCH_SIZE:
        raise SystemExit("v443 calibration is preregistered for inference batch size 4")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v443-close-trainonly-preregistration-v1":
        raise RuntimeError("wrong v443 preregistration")
    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    right_train = sorted(episode for episode in train if arms[episode] == "right")
    if len(train) != 40 or len(validation) != 10 or len(right_train) != 15 or set(train) & set(validation):
        raise RuntimeError("v443 requires immutable public train40/right15")
    if any("right arm" not in prompts[e].lower() or "left arm" in prompts[e].lower() for e in right_train):
        raise RuntimeError("v443 right-train prompt contract failed")

    rows = select_samples(WindowDataset(args.windows, right_train, rollout_horizon=8), right_train, prompts)
    v169 = Track2V169ArmRoutedRuntime(args.v169_release, args.v169_library, args.device)
    v436 = json.loads((args.v436_release / "v436_diagnostic_manifest.json").read_text())
    if v436.get("format") != "track2-v436-v432-step25-diagnostic-release-v1":
        raise RuntimeError("wrong v443 teacher release")
    parent = Track2AutoregressiveUNet(args.v436_release / v436["parent_right"], args.device)
    candidate = Track2AutoregressiveUNet(args.v436_release / v436["candidate_right"], args.device)

    numerator = denominator = residual_square = None
    for begin in range(0, len(rows), INFERENCE_BATCH_SIZE):
        batch = rows[begin:begin + INFERENCE_BATCH_SIZE]
        context = np.stack([row["context_frames"] for row in batch])
        history = np.stack([row["history_actions"] for row in batch])
        future = np.stack([row["future_actions"] for row in batch])
        target = np.stack([row["target_frames"] for row in batch]).astype(np.float64)
        seeds = np.asarray([stable_seed(row["episode"], row["start"]) for row in batch], dtype=np.int64)
        texts = [row["prompt"] for row in batch]
        baseline = v169.predict_batch(context, history, future, seeds, texts).astype(np.float64)
        parent_rgb = parent.predict_batch(context, history, future, seeds, texts)
        candidate_rgb = candidate.predict_batch(context, history, future, seeds, texts)
        teacher = clipped_rgb_delta(candidate_rgb, parent_rgb).astype(np.float64)
        correction = target - baseline
        if numerator is None:
            shape = (len(right_train),) + teacher.shape[1:]
            numerator = np.zeros(shape, dtype=np.float64)
            denominator = np.zeros(shape, dtype=np.float64)
            residual_square = np.zeros(shape, dtype=np.float64)
        for local, row in enumerate(batch):
            e = right_train.index(row["episode"])
            numerator[e] = teacher[local] * correction[local]
            denominator[e] = teacher[local] ** 2
            residual_square[e] = correction[local] ** 2

    total_num = numerator.sum(axis=0)
    total_den = denominator.sum(axis=0)
    fold_beta = []
    fold_improved = []
    for e in range(len(right_train)):
        slope = np.divide(total_num - numerator[e], total_den - denominator[e], out=np.zeros_like(total_num), where=(total_den - denominator[e]) > 0)
        slope = np.clip(slope, -COEFFICIENT_ABS_MAX, COEFFICIENT_ABS_MAX)
        fold_beta.append(slope)
        heldout_sse = residual_square[e] - 2.0 * slope * numerator[e] + slope * slope * denominator[e]
        fold_improved.append(heldout_sse < residual_square[e])
    folds = np.stack(fold_beta)
    improved = np.stack(fold_improved)
    median = np.median(folds, axis=0)
    median_sign = np.sign(median)
    sign_agreement = np.mean(np.sign(folds) == median_sign[None], axis=0)
    improvement_fraction = np.mean(improved, axis=0)
    sign_count = np.sum(np.sign(folds) == median_sign[None], axis=0).astype(np.uint8)
    improvement_count = np.sum(improved, axis=0).astype(np.uint8)
    active = (median_sign != 0) & (sign_agreement >= LOEO_SIGN_AGREEMENT_MIN) & (improvement_fraction >= LOEO_IMPROVEMENT_FRACTION_MIN)
    active[list(PROTECTED_FRAMES)] = False
    beta_map = np.where(active, median, 0.0).astype(np.float32)
    active_pixels = int(np.any(active, axis=(0, 3)).sum())
    active_components = int(active.sum())
    if active_pixels < MIN_ACTIVE_PIXELS:
        raise RuntimeError(f"v443 retained only {active_pixels} active pixels, requires {MIN_ACTIVE_PIXELS}")
    baseline_loeo_sse = float(residual_square.sum())
    corrected_loeo_sse = 0.0
    for e in range(len(right_train)):
        fold = np.where(active, folds[e], 0.0)
        corrected_loeo_sse += float((residual_square[e] - 2.0 * fold * numerator[e] + fold * fold * denominator[e]).sum())
    aggregate_ratio = corrected_loeo_sse / baseline_loeo_sse
    if not np.isfinite(aggregate_ratio) or aggregate_ratio >= 1.0:
        raise RuntimeError(f"v443 LOEO aggregate did not improve: ratio={aggregate_ratio}")
    if np.any(beta_map[list(PROTECTED_FRAMES)] != 0.0) or np.abs(beta_map).max(initial=0.0) > 1.0:
        raise RuntimeError("v443 protected/bounded beta contract failed")

    metadata = {
        "format": "strict-track2-v443-close-trainonly-spatial-alignment-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "right-train15 close-only per-time/pixel/RGB zero-intercept 15-fold LOEO spatial alignment",
        "train_episodes": train, "validation_episodes_excluded": validation,
        "right_train_episodes": right_train, "samples_per_episode": SAMPLES_PER_EPISODE,
        "selection": "among exact eight close windows choose first-close index nearest 4, then earliest start",
        "samples": [{key: row[key] for key in ("episode", "phase", "start", "dataset_index", "first_close_index")} for row in rows],
        "beta_shape": list(beta_map.shape), "protected_frames": list(PROTECTED_FRAMES),
        "loeo": {
            "sign_agreement_min": LOEO_SIGN_AGREEMENT_MIN,
            "heldout_pixel_sse_improvement_fraction_min": LOEO_IMPROVEMENT_FRACTION_MIN,
            "coefficient_abs_max": COEFFICIENT_ABS_MAX,
            "active_pixels_min": MIN_ACTIVE_PIXELS, "active_pixels": active_pixels,
            "active_components": active_components, "aggregate_sse_ratio": aggregate_ratio,
            "morphology_or_smoothing": False,
        },
        "sha256": {
            "split": sha256(args.split), "preregistration": sha256(args.preregistration),
            "v169_manifest": sha256(args.v169_release / "v169_arm_routed_manifest.json"),
            "v436_manifest": sha256(args.v436_release / "v436_diagnostic_manifest.json"),
        },
        "guards": {
            "public_train40_only": True, "right_train15_only": True,
            "validation_or_dev_used": False, "reward_read": False, "outcome_read": False,
            "hidden_or_final_data": False, "coefficient_dev_sweep": False,
            "policy_updates": 0, "real_submission": False, "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, beta_map=beta_map,
        sign_agreement_count=sign_count,
        heldout_improvement_count=improvement_count,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps({"output": str(args.output), "active_pixels": active_pixels, "active_components": active_components, "aggregate_sse_ratio": aggregate_ratio}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
