#!/usr/bin/env python3
"""Calibrate v441 phase/frame/channel coefficients on public right train15 only."""

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
from wam_pipeline.v441_v169_postclose_aligned_projection_runtime import (
    PHASES,
    clipped_rgb_delta,
    gate_decision,
)


SAMPLES_PER_EPISODE_PHASE = 4
LOEO_SIGN_AGREEMENT_MIN = 0.80
LOEO_IMPROVEMENT_FRACTION_MIN = 0.80
COEFFICIENT_ABS_MAX = 1.0
INFERENCE_BATCH_SIZE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(episode: int, start: int) -> int:
    payload = f"v441-trainonly/episode{episode}/start{start}/seed1586"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**31)


def select_samples(dataset: WindowDataset, episodes: list[int], prompts: dict[int, str]) -> list[dict]:
    grouped = {(episode, phase): [] for episode in episodes for phase in PHASES}
    for index, path in enumerate(dataset.paths):
        episode = int(path.name.split("_")[0][7:])
        if episode not in episodes:
            continue
        history, future = dataset.load_actions(index)
        decision = gate_decision(history, future, prompts[episode])
        phase = decision.get("phase")
        if decision.get("gate") and phase in PHASES:
            start = int(path.stem.split("_")[1])
            grouped[(episode, phase)].append((start, index))
    rows = []
    for episode in episodes:
        for phase in PHASES:
            candidates = sorted(grouped[(episode, phase)])
            if len(candidates) < SAMPLES_PER_EPISODE_PHASE:
                raise RuntimeError(f"v441 requires four gated {phase} samples in train episode {episode}, got {len(candidates)}")
            positions = np.linspace(0, len(candidates) - 1, SAMPLES_PER_EPISODE_PHASE, dtype=np.int64)
            chosen = [candidates[int(position)] for position in positions]
            if len({index for _, index in chosen}) != SAMPLES_PER_EPISODE_PHASE:
                raise RuntimeError("v441 deterministic calibration selection duplicated a row")
            for start, index in chosen:
                arrays = dataset.load_arrays(index)
                rows.append({
                    "episode": episode, "phase": phase, "start": start,
                    "dataset_index": index, "prompt": prompts[episode], **arrays,
                })
    expected = len(episodes) * len(PHASES) * SAMPLES_PER_EPISODE_PHASE
    if len(rows) != expected:
        raise RuntimeError(f"v441 expected {expected} calibration rows, got {len(rows)}")
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
    if args.inference_batch_size != INFERENCE_BATCH_SIZE:
        raise SystemExit("v441 calibration is preregistered for inference batch size 4")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v441-postclose-trainonly-preregistration-v1":
        raise RuntimeError("wrong v441 preregistration")
    split = json.loads(args.split.read_text())
    arms = {int(key): str(value) for key, value in split["arm_by_episode"].items()}
    prompts = {int(key): str(value) for key, value in split["episode_to_instruction"].items()}
    train = [int(value) for value in split["train_episodes"]]
    validation = [int(value) for value in split["validation_episodes"]]
    right_train = sorted(episode for episode in train if arms[episode] == "right")
    if len(train) != 40 or len(validation) != 10 or len(right_train) != 15 or set(train) & set(validation):
        raise RuntimeError("v441 requires the immutable public train40/right15 boundary")
    if any("right arm" not in prompts[episode].lower() or "left arm" in prompts[episode].lower() for episode in right_train):
        raise RuntimeError("v441 right-train prompt contract failed")

    dataset = WindowDataset(args.windows, right_train, rollout_horizon=8)
    rows = select_samples(dataset, right_train, prompts)
    v169 = Track2V169ArmRoutedRuntime(args.v169_release, args.v169_library, args.device)
    v436 = json.loads((args.v436_release / "v436_diagnostic_manifest.json").read_text())
    if v436.get("format") != "track2-v436-v432-step25-diagnostic-release-v1":
        raise RuntimeError("wrong v441 teacher release")
    parent = Track2AutoregressiveUNet(args.v436_release / v436["parent_right"], args.device)
    candidate = Track2AutoregressiveUNet(args.v436_release / v436["candidate_right"], args.device)

    shape = (len(right_train), len(PHASES), 8, 3)
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    residual_square = np.zeros(shape, dtype=np.float64)
    counts = np.zeros((len(right_train), len(PHASES)), dtype=np.int64)
    episode_slot = {episode: index for index, episode in enumerate(right_train)}
    phase_slot = {phase: index for index, phase in enumerate(PHASES)}
    for begin in range(0, len(rows), INFERENCE_BATCH_SIZE):
        batch = rows[begin : begin + INFERENCE_BATCH_SIZE]
        context = np.stack([row["context_frames"] for row in batch])
        history = np.stack([row["history_actions"] for row in batch])
        future = np.stack([row["future_actions"] for row in batch])
        target = np.stack([row["target_frames"] for row in batch]).astype(np.int16)
        seeds = np.asarray([stable_seed(row["episode"], row["start"]) for row in batch], dtype=np.int64)
        texts = [row["prompt"] for row in batch]
        baseline = v169.predict_batch(context, history, future, seeds, texts)
        parent_rgb = parent.predict_batch(context, history, future, seeds, texts)
        candidate_rgb = candidate.predict_batch(context, history, future, seeds, texts)
        teacher = clipped_rgb_delta(candidate_rgb, parent_rgb).astype(np.float64)
        correction = target.astype(np.float64) - baseline.astype(np.float64)
        for local, row in enumerate(batch):
            e = episode_slot[row["episode"]]; p = phase_slot[row["phase"]]
            numerator[e, p] += np.sum(teacher[local] * correction[local], axis=(1, 2))
            denominator[e, p] += np.sum(teacher[local] ** 2, axis=(1, 2))
            residual_square[e, p] += np.sum(correction[local] ** 2, axis=(1, 2))
            counts[e, p] += 1

    if not np.all(counts == SAMPLES_PER_EPISODE_PHASE):
        raise RuntimeError(f"v441 calibration count drift: {counts.tolist()}")
    beta = np.zeros((len(PHASES), 8, 3), dtype=np.float64)
    evidence = {}
    for phase, p in phase_slot.items():
        fold_beta = []
        fold_improved = []
        total_num = numerator[:, p].sum(0)
        total_den = denominator[:, p].sum(0)
        for e in range(len(right_train)):
            train_num = total_num - numerator[e, p]
            train_den = total_den - denominator[e, p]
            slope = np.divide(train_num, train_den, out=np.zeros_like(train_num), where=train_den > 0)
            slope = np.clip(slope, -COEFFICIENT_ABS_MAX, COEFFICIENT_ABS_MAX)
            fold_beta.append(slope)
            heldout_sse = residual_square[e, p] - 2.0 * slope * numerator[e, p] + slope * slope * denominator[e, p]
            fold_improved.append(heldout_sse < residual_square[e, p])
        folds = np.stack(fold_beta)
        improved = np.stack(fold_improved)
        median = np.median(folds, axis=0)
        median_sign = np.sign(median)
        sign_agreement = np.mean(np.sign(folds) == median_sign[None], axis=0)
        improvement_fraction = np.mean(improved, axis=0)
        active = (
            (median_sign != 0)
            & (sign_agreement >= LOEO_SIGN_AGREEMENT_MIN)
            & (improvement_fraction >= LOEO_IMPROVEMENT_FRACTION_MIN)
        )
        beta[p] = np.where(active, median, 0.0)
        if int(active.sum()) == 0:
            raise RuntimeError(f"v441 LOEO retained no stable {phase} component")
        evidence[phase] = {
            "median_fold_beta": median.tolist(),
            "sign_agreement": sign_agreement.tolist(),
            "heldout_improvement_fraction": improvement_fraction.tolist(),
            "active": active.tolist(),
            "active_components": int(active.sum()),
        }

    payload = {
        "format": "strict-track2-v441-postclose-trainonly-rgb-alignment-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "public-right-train15 leave-one-episode-out phase/frame/RGB-channel closed-form alignment",
        "train_episodes": train,
        "validation_episodes_excluded": validation,
        "right_train_episodes": right_train,
        "phases": list(PHASES),
        "samples_per_episode_phase": SAMPLES_PER_EPISODE_PHASE,
        "samples": [{key: row[key] for key in ("episode", "phase", "start", "dataset_index")} for row in rows],
        "beta": {phase: beta[index].tolist() for index, phase in enumerate(PHASES)},
        "loeo": {
            "sign_agreement_min": LOEO_SIGN_AGREEMENT_MIN,
            "heldout_improvement_fraction_min": LOEO_IMPROVEMENT_FRACTION_MIN,
            "coefficient_abs_max": COEFFICIENT_ABS_MAX,
            "evidence": evidence,
        },
        "sha256": {
            "split": sha256(args.split),
            "preregistration": sha256(args.preregistration),
            "v169_manifest": sha256(args.v169_release / "v169_arm_routed_manifest.json"),
            "v436_manifest": sha256(args.v436_release / "v436_diagnostic_manifest.json"),
        },
        "guards": {
            "public_train40_only": True, "right_train15_only": True,
            "validation_or_dev_used": False, "reward_read": False,
            "outcome_read": False, "hidden_or_final_data": False,
            "coefficient_dev_sweep": False, "policy_updates": 0,
            "real_submission": False, "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(rows), "active": {phase: evidence[phase]["active_components"] for phase in PHASES}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

