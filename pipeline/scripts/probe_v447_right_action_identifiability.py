#!/usr/bin/env python3
"""Five-fold float-latent probe of action identifiability in public right15."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FORMAT = "strict-track2-v447-right-action-identifiability-probe-v1"
PREREG_FORMAT = "strict-track2-v447-right-action-identifiability-preregistration-v1"
GRID = 16
PCA_DIM = 32
TARGET_DIM = 16
RIDGE_ALPHA = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def block_mean(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames, dtype=np.float32)
    if value.ndim != 4 or value.shape[1:] != (256, 256, 3):
        raise ValueError(f"V447 requires RGB[T,256,256,3], got {value.shape}")
    block = 256 // GRID
    return value.reshape(len(value), GRID, block, GRID, block, 3).mean((2, 4))


def first_close_index(history: np.ndarray, future: np.ndarray) -> int | None:
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    if history.shape != (4, 14) or future.shape != (8, 14):
        raise ValueError("V447 requires history4/future8 joint14")
    if not np.isfinite(history).all() or not np.isfinite(future).all():
        raise ValueError("V447 actions must be finite")
    if float(history[-1, 13]) < 0.5:
        return None
    closed = future[:, 13] < 0.5
    indices = np.flatnonzero(closed)
    if not indices.size or not np.all(closed[indices[0] :]):
        return None
    return int(indices[0])


def load_rows(windows: Path, episodes: list[int]) -> list[dict]:
    rows = []
    counts = {episode: 0 for episode in episodes}
    for episode in episodes:
        for path in sorted(windows.glob(f"episode{episode}_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                context = np.asarray(data["context_frames"], dtype=np.uint8)
                history = np.asarray(data["history_actions"], dtype=np.float32)
                future = np.asarray(data["future_actions"], dtype=np.float32)
                target = np.asarray(data["target_frames"], dtype=np.uint8)
                phase = first_close_index(history, future)
                if phase is None:
                    continue
                source = str(data["source"].item()) if "source" in data.files else ""
                fields = tuple(sorted(data.files))
                scalar = {}
                for key in ("capture_success", "success", "task_success", "intervention", "counterfactual", "action_variant"):
                    if key in data.files and np.asarray(data[key]).ndim == 0:
                        scalar[key] = np.asarray(data[key]).item()
            if context.shape != (5, 256, 256, 3) or target.shape != (8, 256, 256, 3):
                raise RuntimeError(f"V447 frame contract drift: {path}")
            rows.append(
                {
                    "episode": episode,
                    "start": int(path.stem.split("_")[1]),
                    "path": str(path.resolve()),
                    "source": source,
                    "fields": fields,
                    "scalar": scalar,
                    "phase": phase,
                    "context": context,
                    "history": history,
                    "future": future,
                    "target": target,
                }
            )
            counts[episode] += 1
    if len(rows) != 120 or any(count != 8 for count in counts.values()):
        raise RuntimeError(f"V447 requires close-event 8xright15=120, got {counts}")
    return rows


def raw_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contexts, actions, targets = [], [], []
    for row in rows:
        down = block_mean(row["context"][-2:]) / 255.0
        context = np.concatenate((down[-1].reshape(-1), (down[-1] - down[-2]).reshape(-1)))
        action = np.concatenate((row["history"], row["future"]), axis=0).reshape(-1)
        target = (block_mean(row["target"]) - block_mean(row["context"][-1:])[0]).reshape(-1) / 255.0
        contexts.append(context)
        actions.append(action)
        targets.append(target)
    return tuple(np.asarray(value, dtype=np.float64) for value in (contexts, actions, targets))


def fit_pca(values: np.ndarray, dimension: int = PCA_DIM, standardize: bool = False) -> dict:
    mean = values.mean(0)
    scale_in = values.std(0) if standardize else np.ones(values.shape[1], dtype=np.float64)
    scale_in = np.maximum(scale_in, 1e-8)
    centered = (values - mean) / scale_in
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(dimension, len(vt))
    components = vt[:count]
    latent = centered @ components.T
    latent_scale = np.maximum(latent.std(0), 1e-8)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
    rank95 = int(np.searchsorted(cumulative, 0.95) + 1)
    numerical_rank = int(np.sum(singular > max(float(singular[0]), 1.0) * 1e-8))
    return {
        "mean": mean,
        "input_scale": scale_in,
        "components": components,
        "latent_scale": latent_scale,
        "singular": singular,
        "rank95": rank95,
        "numerical_rank": numerical_rank,
    }


def transform(values: np.ndarray, projector: dict, dimension: int = PCA_DIM) -> np.ndarray:
    latent = ((values - projector["mean"]) / projector["input_scale"]) @ projector["components"].T
    latent = latent / projector["latent_scale"]
    if latent.shape[1] < dimension:
        latent = np.pad(latent, ((0, 0), (0, dimension - latent.shape[1])))
    return latent[:, :dimension]


def ridge_fit(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    return design @ weights


def phase_shuffle_actions(rows: list[dict]) -> tuple[np.ndarray, dict]:
    output = [None] * len(rows)
    groups = {}
    for phase in range(8):
        indices = [index for index, row in enumerate(rows) if row["phase"] == phase]
        indices.sort(key=lambda index: (rows[index]["episode"], rows[index]["start"]))
        if len(indices) != 3:
            raise RuntimeError(f"V447 expected three held episodes for close-index {phase}, got {len(indices)}")
        for position, index in enumerate(indices):
            donor = indices[(position + 1) % len(indices)]
            if rows[index]["episode"] == rows[donor]["episode"]:
                raise RuntimeError("V447 phase shuffle must cross episodes")
            output[index] = np.concatenate((rows[index]["history"], rows[donor]["future"]), axis=0).reshape(-1)
        groups[str(phase)] = [rows[index]["episode"] for index in indices]
    return np.asarray(output, dtype=np.float64), groups


def errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((prediction - target) ** 2, axis=1)


def support_statistics(rows: list[dict], context_raw: np.ndarray, action_raw: np.ndarray, target_raw: np.ndarray) -> dict:
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(frame_digest(row["context"]), []).append(index)
    exact_groups = 0
    exact_pairs = 0
    exact_pairs_different_target = 0
    for indices in groups.values():
        valid = []
        for left, right in itertools.combinations(indices, 2):
            if not np.array_equal(rows[left]["future"], rows[right]["future"]):
                valid.append((left, right))
        if valid:
            exact_groups += 1
            exact_pairs += len(valid)
            exact_pairs_different_target += sum(
                not np.array_equal(rows[left]["target"], rows[right]["target"])
                for left, right in valid
            )

    context_scale = np.maximum(context_raw.std(0), 1e-8)
    action_scale = np.maximum(action_raw.std(0), 1e-8)
    same_phase_pairs = []
    for left, right in itertools.combinations(range(len(rows)), 2):
        if rows[left]["episode"] == rows[right]["episode"] or rows[left]["phase"] != rows[right]["phase"]:
            continue
        context_distance = float(np.mean(((context_raw[left] - context_raw[right]) / context_scale) ** 2))
        action_distance = float(np.mean(((action_raw[left] - action_raw[right]) / action_scale) ** 2))
        target_distance = float(np.mean((target_raw[left] - target_raw[right]) ** 2))
        same_phase_pairs.append((context_distance, action_distance, target_distance))
    pair_array = np.asarray(same_phase_pairs, dtype=np.float64)
    if len(pair_array):
        context_cut = float(np.quantile(pair_array[:, 0], 0.10))
        action_cut = float(np.quantile(pair_array[:, 1], 0.50))
        near = pair_array[(pair_array[:, 0] <= context_cut) & (pair_array[:, 1] >= action_cut)]
    else:
        context_cut = action_cut = float("nan")
        near = np.empty((0, 3), dtype=np.float64)

    failure_label_count = failure_count = intervention_label_count = intervention_count = 0
    for row in rows:
        scalar = row["scalar"]
        success_keys = [key for key in ("capture_success", "success", "task_success") if key in scalar]
        failure_label_count += bool(success_keys)
        failure_count += bool(success_keys) and not all(bool(scalar[key]) for key in success_keys)
        intervention_keys = [key for key in ("intervention", "counterfactual", "action_variant") if key in scalar]
        intervention_label_count += bool(intervention_keys)
        source = row["source"].lower()
        is_intervention = bool(intervention_keys) or "intervention" in source or "counterfactual" in source
        intervention_count += is_intervention

    return {
        "exact_same_context_different_action_groups": exact_groups,
        "exact_same_context_different_action_pairs": exact_pairs,
        "exact_pairs_with_different_target": exact_pairs_different_target,
        "cross_episode_same_phase_pair_count": int(len(pair_array)),
        "near_context_threshold_p10": context_cut,
        "different_action_threshold_p50": action_cut,
        "near_context_different_action_pair_count": int(len(near)),
        "near_pair_action_distance_mean": float(near[:, 1].mean()) if len(near) else None,
        "near_pair_target_distance_mean": float(near[:, 2].mean()) if len(near) else None,
        "failure_label_available_rows": int(failure_label_count),
        "failure_target_rows": int(failure_count),
        "intervention_label_available_rows": int(intervention_label_count),
        "intervention_or_counterfactual_target_rows": int(intervention_count),
        "clean_demo_source_rows": int(sum("aloha-agilex_clean_50" in row["source"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != PREREG_FORMAT:
        raise RuntimeError("wrong V447 preregistration")
    if sha256(args.split) != prereg["evidence_sha256"]["split"]:
        raise RuntimeError("V447 split hash drift")
    if args.windows.resolve() != Path(prereg["data"]["windows"]).resolve():
        raise RuntimeError("V447 windows path drift")

    right = [int(value) for value in prereg["data"]["right_train_episodes"]]
    folds = [[int(value) for value in fold] for fold in prereg["data"]["fold_holdout_episodes"]]
    rows = load_rows(args.windows, right)
    context_raw, action_raw, target_raw = raw_features(rows)
    index_by_episode = {episode: np.asarray([index for index, row in enumerate(rows) if row["episode"] == episode]) for episode in right}

    fold_reports = []
    episode_reports = {}
    for fold_index, holdout_episodes in enumerate(folds):
        fit_indices = np.concatenate([index_by_episode[episode] for episode in right if episode not in holdout_episodes])
        held_indices = np.concatenate([index_by_episode[episode] for episode in holdout_episodes])
        fit_rows = [rows[int(index)] for index in fit_indices]
        held_rows = [rows[int(index)] for index in held_indices]

        context_pca = fit_pca(context_raw[fit_indices], dimension=PCA_DIM, standardize=False)
        action_pca = fit_pca(action_raw[fit_indices], standardize=True)
        target_pca = fit_pca(target_raw[fit_indices], dimension=TARGET_DIM, standardize=False)
        fit_context = transform(context_raw[fit_indices], context_pca)
        held_context = transform(context_raw[held_indices], context_pca)
        fit_action = transform(action_raw[fit_indices], action_pca)
        held_action = transform(action_raw[held_indices], action_pca)
        fit_target = transform(target_raw[fit_indices], target_pca, dimension=TARGET_DIM)
        held_target = transform(target_raw[held_indices], target_pca, dimension=TARGET_DIM)

        zero_fit = np.zeros_like(fit_context)
        zero_held = np.zeros_like(held_context)
        context_features = np.concatenate((fit_context, zero_fit), axis=1)
        action_features = np.concatenate((fit_context, fit_action), axis=1)
        context_weights = ridge_fit(context_features, fit_target)
        action_weights = ridge_fit(action_features, fit_target)

        shuffled_action_raw, shuffle_groups = phase_shuffle_actions(held_rows)
        shuffled_action = transform(shuffled_action_raw, action_pca)
        context_error = errors(
            ridge_predict(np.concatenate((held_context, zero_held), axis=1), context_weights),
            held_target,
        )
        action_error = errors(
            ridge_predict(np.concatenate((held_context, held_action), axis=1), action_weights),
            held_target,
        )
        shuffle_error = errors(
            ridge_predict(np.concatenate((held_context, shuffled_action), axis=1), action_weights),
            held_target,
        )
        context_ratio = float(action_error.mean() / max(float(context_error.mean()), 1e-12))
        shuffle_ratio = float(action_error.mean() / max(float(shuffle_error.mean()), 1e-12))
        passed = context_ratio <= 0.95 and shuffle_ratio <= 0.97
        fold_reports.append(
            {
                "fold": fold_index,
                "fit_episodes": sorted(set(row["episode"] for row in fit_rows)),
                "holdout_episodes": holdout_episodes,
                "fit_rows": len(fit_rows),
                "holdout_rows": len(held_rows),
                "input_dimension_equal": {"context_only": 64, "action": 64},
                "same_ridge_lambda": RIDGE_ALPHA,
                "output_latent_dimension": TARGET_DIM,
                "action_error_over_context_error": context_ratio,
                "action_error_over_phase_shuffle_error": shuffle_ratio,
                "action_numerical_rank": action_pca["numerical_rank"],
                "action_rank95": action_pca["rank95"],
                "phase_shuffle_groups": shuffle_groups,
                "fit_only_projectors": True,
                "passed": passed,
            }
        )
        for episode in holdout_episodes:
            mask = np.asarray([row["episode"] == episode for row in held_rows])
            ratio_context = float(action_error[mask].mean() / max(float(context_error[mask].mean()), 1e-12))
            ratio_shuffle = float(action_error[mask].mean() / max(float(shuffle_error[mask].mean()), 1e-12))
            episode_reports[str(episode)] = {
                "rows": int(mask.sum()),
                "action_error_over_context_error": ratio_context,
                "action_error_over_phase_shuffle_error": ratio_shuffle,
                "passed": ratio_context <= 0.95 and ratio_shuffle <= 0.97,
            }

    passing_folds = sum(row["passed"] for row in fold_reports)
    passing_episodes = sum(row["passed"] for row in episode_reports.values())
    support = support_statistics(rows, context_raw, action_raw, target_raw)
    global_action = fit_pca(action_raw, standardize=True)
    passed = passing_folds >= 4 and passing_episodes >= 12
    report = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": "current_right15_action_identifiable" if passed else "current_right15_action_identifiability_not_established",
        "coverage": {
            "right_episodes": right,
            "rows": len(rows),
            "rows_per_episode": {str(episode): int(len(index_by_episode[episode])) for episode in right},
            "close_index_counts": {str(phase): int(sum(row["phase"] == phase for row in rows)) for phase in range(8)},
        },
        "gates": {
            "action_error_over_context_error_max": 0.95,
            "action_error_over_phase_shuffle_error_max": 0.97,
            "passing_folds_min": 4,
            "passing_episodes_min": 12,
            "passing_folds": passing_folds,
            "passing_episodes": passing_episodes,
        },
        "folds": fold_reports,
        "episodes": episode_reports,
        "action_support": {
            "global_numerical_rank": global_action["numerical_rank"],
            "global_rank95": global_action["rank95"],
            "raw_dimension": int(action_raw.shape[1]),
            **support,
        },
        "sha256": {
            "preregistration": sha256(args.preregistration),
            "split": sha256(args.split),
        },
        "guards": {
            "fit_only_pca_and_normalization": True,
            "equal_probe_input_dimension": True,
            "float_latent_no_round_or_cap": True,
            "runtime_gate_used": False,
            "v169_used": False,
            "reward_used": False,
            "outcome_used_for_fit_or_gate": False,
            "outcome_labels_counted_descriptive_only": True,
            "validation_development_or_final_used": False,
            "policy_loaded_or_modified": False,
            "policy_updates": 0,
            "real_submission": False,
            "s1_authorized": False,
            "rl_authorized": False,
            "formal_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "gates": report["gates"], "action_support": report["action_support"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
