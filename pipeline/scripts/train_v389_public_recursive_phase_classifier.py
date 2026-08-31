#!/usr/bin/env python3
"""Fit v389 from public-train teacher and recursive contexts only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from wam_pipeline.arm_routed_autoregressive_runtime import (
    Track2ArmRoutedAutoregressiveUNet,
)
from wam_pipeline.v312_causal_terminal_mirror_runtime import (
    action_features as causal_action_features,
)
from wam_pipeline.v326_blended_phase_terminal_runtime import (
    Track2V326BlendedPhaseTerminal,
)
from wam_pipeline.v378_source_routed_blended_cartesian_runtime import (
    Track2V378SourceRoutedBlendedCartesian,
)
from wam_pipeline.v389_public_recursive_phase_gate import (
    FEATURE_VERSION,
    phase_features,
)


SEED = 1550
CS = (0.001, 0.01, 0.1)
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
SOURCE_KINDS = ("v326_generated", "v355_generated", "v378_generated")


def request_seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (
        2**31
    )


class FrozenActionRoute:
    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as values:
            self.mean = values["feature_mean"].astype(np.float32)
            self.scale = values["feature_scale"].astype(np.float32)
            self.coefficient = values["coefficient"].astype(np.float32)
            self.intercept = float(values["intercept"].item())

    def probability(self, history: np.ndarray, future: np.ndarray) -> float:
        feature = causal_action_features(history, future)
        logit = float(
            ((feature - self.mean) / self.scale) @ self.coefficient + self.intercept
        )
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def eligible(self, history: np.ndarray, future: np.ndarray) -> bool:
        probability = self.probability(history, future)
        post_grasp = bool(
            history[-1, 13] < 0.5 and (future[:, 13] < 0.5).mean() >= 0.75
        )
        history_gripper = float(history[-1, 13])
        future_gripper = float(future[:, 13].mean())
        sequence = np.concatenate((history[-1:, 7:13], future[:, 7:13]), axis=0)
        path = float(np.linalg.norm(np.diff(sequence, axis=0), axis=1).sum())
        failure = (
            (history_gripper <= 0.5 and future_gripper > 0.5)
            or path <= 1e-6
            or (future_gripper <= 0.5 and probability < 0.01)
        )
        return bool(post_grasp and probability >= 0.99 and not failure)


def load_window(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return (
            values["context_frames"].astype(np.uint8),
            values["history_actions"].astype(np.float32),
            values["future_actions"].astype(np.float32),
        )


def load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only the tiny action members during recursive generation."""
    with np.load(path, allow_pickle=False) as values:
        return (
            values["history_actions"].astype(np.float32),
            values["future_actions"].astype(np.float32),
        )


def append_example(
    features: list[np.ndarray],
    labels: list[int],
    groups: list[int],
    kinds: list[str],
    routes: list[bool],
    context: np.ndarray,
    history: np.ndarray,
    future: np.ndarray,
    episode: int,
    start: int,
    onset: dict[int, int],
    kind: str,
    action_route: FrozenActionRoute,
) -> None:
    features.append(phase_features(context, history, future))
    labels.append(int(start >= onset[episode]))
    groups.append(episode)
    kinds.append(kind)
    routes.append(action_route.eligible(history, future))


def teacher_examples(
    windows: Path,
    episodes: list[int],
    onset: dict[int, int],
    action_route: FrozenActionRoute,
    output: tuple[list, list, list, list, list],
) -> int:
    count = 0
    for episode in episodes:
        for path in sorted(windows.glob(f"episode{episode}_*.npz")):
            start = int(path.stem.split("_")[1])
            context, history, future = load_window(path)
            append_example(
                *output,
                context,
                history,
                future,
                episode,
                start,
                onset,
                "teacher",
                action_route,
            )
            count += 1
    return count


def initial_states(windows: Path, episodes: list[int]) -> list[dict]:
    states = []
    for episode in episodes:
        available = {
            int(path.stem.split("_")[1]): path
            for path in windows.glob(f"episode{episode}_*.npz")
        }
        for alignment in range(8):
            if alignment not in available:
                continue
            initial, _, _ = load_window(available[alignment])
            states.append(
                {
                    "episode": episode,
                    "alignment": alignment,
                    "start": alignment,
                    "available": available,
                    "context": initial,
                    "generation": 0,
                }
            )
    return states


def recursive_examples(
    model,
    kind: str,
    windows: Path,
    episodes: list[int],
    onset: dict[int, int],
    action_route: FrozenActionRoute,
    output: tuple[list, list, list, list, list],
    batch_size: int,
) -> tuple[int, int]:
    states = initial_states(windows, episodes)
    examples = 0
    predictions = 0
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        histories, futures, seeds = [], [], []
        for state in active:
            path = state["available"][state["start"]]
            history, future = load_actions(path)
            histories.append(history)
            futures.append(future)
            seeds.append(request_seed(path))
            if state["generation"] > 0:
                append_example(
                    *output,
                    state["context"],
                    history,
                    future,
                    state["episode"],
                    state["start"],
                    onset,
                    kind,
                    action_route,
                )
                examples += 1
        histories_array = np.stack(histories)
        futures_array = np.stack(futures)
        seeds_array = np.asarray(seeds, dtype=np.int64)
        generated = []
        for begin in range(0, len(active), batch_size):
            end = begin + batch_size
            generated.extend(
                model.predict_batch(
                    np.stack([state["context"] for state in active[begin:end]]),
                    histories_array[begin:end],
                    futures_array[begin:end],
                    seeds_array[begin:end],
                    ["Adjust bottle"] * len(active[begin:end]),
                )
            )
        for state, prediction in zip(active, generated, strict=True):
            state["context"] = np.asarray(prediction[-5:], dtype=np.uint8).copy()
            state["start"] += 8
            state["generation"] += 1
            predictions += 1
    return examples, predictions


def rates(
    labels: np.ndarray,
    probability: np.ndarray,
    kinds: np.ndarray,
    route: np.ndarray,
    threshold: float,
) -> dict:
    predicted = probability >= threshold

    def group_metrics(mask: np.ndarray) -> dict:
        positive = mask & (labels == 1)
        negative = mask & (labels == 0)
        return {
            "rows": int(mask.sum()),
            "positive": int(positive.sum()),
            "negative": int(negative.sum()),
            "positive_recall": float(predicted[positive].mean()) if positive.any() else None,
            "negative_specificity": float((~predicted[negative]).mean()) if negative.any() else None,
        }

    result = {
        "threshold": threshold,
        "teacher_all": group_metrics(kinds == "teacher"),
        "generated_route": group_metrics((kinds != "teacher") & route),
        "sources": {},
    }
    for kind in SOURCE_KINDS:
        result["sources"][kind] = group_metrics((kinds == kind) & route)
    return result


def eligible(row: dict) -> bool:
    generated = row["generated_route"]
    sources = row["sources"]
    return bool(
        generated["negative_specificity"] >= 0.99
        and generated["positive_recall"] >= 0.50
        and row["teacher_all"]["negative_specificity"] >= 0.98
        and all(
            value["negative_specificity"] >= 0.985
            and value["positive_recall"] >= 0.40
            for value in sources.values()
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "v326-release",
        "v355-release",
        "v378-release",
        "library",
        "windows",
        "split",
        "phase-gate",
        "action-gate",
        "preregistration",
        "output",
        "report",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing overwrite")
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get("format") != "strict-track2-v389-public-recursive-phase-preregistration-v1":
        raise RuntimeError("wrong preregistration")
    split = json.loads(args.split.read_text())
    train = set(map(int, split["train_episodes"]))
    arm = {int(key): value for key, value in split["arm_by_episode"].items()}
    episodes = sorted(episode for episode in train if arm[episode] == "right")
    if len(episodes) != 15:
        raise RuntimeError(f"expected 15 public train right episodes, got {episodes}")
    with np.load(args.phase_gate, allow_pickle=False) as values:
        onset = {
            int(episode) - 20000: int(start)
            for episode, start in zip(
                values["episode"], values["onset_start"], strict=True
            )
        }
    if set(onset) != set(episodes):
        raise RuntimeError("phase table/public split episode mismatch")
    action_route = FrozenActionRoute(args.action_gate)
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    kinds: list[str] = []
    routes: list[bool] = []
    output_lists = (features, labels, groups, kinds, routes)
    teacher_count = teacher_examples(
        args.windows, episodes, onset, action_route, output_lists
    )
    print(json.dumps({"stage": "teacher_features", "examples": teacher_count}), flush=True)
    generation_reports = {}

    factories = (
        (
            "v326_generated",
            lambda: Track2V326BlendedPhaseTerminal(
                args.v326_release,
                args.library,
                args.device,
                action_gate=args.action_gate,
                phase_gate=args.phase_gate,
            ),
        ),
        (
            "v355_generated",
            lambda: Track2ArmRoutedAutoregressiveUNet(
                args.v355_release / "left_expert",
                args.v355_release / "right_expert",
                args.device,
            ),
        ),
        (
            "v378_generated",
            lambda: Track2V378SourceRoutedBlendedCartesian(
                args.v378_release, args.library, args.device
            ),
        ),
    )
    for kind, factory in factories:
        model = factory()
        count, predictions = recursive_examples(
            model,
            kind,
            args.windows,
            episodes,
            onset,
            action_route,
            output_lists,
            args.batch_size,
        )
        generation_reports[kind] = {
            "examples": count,
            "predictions": predictions,
        }
        print(
            json.dumps(
                {"stage": kind, "examples": count, "predictions": predictions}
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group = np.asarray(groups, dtype=np.int64)
    kind = np.asarray(kinds)
    route = np.asarray(routes, dtype=bool)
    cv_rows = []
    for c_value in CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for heldout in episodes:
            train_mask = group != heldout
            test_mask = ~train_mask
            fold_mean = x[train_mask].mean(axis=0)
            fold_scale = np.maximum(x[train_mask].std(axis=0), 1e-6)
            model = LogisticRegression(
                C=c_value,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            ).fit((x[train_mask] - fold_mean) / fold_scale, y[train_mask])
            oof[test_mask] = model.predict_proba(
                (x[test_mask] - fold_mean) / fold_scale
            )[:, 1]
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete grouped OOF predictions")
        for threshold in THRESHOLDS:
            row = rates(y, oof, kind, route, threshold)
            row["c"] = c_value
            row["eligible"] = eligible(row)
            cv_rows.append(row)

    candidates = [row for row in cv_rows if row["eligible"]]
    selected = (
        max(
            candidates,
            key=lambda row: (
                min(
                    value["positive_recall"] for value in row["sources"].values()
                ),
                row["generated_route"]["positive_recall"],
                row["generated_route"]["negative_specificity"],
                row["threshold"],
                -row["c"],
            ),
        )
        if candidates
        else None
    )
    checks = {
        "fifteen_public_train_right_episodes": len(episodes) == 15,
        "teacher_rows_exact_1926": teacher_count == 1926,
        "three_recursive_sources": set(generation_reports) == set(SOURCE_KINDS),
        "each_recursive_source_examples_ge_1800": all(
            value["examples"] >= 1800 for value in generation_reports.values()
        ),
        "episode_grouped_oof": True,
        "eligible_high_specificity_operating_point": selected is not None,
    }
    passed = all(checks.values())
    final_train_rates = None
    artifact_sha256 = None
    if passed:
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-6)
        model = LogisticRegression(
            C=selected["c"],
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        ).fit((x - mean) / scale, y)
        probability = model.predict_proba((x - mean) / scale)[:, 1]
        final_train_rates = rates(
            y, probability, kind, route, selected["threshold"]
        )
        np.savez_compressed(
            args.output,
            feature_mean=mean.astype(np.float32),
            feature_scale=scale.astype(np.float32),
            coefficient=model.coef_[0].astype(np.float32),
            intercept=np.asarray(model.intercept_[0], dtype=np.float32),
            threshold=np.asarray(selected["threshold"], dtype=np.float32),
            c=np.asarray(selected["c"], dtype=np.float32),
            feature_version=np.asarray(FEATURE_VERSION),
        )
        artifact_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()

    report = {
        "format": "strict-track2-v389-public-recursive-phase-training-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": episodes,
        "phase_onsets": onset,
        "examples": int(len(y)),
        "positives": int(y.sum()),
        "route_examples": int(route.sum()),
        "feature_dimension": int(x.shape[1]),
        "teacher_examples": teacher_count,
        "recursive_generation": generation_reports,
        "selection": selected,
        "final_train_rates": final_train_rates,
        "cv_rows": cv_rows,
        "checks": checks,
        "passed": passed,
        "artifact_sha256": artifact_sha256,
        "guards": {
            "public_train_only": True,
            "labels_use_frozen_public_demo_phase_only": True,
            "runtime_features_are_request_rgb_and_actions_only": True,
            "policy_or_simulator_outcomes_read": False,
            "public_holdout_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "examples": report["examples"],
                "feature_dimension": report["feature_dimension"],
                "recursive_generation": generation_reports,
                "selection": selected,
                "checks": checks,
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
