#!/usr/bin/env python3
"""Prepare the v456 200-context selection from public actions only.

This file is staging only.  It refuses to inspect the public split or any HDF5
file until a v457 reconciliation receipt and the pinned v455 pilot data digest
have both passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SEED = 1609
EXPECTED_V457_RECEIPT_SHA256 = "790b8954c659c8038e778eecfba6cebdb84f8e5a477c2956d2755e58c7cf24d4"
EXPECTED_V457_PREREGISTRATION_SHA256 = "a272626c5e9022a9808088ce5458e249e429802f9b3189ddb24f2e93283562eb"
EXPECTED_CONTRACT_SHA256 = "d23af656e4069ba0c09f61e85ef8ce108e433f53bad28f2e2e07cb9f4f4c608b"
EXPECTED_SCIPY_VERSION = "1.15.3"
EXPECTED_HIGHS_VERSION = "1.8.0"
BRANCHES = (
    "factual",
    "no_transport",
    "scale_0p4",
    "scale_1p25",
    "reverse_direction_0p4",
    "factual_duplicate",
)
FOLDS = (
    ((28, 37, 49), {28: 14, 37: 13, 49: 13}),
    ((15, 20, 42), {15: 13, 20: 13, 42: 14}),
    ((25, 40, 46), {25: 13, 40: 14, 46: 13}),
    ((12, 33, 47), {12: 14, 33: 13, 47: 13}),
    ((30, 32, 44), {30: 13, 32: 13, 44: 14}),
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def reconciliation_gate(path: Path, preregistration_path: Path, pilot_dataset: Path, collector: Path) -> tuple[dict, dict]:
    if file_sha256(path) != EXPECTED_V457_RECEIPT_SHA256 or file_sha256(preregistration_path) != EXPECTED_V457_PREREGISTRATION_SHA256:
        raise RuntimeError("v457 receipt/preregistration immutable SHA mismatch")
    receipt = json.loads(path.read_text())
    preregistration = json.loads(preregistration_path.read_text())
    fmt = str(receipt.get("format", ""))
    if fmt != "strict-track2-v457-v455-immutable-reconciliation-receipt-v1":
        raise RuntimeError(f"unexpected v457 reconciliation format: {fmt!r}")
    if receipt.get("passed") is not True or receipt.get("endpoint_parent_data_authorized") is not True:
        raise RuntimeError("v457 reconciliation did not pass")
    if not all(receipt.get("checks", {}).values()) or not all(receipt.get("hash_checks", {}).values()):
        raise RuntimeError("v457 reconciliation contains a failed immutable check")
    if preregistration.get("format") != "strict-track2-v457-v455-immutable-reconciliation-preregistration-v1":
        raise RuntimeError("bad v457 preregistration format")
    immutable = preregistration["immutable_files"]
    for key, record in immutable.items():
        declared_path = Path(record["path"])
        if not declared_path.is_file() or file_sha256(declared_path) != record["sha256"]:
            raise RuntimeError(f"v457 immutable source closure failed: {key}")
    expected_files = {
        "v455_generation_report": pilot_dataset / "generation_report.json",
        "v455_generator": collector,
    }
    for key, actual_path in expected_files.items():
        if immutable[key]["sha256"] != file_sha256(actual_path):
            raise RuntimeError(f"v457 immutable file mismatch: {key}")
    immutable_npz = {row["name"]: row["sha256"] for row in preregistration["immutable_npz"]}
    observed_npz = {path.name: file_sha256(path) for path in sorted(pilot_dataset.glob("episode*.npz"))}
    if len(observed_npz) != 4 or observed_npz != immutable_npz:
        raise RuntimeError("v455 immutable endpoint NPZ SHA binding failed")
    report = json.loads((pilot_dataset / "generation_report.json").read_text())
    if report.get("passed") is not True or len(report.get("rows", [])) != 4:
        raise RuntimeError("v455 generation report did not pass exactly four contexts")
    return receipt, preregistration


def branch_actions(history: np.ndarray, future: np.ndarray) -> dict[str, np.ndarray]:
    anchor = history[-1, 7:13]
    delta = future[:, 7:13] - anchor
    result = {"factual": future.copy()}
    for name, scale in (
        ("no_transport", 0.0),
        ("scale_0p4", 0.4),
        ("scale_1p25", 1.25),
        ("reverse_direction_0p4", -0.4),
    ):
        value = future.copy()
        value[:, 7:13] = anchor + scale * delta
        result[name] = value
    result["factual_duplicate"] = future.copy()
    return result


def stable_key(episode: int, start: int) -> tuple[str, int]:
    value = hashlib.sha256(f"{SEED}:{episode}:{start}".encode()).hexdigest()
    return value, int(value, 16)


def add_constraint(rows, cols, values, lower, upper, indices, coefficients, lo, hi):
    row = len(lower)
    for index, coefficient in zip(indices, coefficients):
        rows.append(row)
        cols.append(index)
        values.append(float(coefficient))
    lower.append(float(lo))
    upper.append(float(hi))


def run_exact_milp(cost: np.ndarray, rows, cols, values, lower, upper, label: str, allow_failure: bool = False):
    # scipy is imported only after the receipt and action-only candidate set exist.
    import scipy
    import scipy.optimize._highspy._core as highspy_core
    import warnings
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    highs_version = f"{highspy_core.HIGHS_VERSION_MAJOR}.{highspy_core.HIGHS_VERSION_MINOR}.{highspy_core.HIGHS_VERSION_PATCH}"
    if scipy.__version__ != EXPECTED_SCIPY_VERSION or highs_version != EXPECTED_HIGHS_VERSION:
        raise RuntimeError(f"pinned scipy drift: {scipy.__version__} != {EXPECTED_SCIPY_VERSION}")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), len(cost))).tocsr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = milp(
            c=cost.astype(np.float64),
            integrality=np.ones(len(cost), dtype=np.int8),
            bounds=Bounds(np.zeros(len(cost)), np.ones(len(cost))),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
            options={"time_limit": 300.0, "mip_rel_gap": 0.0, "presolve": True, "threads": 1, "parallel": False, "random_seed": SEED},
        )
    raw_warning_text = [str(item.message) for item in caught]
    canonical_warnings = []
    for warning_text in raw_warning_text:
        if warning_text.startswith("Unrecognized options detected:") and "passed to HiGHS verbatim" in warning_text:
            canonical_warnings.append({
                "code": "unrecognized_options_forwarded_to_highs",
                "options": ["parallel", "random_seed", "threads"],
            })
        else:
            raise RuntimeError(f"unexpected noncanonical solver warning: {warning_text}")
    if raw_warning_text:
        print(f"{label} raw solver warnings: {raw_warning_text}", file=sys.stderr)
    meta = {
        "label": label,
        "scipy_version": scipy.__version__,
        "highs_version": highs_version,
        "threads": 1,
        "parallel": False,
        "random_seed": SEED,
        "presolve": True,
        "mip_rel_gap": 0.0,
        "success": bool(result.success and result.x is not None),
        "objective": None if result.fun is None else float(result.fun),
        "status": int(result.status),
        "message": str(result.message),
        "solver_warnings": canonical_warnings,
    }
    if not meta["success"]:
        if allow_failure:
            return None, meta
        raise RuntimeError(f"v456 {label} MILP failed: {result.message}; warnings={canonical_warnings}")
    return result.x, meta


def solve_exact(candidates: list[dict]) -> tuple[list[dict], dict]:
    episodes = sorted(episode for fold_episodes, _ in FOLDS for episode in fold_episodes)
    episode_indices = {}
    for episode in episodes:
        episode_indices[episode] = [i for i, row in enumerate(candidates) if row["episode"] == episode]

    def base_constraints(phase_bounds=(6, 10), motion_bounds=(8, 12), episode_bounds=(0, 40)):
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for fold in range(5):
            fold_indices = [i for i, row in enumerate(candidates) if row["fold"] == fold]
            add_constraint(rows, cols, values, lower, upper, fold_indices, [1] * len(fold_indices), 40, 40)
            for phase_bin in range(5):
                indices = [i for i, row in enumerate(candidates) if row["fold"] == fold and row["phase_bin"] == phase_bin]
                add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), *phase_bounds)
            for motion_bin in range(4):
                indices = [i for i, row in enumerate(candidates) if row["fold"] == fold and row["motion_bin"] == motion_bin]
                add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), *motion_bounds)
        for episode in episodes:
            indices = episode_indices[episode]
            add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), *episode_bounds)
        for episode in episodes:
            by_start = sorted(episode_indices[episode], key=lambda index: candidates[index]["start"])
            for left, right in zip(by_start, by_start[1:]):
                if candidates[right]["start"] - candidates[left]["start"] < 2:
                    add_constraint(rows, cols, values, lower, upper, [left, right], [1, 1], -np.inf, 1)
        return rows, cols, values, lower, upper

    exact_trial = base_constraints((8, 8), (10, 10), (0, 40))
    exact_solution, exact_meta = run_exact_milp(
        np.zeros(len(candidates)), *exact_trial, "exact_phase8_motion10_feasibility", allow_failure=True
    )
    if exact_solution is None:
        raise RuntimeError("v461 exact phase8/motion10 became infeasible; relaxed margins are not authorized")
    chosen_phase_bounds, chosen_motion_bounds = (8, 8), (10, 10)
    margin_resolution = "exact phase8/motion10 is feasible after eligibility-before-binning and is the tightest interval; no relaxation used"

    lower_attempts = []
    tight_lower = None
    for candidate_lower in range(13, -1, -1):
        trial = base_constraints(chosen_phase_bounds, chosen_motion_bounds, (candidate_lower, 40))
        solution, attempt = run_exact_milp(
            np.zeros(len(candidates)), *trial, f"episode_uniform_lower_{candidate_lower}", allow_failure=True
        )
        lower_attempts.append(attempt)
        if solution is not None:
            tight_lower = candidate_lower
            break
    if tight_lower is None:
        raise RuntimeError("v461 no feasible uniform episode lower bound")

    upper_attempts = []
    tight_upper = None
    for candidate_upper in range(14, 41):
        trial = base_constraints(chosen_phase_bounds, chosen_motion_bounds, (tight_lower, candidate_upper))
        solution, attempt = run_exact_milp(
            np.zeros(len(candidates)), *trial, f"episode_uniform_upper_{candidate_upper}", allow_failure=True
        )
        upper_attempts.append(attempt)
        if solution is not None:
            tight_upper = candidate_upper
            break
    if tight_upper is None:
        raise RuntimeError("v461 no feasible uniform episode upper bound")

    rows, cols, values, lower, upper = base_constraints(chosen_phase_bounds, chosen_motion_bounds, (tight_lower, tight_upper))

    def with_range(max_range: int):
        tr, tc, tv, tl, tu = list(rows), list(cols), list(values), list(lower), list(upper)
        for first_position, first in enumerate(episodes):
            for second in episodes[first_position + 1 :]:
                a, b = episode_indices[first], episode_indices[second]
                add_constraint(tr, tc, tv, tl, tu, a + b, [1] * len(a) + [-1] * len(b), -max_range, max_range)
        return tr, tc, tv, tl, tu

    attempts = []
    lo, hi = 1, tight_upper - tight_lower
    while lo < hi:
        middle = (lo + hi) // 2
        trial = with_range(middle)
        solution, attempt = run_exact_milp(np.zeros(len(candidates)), *trial, f"episode_range_feasibility_{middle}", allow_failure=True)
        attempts.append(attempt)
        if solution is None:
            lo = middle + 1
        else:
            hi = middle
    minimum_range = lo
    trial = with_range(minimum_range)
    solution, final_meta = run_exact_milp(
        np.asarray([row["cost_rank"] + 1 for row in candidates]),
        *trial,
        f"selection_rank_at_episode_range_{minimum_range}",
    )
    meta = {
        "exact_margin_feasibility": exact_meta,
        "margin_resolution": margin_resolution,
        "phase_bounds": list(chosen_phase_bounds),
        "motion_bounds": list(chosen_motion_bounds),
        "episode_uniform_lower_search": lower_attempts,
        "episode_uniform_upper_search": upper_attempts,
        "episode_bounds": [tight_lower, tight_upper],
        "minimum_episode_range": minimum_range,
        "range_feasibility_attempts": attempts,
        "rank_objective_solver": final_meta,
    }
    selected = [row for row, take in zip(candidates, solution) if take > 0.5]
    if len(selected) != 200:
        raise RuntimeError(f"v456 expected 200 selected contexts, got {len(selected)}")
    return selected, meta


def solve_batches(selected: list[dict]) -> dict:
    variables = [(index, batch) for index in range(len(selected)) for batch in range(10)]
    index_of = {pair: index for index, pair in enumerate(variables)}
    keys = []
    for context_index, batch in variables:
        row = selected[context_index]
        digest = hashlib.sha256(f"{SEED}:batch:{batch}:{row['episode']}:{row['start']}".encode()).hexdigest()
        keys.append((int(digest, 16), row["episode"], row["start"], batch))
    rank_by_variable = {variable_index: rank for rank, variable_index in enumerate(sorted(range(len(variables)), key=lambda i: keys[i]))}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for context_index in range(len(selected)):
        indices = [index_of[(context_index, batch)] for batch in range(10)]
        add_constraint(rows, cols, values, lower, upper, indices, [1] * 10, 1, 1)
    for batch in range(10):
        all_indices = [index_of[(context_index, batch)] for context_index in range(len(selected))]
        add_constraint(rows, cols, values, lower, upper, all_indices, [1] * len(all_indices), 20, 20)
        for fold in range(5):
            indices = [index_of[(i, batch)] for i, row in enumerate(selected) if row["fold"] == fold]
            add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), 4, 4)
        for phase_bin in range(5):
            indices = [index_of[(i, batch)] for i, row in enumerate(selected) if row["phase_bin"] == phase_bin]
            add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), 3, 5)
        for motion_bin in range(4):
            indices = [index_of[(i, batch)] for i, row in enumerate(selected) if row["motion_bin"] == motion_bin]
            add_constraint(rows, cols, values, lower, upper, indices, [1] * len(indices), 4, 6)
    costs = np.asarray([rank_by_variable[index] + 1 for index in range(len(variables))])
    solution, meta = run_exact_milp(costs, rows, cols, values, lower, upper, "batch_assignment")
    for variable_index, take in enumerate(solution):
        if take > 0.5:
            context_index, batch = variables[variable_index]
            if "batch_id" in selected[context_index]:
                raise RuntimeError("batch MILP assigned a context twice")
            selected[context_index]["batch_id"] = batch
    if any("batch_id" not in row for row in selected):
        raise RuntimeError("batch MILP left contexts unassigned")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("contract", "reconciliation", "reconciliation-preregistration", "pilot-dataset", "v455-collector", "split", "dataset", "seed-file", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()

    # This must remain the first data access beyond the reconciliation file itself.
    reconciliation, reconciliation_preregistration = reconciliation_gate(
        args.reconciliation, args.reconciliation_preregistration, args.pilot_dataset, args.v455_collector
    )
    contract = json.loads(args.contract.read_text())
    if file_sha256(args.contract) != EXPECTED_CONTRACT_SHA256 or contract.get("format") != "strict-track2-v456-endpoint-residual-parent-frozen-contract-v1" or contract.get("status") != "READY_STAGING_V457_BOUND_NOT_DEPLOYED":
        raise RuntimeError("v456 frozen contract SHA/format/status mismatch")

    split = json.loads(args.split.read_text())
    train = sorted(map(int, split["train_episodes"]))
    arms = {int(key): value for key, value in split["arm_by_episode"].items()}
    instructions = {int(key): value for key, value in split["episode_to_instruction"].items()}
    right = sorted(episode for episode in train if arms[episode] == "right")
    expected_right = sorted(episode for episodes, _ in FOLDS for episode in episodes)
    if len(train) != 40 or right != expected_right:
        raise RuntimeError(f"public train40/right15 split drift: right={right}")
    seeds = list(map(int, args.seed_file.read_text().split()))
    if len(seeds) != 50:
        raise RuntimeError("expected exactly 50 public dataset seeds")

    actions: dict[int, np.ndarray] = {}
    for episode in train:
        path = args.dataset / "data" / f"episode{episode}.hdf5"
        with h5py.File(path, "r") as handle:
            actions[episode] = np.asarray(handle["joint_action/vector"], dtype=np.float32)
    stacked = np.concatenate([actions[episode] for episode in train])
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    fold_by_episode = {episode: fold for fold, (episodes, _) in enumerate(FOLDS) for episode in episodes}

    candidates: list[dict] = []
    eligible_stats: list[dict] = []
    for episode in right:
        source = actions[episode]
        closed = np.flatnonzero(source[:, 13] < 0.5)
        if not len(closed):
            raise RuntimeError(f"right episode {episode} has no action-only close event")
        first_close = int(closed[0])
        for start in range(len(source) - 11):
            history = source[start : start + 4]
            future = source[start + 4 : start + 12]
            if not (history[-1, 13] < 0.5 and np.all(future[:, 13] < 0.5)):
                continue
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                continue
            anchor = history[-1, 7:13]
            right_path = np.vstack((anchor, future[:, 7:13]))
            motion = float(np.linalg.norm(np.diff(right_path, axis=0), axis=1).sum())
            phase = float((start + 4 - first_close) / max(1, len(source) - 1 - first_close))
            eligible_sha, eligible_uint = stable_key(episode, start)
            branches = branch_actions(history, future)
            if any(not np.isfinite(value).all() or np.any(value < lower) or np.any(value > upper) for value in branches.values()):
                continue
            endpoint_diffs = {name: float(np.linalg.norm(value[-1, 7:13] - future[-1, 7:13])) for name, value in branches.items() if name not in ("factual", "factual_duplicate")}
            path_diffs = {name: float(np.linalg.norm(value[:, 7:13] - future[:, 7:13])) for name, value in branches.items() if name not in ("factual", "factual_duplicate")}
            if min(endpoint_diffs.values()) < 0.01 or min(path_diffs.values()) < 0.01:
                continue
            eligible_stats.append({"episode": episode, "start": start, "fold": fold_by_episode[episode], "phase": phase, "motion": motion, "cost_sha256": eligible_sha, "cost_uint256": eligible_uint})
            cost_sha256, cost_uint256 = stable_key(episode, start)
            candidates.append({
                "episode": episode,
                "start": start,
                "fold": fold_by_episode[episode],
                "phase": phase,
                "motion": motion,
                "endpoint_diffs": endpoint_diffs,
                "path_diffs": path_diffs,
                "history_action_sha256": array_sha256(history),
                "branch_action_sha256": {name: array_sha256(value) for name, value in branches.items()},
                "cost_sha256": cost_sha256,
                "cost_uint256": cost_uint256,
            })
    bin_by_context = {}
    fold_bin_coverage = {}
    for fold in range(5):
        fold_rows = [row for row in eligible_stats if row["fold"] == fold]
        phase_sorted = sorted(fold_rows, key=lambda row: (row["phase"], row["cost_uint256"], row["episode"], row["start"]))
        for rank, row in enumerate(phase_sorted):
            row["phase_bin"] = min(4, 5 * rank // len(phase_sorted))
        for phase_bin in range(5):
            phase_rows = [row for row in fold_rows if row["phase_bin"] == phase_bin]
            motion_sorted = sorted(phase_rows, key=lambda row: (row["motion"], row["cost_uint256"], row["episode"], row["start"]))
            for rank, row in enumerate(motion_sorted):
                row["motion_bin"] = min(3, 4 * rank // len(motion_sorted))
        for row in fold_rows:
            bin_by_context[(row["episode"], row["start"])] = (row["phase_bin"], row["motion_bin"])
        fold_bin_coverage[str(fold)] = {
            "eligible": len(fold_rows),
            "phase": [sum(row["phase_bin"] == value for row in fold_rows) for value in range(5)],
            "conditional_motion": [sum(row["motion_bin"] == value for row in fold_rows) for value in range(4)],
        }
    for row in candidates:
        row["phase_bin"], row["motion_bin"] = bin_by_context[(row["episode"], row["start"])]

    for rank, row in enumerate(sorted(candidates, key=lambda value: (value["cost_uint256"], value["episode"], value["start"]))):
        row["cost_rank"] = rank

    selected, selection_solver = solve_exact(candidates)
    if len({(row["episode"], row["start"]) for row in selected}) != 200:
        raise RuntimeError("post-MILP unique episode/start audit failed")
    if any(sum(row["fold"] == fold for row in selected) != 40 for fold in range(5)):
        raise RuntimeError("post-MILP fold40 audit failed")
    phase_lower, phase_upper = selection_solver["phase_bounds"]
    motion_lower, motion_upper = selection_solver["motion_bounds"]
    if any(not phase_lower <= sum(row["fold"] == fold and row["phase_bin"] == value for row in selected) <= phase_upper for fold in range(5) for value in range(5)):
        raise RuntimeError("post-MILP fold-phase bounds audit failed")
    if any(not motion_lower <= sum(row["fold"] == fold and row["motion_bin"] == value for row in selected) <= motion_upper for fold in range(5) for value in range(4)):
        raise RuntimeError("post-MILP fold-motion bounds audit failed")
    episode_counts = {episode: sum(row["episode"] == episode for row in selected) for episode in right}
    episode_lower, episode_upper = selection_solver["episode_bounds"]
    if min(episode_counts.values()) < episode_lower or max(episode_counts.values()) > episode_upper or max(episode_counts.values()) - min(episode_counts.values()) != selection_solver["minimum_episode_range"]:
        raise RuntimeError("post-MILP episode bounds/range audit failed")
    batch_solver = solve_batches(selected)
    for batch_id in range(10):
        rows = [row for row in selected if row["batch_id"] == batch_id]
        if len(rows) != 20 or any(sum(row["fold"] == value for row in rows) != 4 for value in range(5)) or any(not 3 <= sum(row["phase_bin"] == value for row in rows) <= 5 for value in range(5)) or any(not 4 <= sum(row["motion_bin"] == value for row in rows) <= 6 for value in range(4)):
            raise RuntimeError(f"post-batch-MILP margin audit failed batch={batch_id}")

    contexts = []
    hdf5_sha_cache: dict[Path, str] = {}
    for row in sorted(selected, key=lambda value: (value["batch_id"], value["fold"], value["phase_bin"], value["motion_bin"], value["episode"], value["start"])):
        episode = row["episode"]
        source_hdf5 = (args.dataset / "data" / f"episode{episode}.hdf5").resolve()
        if source_hdf5 not in hdf5_sha_cache:
            hdf5_sha_cache[source_hdf5] = file_sha256(source_hdf5)
        contexts.append({
            **{key: value for key, value in row.items() if key not in ("cost_uint256",)},
            "dataset_seed": seeds[episode],
            "instruction": instructions[episode],
            "source_hdf5": str(source_hdf5),
            "source_hdf5_sha256": hdf5_sha_cache[source_hdf5],
        })

    payload = {
        "format": "strict-track2-v461-endpoint200-preregistration-v1",
        "classification": "public-train action-only endpoint collection preregistration; no model, policy, RL, reward, hidden/final or submission authority",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "status": "selected_action_only_not_collected",
        "v457_reconciliation_sha256": file_sha256(args.reconciliation),
        "v457_preregistration_sha256": file_sha256(args.reconciliation_preregistration),
        "frozen_contract_path": str(args.contract.resolve()),
        "frozen_contract_sha256": file_sha256(args.contract),
        "v455_generation_report_sha256": reconciliation_preregistration["immutable_files"]["v455_generation_report"]["sha256"],
        "v455_immutable_npz_sha256": {row["name"]: row["sha256"] for row in reconciliation_preregistration["immutable_npz"]},
        "source_sha256": {"split": file_sha256(args.split), "seed_file": file_sha256(args.seed_file)},
        "branches": list(BRANCHES),
        "action_difference_gates": {"right6_endpoint_l2_min": 0.01, "right6_full_path_tensor_l2_min": 0.01, "applied_before_binning": True},
        "per_dim_action_lower": lower.tolist(),
        "per_dim_action_upper": upper.tolist(),
        "fold_local_eligible_bin_coverage": fold_bin_coverage,
        "selected_coverage": {
            "episode_counts": {str(key): value for key, value in episode_counts.items()},
            "episode_count_min": min(episode_counts.values()),
            "episode_count_max": max(episode_counts.values()),
            "episode_count_range": max(episode_counts.values()) - min(episode_counts.values()),
            "fold_counts": [sum(row["fold"] == value for row in selected) for value in range(5)],
            "fold_phase_counts": {str(fold): [sum(row["fold"] == fold and row["phase_bin"] == value for row in selected) for value in range(5)] for fold in range(5)},
            "fold_motion_counts": {str(fold): [sum(row["fold"] == fold and row["motion_bin"] == value for row in selected) for value in range(4)] for fold in range(5)},
        },
        "assignment_objective": {
            "definition": "two-stage exact MILP: minimum episode-count range then full-SHA rank sum; independent exact batch-margin MILP",
            "selected_rank_sum": sum(row["cost_rank"] for row in selected),
            "selection_solver": selection_solver,
            "batch_assignment_solver": batch_solver,
            "parallel": False,
        },
        "contexts": contexts,
        "batches": [{"batch_id": batch_id, "contexts": [index for index, row in enumerate(contexts) if row["batch_id"] == batch_id]} for batch_id in range(10)],
        "expected_generation_outputs": {
            "generation_manifest": "generation_manifest.json",
            "complete_batch_directories": [f"batch_{batch_id:03d}" for batch_id in range(10)],
            "per_batch": "rows/<episode_start> atomic directories, each containing endpoint.npz plus receipt.json, and one batch_report.json",
            "final_report": "generation_report.json",
            "partial_directory_suffix": ".partial",
            "worker_directory_suffix": ".work",
            "forensic_root": "_forensic_uncommitted/batch_XXX/<timestamp-hash>/quarantine_receipt.json",
            "committed_batch_exact_tree": "rows/ plus batch_report.json only",
            "generator_modes": ["--batch-id 0..9", "--finalize"],
        },
        "guards": {
            "selection_inputs_action_only": True,
            "rgb_reward_success_outcome_read": False,
            "exact_contexts": 200,
            "exact_batches": 10,
            "contexts_per_batch": 20,
            "all_selected_rows_must_be_retained": True,
            "simulator_started": False,
            "policy_updates": 0,
            "rl_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite selection receipt: {args.output}")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
