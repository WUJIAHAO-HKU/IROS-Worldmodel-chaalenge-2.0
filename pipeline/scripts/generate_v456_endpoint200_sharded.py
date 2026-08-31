#!/usr/bin/env python3
"""Recoverable 10x20-context endpoint collector for the v460 staging.

The implementation uses the frozen HDF5-only v460 collect_one function. Each worker
writes into a private directory; the parent fsyncs and atomically promotes the
NPZ and its receipt.  A batch becomes visible only after all twenty rows pass
the fixed short gate.  This script must not be run before v457 passes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

EXPECTED_V457_RECEIPT_SHA256 = "790b8954c659c8038e778eecfba6cebdb84f8e5a477c2956d2755e58c7cf24d4"
EXPECTED_V457_PREREGISTRATION_SHA256 = "a272626c5e9022a9808088ce5458e249e429802f9b3189ddb24f2e93283562eb"
EXPECTED_CONTRACT_SHA256 = "d23af656e4069ba0c09f61e85ef8ce108e433f53bad28f2e2e07cb9f4f4c608b"
EXPECTED_V460_CONTRACT_SHA256 = "925aa0a6843b268725f048cae54f68b4883c781c1d655ed33e85f6ebcbbbf923"
EXPECTED_ENDPOINT_COLLECTOR_SHA256 = "6c64e4a42f273f43b9e64523bf12e4dc5f24618d5673848db6c1582b5547e846"
VARIANTS = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4", "factual_duplicate")
TRANSPORT = VARIANTS[1:5]
EXPECTED_PUBLIC_SPLIT_SHA256 = "8c1ecfb7127b2d30862420d311d9b3e0f7d1337bec6b4c214cb9db372fc78a87"
EXPECTED_PUBLIC_SEED_FILE_SHA256 = "2a9302903b92b6828ad6cb99c0b04af50fcbbf176cdd827ad34abeced04ea419"
EXPECTED_RIGHT_EPISODES = (12, 15, 20, 25, 28, 30, 32, 33, 37, 40, 42, 44, 46, 47, 49)
FORBIDDEN_ENDPOINT_FIELDS = {"source_window", "source_window_sha256", "target_frames", "public_factual_endpoint_sha256", "factual_public_endpoint_rgb_mae", "factual_public_endpoint_rgb_mae_diagnostic", "reward", "success", "done", "termination", "truncation", "outcome", "policy_action"}
EXPECTED_ENDPOINT_KEYS = {"episode", "dataset_seed", "start", "variants", "instruction", "branch_context_rgb", "branch_context_state", "branch_context_pose", "branch_context_bottle_position", "history_actions", "future_actions", "endpoint_rgb", "endpoint_state", "endpoint_bottle_position", "context_rgb_sha256", "context_state_sha256", "executed_action_sha256"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def gpu_mib() -> int:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return sum(int(line.strip()) for line in text.splitlines() if line.strip())
    except Exception:
        return -1


def receipt_declared_sha(receipt: dict, key: str):
    direct = receipt.get(key)
    if direct is not None:
        return direct
    return receipt.get("evidence_sha256", {}).get(key)


def validate_spec_sources(spec: dict) -> None:
    source_hdf5 = Path(spec["source_hdf5"])
    if file_sha256(source_hdf5) != spec["source_hdf5_sha256"]:
        raise RuntimeError(f"authoritative source SHA drift episode={spec['episode']} start={spec['start']}")
    with h5py.File(source_hdf5, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"], dtype=np.float32)
    start = int(spec["start"])
    history = actions[start : start + 4]
    future = actions[start + 4 : start + 12]
    if array_sha256(history) != spec["history_action_sha256"]:
        raise RuntimeError(f"source history-action drift episode={spec['episode']} start={start}")
    anchor = history[-1, 7:13]
    delta = future[:, 7:13] - anchor
    branches = {"factual": future.copy()}
    for name, scale in (("no_transport", 0.0), ("scale_0p4", 0.4), ("scale_1p25", 1.25), ("reverse_direction_0p4", -0.4)):
        value = future.copy()
        value[:, 7:13] = anchor + scale * delta
        branches[name] = value
    branches["factual_duplicate"] = future.copy()
    observed = {name: array_sha256(branches[name]) for name in VARIANTS}
    if observed != spec["branch_action_sha256"]:
        raise RuntimeError(f"source future-action drift episode={spec['episode']} start={start}")
    endpoint_diffs = {name: float(np.linalg.norm(value[-1, 7:13] - future[-1, 7:13])) for name, value in branches.items() if name not in ("factual", "factual_duplicate")}
    path_diffs = {name: float(np.linalg.norm(value[:, 7:13] - future[:, 7:13])) for name, value in branches.items() if name not in ("factual", "factual_duplicate")}
    if endpoint_diffs != spec.get("endpoint_diffs") or path_diffs != spec.get("path_diffs") or min(endpoint_diffs.values()) < 0.01 or min(path_diffs.values()) < 0.01:
        raise RuntimeError(f"v461 action-difference selection gate drift episode={spec['episode']} start={start}")


def preflight(args) -> tuple[dict, dict, dict]:
    if file_sha256(args.reconciliation) != EXPECTED_V457_RECEIPT_SHA256 or file_sha256(args.reconciliation_preregistration) != EXPECTED_V457_PREREGISTRATION_SHA256:
        raise RuntimeError("v457 immutable receipt/preregistration SHA mismatch")
    reconciliation = json.loads(args.reconciliation.read_text())
    reconciliation_pre = json.loads(args.reconciliation_preregistration.read_text())
    fmt = str(reconciliation.get("format", ""))
    if not (fmt == "strict-track2-v457-v455-immutable-reconciliation-receipt-v1" and reconciliation.get("passed") is True and reconciliation.get("endpoint_parent_data_authorized") is True):
        raise RuntimeError("v457 reconciliation has not passed")
    if not all(reconciliation.get("checks", {}).values()) or not all(reconciliation.get("hash_checks", {}).values()):
        raise RuntimeError("v457 reconciliation has failed checks")
    if reconciliation_pre.get("format") != "strict-track2-v457-v455-immutable-reconciliation-preregistration-v1":
        raise RuntimeError("bad v457 preregistration format")
    immutable = reconciliation_pre["immutable_files"]
    for key, record in immutable.items():
        declared_path = Path(record["path"])
        if not declared_path.is_file() or file_sha256(declared_path) != record["sha256"]:
            raise RuntimeError(f"v457 immutable source closure failed: {key}")
    declared_collector = immutable["v455_generator"]["sha256"]
    actual_collector = file_sha256(args.v455_collector)
    if declared_collector != actual_collector:
        raise RuntimeError(f"v455 collector binding failed actual={actual_collector} declared={declared_collector}")
    if immutable["v455_generation_report"]["sha256"] != file_sha256(args.pilot_dataset / "generation_report.json"):
        raise RuntimeError("v455 generation-report binding failed")
    expected_npz = {row["name"]: row["sha256"] for row in reconciliation_pre["immutable_npz"]}
    observed_npz = {path.name: file_sha256(path) for path in sorted(args.pilot_dataset.glob("episode*.npz"))}
    if len(observed_npz) != 4 or observed_npz != expected_npz:
        raise RuntimeError("v455 endpoint NPZ immutable binding failed")
    selection = json.loads(args.selection.read_text())
    if selection.get("format") != "strict-track2-v461-endpoint200-preregistration-v1":
        raise RuntimeError("bad v461 selection format")
    if selection.get("v457_reconciliation_sha256") != EXPECTED_V457_RECEIPT_SHA256 or selection.get("v457_preregistration_sha256") != EXPECTED_V457_PREREGISTRATION_SHA256 or selection.get("frozen_contract_sha256") != EXPECTED_CONTRACT_SHA256 or selection.get("v455_generation_report_sha256") != immutable["v455_generation_report"]["sha256"] or selection.get("v455_immutable_npz_sha256") != expected_npz or len(selection.get("contexts", [])) != 200:
        raise RuntimeError("v456 selection binding/count failed")
    contract = json.loads(args.contract.read_text())
    if file_sha256(args.contract) != EXPECTED_CONTRACT_SHA256 or contract.get("format") != "strict-track2-v456-endpoint-residual-parent-frozen-contract-v1":
        raise RuntimeError("bad v456 contract format")
    if contract.get("status") != "READY_STAGING_V457_BOUND_NOT_DEPLOYED":
        raise RuntimeError("unexpected v456 contract state")
    if tuple(selection.get("branches", ())) != VARIANTS:
        raise RuntimeError("v456 branch drift")
    if selection.get("action_difference_gates") != {"right6_endpoint_l2_min": 0.01, "right6_full_path_tensor_l2_min": 0.01, "applied_before_binning": True}:
        raise RuntimeError("v461 selection action-difference gate declaration drift")
    split_sha = file_sha256(args.split)
    seed_file_sha = file_sha256(args.seed_file)
    if split_sha != EXPECTED_PUBLIC_SPLIT_SHA256 or seed_file_sha != EXPECTED_PUBLIC_SEED_FILE_SHA256:
        raise RuntimeError("public split/seed immutable SHA mismatch")
    if selection.get("source_sha256") != {"split": split_sha, "seed_file": seed_file_sha}:
        raise RuntimeError("selection public split/seed binding failed")
    split = json.loads(args.split.read_text())
    train = sorted(map(int, split.get("train_episodes", [])))
    arms = {int(key): value for key, value in split.get("arm_by_episode", {}).items()}
    instructions = {int(key): value for key, value in split.get("episode_to_instruction", {}).items()}
    right = tuple(sorted(episode for episode in train if arms.get(episode) == "right"))
    seeds = list(map(int, args.seed_file.read_text().split()))
    if len(train) != 40 or right != EXPECTED_RIGHT_EPISODES or len(seeds) != 50:
        raise RuntimeError("public train40/right15/seed50 semantic drift")
    contexts = selection["contexts"]
    if tuple(sorted({int(row["episode"]) for row in contexts})) != EXPECTED_RIGHT_EPISODES:
        raise RuntimeError("selection episode set is not exact public right15")
    dataset_data = (args.dataset / "data").resolve()
    for row in contexts:
        episode = int(row["episode"])
        expected_hdf5 = (dataset_data / f"episode{episode}.hdf5").resolve()
        if (
            arms.get(episode) != "right"
            or row.get("instruction") != instructions.get(episode)
            or int(row.get("dataset_seed", -1)) != seeds[episode]
            or Path(row["source_hdf5"]).resolve() != expected_hdf5
            or not expected_hdf5.is_file()
        ):
            raise RuntimeError(f"selection public mapping/canonical HDF5 drift episode={episode}")
    return reconciliation, selection, contract


def validate_saved_row(partial: Path, spec: dict) -> dict | None:
    name = f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"
    row_dir = partial / "rows" / name
    if not row_dir.exists():
        return None
    npz = row_dir / "endpoint.npz"
    receipt = row_dir / "receipt.json"
    if not row_dir.is_dir() or not npz.is_file() or not receipt.is_file() or {path.name for path in row_dir.iterdir()} != {"endpoint.npz", "receipt.json"}:
        raise RuntimeError(f"incomplete atomic row pair for {name}; manual forensic review required")
    row = json.loads(receipt.read_text())
    if set(row).intersection(FORBIDDEN_ENDPOINT_FIELDS) or row.get("endpoint_only") is not True or row.get("public_window_or_endpoint_diagnostic_consumed") is not False:
        raise RuntimeError(f"forbidden/drifted endpoint receipt schema for {name}")
    if row.get("episode") != spec["episode"] or row.get("start") != spec["start"]:
        raise RuntimeError(f"row/spec mismatch for {name}")
    if row.get("npz_sha256") != file_sha256(npz):
        raise RuntimeError(f"row NPZ digest mismatch for {name}")
    if row.get("selection_action_sha256") != spec["branch_action_sha256"]:
        raise RuntimeError(f"row selection-action receipt mismatch for {name}")
    if file_sha256(Path(spec["source_hdf5"])) != spec["source_hdf5_sha256"]:
        raise RuntimeError(f"row authoritative source hash mismatch for {name}")
    with np.load(npz, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_ENDPOINT_KEYS:
            raise RuntimeError(f"non-exact endpoint NPZ schema for {name}: {sorted(payload.files)}")
        if int(payload["episode"]) != spec["episode"] or int(payload["start"]) != spec["start"]:
            raise RuntimeError(f"NPZ episode/start mismatch for {name}")
        if array_sha256(payload["history_actions"]) != spec["history_action_sha256"]:
            raise RuntimeError(f"NPZ history-action mismatch for {name}")
        futures = payload["future_actions"]
        variants = list(map(str, payload["variants"]))
        if variants != list(VARIANTS) or futures.shape != (6, 8, 14):
            raise RuntimeError(f"NPZ branch schema mismatch for {name}")
        observed = {variant: array_sha256(futures[index]) for index, variant in enumerate(variants)}
        if observed != spec["branch_action_sha256"]:
            raise RuntimeError(f"NPZ future-action binding mismatch for {name}")
        endpoint_rgb = payload["endpoint_rgb"]
        endpoint_state = payload["endpoint_state"]
        endpoint_bottle = payload["endpoint_bottle_position"]
        if payload["history_actions"].shape != (4,14) or payload["branch_context_rgb"].shape != (6,256,256,3) or payload["branch_context_rgb"].dtype != np.uint8 or endpoint_rgb.shape != (6,256,256,3) or endpoint_rgb.dtype != np.uint8 or endpoint_state.shape[0] != 6 or endpoint_bottle.shape[0] != 6:
            raise RuntimeError(f"endpoint tensor shape/dtype drift for {name}")
        duplicate = float(np.abs(endpoint_rgb[0].astype(np.float32) - endpoint_rgb[5].astype(np.float32)).mean())
        if abs(duplicate - float(row["factual_duplicate_rgb_mae"])) > 1e-6 or bool(duplicate <= 0.5) != bool(row["duplicate_passed"]):
            raise RuntimeError(f"NPZ duplicate metric mismatch for {name}")
        recomputed_effects = {}
        for index, variant in enumerate(TRANSPORT, 1):
            rgb = float(np.abs(endpoint_rgb[index].astype(np.float32) - endpoint_rgb[0].astype(np.float32)).mean())
            qpos = float(np.linalg.norm(endpoint_state[index] - endpoint_state[0]))
            bottle = float(np.linalg.norm(endpoint_bottle[index] - endpoint_bottle[0]))
            recomputed_effects[variant] = {"endpoint_rgb_mae": rgb, "endpoint_qpos_l2": qpos, "bottle_position_l2": bottle, "passed": rgb >= 1.0 and (qpos >= 0.01 or bottle >= 0.005)}
        for variant in TRANSPORT:
            for metric in ("endpoint_rgb_mae", "endpoint_qpos_l2", "bottle_position_l2"):
                if abs(recomputed_effects[variant][metric] - float(row["effects"][variant][metric])) > 1e-6:
                    raise RuntimeError(f"NPZ effect metric mismatch {name} {variant} {metric}")
            if bool(recomputed_effects[variant]["passed"]) != bool(row["effects"][variant]["passed"]):
                raise RuntimeError(f"NPZ effect pass mismatch {name} {variant}")
        if sum(value["passed"] for value in recomputed_effects.values()) != row["effect_count"]:
            raise RuntimeError(f"NPZ effect-count mismatch for {name}")
    return row


def promote_worker_result(partial: Path, work: Path, spec: dict, row: dict) -> dict:
    name = f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"
    source = work / f"{name}.npz"
    rows_dir = partial / "rows"
    rows_dir.mkdir(exist_ok=True)
    target_dir = rows_dir / name
    if not source.is_file() or target_dir.exists():
        raise RuntimeError(f"bad worker result paths for {name}")
    fsync_path(source)
    endpoint = work / "endpoint.npz"
    os.replace(source, endpoint)
    saved = {**row, "npz_sha256": file_sha256(endpoint), "selection_action_sha256": spec["branch_action_sha256"]}
    atomic_json(work / "receipt.json", saved)
    fsync_dir(work)
    os.replace(work, target_dir)
    fsync_dir(rows_dir)
    return validate_saved_row(partial, spec)


def batch_short_gate(batch_id: int, partial: Path, specs: list[dict], rows: list[dict], peak: int, wall: float) -> dict:
    effect_counts = {name: 0 for name in TRANSPORT}
    for row in rows:
        for name in TRANSPORT:
            effect_counts[name] += int(row["effects"][name]["passed"])
    npz_files = sorted(partial.glob("rows/*/endpoint.npz"))
    output_bytes = sum(path.stat().st_size for path in npz_files)
    forensic_batch = partial.parent / "_forensic_uncommitted" / f"batch_{batch_id:03d}"
    quarantine_receipts = {
        str(path.relative_to(partial.parent)): file_sha256(path)
        for path in sorted(forensic_batch.glob("*/quarantine_receipt.json"))
    } if forensic_batch.exists() else {}
    failure_receipts = {
        str(path.relative_to(partial.parent)): file_sha256(path)
        for path in sorted(forensic_batch.glob("failures/batch_failure_receipt-*.json"))
    } if forensic_batch.exists() else {}
    checks = {
        "exact_20_contexts": len(specs) == len(rows) == len(npz_files) == 20,
        "selection_batch_margins": (
            all(sum(int(spec["fold"]) == value for spec in specs) == 4 for value in range(5))
            and all(3 <= sum(int(spec["phase_bin"]) == value for spec in specs) <= 5 for value in range(5))
            and all(4 <= sum(int(spec["motion_bin"]) == value for spec in specs) <= 6 for value in range(4))
        ),
        "context_and_action_contract": all(
            row.get("context_bitexact") is True
            and row.get("executed_actions_exact") is True
            and row.get("action_bounds_passed") is True
            and row.get("action_diff_passed") is True
            and row.get("duplicate_passed") is True
            and row.get("open3d_import_mode") == "native"
            for row in rows
        ),
        "contexts_with_three_effects_16_of_20": sum(row.get("effect_count", 0) >= 3 for row in rows) >= 16,
        "each_transport_effect_14_of_20": all(value >= 14 for value in effect_counts.values()),
        "gpu_peak_mib_le_24576": 0 <= peak <= 24576,
        "batch_wall_seconds_le_900": wall <= 900,
        "batch_output_bytes_le_41943040": output_bytes <= 41943040,
    }
    return {
        "format": "strict-track2-v461-endpoint200-batch-short-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "passed": all(checks.values()),
        "checks": checks,
        "transport_effect_counts": effect_counts,
        "gpu_peak_mib": peak,
        "wall_seconds": wall,
        "output_bytes": output_bytes,
        "rows": rows,
        "quarantine_receipt_count": len(quarantine_receipts),
        "quarantine_receipts_sha256": quarantine_receipts,
        "failure_receipt_count": len(failure_receipts),
        "failure_receipts_sha256": failure_receipts,
        "guards": {"reward_success_done_consumed": False, "all_selected_rows_retained": True, "policy_updates": 0, "rl_authorized": False},
    }


def validate_complete_batch(path: Path, batch_id: int, specs: list[dict]) -> dict:
    report_path = path / "batch_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"completed batch {batch_id} lacks report")
    report = json.loads(report_path.read_text())
    if report.get("batch_id") != batch_id or report.get("passed") is not True or len(report.get("rows", [])) != 20:
        raise RuntimeError(f"completed batch {batch_id} failed validation")
    expected_names = {f"episode{int(spec['episode'])}_start{int(spec['start']):05d}" for spec in specs}
    rows_dir = path / "rows"
    if not rows_dir.is_dir() or {item.name for item in rows_dir.iterdir()} != expected_names or {item.name for item in path.iterdir()} != {"rows", "batch_report.json"}:
        raise RuntimeError(f"completed batch {batch_id} has missing/extra paths")
    validated = []
    for spec in specs:
        row = validate_saved_row(path, spec)
        if row is None:
            raise RuntimeError(f"completed batch {batch_id} missing row")
        validated.append(row)
    report_rows = {(row["episode"], row["start"]): row["npz_sha256"] for row in report["rows"]}
    actual_rows = {(row["episode"], row["start"]): row["npz_sha256"] for row in validated}
    if report_rows != actual_rows:
        raise RuntimeError(f"completed batch {batch_id} report/receipt mismatch")
    recomputed = batch_short_gate(batch_id, path, specs, validated, int(report["gpu_peak_mib"]), float(report["wall_seconds"]))
    if recomputed["checks"] != report["checks"] or recomputed["transport_effect_counts"] != report["transport_effect_counts"] or recomputed["output_bytes"] != report["output_bytes"]:
        raise RuntimeError(f"completed batch {batch_id} short gate recomputation mismatch")
    return report


def quarantine_stale_work(root: Path, partial: Path, batch_id: int) -> list[str]:
    work_base = root / "work" / f"batch_{batch_id:03d}"
    if not work_base.exists():
        return []
    if not work_base.is_dir() or work_base.resolve().parent != (root / "work").resolve():
        raise RuntimeError(f"unsafe work tree for batch {batch_id}: {work_base}")
    sources = sorted(work_base.iterdir())
    if not sources:
        work_base.rmdir()
        work_root = root / "work"
        if work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()
        return []
    forensic = root / "_forensic_uncommitted" / f"batch_{batch_id:03d}"
    forensic.mkdir(parents=True, exist_ok=True)
    moved = []
    for source in sources:
        if not source.is_dir() or source.resolve().parent != work_base.resolve() or ".attempt_" not in source.name:
            raise RuntimeError(f"unexpected work entry requires manual review: {source}")
        stamp = f"{time.time_ns()}-{hashlib.sha256(str(source).encode()).hexdigest()[:12]}"
        destination = forensic / f"{source.name}-{stamp}"
        if destination.exists():
            raise RuntimeError(f"refusing forensic overwrite: {destination}")
        evidence = {}
        for item in sorted(source.rglob("*")):
            if item.is_file():
                evidence[str(item.relative_to(source))] = file_sha256(item)
        atomic_json(source / "quarantine_receipt.json", {
            "format": "strict-track2-v461-uncommitted-work-quarantine-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_path": str(source),
            "destination": str(destination),
            "files_sha256": evidence,
            "deleted_or_reused": False,
        })
        fsync_dir(source)
        os.replace(source, destination)
        fsync_dir(forensic)
        moved.append(str(destination.relative_to(root)))
    work_base.rmdir()
    work_root = root / "work"
    if work_root.exists() and not any(work_root.iterdir()):
        work_root.rmdir()
    return moved


def terminate_join_kill_pool(pool, pending: dict, join_seconds: float = 15.0) -> dict:
    processes = list(getattr(pool, "_processes", {}).values())
    initial_pids = [process.pid for process in processes if process.pid is not None]
    for future in pending:
        future.cancel()
    terminated = []
    for process in processes:
        if process.is_alive():
            process.terminate()
            terminated.append(process.pid)
    deadline = time.monotonic() + join_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    killed = []
    for process in processes:
        if process.is_alive():
            process.kill()
            killed.append(process.pid)
    for process in processes:
        process.join(timeout=5.0)
    alive_after = [process.pid for process in processes if process.is_alive()]
    exitcodes = {str(process.pid): process.exitcode for process in processes}
    pool.shutdown(wait=False, cancel_futures=True)
    return {
        "initial_pids": initial_pids,
        "terminate_pids": terminated,
        "kill_pids": killed,
        "alive_after_cleanup": alive_after,
        "exitcodes": exitcodes,
        "bounded_join_seconds": join_seconds,
        "pending_futures": len(pending),
    }


def write_batch_failure_receipt(root: Path, batch_id: int, exc: BaseException, cleanup: dict, gpu_before: int, gpu_after: int) -> Path:
    failure_root = root / "_forensic_uncommitted" / f"batch_{batch_id:03d}" / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns()}-{hashlib.sha256(repr(exc).encode()).hexdigest()[:12]}"
    path = failure_root / f"batch_failure_receipt-{stamp}.json"
    if path.exists():
        raise RuntimeError(f"refusing failure receipt overwrite: {path}")
    atomic_json(path, {
        "format": "strict-track2-v461-batch-failure-receipt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "gpu_mib_before_cleanup": gpu_before,
        "gpu_mib_after_cleanup": gpu_after,
        "cleanup": cleanup,
        "cleanup_passed": cleanup["alive_after_cleanup"] == [],
        "deleted_or_overwritten": False,
    })
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("reconciliation", "reconciliation-preregistration", "pilot-dataset", "selection", "contract", "v460-contract", "v455-collector", "endpoint-collector", "support-root", "task-config", "split", "seed-file", "dataset", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch-id", type=int)
    mode.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.batch_id is not None and not 0 <= args.batch_id < 10:
        raise RuntimeError("--batch-id must be in [0,9]")
    reconciliation, selection, contract = preflight(args)
    v460_contract = json.loads(args.v460_contract.read_text())
    if file_sha256(args.v460_contract) != EXPECTED_V460_CONTRACT_SHA256 or v460_contract.get("format") != "strict-track2-v461-endpoint-only-collection-contract-v1":
        raise RuntimeError("bad v460 endpoint-only contract")
    if v460_contract.get("selection_parent", {}).get("parent_contract_sha256") != EXPECTED_CONTRACT_SHA256 or v460_contract.get("collector", {}).get("sha256") != EXPECTED_ENDPOINT_COLLECTOR_SHA256:
        raise RuntimeError("v460 parent/collector contract drift")
    if file_sha256(args.endpoint_collector) != EXPECTED_ENDPOINT_COLLECTOR_SHA256:
        raise RuntimeError("v460 endpoint collector SHA mismatch")

    sys.path.insert(0, str(args.endpoint_collector.resolve().parent))
    collector = importlib.import_module(args.endpoint_collector.stem)
    if tuple(collector.VARIANTS) != VARIANTS or not callable(collector.collect_one) or not callable(collector.runtime_provenance):
        raise RuntimeError("v460 endpoint collector interface drift")
    collector_provenance = collector.runtime_provenance()
    modules = collector_provenance.get("modules", {})
    if Path(collector_provenance.get("sys_executable", "")).resolve() != Path(sys.executable).resolve() or set(modules) != {"open3d", "toppra", "mplib", "sapien"} or any(not Path(record.get("file", "/missing")).is_file() for record in modules.values()):
        raise RuntimeError("v460 endpoint collector runtime provenance failed")

    root = args.output_dir
    if root.exists() and any(root.iterdir()) and not (root / "generation_manifest.json").is_file():
        raise RuntimeError("v461 refuses any preexisting rows/partials without its own immutable generation manifest; v460 rows are not reusable")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "strict-track2-v461-endpoint200-generation-manifest-v1",
        "selection_sha256": file_sha256(args.selection),
        "contract_sha256": file_sha256(args.contract),
        "v460_contract_sha256": file_sha256(args.v460_contract),
        "reconciliation_sha256": file_sha256(args.reconciliation),
        "reconciliation_preregistration_sha256": file_sha256(args.reconciliation_preregistration),
        "v455_collector_sha256": file_sha256(args.v455_collector),
        "endpoint_collector_sha256": file_sha256(args.endpoint_collector),
        "endpoint_collector_runtime_provenance": collector_provenance,
        "v455_generation_report_sha256": selection["v455_generation_report_sha256"],
        "v455_immutable_npz_sha256": selection["v455_immutable_npz_sha256"],
        "public_split_sha256": file_sha256(args.split),
        "public_seed_file_sha256": file_sha256(args.seed_file),
        "dataset_data_root": str((args.dataset / "data").resolve()),
        "workers": 2,
        "contexts_per_batch": 20,
        "batches": 10,
    }
    manifest_path = root / "generation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("refusing resume with a different generation manifest")
    else:
        atomic_json(manifest_path, manifest)

    lower = selection["per_dim_action_lower"]
    upper = selection["per_dim_action_upper"]
    if args.finalize:
        reports = []
        for batch_id in range(10):
            specs = sorted((row for row in selection["contexts"] if row["batch_id"] == batch_id), key=lambda row: (row["fold"], row["phase_bin"], row["motion_bin"], row["episode"], row["start"]))
            reports.append(validate_complete_batch(root / f"batch_{batch_id:03d}", batch_id, specs))
        final = {
            "format": "strict-track2-v461-endpoint200-generation-report-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": len(reports) == 10 and all(report.get("passed") is True for report in reports),
            "exact_contexts": sum(len(report["rows"]) for report in reports),
            "exact_batches": len(reports),
            "wall_seconds": sum(float(report["wall_seconds"]) for report in reports),
            "batch_reports_sha256": {f"batch_{index:03d}": file_sha256(root / f"batch_{index:03d}" / "batch_report.json") for index in range(10)},
            "quarantine_receipt_count": len(list((root / "_forensic_uncommitted").glob("batch_*/*/quarantine_receipt.json"))) if (root / "_forensic_uncommitted").exists() else 0,
            "quarantine_receipts_sha256": {str(path.relative_to(root)): file_sha256(path) for path in sorted((root / "_forensic_uncommitted").glob("batch_*/*/quarantine_receipt.json"))} if (root / "_forensic_uncommitted").exists() else {},
            "failure_receipt_count": len(list((root / "_forensic_uncommitted").glob("batch_*/failures/batch_failure_receipt-*.json"))) if (root / "_forensic_uncommitted").exists() else 0,
            "failure_receipts_sha256": {str(path.relative_to(root)): file_sha256(path) for path in sorted((root / "_forensic_uncommitted").glob("batch_*/failures/batch_failure_receipt-*.json"))} if (root / "_forensic_uncommitted").exists() else {},
            "guards": {"finalize_only": True, "simulator_started": False, "stale_work_never_deleted_or_reused": True, "reward_success_done_consumed": False, "all_selected_rows_retained": True, "policy_updates": 0, "rl_authorized": False},
        }
        if final["exact_contexts"] != 200:
            final["passed"] = False
        final_path = root / "generation_report.json"
        if final_path.exists():
            raise RuntimeError("refusing to overwrite an existing v456 final generation report")
        atomic_json(final_path, final)
        print(json.dumps(final, indent=2))
        return 0 if final["passed"] else 2

    reports = []
    run_started = time.monotonic()
    overall_peak = max(0, gpu_mib())
    for batch_id in (args.batch_id,):
        specs = sorted((row for row in selection["contexts"] if row["batch_id"] == batch_id), key=lambda row: (row["fold"], row["phase_bin"], row["motion_bin"], row["episode"], row["start"]))
        if len(specs) != 20:
            raise RuntimeError(f"selection batch {batch_id} does not contain 20 contexts")
        for spec in specs:
            validate_spec_sources(spec)
        complete = root / f"batch_{batch_id:03d}"
        partial = root / f"batch_{batch_id:03d}.partial"
        if complete.exists():
            if partial.exists():
                raise RuntimeError(f"both complete and partial batch {batch_id} exist")
            reports.append(validate_complete_batch(complete, batch_id, specs))
            continue
        partial.mkdir(exist_ok=True)
        quarantine_stale_work(root, partial, batch_id)
        rows = []
        missing = []
        for spec in specs:
            saved = validate_saved_row(partial, spec)
            if saved is None:
                missing.append(spec)
            else:
                rows.append(saved)
        started = time.monotonic()
        ctx = mp.get_context("spawn")
        pool = cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx)
        pending = {}
        failed = None
        last_gpu_sample = overall_peak
        try:
            for spec in missing:
                name = f"episode{int(spec['episode'])}_start{int(spec['start']):05d}"
                work_base = root / "work" / f"batch_{batch_id:03d}"
                work_base.mkdir(parents=True, exist_ok=True)
                attempt = len(list(work_base.glob(f"{name}.attempt_*")))
                work = work_base / f"{name}.attempt_{attempt:03d}"
                if work.resolve().parent != work_base.resolve() or work.exists():
                    raise RuntimeError(f"unsafe/nonunique worker path: {work}")
                work.mkdir()
                future = pool.submit(
                    collector.collect_one,
                    spec,
                    str(args.support_root),
                    str(args.task_config),
                    str(work),
                    lower,
                    upper,
                )
                pending[future] = (spec, work)
            while pending:
                if time.monotonic() - started > 900:
                    raise TimeoutError(f"batch {batch_id} exceeded 900 seconds")
                sample = gpu_mib()
                last_gpu_sample = sample
                if sample < 0 or sample > 24576:
                    raise RuntimeError(f"GPU monitor/limit failed: {sample} MiB")
                overall_peak = max(overall_peak, sample)
                done, _ = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    spec, work = pending.pop(future)
                    row, technical = future.result()
                    if technical or row is None:
                        raise RuntimeError(f"v460 collector failed episode={spec['episode']} start={spec['start']}: {technical}")
                    rows.append(promote_worker_result(partial, work, spec, row))
        except Exception as exc:
            failed = exc
            cleanup = terminate_join_kill_pool(pool, pending)
            post_cleanup_gpu = gpu_mib()
            write_batch_failure_receipt(root, batch_id, exc, cleanup, last_gpu_sample, post_cleanup_gpu)
            if cleanup["alive_after_cleanup"]:
                failed = RuntimeError(f"worker cleanup left live PIDs {cleanup['alive_after_cleanup']}; original={exc!r}")
        else:
            pool.shutdown(wait=True)
        if failed is not None:
            raise failed
        quarantine_stale_work(root, partial, batch_id)
        rows = [validate_saved_row(partial, spec) for spec in specs]
        if any(row is None for row in rows):
            raise RuntimeError(f"batch {batch_id} missing rows after collection")
        report = batch_short_gate(batch_id, partial, specs, rows, overall_peak, time.monotonic() - started)
        atomic_json(partial / "batch_report.json", report)
        if report["passed"] is not True:
            raise RuntimeError(f"batch {batch_id} short gate failed; retaining partial batch and stopping")
        os.replace(partial, complete)
        reports.append(validate_complete_batch(complete, batch_id, specs))
    print(json.dumps(reports[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
