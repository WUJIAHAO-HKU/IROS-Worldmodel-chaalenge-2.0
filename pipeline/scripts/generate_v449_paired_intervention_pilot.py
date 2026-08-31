#!/usr/bin/env python3
"""Two-process, branch-serial fixed-horizon simulator collection for v449."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

VARIANTS = (
    "factual",
    "open",
    "static_transport",
    "reverse_transport",
    "hold_all",
    "factual_duplicate",
)


def ensure_open3d_importable() -> bool:
    """Install a deliberately empty import stub only when open3d itself is absent."""
    existing = sys.modules.get("open3d")
    if existing is not None and getattr(existing, "__v449_rgb_only_stub__", False):
        return True
    try:
        __import__("open3d")
        return False
    except ImportError as exc:
        if getattr(exc, "name", None) != "open3d":
            raise
        stub = types.ModuleType("open3d")
        stub.__v449_rgb_only_stub__ = True
        stub.__all__ = []
        sys.modules["open3d"] = stub
        return True


def digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def branches(history: np.ndarray, future: np.ndarray) -> dict[str, np.ndarray]:
    factual = future.copy()
    opened = factual.copy()
    opened[:, 13] = 1.0
    static = factual.copy()
    static[:, 7:13] = history[-1, 7:13]
    reverse = factual.copy()
    reverse[:, 7:13] = future[::-1, 7:13]
    hold = np.repeat(history[-1:], 8, axis=0)
    return {
        "factual": factual,
        "open": opened,
        "static_transport": static,
        "reverse_transport": reverse,
        "hold_all": hold,
        "factual_duplicate": factual.copy(),
    }


def image(obs: dict) -> np.ndarray:
    x = np.asarray(obs["full_image"])
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
        raise RuntimeError(f"bad simulator RGB {x.shape} {x.dtype}")
    return x.copy()


def pose(task) -> np.ndarray:
    raw = task.get_obs()["endpose"]
    return np.concatenate(
        (
            np.asarray(raw["left_endpose"]).reshape(-1),
            np.asarray(raw["left_gripper"]).reshape(-1),
            np.asarray(raw["right_endpose"]).reshape(-1),
            np.asarray(raw["right_gripper"]).reshape(-1),
        )
    ).astype(np.float64)


def bottle(task) -> np.ndarray:
    return np.asarray(task.bottle.get_functional_point(0), dtype=np.float64).reshape(-1)


def timeout_handler(_signum, _frame) -> None:
    raise TimeoutError("v449 fixed context exceeded 300 seconds")


def collect_one(spec: dict, support_root: str, task_config: str, out_dir: str):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    support_root_path = Path(support_root)
    out = Path(out_dir)
    cfg = yaml.safe_load(Path(task_config).read_text())
    cfg.update(
        {
            "task_name": "adjust_bottle",
            "planner_backend": "mplib",
            "step_lim": 100000,
            "clear_cache_freq": 1,
        }
    )
    os.environ["ASSETS_PATH"] = str(support_root_path.resolve())
    open3d_stub_active = ensure_open3d_importable()
    from robotwin.envs.vector_env import VectorEnv

    env = None
    technical: list[dict] = []
    try:
        env = VectorEnv(
            task_config=cfg,
            n_envs=1,
            env_seeds=[int(spec["dataset_seed"])],
        )
        start = int(spec["start"])
        with h5py.File(spec["source_hdf5"], "r") as h5:
            actions = np.asarray(h5["joint_action/vector"], dtype=np.float32)
        if start < 0 or start + 12 > len(actions):
            raise RuntimeError(f"source action boundary failed: start={start} len={len(actions)}")
        history = actions[start : start + 4].copy()
        future = actions[start + 4 : start + 12].copy()
        action_map = branches(history, future)
        contexts: list[np.ndarray] = []
        states: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        context_bottles: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        final_states: list[np.ndarray] = []
        final_bottles: list[np.ndarray] = []
        executed: list[str] = []
        for name in VARIANTS:
            task = None
            original = None
            try:
                env.reset(env_idx=[0], env_seeds=[int(spec["dataset_seed"])])
                task = env.envs[0].task
                task.eval_success = False
                original = task.check_success
                task.check_success = types.MethodType(lambda self: False, task)
                prefix = actions[:start]
                if len(prefix):
                    env.step(prefix[None])
                context = [image(env.get_obs()[0])]
                for action in history:
                    env.step(action[None, None])
                    context.append(image(env.get_obs()[0]))
                state = np.asarray(env.get_obs()[0]["state"], dtype=np.float32).copy()
                context_pose = pose(task)
                context_bottle = bottle(task)
                prediction = []
                for action in action_map[name]:
                    env.step(action[None, None])
                    prediction.append(image(env.get_obs()[0]))
                contexts.append(np.stack(context))
                states.append(state)
                poses.append(context_pose)
                context_bottles.append(context_bottle)
                targets.append(np.stack(prediction))
                final_states.append(
                    np.asarray(env.get_obs()[0]["state"], dtype=np.float32).copy()
                )
                final_bottles.append(bottle(task))
                executed.append(digest(action_map[name]))
            except Exception as exc:
                technical.append(
                    {
                        "episode": int(spec["episode"]),
                        "start": start,
                        "variant": name,
                        "error": repr(exc),
                    }
                )
                break
            finally:
                if task is not None and original is not None:
                    task.check_success = original
        if len(targets) != 6:
            return None, technical
        contexts_array = np.stack(contexts)
        states_array = np.stack(states)
        poses_array = np.stack(poses)
        context_bottles_array = np.stack(context_bottles)
        targets_array = np.stack(targets)
        final_states_array = np.stack(final_states)
        final_bottles_array = np.stack(final_bottles)
        futures = np.stack([action_map[name] for name in VARIANTS])
        context_rgb_bitexact = all(
            np.array_equal(contexts_array[0], x) for x in contexts_array[1:]
        )
        context_bitexact = context_rgb_bitexact and all(
            np.array_equal(states_array[0], states_array[index])
            and np.array_equal(poses_array[0], poses_array[index])
            and np.array_equal(context_bottles_array[0], context_bottles_array[index])
            for index in range(1, 6)
        )
        context_mae = max(
            float(
                np.abs(x.astype(np.float32) - contexts_array[0].astype(np.float32)).mean()
            )
            for x in contexts_array[1:]
        )
        qpos_max = max(
            float(np.abs(x - states_array[0]).max()) for x in states_array[1:]
        )
        pose_max = max(
            float(np.abs(x - poses_array[0]).max()) for x in poses_array[1:]
        )
        bottle_max = max(
            float(np.abs(x - context_bottles_array[0]).max())
            for x in context_bottles_array[1:]
        )
        context_pass = context_bitexact or (
            context_mae <= 0.25
            and qpos_max <= 1e-5
            and pose_max <= 1e-5
            and bottle_max <= 1e-5
        )
        action_exact = executed == [digest(x) for x in futures]
        with np.load(spec["source_window"], allow_pickle=False) as source:
            public = source["target_frames"].astype(np.uint8)
        fidelity = float(
            np.abs(targets_array[0].astype(np.float32) - public.astype(np.float32)).mean()
        )
        duplicate = float(
            np.abs(
                targets_array[0].astype(np.float32)
                - targets_array[5].astype(np.float32)
            ).mean()
        )
        noise = max(1.0, 5.0 * duplicate)
        counterfactuals = []
        for index, name in enumerate(VARIANTS[1:5], 1):
            rgb = float(
                np.abs(
                    targets_array[index, -1].astype(np.float32)
                    - targets_array[0, -1].astype(np.float32)
                ).mean()
            )
            qpos = float(np.linalg.norm(final_states_array[index] - final_states_array[0]))
            object_l2 = float(
                np.linalg.norm(final_bottles_array[index] - final_bottles_array[0])
            )
            counterfactuals.append(
                {
                    "variant": name,
                    "final_rgb_mae": rgb,
                    "final_qpos_l2": qpos,
                    "bottle_position_l2": object_l2,
                    "passed": rgb >= noise and (qpos >= 0.01 or object_l2 >= 0.005),
                }
            )
        causal_count = sum(x["passed"] for x in counterfactuals)
        path = out / f"episode{int(spec['episode'])}_start{start:05d}.npz"
        np.savez_compressed(
            path,
            episode=np.int64(spec["episode"]),
            dataset_seed=np.int64(spec["dataset_seed"]),
            start=np.int64(start),
            variants=np.asarray(VARIANTS),
            instruction=np.asarray(spec["instruction"]),
            context_frames=contexts_array[0],
            branch_context_frames=contexts_array,
            branch_context_state=states_array,
            branch_context_pose=poses_array,
            branch_context_bottle_position=context_bottles_array,
            history_actions=history,
            future_actions=futures,
            target_frames=targets_array,
            branch_final_state=final_states_array,
            branch_final_bottle_position=final_bottles_array,
            context_sha256=np.asarray([digest(x) for x in contexts_array]),
            state_sha256=np.asarray([digest(x) for x in states_array]),
            pose_sha256=np.asarray([digest(x) for x in poses_array]),
            bottle_position_sha256=np.asarray([digest(x) for x in context_bottles_array]),
            executed_action_sha256=np.asarray(executed),
            public_factual_target_sha256=np.asarray(digest(public)),
            factual_replay_public_target_rgb_mae=np.float64(fidelity),
            factual_duplicate_rgb_mae=np.float64(duplicate),
        )
        row = {
            "episode": int(spec["episode"]),
            "start": start,
            "file": path.name,
            "context_bitexact": context_bitexact,
            "context_rgb_bitexact": context_rgb_bitexact,
            "context_rgb_mae_max": context_mae,
            "context_qpos_max_abs": qpos_max,
            "context_pose_max_abs": pose_max,
            "context_bottle_position_max_abs": bottle_max,
            "context_passed": context_pass,
            "executed_actions_exact": action_exact,
            "factual_replay_public_target_rgb_mae": fidelity,
            "fidelity_passed": fidelity <= 8.0,
            "factual_duplicate_rgb_mae": duplicate,
            "duplicate_passed": duplicate <= 0.5,
            "noise_threshold": noise,
            "counterfactuals": counterfactuals,
            "causal_counterfactual_count": causal_count,
            "causal_passed": causal_count >= 2,
            "open3d_import_mode": "rgb_only_stub" if open3d_stub_active else "native",
        }
        return row, technical
    finally:
        signal.alarm(0)
        if env is not None:
            env.close(clear_cache=True)


def gpu_mib() -> int:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return sum(int(x.strip()) for x in text.splitlines() if x.strip())
    except Exception:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "support-root", "task-config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    out = args.output_dir
    if (
        prereg.get("format")
        != "strict-track2-v449-paired-intervention-pilot-preregistration-v1"
        or out.exists()
    ):
        raise RuntimeError("v449 prereg/output contract failed")
    out.mkdir(parents=True)
    started = time.monotonic()
    rows: list[dict] = []
    technical: list[dict] = []
    initial_gpu_mib = gpu_mib()
    peak = max(0, initial_gpu_mib)
    if initial_gpu_mib < 0:
        technical.append(
            {"variant": "resource_monitor", "error": "nvidia-smi memory query failed"}
        )
    context = mp.get_context("spawn")
    pool = cf.ProcessPoolExecutor(max_workers=2, mp_context=context)
    aborted = bool(technical)
    pending = {}
    if not aborted:
        pending = {
            pool.submit(
                collect_one,
                spec,
                str(args.support_root),
                str(args.task_config),
                str(out),
            ): spec
            for spec in prereg["public_train"]["contexts"]
        }
        while pending:
            if time.monotonic() - started > 1200:
                technical.append(
                    {"variant": "wall_timeout", "error": "pilot exceeded 1200 seconds"}
                )
                aborted = True
                break
            done, _ = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
            sample_gpu_mib = gpu_mib()
            if sample_gpu_mib < 0:
                technical.append(
                    {
                        "variant": "resource_monitor",
                        "error": "nvidia-smi memory query failed",
                    }
                )
                aborted = True
                break
            peak = max(peak, sample_gpu_mib)
            if peak > 8192:
                technical.append(
                    {
                        "variant": "resource_limit",
                        "error": f"GPU peak {peak} MiB exceeded 8192 MiB",
                    }
                )
                aborted = True
                break
            for future in done:
                spec = pending.pop(future)
                try:
                    row, errors = future.result()
                    technical.extend(errors)
                    if row is not None:
                        rows.append(row)
                    if errors or row is None:
                        aborted = True
                        break
                except Exception as exc:
                    technical.append(
                        {
                            "episode": int(spec["episode"]),
                            "start": int(spec["start"]),
                            "variant": "context_worker",
                            "error": repr(exc),
                        }
                    )
                    aborted = True
                    break
            if aborted:
                break
    if aborted:
        for future in pending:
            future.cancel()
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
        for process in list(getattr(pool, "_processes", {}).values()):
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
    else:
        pool.shutdown(wait=True)
    rows.sort(key=lambda x: x["episode"])
    wall = time.monotonic() - started
    causal = sum(row.get("causal_passed", False) for row in rows)
    output_bytes = sum(path.stat().st_size for path in out.glob("episode*_start*.npz"))
    pointcloud_files = sorted(
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcd", ".ply"}
    )
    passed = (
        len(rows) == 4
        and not technical
        and causal >= 3
        and peak <= 8192
        and wall <= 1200
        and output_bytes <= 16_777_216
        and not pointcloud_files
        and all(
            row["context_passed"]
            and row["executed_actions_exact"]
            and row["fidelity_passed"]
            and row["duplicate_passed"]
            for row in rows
        )
    )
    report = {
        "format": "strict-track2-v449-paired-intervention-pilot-generation-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "rows": rows,
        "causal_contexts": causal,
        "gpu_peak_mib": peak,
        "wall_seconds": wall,
        "output_bytes": output_bytes,
        "pointcloud_files": pointcloud_files,
        "technical_failures": technical,
        "guards": {
            "context_workers": 2,
            "branches_serial": True,
            "retry": 0,
            "context_timeout_seconds": 300,
            "planner_backend": "mplib",
            "curobo_used": False,
            "fixed_future_frames": 8,
            "success_check_forced_false": True,
            "reward_success_done_consumed": False,
            "technical_intervention_effect_gate": True,
            "task_outcome_or_reward_selection": False,
            "all_fixed_rows_branches_retained": True,
            "policy_updates": 0,
            "open3d_stub_rgb_only": True,
            "pointcloud_calls": 0,
        },
    }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
