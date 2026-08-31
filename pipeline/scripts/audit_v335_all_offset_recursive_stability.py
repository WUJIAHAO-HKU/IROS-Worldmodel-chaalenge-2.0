#!/usr/bin/env python3
"""All-offset public recursive replay for the sparse v335 repair.

Every public window in the four frozen right-arm holdout episodes belongs to
one of eight independent chunk-alignment chains (start modulo eight).  Each
chain feeds prediction[-5:] into its next request, exactly matching the bridge.
V335 is compared with v326 only where its teacher-forced sparse branch changes;
validation, where the branch never changes, must remain recursively bit-exact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from audit_v310_full_mirror_causal_gate import RIGHT_EPISODES, score_terminal, sha256
from wam_pipeline.v324_phase_guarded_terminal_runtime import ACTION_PROBABILITY_MIN, BASE_ALPHA_ZERO_MAX
from wam_pipeline.v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal
from wam_pipeline.v334_bounded_onset_terminal_runtime import ONSET_REPAIR_MAX_OFFSET
from wam_pipeline.v335_all_offset_bounded_onset_runtime import Track2V335AllOffsetBoundedOnset


def deterministic_seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:8], "little") % (2**31)


def rgb_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.int16) - target.astype(np.int16)).mean())


def temporal_delta_error(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(
        np.diff(prediction.astype(np.float32), axis=0)
        - np.diff(target.astype(np.float32), axis=0)
    ).mean())


def digest_frames(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def instantiate(name: str, args):
    cls = Track2V326BlendedPhaseTerminal if name == "v326" else Track2V335AllOffsetBoundedOnset
    return cls(
        args.checkpoint_dir, args.library_index, args.device,
        args.action_gate, args.phase_gate,
    )


def changed_branch(runtime, context, history, future) -> tuple[bool, dict]:
    if not runtime._post_grasp(history, future):
        return False, {"reason": "pregrasp"}
    probability = runtime._probability(history, future)
    signature = runtime._signature(history, future, probability)
    if signature is not None:
        return False, {"reason": signature, "probability": probability}
    base, _ = runtime._nearest_clean(context, history, future)
    baseline_alpha = runtime._alpha_with_context(context, history, future, base)
    phase_ready, episode, start, onset = runtime._phase(base)
    override = (
        baseline_alpha <= BASE_ALPHA_ZERO_MAX
        and probability >= ACTION_PROBABILITY_MIN
        and phase_ready
    )
    phase_offset = start - onset
    changed = bool(
        override
        and not runtime.eligible_episode.get(episode, False)
        and 0 <= phase_offset <= ONSET_REPAIR_MAX_OFFSET
    )
    return changed, {
        "reason": "bounded_onset" if changed else "unchanged",
        "probability": probability,
        "baseline_alpha": baseline_alpha,
        "phase_ready": phase_ready,
        "phase_episode": episode,
        "phase_start": start,
        "phase_onset": onset,
        "phase_offset": phase_offset,
        "terminal_eligible": bool(runtime.eligible_episode.get(episode, False)),
    }


def run_model(name: str, args, instructions: dict[str, str], reward) -> list[dict]:
    runtime = instantiate(name, args)
    states = []
    for split_name, episodes in RIGHT_EPISODES.items():
        for episode in episodes:
            available = {
                int(path.stem.split("_")[1]): path
                for path in args.windows.glob(f"episode{episode}_*.npz")
            }
            for alignment in range(8):
                if alignment not in available:
                    continue
                with np.load(available[alignment], allow_pickle=False) as values:
                    initial = values["context_frames"].astype(np.uint8)
                states.append({
                    "split": split_name,
                    "episode": episode,
                    "alignment": alignment,
                    "start": alignment,
                    "available": available,
                    "recursive_context": initial,
                    "instruction": str(instructions[str(episode)]),
                })
    rows = []
    reward_frames = []
    reward_prompts = []
    while True:
        active = [state for state in states if state["start"] in state["available"]]
        if not active:
            break
        requests = []
        pair_contexts = []
        pair_histories = []
        pair_futures = []
        pair_seeds = []
        pair_prompts = []
        for state in active:
            path = state["available"][state["start"]]
            with np.load(path, allow_pickle=False) as values:
                real_context = values["context_frames"].astype(np.uint8)
                history = values["history_actions"].astype(np.float32)
                future = values["future_actions"].astype(np.float32)
                target = values["target_frames"].astype(np.uint8)
            changed, branch = (
                changed_branch(runtime, real_context, history, future)
                if name == "v335" else (False, {})
            )
            requests.append({
                "state": state, "path": path, "real_context": real_context,
                "history": history, "future": future, "target": target,
                "changed": changed, "branch": branch,
            })
            for context in (real_context, state["recursive_context"]):
                pair_contexts.append(context)
                pair_histories.append(history)
                pair_futures.append(future)
                pair_seeds.append(deterministic_seed(path))
                pair_prompts.append(state["instruction"])
        predictions = []
        for begin in range(0, len(pair_contexts), args.inference_batch_size):
            end = begin + args.inference_batch_size
            predictions.extend(runtime.predict_batch(
                np.stack(pair_contexts[begin:end]),
                np.stack(pair_histories[begin:end]),
                np.stack(pair_futures[begin:end]),
                np.asarray(pair_seeds[begin:end], dtype=np.int64),
                pair_prompts[begin:end],
            ))
        for index, request in enumerate(requests):
            teacher = predictions[2 * index]
            recursive = predictions[2 * index + 1]
            target_context = request["target"][-5:]
            state = request["state"]
            key = f"{state['split']}/episode{state['episode']}/a{state['alignment']}/s{state['start']:05d}"
            offset = len(reward_frames)
            reward_frames.extend((
                request["real_context"][-1], request["target"][-1],
                teacher[-1], recursive[-1],
            ))
            reward_prompts.extend([state["instruction"]] * 4)
            rows.append({
                "key": key,
                "split": state["split"],
                "episode": state["episode"],
                "alignment": state["alignment"],
                "start": state["start"],
                "teacher_changed_branch": request["changed"],
                "branch": request["branch"],
                "teacher_next_context_rgb_mae": rgb_mae(teacher[-5:], target_context),
                "recursive_next_context_rgb_mae": rgb_mae(recursive[-5:], target_context),
                "teacher_temporal_delta_error": temporal_delta_error(teacher[-5:], target_context),
                "recursive_temporal_delta_error": temporal_delta_error(recursive[-5:], target_context),
                "recursive_vs_teacher_rgb_mae": rgb_mae(recursive[-5:], teacher[-5:]),
                "teacher_next_context_sha256": digest_frames(teacher[-5:]),
                "recursive_next_context_sha256": digest_frames(recursive[-5:]),
                "reward_offset": offset,
            })
            state["recursive_context"] = recursive[-5:].copy()
            state["start"] += 8

    scores = score_terminal(
        reward, np.stack(reward_frames), reward_prompts,
        torch.device(args.device), args.reward_batch_size,
    )
    for row in rows:
        offset = row.pop("reward_offset")
        values = scores[offset : offset + 4]
        row.update({
            "context_terminal_reward": float(values[0]),
            "gt_terminal_reward": float(values[1]),
            "teacher_terminal_reward": float(values[2]),
            "recursive_terminal_reward": float(values[3]),
            "teacher_reward_absolute_error": float(abs(values[2] - values[1])),
            "recursive_reward_absolute_error": float(abs(values[3] - values[1])),
        })
    del runtime
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"ALL_OFFSET_MODEL_SCORED {name} {len(rows)}", flush=True)
    return rows


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def aggregate(rows: list[dict], changed_keys: set[str]) -> dict:
    result = {}
    for split_name in RIGHT_EPISODES:
        split_rows = [row for row in rows if row["split"] == split_name]
        result[split_name] = {}
        for group_name, selected in (
            ("all", split_rows),
            ("changed_branch", [row for row in split_rows if row["key"] in changed_keys]),
        ):
            value = {
                metric: stats([float(row[metric]) for row in selected])
                for metric in (
                    "teacher_next_context_rgb_mae", "recursive_next_context_rgb_mae",
                    "teacher_temporal_delta_error", "recursive_temporal_delta_error",
                    "recursive_vs_teacher_rgb_mae", "teacher_reward_absolute_error",
                    "recursive_reward_absolute_error", "teacher_terminal_reward",
                    "recursive_terminal_reward",
                )
            }
            value["recursive_reward_hit_rate_at_0p9"] = (
                float(np.mean([row["recursive_terminal_reward"] >= 0.90 for row in selected]))
                if selected else None
            )
            result[split_name][group_name] = value
    return result


def mean(aggregates, model, split_name, group, metric) -> float:
    value = aggregates[model][split_name][group][metric]["mean"]
    if value is None:
        raise RuntimeError(f"empty metric {model}/{split_name}/{group}/{metric}")
    return float(value)


def ratio(value: float, baseline: float) -> float:
    return float(value / max(baseline, 1e-9))


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "checkpoint-dir", "library-index", "action-gate", "phase-gate", "windows",
        "instruction-map", "reward-checkpoint", "t5-model", "contract-report",
        "causal-report", "preregistration", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract_report.read_text())
    causal = json.loads(args.causal_report.read_text())
    prereg = json.loads(args.preregistration.read_text())
    if contract.get("passed") is not True or causal.get("passed") is not True:
        raise RuntimeError("prerequisite gate failed")
    if prereg.get("format") != "strict-track2-v335-all-offset-bounded-onset-preregistration-v1":
        raise RuntimeError("wrong preregistration")
    mapping = json.loads(args.instruction_map.read_text())
    if mapping.get("hidden_or_final_evaluation_data") is not False:
        raise RuntimeError("public data boundary is invalid")

    from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

    reward = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        str(args.reward_checkpoint), config={"t5_model_name": str(args.t5_model)}
    ).to(args.device).eval().requires_grad_(False)
    instructions = mapping["episode_to_instruction"]
    rows = {name: run_model(name, args, instructions, reward) for name in ("v326", "v335")}
    if [row["key"] for row in rows["v326"]] != [row["key"] for row in rows["v335"]]:
        raise RuntimeError("model rows are misaligned")
    if len(rows["v335"]) != 512:
        raise RuntimeError(f"expected 512 all-offset rows, found {len(rows['v335'])}")
    changed_keys = {row["key"] for row in rows["v335"] if row["teacher_changed_branch"]}
    aggregates = {name: aggregate(value, changed_keys) for name, value in rows.items()}
    changed_counts = {
        split_name: aggregates["v335"][split_name]["changed_branch"]
        ["recursive_next_context_rgb_mae"]["count"]
        for split_name in RIGHT_EPISODES
    }
    comparisons = {}
    for split_name in RIGHT_EPISODES:
        comparisons[split_name] = {}
        groups = ["all"] + (["changed_branch"] if changed_counts[split_name] else [])
        for group in groups:
            comparisons[split_name][group] = {
                metric: ratio(
                    mean(aggregates, "v335", split_name, group, metric),
                    mean(aggregates, "v326", split_name, group, metric),
                )
                for metric in (
                    "recursive_next_context_rgb_mae",
                    "recursive_temporal_delta_error",
                    "recursive_reward_absolute_error",
                )
            }
    validation_pairs = [
        (base, candidate)
        for base, candidate in zip(rows["v326"], rows["v335"], strict=True)
        if candidate["split"] == "validation"
    ]
    validation_teacher_exact = all(
        base["teacher_next_context_sha256"] == candidate["teacher_next_context_sha256"]
        for base, candidate in validation_pairs
    )
    validation_recursive_exact = all(
        base["recursive_next_context_sha256"] == candidate["recursive_next_context_sha256"]
        for base, candidate in validation_pairs
    )
    local_changed = aggregates["v335"]["local_test"]["changed_branch"]
    checks = {
        "contract_gate": contract.get("passed") is True,
        "causal_gate": causal.get("passed") is True,
        "all_offset_rows_exact_512": len(rows["v335"]) == 512,
        "validation_changed_branch_count_zero": changed_counts["validation"] == 0,
        "local_changed_branch_count_ge_3": changed_counts["local_test"] >= 3,
        "validation_teacher_next_context_bit_exact_v326": validation_teacher_exact,
        "validation_recursive_next_context_bit_exact_v326": validation_recursive_exact,
        "local_changed_recursive_rgb_ratio_le_0p90": comparisons["local_test"]["changed_branch"]
        ["recursive_next_context_rgb_mae"] <= 0.90,
        "local_changed_recursive_temporal_ratio_le_0p98": comparisons["local_test"]["changed_branch"]
        ["recursive_temporal_delta_error"] <= 0.98,
        "local_changed_recursive_reward_error_ratio_le_1p50": comparisons["local_test"]["changed_branch"]
        ["recursive_reward_absolute_error"] <= 1.50,
        "local_changed_recursive_reward_mean_ge_0p90": local_changed["recursive_terminal_reward"]["mean"] >= 0.90,
        "local_changed_recursive_reward_hit_rate_ge_0p85": local_changed["recursive_reward_hit_rate_at_0p9"] >= 0.85,
        "local_all_recursive_rgb_ratio_le_1p02": comparisons["local_test"]["all"]
        ["recursive_next_context_rgb_mae"] <= 1.02,
        "local_all_recursive_temporal_ratio_le_1p02": comparisons["local_test"]["all"]
        ["recursive_temporal_delta_error"] <= 1.02,
        "local_all_recursive_reward_error_ratio_le_1p02": comparisons["local_test"]["all"]
        ["recursive_reward_absolute_error"] <= 1.02,
    }
    report = {
        "format": "strict-track2-v335-all-offset-recursive-stability-gate-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "track2-v335-all-offset-bounded-onset-v334-v333-v326-v325-v317",
        "replay_rule": {
            "episodes": RIGHT_EPISODES,
            "alignments": list(range(8)),
            "chain": "start=alignment, alignment+8, ...; next context=prediction[-5:]",
            "row_coverage": "every one of the 512 public holdout right-arm windows exactly once",
            "changed_subset": "teacher-forced v335 branch differs structurally from v326",
        },
        "changed_branch_counts": changed_counts,
        "validation_bit_exact": {
            "teacher_next_context": validation_teacher_exact,
            "recursive_next_context": validation_recursive_exact,
        },
        "aggregates": aggregates,
        "v335_over_v326_comparisons": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
        "authorization": {
            "expensive_rl_allowed": all(checks.values()),
            "all_checks_required": True,
            "waivers_or_excluded_failed_checks": False,
        },
        "evidence_sha256": {
            "preregistration": sha256(args.preregistration),
            "contract_report": sha256(args.contract_report),
            "causal_report": sha256(args.causal_report),
            "action_gate": sha256(args.action_gate),
            "phase_gate": sha256(args.phase_gate),
            "instruction_map": sha256(args.instruction_map),
            "reward_checkpoint": sha256(args.reward_checkpoint),
        },
        "guards": {
            "declared_public_world_model_data_only": True,
            "runtime_reads_reward_or_outcome": False,
            "public_evaluation_outcomes_read": False,
            "hidden_or_final_data": False,
            "real_submission": False,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "changed_branch_counts": changed_counts,
        "validation_bit_exact": report["validation_bit_exact"],
        "v335_over_v326_comparisons": comparisons,
        "checks": checks,
    }, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
