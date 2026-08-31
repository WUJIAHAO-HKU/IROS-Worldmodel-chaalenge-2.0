#!/usr/bin/env python3
"""Independent scalar-evidence audit for the preregistered v482 public-train S0."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


BRANCHES = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4")
EXPECTED_FOLD_KEYS = {
    "enabled_sequences", "protected_sequences", "continuous_ratio", "uint8_ratio",
    "branch_ratio", "prefix_ratio", "first7_ratio", "action_over_context_only",
    "action_over_phase_shuffle", "positive_delta_cosine_fraction",
    "nonzero_prediction_delta_fraction", "improved_episodes", "episode_uint8_ratio",
    "episode_uint8_ratio_max", "protected_uint8_bitexact", "passed", "fold",
    "fit_sequences", "holdout_sequences", "schedule_sha256", "phase_shuffle_fit_sha256",
    "phase_shuffle_holdout_sha256", "initial_state_sha256", "loss_first_last",
    "all_losses_finite",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def stable_seed(episode, start):
    payload = f"v482-v169-cache/seed1624/episode{int(episode)}/start{int(start)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, default=json_default)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)


def ratio(numerator, denominator, mask):
    n = float(np.asarray(numerator, np.float64)[mask].sum())
    d = float(np.asarray(denominator, np.float64)[mask].sum())
    return 0.0 if d == 0.0 and n == 0.0 else (math.inf if d == 0.0 else n / d)


def metrics(data, mask, fold):
    enabled = data["enabled"] & mask
    protected = (~data["enabled"]) & mask
    baseline = data["baseline_mae"]
    action = data["action_mae"]
    uint = data["uint_mae"]
    context = data["context_mae"]
    phase = data["phase_mae"]
    branches = {
        name: ratio(action, baseline, enabled & (data["branch"] == branch))
        for branch, name in enumerate(BRANCHES)
    }
    prefixes = [ratio(action[:, k], baseline[:, k], enabled) for k in range(8)]
    episodes = sorted(set(map(int, data["episode"][mask])))
    episode_ratios = {
        str(ep): ratio(uint, baseline, mask & (data["episode"] == ep)) for ep in episodes
    }
    cosine_mask = np.ones(len(data["cosine_dot"]), bool) if fold is None else data["cosine_fold"] == fold
    cosine = data["cosine_dot"][cosine_mask]
    cosine_target_norm = data["cosine_target_norm"][cosine_mask]
    cosine_eligible = cosine_target_norm > 1e-8
    if not cosine_eligible.any():
        raise RuntimeError("v482 empty eligible causal-cosine evidence")
    nonzero = data["delta_nonzero"] if fold is None else data["delta_nonzero"][data["delta_nonzero_fold"] == fold]
    improved = sum(
        float(action[mask & (data["episode"] == ep)].sum())
        < float(baseline[mask & (data["episode"] == ep)].sum())
        for ep in episodes
    )
    result = {
        "enabled_sequences": int(enabled.sum()),
        "protected_sequences": int(protected.sum()),
        "continuous_ratio": ratio(action, baseline, enabled),
        "uint8_ratio": ratio(uint, baseline, enabled),
        "branch_ratio": branches,
        "prefix_ratio": prefixes,
        "first7_ratio": ratio(action[:, :7], baseline[:, :7], enabled),
        "action_over_context_only": ratio(action, context, enabled),
        "action_over_phase_shuffle": ratio(action, phase, enabled),
        "positive_delta_cosine_fraction": float(np.mean(cosine[cosine_eligible] > 0)),
        "nonzero_prediction_delta_fraction": float(np.mean(nonzero)),
        "improved_episodes": int(improved),
        "episode_uint8_ratio": episode_ratios,
        "episode_uint8_ratio_max": max(episode_ratios.values()),
        "protected_uint8_bitexact": bool(np.all(data["protected_pixel_mismatch_count"][protected] == 0)),
    }
    result["passed"] = bool(
        result["continuous_ratio"] <= .95 and result["uint8_ratio"] <= 1
        and branches["factual"] <= 1 and all(branches[x] <= .98 for x in BRANCHES[1:])
        and max(prefixes) <= 1 and result["first7_ratio"] <= .98
        and result["action_over_context_only"] <= .95
        and result["action_over_phase_shuffle"] <= .97
        and result["positive_delta_cosine_fraction"] >= .60
        and result["nonzero_prediction_delta_fraction"] >= .95
        and result["episode_uint8_ratio_max"] <= 1.05
        and result["protected_uint8_bitexact"]
    )
    return result


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre = json.loads(args.preregistration.read_text())
    report_path = args.result_dir / "s0_report.json"
    report = json.loads(report_path.read_text())
    checks = {}
    runtime_spec = importlib.util.spec_from_file_location("v482_audit_runtime", pre["source"]["runtime_path"])
    runtime_module = importlib.util.module_from_spec(runtime_spec); runtime_spec.loader.exec_module(runtime_module)
    closure_path = Path(pre["v169"]["closure_path"])
    dataset_root = Path(pre["dataset"]["root"])
    declared_dataset = {row["relative"]: row["sha256"] for row in pre["dataset"]["files"]}
    actual_dataset = {}
    dataset_no_links = True
    for path in sorted(dataset_root.rglob("*")):
        if path.is_symlink(): dataset_no_links = False
        elif path.is_file(): actual_dataset[path.relative_to(dataset_root).as_posix()] = sha(path)
    checks["source_closure"] = bool(
        sha(Path(__file__)) == pre["source"]["auditor_sha256"]
        and sha(args.contract) == pre["source"]["contract_sha256"]
        and sha(pre["source"]["trainer_path"]) == pre["source"]["trainer_sha256"]
        and sha(pre["source"]["runtime_path"]) == pre["source"]["runtime_sha256"]
        and sha(args.preregistration) == report.get("preregistration_sha256")
        and sha(args.contract) == report.get("contract_sha256")
        and pre["source"]["trainer_sha256"] == report.get("trainer_sha256")
        and pre["source"]["runtime_sha256"] == report.get("runtime_sha256")
        and dataset_no_links and actual_dataset == declared_dataset
        and sha(pre["dataset"]["final_report"]["path"]) == pre["dataset"]["final_report"]["sha256"]
        and sha(pre["dataset"]["final_audit"]["path"]) == pre["dataset"]["final_audit"]["sha256"]
        and sha(closure_path) == pre["v169"]["closure_sha256"]
        and runtime_module.verify_v169(json.loads(closure_path.read_text())) == pre["v169"]["closure_digest"]
    )
    evidence_path = args.result_dir / "oof_scalar_evidence.npz"
    checks["evidence_sha"] = sha(evidence_path) == report.get("oof_scalar_evidence_sha256")
    with np.load(evidence_path, allow_pickle=False) as raw:
        required = {
            "baseline_mae", "action_mae", "uint_mae", "context_mae", "phase_mae", "enabled",
            "protected_pixel_mismatch_count", "episode", "branch", "fold", "cosine_positive",
            "cosine_dot", "cosine_target_norm", "cosine_prediction_norm", "cosine_fold",
            "delta_nonzero", "delta_prediction_norm", "delta_nonzero_fold",
        }
        checks["evidence_schema"] = set(raw.files) == required
        data = {key: np.asarray(raw[key]) for key in raw.files}
    selection = json.loads(Path(pre["dataset"]["selection"]["path"]).read_text())
    completed = int(report["completed_folds"])
    if completed < 1 or completed > 5:
        raise RuntimeError("v482 invalid completed fold count")
    expected_episode, expected_branch, expected_fold, expected_enabled = [], [], [], []
    expected_seeds, expected_request_sha, expected_context_repeat_sha = [], [], []
    for order, spec in enumerate(selection["contexts"]):
        with np.load(pre["temporal_contexts"][order]["row_npz_path"], allow_pickle=False) as row:
            history = np.asarray(row["history_actions"])
            futures = np.asarray(row["future_actions"])
            stored_context = np.asarray(row["pre_future_context_rgb"])
        repeat5 = np.repeat(np.ascontiguousarray(stored_context[0, 0])[None], 5, axis=0)
        expected_context_repeat_sha.append(arrsha(repeat5))
        parser_value = str(spec["instruction"]).lower()
        right = "right arm" in parser_value and "left arm" not in parser_value
        for branch in range(5):
            seed = stable_seed(spec["episode"], spec["start"])
            request_digest = hashlib.sha256()
            request_digest.update(np.ascontiguousarray(repeat5).view(np.uint8))
            request_digest.update(np.ascontiguousarray(history, np.float32).view(np.uint8))
            request_digest.update(np.ascontiguousarray(futures[branch], np.float32).view(np.uint8))
            request_digest.update(np.asarray([seed], dtype="<i8").view(np.uint8))
            request_digest.update(str(spec["instruction"]).encode("utf-8"))
            expected_seeds.append(seed); expected_request_sha.append(request_digest.hexdigest())
            expected_episode.append(int(spec["episode"])); expected_branch.append(branch); expected_fold.append(int(spec["fold"]))
            expected_enabled.append(bool(right and history[-1, 13] < .5 and np.all(futures[branch, :, 13] < .5)))
    expected_enabled = np.asarray(expected_enabled, bool)
    computed_mask = data["fold"] < completed
    checks["request_only_evidence_identity"] = bool(
        np.array_equal(data["episode"], np.asarray(expected_episode, np.int64))
        and np.array_equal(data["branch"], np.asarray(expected_branch, np.int64))
        and np.array_equal(data["fold"], np.asarray(expected_fold, np.int64))
        and np.array_equal(data["enabled"][computed_mask], expected_enabled[computed_mask])
        and not data["enabled"][~computed_mask].any()
    )
    scalar_names = ("baseline_mae", "action_mae", "uint_mae", "context_mae", "phase_mae")
    pair_count = 160 * completed
    checks["evidence_shapes_dtypes"] = bool(
        all(data[k].shape == (1000, 8) and data[k].dtype == np.float64 for k in scalar_names)
        and data["enabled"].shape == (1000,) and data["enabled"].dtype == np.bool_
        and data["protected_pixel_mismatch_count"].shape == (1000,) and data["protected_pixel_mismatch_count"].dtype == np.float64
        and all(data[k].shape == (1000,) and np.issubdtype(data[k].dtype, np.integer)
                for k in ("episode", "branch", "fold"))
        and all(np.isfinite(data[k][computed_mask]).all() and np.isnan(data[k][~computed_mask]).all() for k in scalar_names)
        and np.isfinite(data["protected_pixel_mismatch_count"][computed_mask]).all()
        and np.isnan(data["protected_pixel_mismatch_count"][~computed_mask]).all()
        and np.isfinite(np.concatenate([data[k].reshape(-1) for k in
            ("cosine_dot", "cosine_target_norm", "cosine_prediction_norm")])).all()
        and np.isfinite(data["delta_prediction_norm"]).all()
        and np.array_equal(data["cosine_positive"], (data["cosine_target_norm"] > 1e-8) & (data["cosine_dot"] > 0))
        and np.array_equal(data["delta_nonzero"], data["delta_prediction_norm"] > 1e-8)
        and data["cosine_positive"].dtype == data["delta_nonzero"].dtype == np.bool_
        and data["cosine_dot"].shape == data["cosine_target_norm"].shape == data["cosine_prediction_norm"].shape == data["cosine_fold"].shape == (pair_count,)
        and data["delta_nonzero"].shape == data["delta_prediction_norm"].shape == data["delta_nonzero_fold"].shape == (pair_count,)
        and np.all(data["cosine_target_norm"] >= 0)
        and np.all(data["cosine_prediction_norm"] >= 0)
        and all(np.count_nonzero(data["cosine_fold"] == fold) == 160 for fold in range(completed))
        and all(np.count_nonzero(data["delta_nonzero_fold"] == fold) == 160 for fold in range(completed))
        and not np.any(data["cosine_fold"] >= completed)
        and not np.any(data["delta_nonzero_fold"] >= completed)
    )
    cache_receipt = report.get("scalar_v169_cache", {})
    cache_manifest_path = Path(cache_receipt.get("manifest_path", ""))
    cache_path = Path(cache_receipt.get("path", ""))
    cache_manifest = json.loads(cache_manifest_path.read_text())
    expected_cache_receipt = {
        **cache_manifest,
        "path": str(cache_path),
        "manifest_path": str(cache_manifest_path),
        "manifest_sha256": sha(cache_manifest_path),
    }
    checks["cache_receipt_manifest_closure"] = bool(
        cache_receipt == expected_cache_receipt
        and cache_manifest.get("format") == "strict-track2-v482-scalar-v169-temporal-cache-v1"
        and cache_manifest.get("ordered_scalar_requests") == 1000
        and cache_manifest.get("array_sha256") == sha(cache_path)
        and cache_manifest.get("context_repeat5_sha256") == expected_context_repeat_sha
        and cache_manifest.get("context_repeat5_digest_sha256") == hashlib.sha256("".join(expected_context_repeat_sha).encode()).hexdigest()
        and cache_manifest.get("context_repeat5_digest_sha256") == pre["v169_cache"]["context_repeat5_digest_sha256"]
    )
    with np.load(cache_path, allow_pickle=False) as cache_raw:
        cache_keys = set(cache_raw.files)
        baseline = np.asarray(cache_raw["baseline"])
        seeds = np.asarray(cache_raw["seed"])
        request_sha = np.asarray(cache_raw["request_sha256"]).astype("U")
        output_sha = np.asarray(cache_raw["output_sha256"]).astype("U")
        sample_id = np.asarray(cache_raw["sample_id"])
    recomputed_output_sha = np.asarray(
        [arrsha(baseline[context, branch]) for context in range(200) for branch in range(5)]
    ).reshape(200, 5)
    checks["cache_arrays_order_and_digests"] = bool(
        cache_keys == {"baseline", "seed", "request_sha256", "output_sha256", "sample_id"}
        and baseline.shape == (200, 5, 8, 256, 256, 3) and baseline.dtype == np.uint8
        and seeds.shape == request_sha.shape == output_sha.shape == sample_id.shape == (200, 5)
        and seeds.dtype == sample_id.dtype == np.int64
        and np.array_equal(sample_id, np.arange(1000, dtype=np.int64).reshape(200, 5))
        and np.array_equal(seeds, np.asarray(expected_seeds, np.int64).reshape(200, 5))
        and np.array_equal(request_sha, np.asarray(expected_request_sha).reshape(200, 5))
        and np.array_equal(output_sha, recomputed_output_sha)
        and cache_manifest.get("seed_digest_sha256") == arrsha(seeds)
        and cache_manifest.get("request_digest_sha256") == hashlib.sha256("".join(request_sha.reshape(-1)).encode()).hexdigest()
        and cache_manifest.get("output_digest_sha256") == hashlib.sha256("".join(output_sha.reshape(-1)).encode()).hexdigest()
    )
    recomputed, folded = [], []
    for fold in range(report["completed_folds"]):
        stored = json.loads((args.result_dir / f"fold{fold}_receipt.json").read_text())
        fresh = metrics(data, data["fold"] == fold, fold)
        semantic = {key: stored[key] for key in fresh}
        folded.append(
            set(stored) == EXPECTED_FOLD_KEYS
            and close(semantic, fresh)
            and stored["all_losses_finite"] is True
            and all(np.isfinite(np.asarray(value, np.float64)).all() for value in stored["loss_first_last"].values())
        )
        recomputed.append(fresh)
    expected_names = {f"fold{x}_receipt.json" for x in range(report["completed_folds"])}
    actual_names = {p.name for p in args.result_dir.glob("fold*_receipt.json")}
    statuses = [row["passed"] for row in recomputed]
    failed = sum(not value for value in statuses)
    first_unreachable = failed >= 2 and all(sum(not x for x in statuses[:k]) < 2 for k in range(1, len(statuses)))
    early = failed >= 2
    checks["folds_independent"] = all(folded) and actual_names == expected_names
    checks["counts_early_stop"] = bool(
        report["completed_folds"] == len(recomputed) == len(report["folds"])
        and report["failed_folds"] == failed
        and report["early_stop_mathematically_unreachable"] == early
        and (not early or first_unreachable)
    )
    checks["report_fold_semantics"] = all(
        close({key: stored[key] for key in fresh}, fresh)
        for stored, fresh in zip(report["folds"], recomputed)
    )
    checks["authorization_guards"] = report.get("guards") == {
        "public_train_only": True,
        "factual_duplicate_training_samples": 0,
        "reward_loaded": False,
        "outcome_loaded": False,
        "development_or_final_used": False,
        "training_launch_required_full200_final_pass": True,
        "s1_authorized": False,
        "zero_update_authorized": False,
        "policy_updates": 0,
        "rl_authorized": False,
    }
    if not early:
        fresh_global = metrics(data, np.ones(1000, bool), None)
        aggregate = dict(fresh_global)
        aggregate.pop("passed")
        aggregate["passing_folds"] = sum(statuses)
        global_pass = bool(fresh_global["passed"] and aggregate["passing_folds"] >= 4 and aggregate["improved_episodes"] >= 12)
        checks["aggregate_independent"] = close(report["aggregate"], aggregate) and report["passed"] == global_pass
    else:
        checks["aggregate_independent"] = report["aggregate"] is None and report["passed"] is False
    passed = all(checks.values())
    checks["promotion_branch"] = bool(
        (passed and report["passed"] and (args.result_dir / "all200_action.pt").is_file()
         and report["all200_checkpoint_sha256"] == sha(args.result_dir / "all200_action.pt"))
        or (not report["passed"] and not (args.result_dir / "all200_action.pt").exists()
            and report["all200_checkpoint_sha256"] is None and report["all200_training_performed"] is False)
    )
    passed = all(checks.values())
    receipt = {
        "format": "strict-track2-v482-temporal8-residual-s0-audit-v1",
        "passed": passed,
        "candidate_passed": bool(passed and report["passed"]),
        "checks": checks,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha(report_path),
        "verified_scalar_v169_cache": {
            "manifest_path": str(cache_manifest_path.resolve()),
            "manifest_sha256": sha(cache_manifest_path),
            "array_path": str(cache_path.resolve()),
            "array_sha256": sha(cache_path),
            "ordered_scalar_requests": 1000,
            "request_digest_sha256": cache_manifest["request_digest_sha256"],
            "output_digest_sha256": cache_manifest["output_digest_sha256"],
            "seed_digest_sha256": cache_manifest["seed_digest_sha256"],
            "context_repeat5_digest_sha256": cache_manifest["context_repeat5_digest_sha256"],
        },
        "guards": {"s1_authorized": False, "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False},
    }
    atomic_json(args.output, receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
