#!/usr/bin/env python3
"""Independent in-boundary auditor for v532 repaired actual-OOF output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "strict-track2-v532-v531-actual-oof-independent-audit-v1"
EVENT_FORMAT = "strict-track2-v532-v531-actual-oof-event-v1"
ACTIVE_SOURCE_ROLE_ORDER = [
    "authority_design_contract", "authority_materializer", "actual_oof_execution_preregistration",
    "actual_oof_execution_manifest", "actual_oof_executor", "actual_oof_auditor", "actual_oof_launcher",
]
ACTIVE_SOURCE_PATHS = {
    "authority_design_contract": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_contract.json",
    "authority_materializer": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority.py",
    "actual_oof_execution_preregistration": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_execution_preregistration.json",
    "actual_oof_execution_manifest": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v532_v531_actual_oof_execution_manifest.json",
    "actual_oof_executor": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/execute_v532_v531_actual_oof.py",
    "actual_oof_auditor": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/audit_v532_v531_actual_oof.py",
    "actual_oof_launcher": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v532_v531_actual_oof.py",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_events(path: Path, expected_keys: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1000:
        raise RuntimeError("actual OOF event exact1000")
    keys = set(expected_keys)
    for ordinal, row in enumerate(rows):
        if set(row) != keys or row["oof_ordinal"] != ordinal:
            raise RuntimeError("actual OOF event schema/order")
        unsigned = {key: value for key, value in row.items() if key != "event_canonical_sha256"}
        if row["event_canonical_sha256"] != canonical_sha(unsigned):
            raise RuntimeError("actual OOF event digest")
    return rows


def validate_metric_arrays(path: Path, schema: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        if list(raw.files) != schema["metrics_npz_exact_keys"]:
            raise RuntimeError("metrics npz exact key order")
        arrays = {key: np.asarray(raw[key]) for key in raw.files}
    required = {
        "absolute_error_sum_uint64": ((1000,), np.dtype("uint64")),
        "branch": ((1000,), np.dtype("int64")),
        "episode": ((1000,), np.dtype("int64")),
        "fold": ((1000,), np.dtype("int64")),
        "pixel_count_uint64": ((1000,), np.dtype("uint64")),
        "sample_id": ((1000,), np.dtype("int64")),
        "selection_order": ((1000,), np.dtype("int64")),
        "target_sha256": ((1000,), np.dtype("<U64")),
        "prediction_sha256": ((1000,), np.dtype("<U64")),
    }
    for key, (shape, dtype) in required.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise RuntimeError(f"metrics array schema:{key}")
    return arrays


def recompute_row(row: dict, prediction: np.ndarray) -> tuple[int, int, str, str]:
    row_path = Path(row["row_npz_path"])
    receipt_path = Path(row["row_receipt_path"])
    if row_path.is_symlink() or receipt_path.is_symlink() or sha(row_path) != row["row_npz_sha256"] or sha(receipt_path) != row["row_receipt_sha256"]:
        raise RuntimeError("dataset row current")
    with np.load(row_path, allow_pickle=False) as data:
        if list(np.asarray(data["variants"]).astype("U")) != ["factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4", "factual_duplicate"]:
            raise RuntimeError("dataset variants")
        target = np.ascontiguousarray(data[row["target_array_key"]][row["target_branch_index"]])
    if target.shape != (8, 256, 256, 3) or target.dtype != np.uint8 or prediction.shape != target.shape or prediction.dtype != np.uint8:
        raise RuntimeError("prediction/target schema")
    delta = np.abs(prediction.astype(np.int16) - target.astype(np.int16)).astype(np.uint16)
    return int(delta.sum(dtype=np.uint64)), int(delta.size), arrsha(prediction), arrsha(target)


def audit_staged(staged_root: Path, manifest: dict, cache_path: Path) -> dict:
    staged_root = Path(staged_root)
    if (manifest.get("active_source_role_order") != ACTIVE_SOURCE_ROLE_ORDER
            or set(manifest.get("active_source_paths", {})) != set(ACTIVE_SOURCE_ROLE_ORDER)
            or manifest["active_source_paths"] != ACTIVE_SOURCE_PATHS):
        raise RuntimeError("auditor active source role/path bijection")
    schema = manifest["output_schema"]
    events = load_events(staged_root / "oof_call_events.ndjson", schema["event_exact_keys"])
    arrays = validate_metric_arrays(staged_root / "metrics.npz", schema)
    rows = manifest["ordered_rows"]
    if len(rows) != 1000 or manifest["ordered_rows_canonical_sha256"] != canonical_sha(rows):
        raise RuntimeError("manifest rows")
    with np.load(cache_path, allow_pickle=False) as raw:
        if list(raw.files) != ["baseline", "seed", "request_sha256", "output_sha256", "sample_id"]:
            raise RuntimeError("qualified cache exact keys")
        baseline = np.asarray(raw["baseline"])
        request_hashes = np.asarray(raw["request_sha256"])
        output_hashes = np.asarray(raw["output_sha256"])
        sample_ids = np.asarray(raw["sample_id"])
    if baseline.shape != (200, 5, 8, 256, 256, 3) or baseline.dtype != np.uint8:
        raise RuntimeError("qualified cache baseline schema")
    computed = []
    fold_counts = [0] * 5
    loaded_selection = None
    loaded_targets = None
    for ordinal, (frozen, event) in enumerate(zip(rows, events)):
        selection = frozen["selection_order"]
        branch = frozen["branch_index"]
        sample_id = frozen["sample_id"]
        if int(sample_ids[selection, branch]) != sample_id or str(request_hashes[selection, branch]) != frozen["cache_request_sha256"] or str(output_hashes[selection, branch]) != frozen["cache_output_sha256"]:
            raise RuntimeError("cache row identity")
        if loaded_selection != selection:
            row_path = Path(frozen["row_npz_path"]); receipt_path = Path(frozen["row_receipt_path"])
            if row_path.is_symlink() or receipt_path.is_symlink() or sha(row_path) != frozen["row_npz_sha256"] or sha(receipt_path) != frozen["row_receipt_sha256"]:
                raise RuntimeError("dataset row current")
            with np.load(row_path, allow_pickle=False) as data:
                variants = list(np.asarray(data["variants"]).astype("U"))
                loaded_targets = np.ascontiguousarray(data["temporal_rgb"][:5])
            if variants != ["factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4", "factual_duplicate"] or loaded_targets.shape != (5, 8, 256, 256, 3) or loaded_targets.dtype != np.uint8:
                raise RuntimeError("dataset target group schema")
            loaded_selection = selection
        target = np.ascontiguousarray(loaded_targets[branch])
        prediction = np.ascontiguousarray(baseline[selection, branch])
        delta = np.abs(prediction.astype(np.int16) - target.astype(np.int16)).astype(np.uint16)
        result = int(delta.sum(dtype=np.uint64)), int(delta.size), arrsha(prediction), arrsha(target)
        error_sum, pixel_count, prediction_sha, target_sha = result
        if prediction_sha != frozen["cache_output_sha256"]:
            raise RuntimeError("cached prediction payload hash")
        expected = {
            "absolute_error_sum_uint64": error_sum,
            "branch": frozen["branch"],
            "branch_index": branch,
            "cache_output_sha256": frozen["cache_output_sha256"],
            "episode": frozen["episode"],
            "fold": frozen["fold"],
            "input_raw_warning": frozen["raw_warning"],
            "oof_computation_warnings": [],
            "oof_ordinal": ordinal,
            "pixel_count_uint64": pixel_count,
            "prediction_sha256": prediction_sha,
            "sample_id": sample_id,
            "selection_order": selection,
            "start": frozen["start"],
            "target_sha256": target_sha,
        }
        expected["event_canonical_sha256"] = canonical_sha(expected)
        if event != expected:
            raise RuntimeError("OOF event recomputation")
        if any((
            int(arrays["absolute_error_sum_uint64"][ordinal]) != error_sum,
            int(arrays["pixel_count_uint64"][ordinal]) != pixel_count,
            int(arrays["fold"][ordinal]) != frozen["fold"],
            int(arrays["branch"][ordinal]) != branch,
            int(arrays["episode"][ordinal]) != frozen["episode"],
            int(arrays["sample_id"][ordinal]) != sample_id,
            int(arrays["selection_order"][ordinal]) != selection,
            str(arrays["prediction_sha256"][ordinal]) != prediction_sha,
            str(arrays["target_sha256"][ordinal]) != target_sha,
        )):
            raise RuntimeError("metrics/event bijection")
        computed.append(expected["event_canonical_sha256"])
        fold_counts[frozen["fold"]] += 1
    if fold_counts != [200] * 5:
        raise RuntimeError("audited fold counts")
    fold_receipts = []
    for fold in range(5):
        receipt_path = staged_root / f"fold_{fold}_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("fold") != fold or receipt.get("rows") != 200 or receipt.get("event_digest_sha256") != canonical_sha(computed[fold * 200:(fold + 1) * 200]):
            raise RuntimeError("fold receipt")
        fold_receipts.append({"path": receipt_path.name, "sha256": sha(receipt_path), "logical_bytes": receipt_path.stat().st_size})
    return {
        "format": FORMAT,
        "status": "passed_independent_actual_oof_recomputation",
        "passed": True,
        "events": 1000,
        "folds": 5,
        "fold_row_counts": fold_counts,
        "ordered_event_digest_sha256": canonical_sha(computed),
        "metrics_npz_sha256": sha(staged_root / "metrics.npz"),
        "events_sha256": sha(staged_root / "oof_call_events.ndjson"),
        "fold_receipts": fold_receipts,
        "cache_sha256": sha(cache_path),
        "qualification_mutated": False,
        "model_runtime_delegate_invocations": 0,
        "phase_a_replay_invocations": 0,
        "training_invocations": 0,
        "reward_read_invocations": 0,
        "dev_hidden_final_outcome_read_invocations": 0,
        "submission_invocations": 0,
    }


def synthetic_self_test() -> bool:
    sample = {"a": 1, "b": [2, 3]}
    return canonical_sha(sample) == hashlib.sha256(b'{"a":1,"b":[2,3]}').hexdigest()


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--synthetic-self-test"]:
        raise SystemExit("auditor is import-only inside the authorized actual-OOF boundary")
    print(json.dumps({"passed": synthetic_self_test()}, sort_keys=True))
