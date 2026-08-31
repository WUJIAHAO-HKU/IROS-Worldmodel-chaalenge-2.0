#!/usr/bin/env python3
"""Materialize the action-only/frozen-data execution preregistration for v482."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch


SEED = 1624
CONTRACT_SHA = "19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb"
SELECTION_SHA = "f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398"
ACTION_SOURCE_SHA = "0dd886524e8e65f9cd7d6ca46d8590cf68b9d0962c238168e5ae142e92e83fca"
ACTION_BOUNDS_SHA = "4471e2df7da03dffa758ef1597ac0db81bb6d77c6f9394beb193ec998fbbe69f"
BRANCHES = ("factual", "no_transport", "scale_0p4", "scale_1p25", "reverse_direction_0p4")
EXPECTED_FOLDS = {
    0: (28, 37, 49),
    1: (15, 20, 42),
    2: (25, 40, 46),
    3: (12, 33, 47),
    4: (30, 32, 44),
}
EXPECTED_DATASET_ROOT = "/root/v478_temporal8_dataset_seed1622_20260824"
FINAL_REPORT_FORMAT = "strict-track2-v478-public-train-temporal200-generation-report-v1"
FINAL_AUDIT_FORMAT = "strict-track2-v478-public-train-temporal200-final-audit-v1"
PREREG_FORMAT = "strict-track2-v482-temporal8-residual-preregistration-v1"
EXECUTION_PYTHON_LEXICAL = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
EXECUTION_PYTHON_LINK_TARGET = "/root/autodl-tmp/conda_envs/isaacsim51/bin/python"
EXECUTION_PYTHON_RESOLVED = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
EXECUTION_PYTHON_SHA256 = "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788"
EXECUTION_PYTHON_BYTES = 25555040
SUPERSEDED_FORMALS = (
    (
        Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v482_temporal8_residual_s0_seed1624_20260824/preregistration.json"),
        "a70731adf473582fb63a06e70efc61eb938d2045c2879ebc1a55a4de7c54a4f6",
        "non_execution_interpreter_framework_provenance",
    ),
    (
        Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v482_temporal8_residual_s0_r2_seed1624_20260824/preregistration.json"),
        "0431814403f8cace842765b17fbee098427187cdbfcb7cc05ffde986f0ece665",
        "unbound_postprocessing_materializer_provenance",
    ),
)
FINAL_INTEGRITY_KEYS = {
    "ten_batches",
    "exact200",
    "all_batch_integrity_passed",
    "whole_cumulative_attempt_wall_le_43200",
    "total_output_including_forensics_le_3221225472",
    "no_partial_batches",
    "no_effect_used_for_batch_pass_retry_filter",
}
FINAL_EFFECT_KEYS = {
    "all_200_contexts_have_at_least_three_of_four_transport_effects",
    "each_named_transport_effect_contexts_at_least_150",
    "computed_after_all200_integrity_complete",
    "never_used_for_batch_stop_resume_retry_filter",
}
FINAL_AUDIT_CHECK_KEYS = {
    "input_closure",
    "ten_independent_batch_audits",
    "exact200_9600_replays",
    "collection_integrity_passed",
    "technical_effect_whole200_only",
    "generation_report_exact",
    "dataset_tree_no_partial_or_extra",
    "whole_output_cap",
    "no_training_s1_rl",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_interpreter_and_supersession_evidence():
    lexical = EXECUTION_PYTHON_LEXICAL
    resolved = lexical.resolve(strict=True)
    if (
        Path(sys.executable) != lexical
        or not lexical.is_symlink()
        or os.readlink(lexical) != EXECUTION_PYTHON_LINK_TARGET
        or resolved != EXECUTION_PYTHON_RESOLVED
        or not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size != EXECUTION_PYTHON_BYTES
        or sha(resolved) != EXECUTION_PYTHON_SHA256
        or platform.python_version() != "3.11.15"
        or np.__version__ != "1.26.4"
        or torch.__version__ != "2.7.0+cu128"
    ):
        raise RuntimeError("v482 execution interpreter provenance drift")
    superseded = []
    for path, digest, reason in SUPERSEDED_FORMALS:
        if (
            not path.is_file()
            or path.is_symlink()
            or sha(path) != digest
            or {child.name for child in path.parent.iterdir()} != {"preregistration.json"}
        ):
            raise RuntimeError("v482 superseded unexecuted formal ancestry drift")
        superseded.append({
            "path": str(path.resolve()),
            "sha256": digest,
            "executed": False,
            "superseded": True,
            "supersession_reason": reason,
            "parent_directory_exact_files": ["preregistration.json"],
        })
    return ({
        "lexical_path": str(lexical),
        "lexical_is_symlink": True,
        "symlink_target": EXECUTION_PYTHON_LINK_TARGET,
        "resolved_path": str(resolved),
        "resolved_is_regular_file": True,
        "resolved_bytes": EXECUTION_PYTHON_BYTES,
        "resolved_sha256": EXECUTION_PYTHON_SHA256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }, superseded)


def arrsha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def directory_target_sha(path):
    path = Path(path).resolve()
    items = []
    for parent, dirs, files in os.walk(path, followlinks=False):
        for name in sorted(dirs + files):
            item = Path(parent) / name
            relative = item.relative_to(path).as_posix()
            if item.is_symlink():
                items.append(["link", relative, os.readlink(item)])
            elif item.is_file():
                items.append(["file", relative, sha(item)])
            else:
                items.append(["dir", relative, None])
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_record(base, base_kind, path, allowed_root, relative=None, declared_sha=None):
    base = Path(base).absolute()
    lexical = (
        base / (relative if relative is not None else Path(path).absolute().relative_to(base))
    ).absolute()
    resolved = lexical.resolve()
    if (
        os.path.commonpath((str(base), str(lexical))) != str(base)
        or not inside(resolved, allowed_root)
        or not resolved.is_file()
    ):
        raise RuntimeError(f"v482 closure path escape/missing: {lexical} -> {resolved}")
    digest = sha(resolved)
    if declared_sha is not None and digest != declared_sha:
        raise RuntimeError(f"v482 manifest SHA drift: {lexical}")
    return {
        "record_type": "file",
        "base_kind": base_kind,
        "relative": str(lexical.relative_to(base)),
        "lexical_path": str(lexical),
        "link_target": os.readlink(lexical) if lexical.is_symlink() else None,
        "resolved_path": str(resolved),
        "target_sha": digest,
    }


def release_closure(release, artifact_root):
    release, artifact_root = Path(release).absolute(), Path(artifact_root).resolve()
    records, seen_files, seen_dirs, active = [], set(), set(), set()

    def visit(logical, relative):
        resolved = logical.resolve()
        if not inside(resolved, artifact_root) or not resolved.is_dir():
            raise RuntimeError("v482 release directory escape/missing")
        key = (resolved.stat().st_dev, resolved.stat().st_ino)
        if key in active or key in seen_dirs:
            raise RuntimeError("v482 release symlink cycle/duplicate")
        active.add(key)
        seen_dirs.add(key)
        for child in sorted(logical.iterdir(), key=lambda item: item.name):
            rel, target = relative / child.name, child.resolve()
            if not inside(target, artifact_root):
                raise RuntimeError("v482 release target escape")
            if child.is_symlink():
                records.append({
                    "record_type": "directory_symlink" if target.is_dir() else "file_symlink",
                    "base_kind": "v169_release_link",
                    "relative": str(rel),
                    "lexical_path": str(child.absolute()),
                    "link_target": os.readlink(child),
                    "resolved_path": str(target),
                    "target_sha": directory_target_sha(target) if target.is_dir() else sha(target),
                })
            if target.is_dir():
                visit(child, rel)
            elif target.is_file():
                target_key = (target.stat().st_dev, target.stat().st_ino)
                if target_key in seen_files:
                    raise RuntimeError("v482 duplicate release file target")
                seen_files.add(target_key)
                if not child.is_symlink():
                    records.append(file_record(
                        release,
                        "v169_release_target" if not inside(target, release.resolve()) else "v169_release",
                        child,
                        artifact_root,
                        rel,
                    ))
            else:
                raise RuntimeError("v482 unsupported release target")
        active.remove(key)

    visit(release, Path("."))
    keys = [(row["record_type"], row["relative"], row["resolved_path"]) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("v482 duplicate release closure record")
    return sorted(records, key=lambda row: (row["relative"], row["record_type"]))


def library_closure(release, library):
    release, library = Path(release).absolute(), Path(library).resolve()
    base = (release / "v168_release/base_release").resolve()
    manifest_path = base / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = [
        file_record(base, "v169_base_release", base / relative, library, declared_sha=digest)
        for relative, digest in sorted(manifest["sha256"].items())
    ]
    split = manifest["retrieval"]["split_manifest"]
    records.append(file_record(library, "v169_library", library / split, library))
    windows = (library / manifest["retrieval"]["windows_directory"]).resolve()
    if not inside(windows, library):
        raise RuntimeError("v482 retrieval windows escape")
    records.extend(
        file_record(library, "v169_library", path, library)
        for path in sorted(windows.rglob("*")) if path.is_file()
    )
    keys = [(row["base_kind"], row["relative"], row["resolved_path"]) for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("v482 duplicate library closure record")
    return records


def verify_v169_independently(value, release, library):
    if not isinstance(value, dict):
        raise RuntimeError("v482 v169 closure schema")
    release, library = Path(release).absolute(), Path(library).resolve()
    if Path(value.get("release", "")).absolute() != release or Path(value.get("library", "")).resolve() != library:
        raise RuntimeError("v482 v169 closure roots")
    actual_release = release_closure(release, release.resolve().parent)
    actual_library = library_closure(release, library)
    if value.get("release_files") != actual_release or value.get("library_files") != actual_library:
        raise RuntimeError("v482 independently recomputed v169 inventory drift")
    if sum(row["record_type"] == "directory_symlink" for row in actual_release) != 4:
        raise RuntimeError("v482 v169 directory-symlink count")
    if sum(row["record_type"] == "file_symlink" for row in actual_release) != 2:
        raise RuntimeError("v482 v169 file-symlink count")
    if len({row["resolved_path"] for row in actual_release}) != len(actual_release):
        raise RuntimeError("v482 v169 duplicate resolved release target")
    manifest = release / "v169_arm_routed_manifest.json"
    library_manifest = Path(value["library_manifest"])
    runtime_source = Path(value["runtime_source"]["path"])
    if (
        not manifest.is_file()
        or sha(manifest) != value["release_manifest_sha256"]
        or not library_manifest.is_file()
        or sha(library_manifest) != value["library_manifest_sha256"]
        or not runtime_source.is_file()
        or sha(runtime_source) != value["runtime_source"]["sha256"]
    ):
        raise RuntimeError("v482 v169 manifest/runtime closure")
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, {
        "release_records": len(actual_release),
        "library_records": len(actual_library),
        "directory_symlinks": 4,
        "file_symlinks": 2,
        "canonical_closure_digest_sha256": digest,
    }


def validate_selection(selection):
    if (
        selection.get("format") != "strict-track2-v478-window-eligible-action-only-selection-v1"
        or selection.get("status") != "selected_action_only_not_collected"
        or selection.get("guards", {}).get("training_started") is not False
        or selection.get("guards", {}).get("hidden_final_submission_used") is not False
    ):
        raise RuntimeError("v482 selection authority/schema")
    rows = selection.get("contexts", [])
    if len(rows) != 200 or len({(int(row["episode"]), int(row["start"])) for row in rows}) != 200:
        raise RuntimeError("v482 selection identity")
    fold_by_episode = {episode: fold for fold, episodes in EXPECTED_FOLDS.items() for episode in episodes}
    for row in rows:
        episode = int(row["episode"])
        if (
            episode not in fold_by_episode
            or int(row["fold"]) != fold_by_episode[episode]
            or int(row["batch_id"]) not in range(10)
            or int(row["phase_bin"]) not in range(5)
            or int(row["motion_bin"]) not in range(4)
        ):
            raise RuntimeError("v482 selection fold/bin identity")
    def count(key, value, subset=rows):
        return sum(int(row[key]) == value for row in subset)
    if [count("fold", fold) for fold in range(5)] != [40] * 5:
        raise RuntimeError("v482 selection fold40")
    if [count("phase_bin", phase) for phase in range(5)] != [40] * 5:
        raise RuntimeError("v482 selection phase40")
    if [count("motion_bin", motion) for motion in range(4)] != [50] * 4:
        raise RuntimeError("v482 selection motion50")
    for fold in range(5):
        subset = [row for row in rows if int(row["fold"]) == fold]
        if [count("phase_bin", phase, subset) for phase in range(5)] != [8] * 5:
            raise RuntimeError("v482 selection fold-phase8")
        if [count("motion_bin", motion, subset) for motion in range(4)] != [10] * 4:
            raise RuntimeError("v482 selection fold-motion10")
    for batch in range(10):
        subset = [row for row in rows if int(row["batch_id"]) == batch]
        if (
            len(subset) != 20
            or [count("fold", fold, subset) for fold in range(5)] != [4] * 5
            or [count("phase_bin", phase, subset) for phase in range(5)] != [4] * 5
            or [count("motion_bin", motion, subset) for motion in range(4)] != [5] * 4
        ):
            raise RuntimeError("v482 selection batch margins")
    episode_counts = {}
    for episode in fold_by_episode:
        starts = sorted(int(row["start"]) for row in rows if int(row["episode"]) == episode)
        episode_counts[str(episode)] = len(starts)
        if len(starts) not in range(6, 18) or any(b - a <= 1 for a, b in zip(starts, starts[1:])):
            raise RuntimeError("v482 selection episode bound/nonadjacency")
    if max(episode_counts.values()) - min(episode_counts.values()) != 11:
        raise RuntimeError("v482 selection tight range11")
    return {
        "fold_episodes": {str(key): list(value) for key, value in EXPECTED_FOLDS.items()},
        "fold_counts": [40] * 5,
        "phase_counts": [40] * 5,
        "motion_counts": [50] * 4,
        "episode_counts": episode_counts,
        "batch_count": 10,
        "batch_size": 20,
    }


def reject_placeholders(value, location="root"):
    forbidden = ("REQUIRED", "PLACEHOLDER", "TBD", "TO_BE_FILLED")
    if value is None:
        raise RuntimeError(f"v482 placeholder/null at {location}")
    if isinstance(value, str) and any(token in value.upper() for token in forbidden):
        raise RuntimeError(f"v482 placeholder string at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_placeholders(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_placeholders(child, f"{location}[{index}]")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY); os.fsync(descriptor); os.close(descriptor)


def schedule(ids, seed):
    ids = np.asarray(ids, np.int64)
    ordered = ids[np.random.default_rng(seed).permutation(len(ids))]
    if len(ordered) % 2:
        raise RuntimeError("v482 schedule divisibility")
    return ordered.reshape(-1, 2)


def schedule_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, np.int64).view(np.uint8)).hexdigest()


def donor_map(contexts, fit_ids, query_ids):
    fit_set = set(map(int, fit_ids))
    groups = {}
    for index in fit_ids:
        cid = int(index) // 5
        groups.setdefault(contexts[cid]["phase"], set()).add(cid)
    ordered = {
        phase: sorted(ids, key=lambda cid: (contexts[cid]["episode"], contexts[cid]["start"], cid))
        for phase, ids in groups.items()
    }
    result = {}
    for index in query_ids:
        index = int(index); cid = index // 5; branch = index % 5
        query = contexts[cid]; group = ordered[query["phase"]]
        key = (query["episode"], query["start"], cid)
        insertion = next((i for i, candidate in enumerate(group)
                          if (contexts[candidate]["episode"], contexts[candidate]["start"], candidate) > key), 0)
        donor = next((group[(insertion + offset) % len(group)] for offset in range(len(group))
                      if contexts[group[(insertion + offset) % len(group)]]["episode"] != query["episode"]), None)
        if donor is None or 5 * donor + branch not in fit_set:
            raise RuntimeError("v482 phase donor closure")
        result[index] = 5 * donor + branch
    return result


def state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(str(value.dtype).encode()); digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode()); digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("contract", "selection", "dataset-root", "final-report", "final-audit",
                 "trainer", "runtime", "auditor", "packager", "release-auditor", "v169-release",
                 "v169-library", "v169-closure", "release-manifest", "library-manifest",
                 "batch-audit-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    execution_interpreter, superseded_formals = execution_interpreter_and_supersession_evidence()
    obsolete = Path(__file__).with_name("v482_temporal8_residual_s0_contract.json")
    if obsolete.exists():
        raise RuntimeError("v482 superseded staging S0 contract is present")
    source_paths = {
        "prepare": Path(__file__).resolve(),
        "trainer": args.trainer.resolve(),
        "runtime": args.runtime.resolve(),
        "s0_auditor": args.auditor.resolve(),
        "packager": args.packager.resolve(),
        "release_auditor": args.release_auditor.resolve(),
    }
    if (
        len(set(source_paths.values())) != 6
        or any(not path.is_file() or path.is_symlink() for path in source_paths.values())
    ):
        raise RuntimeError("v482 exact six-source closure")
    source_closure = {
        name: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        for name, path in source_paths.items()
    }
    contract = json.loads(args.contract.read_text())
    selection = json.loads(args.selection.read_text())
    final_report = json.loads(args.final_report.read_text())
    final_audit = json.loads(args.final_audit.read_text())
    selection_evidence = validate_selection(selection)
    runtime_spec = importlib.util.spec_from_file_location("v482_pre_runtime", args.runtime)
    module = importlib.util.module_from_spec(runtime_spec); runtime_spec.loader.exec_module(module)
    v169_closure = json.loads(args.v169_closure.read_text())
    v169_digest, v169_inventory = verify_v169_independently(
        v169_closure, args.v169_release, args.v169_library
    )
    root = args.dataset_root.resolve()
    expected_report_path = root / "generation_report.json"
    if (
        sha(args.contract) != CONTRACT_SHA
        or contract.get("format") != "strict-track2-v482-public-train-temporal-film-residual-model-design-contract-v5"
        or contract.get("status") != "frozen_design_pending_temporal200_final_pass_and_separate_execution_preregistration"
        or sha(args.selection) != SELECTION_SHA
        or len(selection.get("contexts", [])) != 200
        or root != Path(EXPECTED_DATASET_ROOT)
        or args.final_report.resolve() != expected_report_path
        or final_report.get("format") != FINAL_REPORT_FORMAT
        or final_audit.get("format") != FINAL_AUDIT_FORMAT
        or final_report.get("passed") is not True
        or final_audit.get("passed") is not True
        or final_report.get("collection_integrity_passed") is not True
        or final_report.get("technical_effect_diagnostic_qualification_passed") is not True
        or final_report.get("guards", {}).get("temporal_model_design_authorized_if_pass") is not True
        or final_audit.get("authorization", {}).get("temporal_model_design_authorized") is not True
        or final_audit.get("generation_report_sha256") != sha(args.final_report)
        or set(final_report.get("integrity_checks", {})) != FINAL_INTEGRITY_KEYS
        or set(final_report.get("technical_effect_diagnostic_checks", {})) != FINAL_EFFECT_KEYS
        or set(final_audit.get("checks", {})) != FINAL_AUDIT_CHECK_KEYS
        or not all(final_report.get("integrity_checks", {}).values())
        or not all(final_report.get("technical_effect_diagnostic_checks", {}).values())
        or not all(final_audit.get("checks", {}).values())
        or final_report.get("guards", {}).get("training_authorized") is not False
        or final_report.get("guards", {}).get("s1_authorized") is not False
        or final_report.get("guards", {}).get("policy_updates") != 0
        or final_report.get("guards", {}).get("rl_authorized") is not False
        or final_audit.get("authorization", {}).get("training_authorized") is not False
        or final_audit.get("authorization", {}).get("s1_authorized") is not False
        or final_audit.get("authorization", {}).get("zero_update_authorized") is not False
        or final_audit.get("authorization", {}).get("policy_updates") != 0
        or final_audit.get("authorization", {}).get("rl_authorized") is not False
        or v169_digest != "5fc181bf1049b0e763ef47716f5e3356445de8eb6b2eb3de07dd2d70cd4346f3"
        or Path(v169_closure["release"]).resolve() != args.v169_release.resolve()
        or Path(v169_closure["library"]).resolve() != args.v169_library.resolve()
        or args.release_manifest.resolve() != (args.v169_release / "v169_arm_routed_manifest.json").resolve()
        or args.library_manifest.resolve() != Path(v169_closure["library_manifest"]).resolve()
        or sha(args.release_manifest) != v169_closure["release_manifest_sha256"]
        or sha(args.library_manifest) != v169_closure["library_manifest_sha256"]
    ):
        raise RuntimeError("v482 execution remains blocked")
    # Pass 1 is action-only: freeze normalization before reading any RGB/target member
    # or hashing the full compressed row file.
    contexts, actions, action_manifest = [], [], []
    for order, spec in enumerate(selection["contexts"]):
        episode, start, batch = int(spec["episode"]), int(spec["start"]), int(spec["batch_id"])
        rowdir = root / f"batch_{batch:03d}" / "rows" / f"episode{episode}_start{start:05d}"
        npz_path = rowdir / "temporal.npz"
        with np.load(npz_path, allow_pickle=False) as data:
            history = np.asarray(data["history_actions"])
            future = np.asarray(data["future_actions"])
            if (history.shape, history.dtype) != ((4, 14), np.dtype("float32")):
                raise RuntimeError("v482 history schema")
            if (future.shape, future.dtype) != ((6, 8, 14), np.dtype("float32")):
                raise RuntimeError("v482 future schema")
            if not np.isfinite(history).all() or not np.isfinite(future).all() or not np.array_equal(future[0], future[5]):
                raise RuntimeError("v482 action/duplicate closure")
            if arrsha(history) != spec["history_action_sha256"]:
                raise RuntimeError("v482 history identity")
            for branch, name in enumerate(BRANCHES):
                if arrsha(future[branch]) != spec["branch_action_sha256"][name]:
                    raise RuntimeError("v482 branch identity")
            actions.append(np.concatenate([history] + [future[x] for x in range(5)], axis=0))
            action_manifest.append({
                "selection_order": order,
                "episode": episode,
                "start": start,
                "history_action_sha256": arrsha(history),
                "branch_action_sha256": {
                    name: arrsha(future[branch]) for branch, name in enumerate(BRANCHES)
                },
            })
        contexts.append({"episode": episode, "start": start, "fold": int(spec["fold"]), "phase": int(spec["phase_bin"])})
    source = np.ascontiguousarray(np.stack(actions), np.float32)
    lower, upper = source.min((0, 1)).astype(np.float32), source.max((0, 1)).astype(np.float32)
    if (
        source.shape != (200, 44, 14)
        or arrsha(source) != ACTION_SOURCE_SHA
        or arrsha(np.concatenate((lower, upper))) != ACTION_BOUNDS_SHA
        or np.any(upper < lower) or lower[13] != 0 or upper[13] != 0
    ):
        raise RuntimeError("v482 action normalization source")
    if np.any(source[..., upper == lower] != lower[upper == lower]):
        raise RuntimeError("v482 zero span source")
    action_manifest_sha = hashlib.sha256(
        json.dumps(action_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Pass 2 may now bind the public temporal RGB context and full immutable tree.
    context_manifest, repeat_hashes = [], []
    for order, spec in enumerate(selection["contexts"]):
        episode, start, batch = int(spec["episode"]), int(spec["start"]), int(spec["batch_id"])
        rowdir = root / f"batch_{batch:03d}" / "rows" / f"episode{episode}_start{start:05d}"
        npz_path, receipt_path = rowdir / "temporal.npz", rowdir / "receipt.json"
        receipt = json.loads(receipt_path.read_text()); npz_digest = sha(npz_path)
        if receipt.get("npz_sha256") != npz_digest: raise RuntimeError("v482 row receipt/NPZ closure drift")
        with np.load(npz_path, allow_pickle=False) as data:
            stored_context = np.asarray(data["pre_future_context_rgb"])
        if stored_context.shape != (6, 8, 256, 256, 3) or stored_context.dtype != np.uint8 or not stored_context.flags.c_contiguous:
            raise RuntimeError("v482 authoritative temporal context schema")
        redundancy_count = sum(
            np.array_equal(stored_context[branch, frame], stored_context[0, 0])
            for branch in range(6) for frame in range(8)
        )
        if redundancy_count != 48 or not np.array_equal(stored_context, np.broadcast_to(stored_context[0, 0], stored_context.shape)):
            raise RuntimeError("v482 stored temporal context broadcast drift")
        context = np.ascontiguousarray(stored_context[0, 0]); repeat5 = np.repeat(context[None], 5, axis=0)
        if repeat5.shape != (5, 256, 256, 3) or repeat5.dtype != np.uint8 or not repeat5.flags.c_contiguous:
            raise RuntimeError("v482 repeat5 construction")
        context_manifest.append({
            "selection_order": order, "episode": episode, "start": start, "batch_id": batch,
            "row_npz_path": str(npz_path.resolve()), "row_npz_sha256": npz_digest,
            "row_receipt_path": str(receipt_path.resolve()), "row_receipt_sha256": sha(receipt_path),
            "redundancy_identity_gate": True,
            "redundancy_equality_count": redundancy_count,
            "stored_pre_future_context_rgb_sha256": arrsha(stored_context),
            "pre_future_context_rgb_sha256": arrsha(context),
            "constructed_repeat5_context_sha256": arrsha(repeat5),
        })
        repeat_hashes.append(arrsha(repeat5))
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): raise RuntimeError("v482 dataset symlink")
        if path.is_file(): files.append({"relative": path.relative_to(root).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    dataset_bytes = sum(row["bytes"] for row in files)
    allowed_top = {*(f"batch_{batch:03d}" for batch in range(10)), "generation_report.json", "_forensic_uncommitted"}
    actual_top = {path.name for path in root.iterdir()}
    if (
        not {*(f"batch_{batch:03d}" for batch in range(10)), "generation_report.json"}.issubset(actual_top)
        or not actual_top.issubset(allowed_top)
        or any(root.glob("batch_*.partial"))
        or any(not (root / f"batch_{batch:03d}").is_dir() for batch in range(10))
    ):
        raise RuntimeError("v482 final dataset exact top-level tree")
    if final_report.get("output_bytes_including_forensics") != dataset_bytes - args.final_report.stat().st_size or final_audit.get("output_bytes_including_forensics") != dataset_bytes:
        raise RuntimeError("v482 final dataset byte closure drift")
    batch_audits = {}
    for batch in range(10):
        path = args.batch_audit_dir / f"batch_{batch:03d}_audit.json"
        if sha(path) != final_audit["batch_audits_sha256"][f"batch_{batch:03d}"] or json.loads(path.read_text()).get("passed") is not True:
            raise RuntimeError("v482 final batch audit closure drift")
        batch_audits[f"batch_{batch:03d}"] = {"path": str(path.resolve()), "sha256": sha(path)}
    actual_batch_reports = {
        f"batch_{batch:03d}": sha(root / f"batch_{batch:03d}" / "batch_report.json")
        for batch in range(10)
    }
    if final_report.get("batch_reports_sha256") != actual_batch_reports:
        raise RuntimeError("v482 final report batch-report closure")
    tree_pairs = [[row["relative"], row["sha256"]] for row in files]
    tree_sha256 = hashlib.sha256(
        json.dumps(tree_pairs, separators=(",", ":")).encode()
    ).hexdigest()
    schedules, shuffles, initial_states = [], [], []
    model_parameter_count = None
    for fold in range(5):
        fit = np.asarray([i for i in range(1000) if contexts[i // 5]["fold"] != fold], np.int64)
        hold = np.asarray([i for i in range(1000) if contexts[i // 5]["fold"] == fold], np.int64)
        epoch = schedule(fit, SEED + fold)
        fit_donor, hold_donor = donor_map(contexts, fit, fit), donor_map(contexts, fit, hold)
        if epoch.shape != (400, 2) or len(fit_donor) != 800 or len(hold_donor) != 200:
            raise RuntimeError("v482 schedule/donor shape")
        schedules.append({
            "fold": fold,
            "seed": SEED + fold,
            "shape": [400, 2],
            "sha256": schedule_sha(epoch),
            "steps_per_head": len(epoch),
            "heads": ["action", "context_only", "phase_shuffle"],
            "identical_order_for_all_heads": True,
        })
        shuffles.append({
            "fold": fold,
            "fit_sha256": hashlib.sha256(json.dumps(fit_donor, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "holdout_sha256": hashlib.sha256(json.dumps(hold_donor, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "fit_count": len(fit_donor),
            "holdout_count": len(hold_donor),
            "different_episode_required": True,
        })
        torch.manual_seed(SEED + fold)
        model = module.TemporalResidualUNet128FiLM(16)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if count > 1200000 or any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise RuntimeError("v482 initial model parameter/nonfinite guard")
        if model_parameter_count is None:
            model_parameter_count = count
        elif model_parameter_count != count:
            raise RuntimeError("v482 model parameter-count drift")
        initial_states.append({
            "fold": fold,
            "seed": SEED + fold,
            "state_sha256": state_sha(copy.deepcopy(model.state_dict())),
            "shared_by_all_three_heads": True,
        })
    all_epoch = schedule(np.arange(1000, dtype=np.int64), SEED + 1000)
    if all_epoch.shape != (500, 2):
        raise RuntimeError("v482 all200 schedule shape")
    torch.manual_seed(SEED + 1000)
    all_model = module.TemporalResidualUNet128FiLM(16)
    if sum(parameter.numel() for parameter in all_model.parameters() if parameter.requires_grad) != model_parameter_count:
        raise RuntimeError("v482 all200 model capacity drift")
    all_state = state_sha(copy.deepcopy(all_model.state_dict()))
    source_alias = {
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": CONTRACT_SHA,
        "trainer_path": source_closure["trainer"]["path"],
        "trainer_sha256": source_closure["trainer"]["sha256"],
        "runtime_path": source_closure["runtime"]["path"],
        "runtime_sha256": source_closure["runtime"]["sha256"],
        "auditor_path": source_closure["s0_auditor"]["path"],
        "auditor_sha256": source_closure["s0_auditor"]["sha256"],
        "packager_path": source_closure["packager"]["path"],
        "packager_sha256": source_closure["packager"]["sha256"],
        "release_auditor_path": source_closure["release_auditor"]["path"],
        "release_auditor_sha256": source_closure["release_auditor"]["sha256"],
        "prepare_path": source_closure["prepare"]["path"],
        "prepare_sha256": source_closure["prepare"]["sha256"],
    }
    alias_pairs = {
        "trainer": "trainer",
        "runtime": "runtime",
        "auditor": "s0_auditor",
        "packager": "packager",
        "release_auditor": "release_auditor",
        "prepare": "prepare",
    }
    for alias_name, closure_name in alias_pairs.items():
        if (
            source_alias[f"{alias_name}_path"] != source_closure[closure_name]["path"]
            or source_alias[f"{alias_name}_sha256"] != source_closure[closure_name]["sha256"]
        ):
            raise RuntimeError("v482 source alias drift")
    payload = {
        "format": PREREG_FORMAT,
        "status": "preregistered_public_train_temporal_s0_authorized",
        "seed": SEED,
        "design_contract": {
            "path": str(args.contract.resolve()),
            "sha256": CONTRACT_SHA,
            "format": contract["format"],
            "status": contract["status"],
        },
        "source": source_alias,
        "source_alias_contract": {
            "compatibility_consumers": ["trainer", "s0_auditor", "packager"],
            "contract_fields_equal_design_contract": True,
            "six_code_path_and_sha_fields_equal_execution_source_six": True,
            "alias_mapping": alias_pairs,
            "source_alias_sha256": hashlib.sha256(
                json.dumps(source_alias, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "execution_source_six": source_closure,
        "execution_source_six_count": 6,
        "execution_source_six_digest_sha256": hashlib.sha256(
            json.dumps(source_closure, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "framework": {
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "execution_interpreter": execution_interpreter,
            "pythonhashseed_required": "0",
            "torch_deterministic_algorithms_required": True,
            "cublas_workspace_config_required": ":4096:8",
        },
        "supersession": {
            "only_authorized_preregistration": "this native r3 preregistration after independent audit",
            "previous_formals": superseded_formals,
            "all_previous_executed": False,
            "native_prepare_author": {
                "path": source_closure["prepare"]["path"],
                "sha256": source_closure["prepare"]["sha256"],
                "postprocessing_materializer_used": False,
            },
        },
        "dataset": {
            "root": str(root), "selection": {"path": str(args.selection.resolve()), "sha256": SELECTION_SHA},
            "selection_recomputed_evidence": selection_evidence,
            "final_report": {"path": str(args.final_report.resolve()), "sha256": sha(args.final_report)},
            "final_audit": {"path": str(args.final_audit.resolve()), "sha256": sha(args.final_audit)},
            "batch_audits": batch_audits,
            "files": files,
            "tree_algorithm": "sha256(canonical compact JSON list of [relative POSIX path,file_sha256], sorted by relative path)",
            "tree_sha256": tree_sha256,
            "file_count": len(files), "bytes": dataset_bytes,
            "batch_reports_sha256": actual_batch_reports,
            "no_symlinks_or_partial_batches": True,
        },
        "temporal_contexts": context_manifest,
        "v169_cache": {
            "context_repeat5_digest_sha256": hashlib.sha256("".join(repeat_hashes).encode()).hexdigest(),
            "context_manifest_sha256": hashlib.sha256(json.dumps(context_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "exact_scalar_calls": 1000,
            "redundancy_rows_passed": 200,
            "redundancy_equality_count": 9600,
        },
        "action_bounds": {
            "source_array_shape": [200, 44, 14], "source_array_sha256": arrsha(source),
            "ordered_action_manifest_sha256": action_manifest_sha,
            "lower": lower.tolist(), "upper": upper.tolist(),
            "lower_upper_sha256": arrsha(np.concatenate((lower, upper))),
            "lower_float32_bits_hex": [f"0x{int(value):08x}" for value in lower.astype("<f4").view("<u4")],
            "upper_float32_bits_hex": [f"0x{int(value):08x}" for value in upper.astype("<f4").view("<u4")],
            "zero_span_dimensions": np.flatnonzero(upper == lower).tolist(),
            "dimension13_exact_zero_span": bool(lower[13] == 0 and upper[13] == 0),
        },
        "schedules": schedules, "phase_shuffle": shuffles,
        "initial_state_sha256": [row["state_sha256"] for row in initial_states],
        "initial_states_evidence": initial_states,
        "trainable_parameter_count": model_parameter_count,
        "all200_schedule_sha256": schedule_sha(all_epoch),
        "all200_initial_state_sha256": all_state,
        "all200_schedule": {
            "seed": SEED + 1000,
            "shape": [500, 2],
            "sha256": schedule_sha(all_epoch),
            "steps": 500,
        },
        "all200_initial_state": {"seed": SEED + 1000, "state_sha256": all_state},
        "v169": {
            "release_path": str(args.v169_release.resolve()),
            "library_path": str(args.v169_library.resolve()),
            "release_manifest_path": str(args.release_manifest.resolve()), "release_manifest_sha256": sha(args.release_manifest),
            "library_manifest_path": str(args.library_manifest.resolve()), "library_manifest_sha256": sha(args.library_manifest),
            "required_closure_digest": "5fc181bf1049b0e763ef47716f5e3356445de8eb6b2eb3de07dd2d70cd4346f3",
            "closure_path": str(args.v169_closure.resolve()), "closure_sha256": sha(args.v169_closure),
            "closure_digest": v169_digest,
            "independent_inventory_recomputation": v169_inventory,
        },
        "launcher_authority_and_timeouts": {
            "launcher_is_postregistration_orchestrator": True,
            "launcher_sha256_is_not_in_this_preregistration_to_avoid_self_reference": True,
            "launcher_must_bind_this_preregistration_path_and_sha256_before_execution": True,
            "one_shot_attempt": True,
            "authorized_stages_in_order": [
                "immutable_preflight",
                "ordered_1000_call_v169_cache",
                "episode_grouped_outer5_three_head_s0",
                "all200_action_head_only_if_complete_s0_pass",
                "package_and_static_release_audit_only_if_all200_passes",
                "restore_v218_and_write_terminal_receipt",
            ],
            "stage_timeout_seconds": {
                "v169_cache": 1800,
                "each_fold_all_three_heads": 1200,
                "s0_whole": 7200,
                "all200": 1200,
                "full_chain": 8400,
                "term_to_kill_grace": 15,
            },
            "resource_gates": {
                "cpu_affinity": "0-11",
                "total_threads": 6,
                "minimum_shm_free_bytes": 6442450944,
                "gpu_peak_mib_hard_max": 28672,
                "persistent_release_bytes_hard_max": 536870912,
                "no_oom_retry_or_batch_change": True,
            },
            "failure": "TERM process group, bounded join, KILL/join, verify no children/GPU process, atomic failure receipt, restore v218; no retry under this lineage",
            "metric_failure": "a nonfinite prediction/loss/gradient/metric is a terminal execution failure, never a fold FAIL",
            "second_failed_fold": "terminal S0 candidate failure because 4/5 is unreachable; no remaining fold/all200/package",
            "success_receipt_after_v218_restore_health_only": True,
        },
        "guards": {"public_train_only": True, "training_authorized": True, "s1_authorized": False,
                   "zero_update_authorized": False, "policy_updates": 0, "rl_authorized": False,
                   "dev_reward_success_outcome_final_hidden_used": False,
                   "submission_authorized": False},
    }
    reject_placeholders(payload)
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
