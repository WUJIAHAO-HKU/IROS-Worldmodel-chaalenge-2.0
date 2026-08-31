#!/usr/bin/env python3
"""Audit what the public WorldArena 2.0 sources do and do not prove.

The public full Wan/Pi0.5 GRPO example is a reproducible local reference
budget.  The benchmark description separately says that official held-out
sample counts and aggregation parameters are not public.  Keeping those two
facts separate prevents a local reproduction from being mislabeled as an
organizer-certified evaluation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


OFFICIAL_COMMIT = "6f5a981b34232fe77812b818a6ad7a4e6b8728ac"
EXPECTED = {
    "benchmark": "201d401c4440efb76c52719305abb821ccb4703f62387538f0480980b34269fe",
    "profile": "cb64564f2fb6498c65fd75102735229c28a30362927bfff7fa395db80c71794f",
    "full_config": "97c4a6944e55ffb0f263e02d8ef8eb9cd176b821aee6d5776fd59b3df19136fb",
    "http_smoke_config": "46ddd1906355d81dfb98bf95b182de26d0ce21d3bb41c195152ca0deb5d0df82",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(text: str, key: str) -> int:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([0-9]+)\s*(?:#.*)?$", text)
    if match is None:
        raise ValueError(f"missing integer key: {key}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.official_root.resolve()
    paths = {
        "benchmark": root / "assets/track2_benchmark_description_en.md",
        "profile": root / "assets/track2_official_profile.json",
        "full_config": root
        / "RL_env_benchmark/examples/embodiment/config/wan_robotwin_adjust_bottle_grpo_openpi_pi05.yaml",
        "http_smoke_config": root
        / "RL_env_benchmark/examples/embodiment/config/wan_robotwin_adjust_bottle_http_grpo_openpi_pi05.yaml",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    hash_checks = {name: hashes[name] == expected for name, expected in EXPECTED.items()}
    benchmark = paths["benchmark"].read_text(encoding="utf-8")
    full = paths["full_config"].read_text(encoding="utf-8")
    smoke = paths["http_smoke_config"].read_text(encoding="utf-8")
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))

    public_reference = {
        "max_epochs": scalar(full, "max_epochs"),
        "total_num_envs": scalar(full, "total_num_envs"),
        "group_size": scalar(full, "group_size"),
        "rollout_epoch": scalar(full, "rollout_epoch"),
        "max_episode_steps": scalar(full, "max_episode_steps"),
        "max_steps_per_rollout_epoch": scalar(full, "max_steps_per_rollout_epoch"),
        "save_interval": scalar(full, "save_interval"),
    }
    smoke_reference = {
        "max_epochs": scalar(smoke, "max_epochs"),
        "total_num_envs": scalar(smoke, "total_num_envs"),
        "group_size": scalar(smoke, "group_size"),
        "rollout_epoch": scalar(smoke, "rollout_epoch"),
        "max_episode_steps": scalar(smoke, "max_episode_steps"),
    }
    public_reference_checks = {
        "max_epochs_1000": public_reference["max_epochs"] == 1000,
        "total_num_envs_32": public_reference["total_num_envs"] == 32,
        "group_size_4": public_reference["group_size"] == 4,
        "rollout_epoch_8": public_reference["rollout_epoch"] == 8,
        "max_episode_steps_200": public_reference["max_episode_steps"] == 200,
        "max_steps_per_rollout_epoch_200": public_reference["max_steps_per_rollout_epoch"] == 200,
        "save_interval_10": public_reference["save_interval"] == 10,
    }
    smoke_checks = {
        "max_epochs_1": smoke_reference["max_epochs"] == 1,
        "total_num_envs_2": smoke_reference["total_num_envs"] == 2,
        "group_size_2": smoke_reference["group_size"] == 2,
        "rollout_epoch_1": smoke_reference["rollout_epoch"] == 1,
        "max_episode_steps_8": smoke_reference["max_episode_steps"] == 8,
    }
    disclosure_checks = {
        "organizer_fixes_interaction_budget": "Fix the policy starting point, optimization algorithm, and interaction budget" in benchmark,
        "official_sample_counts_not_public": "sample counts, and aggregation parameters are not included in public integration data" in benchmark,
        "participants_submit_no_policy": "Participants submit neither a robot policy nor a reward model" in benchmark,
    }
    service_limits = profile.get("limits", {})
    profile_checks = {
        "max_batch_size_8": service_limits.get("max_batch_size") == 8,
        "max_concurrency_1": service_limits.get("max_concurrency") == 1,
        "recommended_timeout_ms_600000": service_limits.get("recommended_timeout_ms") == 600000,
    }
    passed = all(hash_checks.values()) and all(public_reference_checks.values()) and all(
        smoke_checks.values()
    ) and all(disclosure_checks.values()) and all(profile_checks.values())
    report = {
        "format": "strict-track2-public-budget-provenance-v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "official_repository": "https://github.com/WorldArena2/WorldArena-2.0",
        "official_commit": OFFICIAL_COMMIT,
        "passed": passed,
        "source_paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "hash_checks": hash_checks,
        "public_full_reference_budget": public_reference,
        "public_full_reference_checks": public_reference_checks,
        "public_http_smoke_budget": smoke_reference,
        "public_http_smoke_checks": smoke_checks,
        "public_disclosure_checks": disclosure_checks,
        "official_profile_checks": profile_checks,
        "classification": {
            "local_1000_epoch_run": "official-public-full-reference-budget reproduction",
            "organizer_certified_hidden_evaluation": False,
            "reason": "The 1000-epoch values are byte-verified in the public full reference YAML, while the benchmark explicitly withholds held-out sample counts and aggregation parameters.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
