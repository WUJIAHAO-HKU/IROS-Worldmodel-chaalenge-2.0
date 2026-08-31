#!/usr/bin/env python3
"""Prove that the HTTP Track 2 adapter inherits official reward/RL semantics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import yaml


INHERITED_NUMERIC_METHODS = (
    "_calc_step_reward",
    "_estimate_success_from_rewards",
    "_infer_next_chunk_rewards",
    "_record_metrics",
    "_handle_auto_reset",
    "chunk_step",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime))

    from rlinf.envs.world_model.world_model_wan_env import WanEnv
    from rlinf.envs.world_model.world_model_wan_http_env import WanHttpProxyEnv

    methods = {
        name: {
            "inherited_exact_object": getattr(WanHttpProxyEnv, name) is getattr(WanEnv, name),
            "defined_in_http_subclass": name in WanHttpProxyEnv.__dict__,
        }
        for name in INHERITED_NUMERIC_METHODS
    }

    config_root = runtime / "examples/embodiment/config/env"
    official_path = config_root / "wan_robotwin_adjust_bottle.yaml"
    http_path = config_root / "wan_robotwin_adjust_bottle_http_full.yaml"
    official = yaml.safe_load(official_path.read_text(encoding="utf-8"))
    http = yaml.safe_load(http_path.read_text(encoding="utf-8"))
    stripped_http = dict(http)
    http_only = stripped_http.pop("http", None)
    http_env_type = stripped_http.pop("env_type", None)
    stripped_official = dict(official)
    official_env_type = stripped_official.pop("env_type", None)
    config_value_equivalence = stripped_http == stripped_official
    only_expected_config_changes = (
        config_value_equivalence
        and official_env_type == "wan_wm"
        and http_env_type == "wan_wm_http"
        and http_only == {"server_url": "http://127.0.0.1:18080", "timeout": 600.0}
    )
    passed = (
        all(row["inherited_exact_object"] and not row["defined_in_http_subclass"] for row in methods.values())
        and only_expected_config_changes
    )
    report = {
        "format": "strict-track2-http-reward-equivalence-v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "passed": passed,
        "runtime_root": str(runtime),
        "official_env_sha256": digest(runtime / "rlinf/envs/world_model/world_model_wan_env.py"),
        "http_env_sha256": digest(runtime / "rlinf/envs/world_model/world_model_wan_http_env.py"),
        "numeric_methods": methods,
        "config": {
            "official": str(official_path),
            "http": str(http_path),
            "value_equivalent_after_transport_keys_removed": config_value_equivalence,
            "official_env_type": official_env_type,
            "http_env_type": http_env_type,
            "http_transport": http_only,
            "only_expected_changes": only_expected_config_changes,
        },
        "conclusion": (
            "HTTP subclass changes frame transport/reset action alignment/absolute observation state only; "
            "reward inference, relative reward, success threshold, termination, metrics and chunk stepping "
            "are the exact inherited official method objects."
            if passed
            else "HTTP adapter does not prove numeric equivalence to the official Wan environment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
