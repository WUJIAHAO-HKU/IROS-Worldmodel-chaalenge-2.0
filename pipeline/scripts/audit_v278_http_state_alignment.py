#!/usr/bin/env python3
"""Verify that Track 2 world-model rollouts expose evolving absolute state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FakeHttpClient:
    def __init__(self, current_obs: torch.Tensor) -> None:
        self.current_obs = current_obs

    def chunk_step(self, _actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"current_obs": self.current_obs.clone()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.rlinf_root))
    from rlinf.envs.world_model.world_model_wan_http_env import WanHttpProxyEnv

    source = (
        args.rlinf_root
        / "rlinf/envs/world_model/world_model_wan_http_env.py"
    )
    text = source.read_text(encoding="utf-8")
    env = WanHttpProxyEnv.__new__(WanHttpProxyEnv)
    env.num_envs = 2
    env.action_dim = 14
    env.chunk = 8
    env.device = torch.device("cpu")
    env.image_size = (4, 4)
    env.task_descriptions = ["adjust bottle", "adjust bottle"]
    env.current_obs = torch.zeros((2, 3, 1, 13, 4, 4), dtype=torch.float32)
    env.condition_action = torch.arange(2 * 5 * 14, dtype=torch.float32).reshape(2, 5, 14)

    initial_expected = env.condition_action[:, -1].clone()
    initial_obs = env._wrap_obs()
    new_frames = torch.ones_like(env.current_obs)
    env._http_client = FakeHttpClient(new_frames)
    actions = torch.arange(2 * 8 * 14, dtype=torch.float32).reshape(2, 8, 14) + 1000
    env._infer_next_chunk_frames(actions)
    next_obs = env._wrap_obs()

    checks = {
        "source_exists": source.is_file(),
        "http_override_present": "def _wrap_obs(self):" in text,
        "latest_condition_action_used_as_state": "self.condition_action[:, -1]" in text,
        "reference_reset_action_injected": "def _inject_reference_actions" in text,
        "chunk_history_updated": "actions_tensor[:, -4:, :]" in text,
        "initial_state_runtime_exact": torch.equal(initial_obs["states"], initial_expected),
        "next_state_runtime_exact": torch.equal(next_obs["states"], actions[:, -1]),
        "history_runtime_exact": torch.equal(env.condition_action[:, 1:], actions[:, -4:]),
        "state_shape_14d": tuple(next_obs["states"].shape) == (2, 14),
        "state_cpu_contiguous": next_obs["states"].device.type == "cpu"
        and next_obs["states"].is_contiguous(),
    }
    report = {
        "format": "strict-track2-v278-http-state-alignment-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": sha256(source),
        "official_issue": "https://github.com/WorldArena2/WorldArena-2.0/issues/5",
        "contract": "Pi0.5 receives latest 14-D absolute joint action as state at reset and after every generated chunk",
        "checks": checks,
        "passed": all(checks.values()),
        "policy_outcomes_read": False,
        "official_submission": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
