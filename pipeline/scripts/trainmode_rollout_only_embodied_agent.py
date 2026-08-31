#!/usr/bin/env python3
"""Run RLinf training-mode world-model rollouts without an actor update."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def scalar(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    return value


@hydra.main(
    version_base="1.1",
    config_path="../../third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark/examples/embodiment/config",
    config_name="wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05",
)
def main(cfg) -> None:
    if cfg.runner.get("only_eval", False):
        raise RuntimeError("training-mode rollout-only requires runner.only_eval=false")
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    env = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    rollout_channel = Channel.create("RolloutOnlyRollout")
    env_channel = Channel.create("RolloutOnlyEnv")

    rollout_init = rollout.init_worker()
    env_init = env.init_worker()
    rollout_init.wait()
    env_init.wait()

    env_handle = env.interact(
        input_channel=env_channel,
        rollout_channel=rollout_channel,
        reward_channel=None,
        actor_channel=None,
    )
    rollout_handle = rollout.generate(
        input_channel=rollout_channel,
        output_channel=env_channel,
    )
    env_results = env_handle.wait()
    rollout_handle.wait()
    metrics = compute_evaluate_metrics(
        [result for result in env_results if result is not None]
    )
    metrics = {key: scalar(value) for key, value in metrics.items()}

    output = Path(os.environ["TRACK2_ROLLOUT_ONLY_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "strict-track2-training-mode-rollout-only-raw-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "protocol": {
            "rollout_mode": "train",
            "flow_sampling": "training flow_sde/flow_ode mixture",
            "actor_worker_created": False,
            "policy_updates": 0,
            "gradient_steps": 0,
            "checkpoint_writes": 0,
            "rollout_epochs": int(cfg.algorithm.rollout_epoch),
            "total_envs": int(cfg.env.train.total_num_envs),
            "group_size": int(cfg.algorithm.group_size),
            "episode_steps": int(cfg.env.train.max_episode_steps),
            "actor_seed": int(cfg.actor.seed),
            "environment_seed": int(cfg.env.train.seed),
        },
        "guards": {
            "hidden_or_final_data": False,
            "real_submission": False,
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
