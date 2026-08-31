# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side Wan HTTP proxy environment."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional, Union

import numpy as np
import torch

from rlinf.envs.world_model.http_payload import decode_payload, encode_payload
from rlinf.envs.world_model.world_model_wan_env import WanEnv


class _WanHttpClient:
    def __init__(self, server_url: str, timeout: float = 600.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _post_payload(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"payload": encode_payload(payload)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Wan HTTP server returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to connect to Wan HTTP server: {exc}") from exc
        return decode_payload(result["payload"])

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._post_payload("/reset", state)

    def chunk_step(self, actions: Any) -> dict[str, Any]:
        return self._post_payload("/chunk_step", {"actions": actions})


class WanHttpProxyEnv(WanEnv):
    """WanEnv-compatible proxy that delegates frame generation to HTTP."""

    def _build_pipeline(self):
        # The host process keeps dataset/reward/training only; Wan inference runs
        # in the HTTP server process.
        return None

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
        record_metrics: bool = True,
    ):
        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
            record_metrics=record_metrics,
        )
        http_cfg = getattr(cfg, "http", {})
        server_url = getattr(http_cfg, "server_url", "http://127.0.0.1:18080")
        timeout = float(getattr(http_cfg, "timeout", 600.0))
        self._http_client = _WanHttpClient(server_url=server_url, timeout=timeout)

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        # Resolve the episode IDs before delegating to WanEnv so that we can
        # recover the action attached to the reference frame.  Upstream
        # WanEnv intentionally leaves condition_action[:, 0] as a placeholder,
        # but Track 2 history is defined as a0..a3 for image context o0..o4.
        # Passing the resolved IDs into super().reset also avoids drawing a
        # second random sample while preserving upstream reset semantics.
        resolved_episode_indices = episode_indices
        if self.is_start and self.use_fixed_reset_state_ids:
            resolved_episode_indices = self.reset_state_ids
        if resolved_episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            resolved_episode_indices = np.random.choice(
                len(self.dataset), size=self.num_envs, replace=False
            )
        if isinstance(resolved_episode_indices, torch.Tensor):
            resolved_episode_indices = resolved_episode_indices.detach().cpu().numpy()
        resolved_episode_indices = np.asarray(resolved_episode_indices, dtype=np.int64)
        if resolved_episode_indices.shape != (self.num_envs,):
            raise RuntimeError(
                "resolved Track 2 reset episode indices have shape "
                f"{resolved_episode_indices.shape}, expected ({self.num_envs},)"
            )
        obs, infos = super().reset(
            seed=seed,
            options=options,
            episode_indices=resolved_episode_indices,
        )
        self._inject_reference_actions(resolved_episode_indices)
        # _wrap_obs ran inside WanEnv.reset before slot zero was repaired.  The
        # latest state (slot four) is unchanged, but rebuild the observation to
        # keep this override correct if upstream starts exposing history later.
        obs = self._wrap_obs()
        self._http_client.reset(
            {
                "current_obs": self.current_obs.detach().cpu(),
                "condition_action": self.condition_action.detach().cpu(),
                "task_descriptions": list(self.task_descriptions),
                "elapsed_steps": self.elapsed_steps,
            }
        )
        return obs, infos

    def _inject_reference_actions(self, episode_indices: np.ndarray) -> None:
        """Fill slot zero with a0, yielding aligned initial history a0..a3."""
        reference_actions = []
        for episode_idx in episode_indices:
            episode = self.dataset[int(episode_idx)]
            start_items = episode.get("start_items", [])
            if not start_items or "action" not in start_items[0]:
                raise RuntimeError(
                    f"Track 2 episode {int(episode_idx)} has no reference-frame action"
                )
            action = torch.as_tensor(start_items[0]["action"], dtype=torch.float32)
            action = action.flatten()
            if action.shape != (self.action_dim,):
                raise RuntimeError(
                    f"Track 2 reference action has shape {tuple(action.shape)}, "
                    f"expected ({self.action_dim},)"
                )
            reference_actions.append(action)
        stacked = torch.stack(reference_actions, dim=0).to(
            device=self.condition_action.device,
            dtype=self.condition_action.dtype,
        )
        self.condition_action[:, 0, :] = stacked

    def _wrap_obs(self):
        """Expose the absolute joint state expected by the Pi0 RoboTwin adapter.

        ``WanEnv`` historically emitted a zero placeholder because the original
        diffusion-only environment did not consume robot state.  The official
        Pi0.5 RoboTwin data transform, however, converts predicted joint deltas
        back to absolute actions by adding ``obs["states"]``.  The HTTP Track 2
        contract and reset dataset both use absolute 14-D actions, so the most
        recent condition action is the required state here.
        """
        obs = super()._wrap_obs()
        if self.condition_action is None:
            raise RuntimeError("condition_action is unavailable while wrapping HTTP observation")
        states = self.condition_action[:, -1].to(
            self.device, dtype=torch.float32
        ).contiguous()
        if states.shape != (self.num_envs, self.action_dim):
            raise RuntimeError(
                f"absolute Track 2 state has shape {tuple(states.shape)}, expected "
                f"({self.num_envs}, {self.action_dim})"
            )
        # The HTTP proxy is a process boundary.  Returning CPU-contiguous
        # observations avoids pickling a nested dictionary of CUDA tensors in
        # RLinf's Env channel (which can deadlock at the official batch of 32).
        # OpenPI's precision processor moves these tensors onto its own device,
        # so this is a value-preserving transport change only.
        obs["main_images"] = obs["main_images"].cpu().contiguous()
        obs["states"] = states.cpu().contiguous()
        return obs

    @torch.no_grad()
    def _infer_next_chunk_frames(self, actions):
        actions_to_send = (
            actions.detach().cpu() if isinstance(actions, torch.Tensor) else actions
        )
        result = self._http_client.chunk_step(actions_to_send)
        self.current_obs = result["current_obs"].to(self.device)
        actions_tensor = torch.as_tensor(
            actions_to_send, dtype=self.condition_action.dtype, device=self.device
        )
        if actions_tensor.shape != (self.num_envs, self.chunk, self.action_dim):
            raise RuntimeError(
                f"absolute Track 2 actions have shape {tuple(actions_tensor.shape)}, expected "
                f"({self.num_envs}, {self.chunk}, {self.action_dim})"
            )
        # Mirror the bridge's temporal update so the next Pi0.5 call is anchored
        # at the latest absolute joint position rather than the reset state.
        self.condition_action = self.condition_action.to(
            device=self.device, dtype=actions_tensor.dtype
        )
        self.condition_action[:, 1:, :] = actions_tensor[:, -4:, :]

    def offload(self):
        if self._is_offloaded:
            return
        self.reward_model = self.reward_model.to("cpu")
        self.current_obs = self.current_obs.cpu() if self.current_obs is not None else None
        self.gt_last_frames = (
            self.gt_last_frames.cpu() if self.gt_last_frames is not None else None
        )
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self.reward_model = self.reward_model.to(self.device)
        self.current_obs = self.current_obs.to(self.device)
        self.gt_last_frames = self.gt_last_frames.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False
