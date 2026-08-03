"""Track 2 HTTP world-model environment for RLinf.

This environment deliberately does not import Wan or DiffSynth.  It uses the
local bridge to call the Track 2 submission API, while keeping the published
RoboTwin reward model and RLinf chunk-return contract in the training process.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.envs.world_model.http_payload import decode_payload, encode_payload


class _Track2HttpClient:
    """Client for the local pickle transport between RLinf and the bridge."""

    def __init__(self, server_url: str, timeout: float) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps({"payload": encode_payload(payload)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Track 2 bridge returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Track 2 bridge: {exc}") from exc
        return decode_payload(response_body["payload"])

    def reset(self, state: dict[str, Any]) -> dict[str, Any]:
        return self._post("/reset", state)

    def chunk_step(self, actions: torch.Tensor) -> dict[str, Any]:
        return self._post("/chunk_step", {"actions": actions.cpu()})


class Track2HttpEnv(BaseWorldEnv):
    """RLinf environment backed by a Track 2 submission-service bridge.

    Reset files contain five public RGB frames and their absolute 14-D actions.
    Each rollout consumes one policy chunk of eight actions and receives eight
    predicted RGB frames.  The reward is computed on those frames by the
    published RoboTwin T5 checkpoint.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        record_metrics: bool = True,
        worker_info=None,
    ) -> None:
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )
        self.group_size = int(cfg.group_size)
        if self.num_envs % self.group_size:
            raise ValueError("total_num_envs must be divisible by group_size")
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = bool(cfg.get("use_fixed_reset_state_ids", True))
        self._generator = torch.Generator().manual_seed(self.seed)
        self.update_reset_state_ids()

        self.chunk = int(cfg.get("chunk", 8))
        self.condition_frame_length = int(cfg.get("condition_frame_length", 5))
        self.action_dim = int(cfg.get("action_dim", 14))
        self.image_size = tuple(cfg.get("image_size", [256, 256]))
        self.max_episode_steps = int(cfg.max_episode_steps)
        if self.chunk != 8 or self.condition_frame_length != 5 or self.action_dim != 14:
            raise ValueError("Track 2 requires chunk=8, condition_frame_length=5, action_dim=14")
        if self.image_size != (256, 256):
            raise ValueError("Track 2 requires 256x256 RGB frames")

        http_cfg = cfg.get("http", {})
        self._http_client = _Track2HttpClient(
            server_url=str(http_cfg.get("server_url", "http://127.0.0.1:18080")),
            timeout=float(http_cfg.get("timeout", 600.0)),
        )

        from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

        reward_cfg = cfg.reward_model
        self.reward_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
            reward_cfg.from_pretrained,
            config={"t5_model_name": reward_cfg.get("t5_model_name", "t5-base")},
        ).eval().to(self.device)

        self.current_obs: torch.Tensor | None = None
        self.condition_action: torch.Tensor | None = None
        self.task_descriptions: list[str] = [""] * self.num_envs
        self._is_offloaded = False

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(
            cfg.initial_image_path,
            action_key=cfg.get("action_key", "abs_action"),
            enable_kir=False,
        )

    def update_reset_state_ids(self) -> None:
        reset_state_ids = torch.randint(
            low=0,
            high=len(self.dataset),
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(self.group_size).to(
            self.device
        )

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _record_metrics(self, step_reward, terminations, infos):
        if not self.record_metrics:
            return infos
        self.success_once |= terminations.to(self.device)
        self.returns += step_reward
        episode_len = max(self.elapsed_steps, 1)
        infos["episode"] = {
            "success_once": self.success_once.clone(),
            "return": self.returns.clone(),
            "episode_len": torch.full(
                (self.num_envs,), float(episode_len), dtype=torch.float32, device=self.device
            ),
            "reward": self.returns / float(episode_len),
        }
        return infos

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"reset image must be CHW RGB, got {tuple(image.shape)}")
        image = image.float()
        if tuple(image.shape[-2:]) != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return image.mul(2.0).sub(1.0)

    def _wrap_obs(self) -> dict[str, Any]:
        if self.current_obs is None:
            raise RuntimeError("reset must be called before observations are requested")
        latest = self.current_obs[:, :, 0, -1]
        images = latest.permute(0, 2, 3, 1).add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8)
        return {
            "main_images": images,
            "wrist_images": None,
            "states": torch.zeros(
                (self.num_envs, self.action_dim), dtype=torch.float32, device=self.device
            ),
            "task_descriptions": self.task_descriptions,
        }

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        del options
        self.onload()
        if self.is_start and self.use_fixed_reset_state_ids:
            episode_indices = self.reset_state_ids
        self.is_start = False
        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            episode_indices = np.random.choice(len(self.dataset), size=self.num_envs, replace=False)
        if isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()
        if len(episode_indices) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset IDs, got {len(episode_indices)}")

        windows = []
        conditions = []
        instructions = []
        for episode_index in episode_indices:
            episode = self.dataset[int(episode_index)]
            start_items = episode["start_items"]
            target_items = episode["target_items"]
            if len(start_items) != 1 or len(target_items) != self.condition_frame_length - 1:
                raise ValueError(
                    "each Track 2 reset must provide one reference and four history frames"
                )
            first = start_items[0]
            history = [first, *target_items]
            if any("image" not in frame or "action" not in frame for frame in history):
                raise ValueError("reset frames must contain image and abs_action")
            images = torch.stack([self._prepare_image(frame["image"]) for frame in history])
            action_history = torch.zeros(
                self.condition_frame_length, self.action_dim, dtype=torch.float32
            )
            action_history[1:] = torch.stack(
                [frame["action"].float() for frame in target_items]
            )
            windows.append(images.permute(1, 0, 2, 3))
            conditions.append(action_history)
            instructions.append(str(episode.get("task", "")))

        self.current_obs = torch.stack(windows).unsqueeze(2).to(self.device)
        self.condition_action = torch.stack(conditions).to(self.device)
        self.task_descriptions = instructions
        self._reset_metrics()
        response = self._http_client.reset(
            {
                "current_obs": self.current_obs.detach().cpu(),
                "condition_action": self.condition_action.detach().cpu(),
                "task_descriptions": self.task_descriptions,
                "elapsed_steps": self.elapsed_steps,
            }
        )
        if response.get("status") != "reset":
            raise RuntimeError(f"unexpected Track 2 bridge reset response: {response}")
        return self._wrap_obs(), {}

    def _infer_next_chunk_rewards(self) -> torch.Tensor:
        if self.current_obs is None:
            raise RuntimeError("reset must be called before reward inference")
        frames = self.current_obs[:, :, 0, -self.chunk :]
        flat_frames = frames.permute(0, 2, 1, 3, 4).reshape(
            self.num_envs * self.chunk, 3, *self.image_size
        )
        instructions = [task for task in self.task_descriptions for _ in range(self.chunk)]
        rewards = self.reward_model.compute_reward(
            flat_frames.add(1.0).div(2.0).float(), task_descriptions=instructions
        )
        return rewards.reshape(self.num_envs, self.chunk)

    def _calc_step_reward(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        scaled = float(self.cfg.reward_coef) * chunk_rewards
        reward_diffs = torch.empty_like(scaled)
        reward_diffs[:, 0] = scaled[:, 0] - self.prev_step_reward
        reward_diffs[:, 1:] = scaled[:, 1:] - scaled[:, :-1]
        self.prev_step_reward = scaled[:, -1]
        return reward_diffs if self.use_rel_reward else chunk_rewards

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        actions = torch.as_tensor(policy_output_action, dtype=torch.float32)
        if actions.shape != (self.num_envs, self.chunk, self.action_dim):
            raise ValueError(
                f"Track 2 policy actions must be [{self.num_envs},{self.chunk},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )
        response = self._http_client.chunk_step(actions)
        current_obs = response.get("current_obs")
        expected_shape = (self.num_envs, 3, 1, self.condition_frame_length + self.chunk, *self.image_size)
        if not isinstance(current_obs, torch.Tensor) or tuple(current_obs.shape) != expected_shape:
            actual_shape = tuple(current_obs.shape) if isinstance(current_obs, torch.Tensor) else None
            raise RuntimeError(f"bridge returned {actual_shape}, expected {expected_shape}")
        self.current_obs = current_obs.to(self.device, dtype=torch.float32)
        self.condition_action[:, 1:] = actions.to(self.device)[:, -4:]
        self.elapsed_steps = int(response.get("elapsed_steps", self.elapsed_steps + self.chunk))

        chunk_rewards = self._infer_next_chunk_rewards()
        step_rewards = self._calc_step_reward(chunk_rewards)
        threshold = float(self.cfg.get("success_reward_threshold", 0.9))
        terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        terminations[:, -1] = chunk_rewards.max(dim=1).values >= threshold
        truncations = torch.zeros_like(terminations)
        if self.elapsed_steps >= self.max_episode_steps:
            truncations[:, -1] = True
        dones = torch.logical_or(terminations, truncations).any(dim=1)
        obs = self._wrap_obs()
        infos: dict[str, Any] = {}
        if dones.any() and self.auto_reset:
            final_obs = obs
            final_info = infos
            obs, infos = self.reset()
            infos.update(
                {
                    "final_observation": final_obs,
                    "final_info": final_info,
                    "_final_info": dones,
                    "_final_observation": dones,
                    "_elapsed_steps": dones,
                }
            )
        infos = self._record_metrics(step_rewards.sum(dim=1), terminations.any(dim=1), infos)
        return [obs], step_rewards, terminations, truncations, [infos]

    def step(self, actions=None, auto_reset=True):
        del actions, auto_reset
        raise NotImplementedError("Track2HttpEnv uses chunk_step()")

    def offload(self) -> None:
        if self._is_offloaded:
            return
        self.reward_model = self.reward_model.to("cpu")
        self.prev_step_reward = self.prev_step_reward.cpu()
        self.reset_state_ids = self.reset_state_ids.cpu()
        if self.current_obs is not None:
            self.current_obs = self.current_obs.cpu()
        if self.condition_action is not None:
            self.condition_action = self.condition_action.cpu()
        if self.record_metrics:
            self.success_once = self.success_once.cpu()
            self.returns = self.returns.cpu()
        self._is_offloaded = True

    def onload(self) -> None:
        if not self._is_offloaded:
            return
        self.reward_model = self.reward_model.to(self.device)
        self.prev_step_reward = self.prev_step_reward.to(self.device)
        self.reset_state_ids = self.reset_state_ids.to(self.device)
        if self.current_obs is not None:
            self.current_obs = self.current_obs.to(self.device)
        if self.condition_action is not None:
            self.condition_action = self.condition_action.to(self.device)
        if self.record_metrics:
            self.success_once = self.success_once.to(self.device)
            self.returns = self.returns.to(self.device)
        self._is_offloaded = False

    def get_state(self) -> bytes:
        """Serialize local state for RLinf offload/checkpoint hooks."""
        buffer = io.BytesIO()
        torch.save(
            {
                "current_obs": None if self.current_obs is None else self.current_obs.cpu(),
                "condition_action": None
                if self.condition_action is None
                else self.condition_action.cpu(),
                "task_descriptions": self.task_descriptions,
                "elapsed_steps": self.elapsed_steps,
                "prev_step_reward": self.prev_step_reward.cpu(),
            },
            buffer,
        )
        return buffer.getvalue()
