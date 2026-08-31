"""Hard-routed frozen/adapted autoregressive parent for on-policy states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autoregressive_unet_runtime import Track2AutoregressiveUNet


class Track2GatedAutoregressiveUNet:
    """Select one complete AR expert from the last real context frame.

    The route is computed before rollout and is hard at the checkpointed
    threshold.  Familiar states therefore remain bit-identical to the frozen
    parent and only one expert executes, avoiding a two-model latency penalty.
    """

    def __init__(
        self,
        baseline_dir: str | Path,
        candidate_dir: str | Path,
        gate_path: str | Path,
        device: str = "cuda",
        right_candidate_dir: str | Path | None = None,
        candidate_blend: float = 1.0,
        candidate_blend_schedule: list[float] | tuple[float, ...] | None = None,
        right_candidate_blend_schedule: list[float] | tuple[float, ...] | None = None,
        instruction_arm_router: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.baseline = Track2AutoregressiveUNet(baseline_dir, device)
        self.candidate = Track2AutoregressiveUNet(candidate_dir, device)
        self.right_candidate = (
            Track2AutoregressiveUNet(right_candidate_dir, device)
            if right_candidate_dir is not None
            else None
        )
        self.candidate_blend = float(candidate_blend)
        if not 0.0 < self.candidate_blend <= 1.0:
            raise ValueError("candidate blend must be in (0, 1]")
        self.candidate_blend_schedule = self._validate_schedule(candidate_blend_schedule)
        self.right_candidate_blend_schedule = self._validate_schedule(
            right_candidate_blend_schedule
        )
        self.instruction_arm_router = bool(instruction_arm_router)
        gate = torch.load(gate_path, map_location="cpu", weights_only=True)
        self.gate_format = gate.get("format")
        if self.gate_format not in {
            "strict-track2-visual-source-gate-v1",
            "strict-track2-action-source-gate-v1",
        }:
            raise RuntimeError("unsupported visual source gate")
        self.feature_mean = gate["feature_mean"].to(self.device).float()
        self.feature_std = gate["feature_std"].to(self.device).float()
        if self.gate_format == "strict-track2-visual-source-gate-v1":
            self.pool_size = int(gate["pool_size"])
            self.weight = gate["weight"].to(self.device).float()
            self.bias = gate["bias"].to(self.device).float()
            self.action_gate_state = None
        else:
            if (
                int(gate["history_actions"]) != 4
                or int(gate["future_actions"]) != 8
                or int(gate["action_dim"]) != 14
            ):
                raise RuntimeError("action source gate does not match Track 2 dimensions")
            self.action_gate_state = {
                name: value.to(self.device).float() for name, value in gate["state_dict"].items()
            }
        self.threshold = float(gate["threshold"])
        self.last_probability_synthetic: float | None = None
        self.last_route: str | None = None
        self.last_arm_route: str | None = None
        self.last_arm_route_source: str | None = None

    @staticmethod
    def _validate_schedule(value):
        if value is None:
            return None
        schedule = tuple(float(item) for item in value)
        if len(schedule) != 8 or any(not 0.0 <= item <= 1.0 for item in schedule):
            raise ValueError("candidate blend schedule must contain eight values in [0, 1]")
        return schedule

    @staticmethod
    def active_arm_from_actions(history_actions: np.ndarray, future_actions: np.ndarray) -> str:
        """Infer the commanded arm from raw joint motion, without task metadata."""
        actions = np.concatenate((history_actions, future_actions), axis=0)
        delta = np.abs(np.diff(actions, axis=0))
        return "right" if delta[:, 7:].mean() > delta[:, :7].mean() else "left"

    def active_arm(
        self,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        instruction: str | None,
    ) -> str:
        if self.instruction_arm_router and instruction:
            lowered = instruction.casefold()
            has_left = "left arm" in lowered
            has_right = "right arm" in lowered
            if has_left != has_right:
                self.last_arm_route_source = "instruction"
                return "left" if has_left else "right"
        self.last_arm_route_source = "raw_action_delta_fallback"
        return self.active_arm_from_actions(history_actions, future_actions)

    def probability_synthetic(self, context_frames: np.ndarray) -> float:
        if self.gate_format != "strict-track2-visual-source-gate-v1":
            raise RuntimeError("the action source gate requires request actions")
        torch = self.torch
        frame = torch.from_numpy(np.ascontiguousarray(context_frames[-1])).permute(2, 0, 1)
        frame = frame.to(self.device).float().div(255).unsqueeze(0)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            frame, (self.pool_size, self.pool_size)
        ).flatten(1)
        feature = torch.cat(
            (pooled, frame.mean((2, 3)), frame.std((2, 3), unbiased=False)), 1
        )
        normalized = (feature - self.feature_mean) / self.feature_std
        with torch.inference_mode():
            return float(torch.sigmoid(torch.nn.functional.linear(normalized, self.weight, self.bias))[0, 0])

    def probability_onpolicy_actions(
        self, history_actions: np.ndarray, future_actions: np.ndarray
    ) -> float:
        if self.gate_format != "strict-track2-action-source-gate-v1":
            raise RuntimeError("the visual source gate requires context RGB")
        actions = np.concatenate((history_actions, future_actions), axis=0).astype(np.float32)
        if actions.shape != (12, 14):
            raise ValueError(f"unexpected Track 2 action sequence {actions.shape}")
        delta = np.diff(actions, axis=0)
        summary = np.concatenate((
            actions.mean(0),
            actions.std(0),
            np.abs(delta[:, :7]).mean(0),
            np.abs(delta[:, 7:]).mean(0),
            np.asarray(
                [np.abs(delta[:, :7]).mean(), np.abs(delta[:, 7:]).mean()],
                dtype=np.float32,
            ),
        ))
        feature = np.concatenate((actions.reshape(-1), delta.reshape(-1), summary))
        value = self.torch.from_numpy(feature).to(self.device).float().unsqueeze(0)
        normalized = (value - self.feature_mean) / self.feature_std
        state = self.action_gate_state
        assert state is not None
        with self.torch.inference_mode():
            hidden = self.torch.nn.functional.silu(
                self.torch.nn.functional.linear(
                    normalized, state["first.weight"], state["first.bias"]
                )
            )
            logit = self.torch.nn.functional.linear(
                hidden, state["second.weight"], state["second.bias"]
            )
            return float(logit.sigmoid()[0, 0])

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self.gate_format == "strict-track2-action-source-gate-v1":
            probability = self.probability_onpolicy_actions(history_actions, future_actions)
        else:
            probability = self.probability_synthetic(context_frames)
        use_candidate = probability >= self.threshold
        self.last_probability_synthetic = probability
        arm = self.active_arm(history_actions, future_actions, instruction)
        self.last_arm_route = arm
        if not use_candidate:
            self.last_route = "baseline"
            expert = self.baseline
        elif arm == "right" and self.right_candidate is not None:
            self.last_route = "candidate_right"
            expert = self.right_candidate
        else:
            self.last_route = "candidate_left" if self.right_candidate is not None else "candidate"
            expert = self.candidate
        candidate_prediction = expert.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        schedule = (
            self.right_candidate_blend_schedule
            if arm == "right" and self.right_candidate_blend_schedule is not None
            else self.candidate_blend_schedule
        )
        if not use_candidate or (schedule is None and self.candidate_blend == 1.0):
            return candidate_prediction
        baseline_prediction = self.baseline.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )
        self.last_route += "_blend"
        alpha = (
            np.asarray(schedule, dtype=np.float32)[:, None, None, None]
            if schedule is not None
            else self.candidate_blend
        )
        blended = baseline_prediction.astype(np.float32) + alpha * (
            candidate_prediction.astype(np.float32) - baseline_prediction.astype(np.float32)
        )
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        """Route samples first, then execute each selected expert as a batch."""
        batch = int(context_frames.shape[0])
        if history_actions.shape[0] != batch or future_actions.shape[0] != batch:
            raise ValueError("batched V15 gate inputs have inconsistent leading dimensions")
        if len(seeds) != batch or len(instructions) != batch:
            raise ValueError("batched V15 gate metadata has inconsistent length")
        probabilities, arms, routes = [], [], []
        for index in range(batch):
            probability = (
                self.probability_onpolicy_actions(history_actions[index], future_actions[index])
                if self.gate_format == "strict-track2-action-source-gate-v1"
                else self.probability_synthetic(context_frames[index])
            )
            arm = self.active_arm(
                history_actions[index], future_actions[index], instructions[index]
            )
            use_candidate = probability >= self.threshold
            if not use_candidate:
                route = "baseline"
            elif arm == "right" and self.right_candidate is not None:
                route = "candidate_right"
            else:
                route = "candidate_left" if self.right_candidate is not None else "candidate"
            probabilities.append(probability)
            arms.append(arm)
            routes.append(route)

        def run(expert, indices):
            selected = np.asarray(indices, dtype=np.int64)
            return expert.predict_batch(
                context_frames[selected],
                history_actions[selected],
                future_actions[selected],
                np.asarray(seeds)[selected],
                [instructions[index] for index in indices],
            )

        output = np.empty((batch, 8, 256, 256, 3), dtype=np.uint8)
        def blend_schedule(arm):
            return (
                self.right_candidate_blend_schedule
                if arm == "right" and self.right_candidate_blend_schedule is not None
                else self.candidate_blend_schedule
            )

        baseline_indices = [
            index
            for index, (route, arm) in enumerate(zip(routes, arms))
            if route == "baseline"
            or (route != "baseline" and (blend_schedule(arm) is not None or self.candidate_blend != 1.0))
        ]
        baseline_by_index = {}
        if baseline_indices:
            values = run(self.baseline, baseline_indices)
            baseline_by_index.update(zip(baseline_indices, values))
            for index in baseline_indices:
                if routes[index] == "baseline":
                    output[index] = baseline_by_index[index]

        for route, expert in (
            ("candidate", self.candidate),
            ("candidate_left", self.candidate),
            ("candidate_right", self.right_candidate),
        ):
            indices = [index for index, value in enumerate(routes) if value == route]
            if not indices:
                continue
            if expert is None:
                raise RuntimeError(f"missing expert for route {route}")
            values = run(expert, indices)
            for index, candidate_prediction in zip(indices, values):
                schedule = blend_schedule(arms[index])
                if schedule is None and self.candidate_blend == 1.0:
                    output[index] = candidate_prediction
                    continue
                alpha = (
                    np.asarray(schedule, dtype=np.float32)[:, None, None, None]
                    if schedule is not None
                    else self.candidate_blend
                )
                baseline_prediction = baseline_by_index[index]
                blended = baseline_prediction.astype(np.float32) + alpha * (
                    candidate_prediction.astype(np.float32)
                    - baseline_prediction.astype(np.float32)
                )
                output[index] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
                routes[index] += "_blend"

        self.last_probability_synthetic = probabilities[-1] if probabilities else None
        self.last_route = routes[-1] if routes else None
        self.last_arm_route = arms[-1] if arms else None
        return output
