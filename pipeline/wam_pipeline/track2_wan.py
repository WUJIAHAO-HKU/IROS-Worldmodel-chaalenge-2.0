"""Action-conditioned Wan primitives for the exact Track 2 transition.

The public Track 2 request contains 13 RGB positions: five observed frames
followed by eight frames to predict.  Wan's VAE compresses time in groups of
four after the first frame, therefore these positions become four latent time
slots.  The first two latent slots are observed context and the final two are
the only slots supervised and sampled by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES


TOTAL_FRAMES = CONTEXT_FRAMES + PREDICTION_FRAMES
TOTAL_ACTION_SLOTS = TOTAL_FRAMES
WAN_TEMPORAL_COMPRESSION = 4


def wan_latent_frames(raw_frames: int, temporal_compression: int = WAN_TEMPORAL_COMPRESSION) -> int:
    """Return Wan's causal VAE length for a sequence of RGB frames."""
    if raw_frames < 1:
        raise ValueError("raw_frames must be positive")
    if temporal_compression < 1:
        raise ValueError("temporal_compression must be positive")
    if (raw_frames - 1) % temporal_compression:
        raise ValueError(
            "Wan raw-frame count must be one plus a whole number of temporal groups; "
            f"got raw_frames={raw_frames}, temporal_compression={temporal_compression}"
        )
    return 1 + (raw_frames - 1) // temporal_compression


TRACK2_LATENT_FRAMES = wan_latent_frames(TOTAL_FRAMES)
TRACK2_CONTEXT_LATENT_FRAMES = wan_latent_frames(CONTEXT_FRAMES)


def build_action_slots(history_actions: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
    """Construct the 13-slot Wan action sequence with its explicit zero anchor."""
    if history_actions.ndim == 2:
        history_actions = history_actions.unsqueeze(0)
    if future_actions.ndim == 2:
        future_actions = future_actions.unsqueeze(0)
    expected_history = (CONTEXT_ACTIONS, ACTION_DIM)
    expected_future = (PREDICTION_FRAMES, ACTION_DIM)
    if history_actions.ndim != 3 or tuple(history_actions.shape[1:]) != expected_history:
        raise ValueError(f"history_actions must be [B,{expected_history[0]},{expected_history[1]}]")
    if future_actions.ndim != 3 or tuple(future_actions.shape[1:]) != expected_future:
        raise ValueError(f"future_actions must be [B,{expected_future[0]},{expected_future[1]}]")
    if history_actions.shape[0] != future_actions.shape[0]:
        raise ValueError("history and future action batches do not match")
    anchor = torch.zeros_like(history_actions[:, :1])
    return torch.cat((anchor, history_actions, future_actions), dim=1)


def normalize_action_slots(action_slots: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalize physical actions while retaining the slot-zero zero anchor."""
    if action_slots.ndim != 3 or tuple(action_slots.shape[1:]) != (TOTAL_ACTION_SLOTS, ACTION_DIM):
        raise ValueError(f"action_slots must be [B,{TOTAL_ACTION_SLOTS},{ACTION_DIM}]")
    if tuple(mean.shape) != (ACTION_DIM,) or tuple(std.shape) != (ACTION_DIM,):
        raise ValueError("action normalization tensors must both be [14]")
    if not torch.isfinite(action_slots).all() or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("actions and normalization tensors must be finite")
    if torch.any(std <= 0):
        raise ValueError("action normalization standard deviations must be positive")
    normalized = (action_slots[:, 1:] - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    # Slot zero represents RGB frame 0 and is intentionally not a normalized action.
    return torch.cat((torch.zeros_like(action_slots[:, :1]), normalized), dim=1)


class Track2ActionConditioner(nn.Module):
    """Map all 13 aligned abs14 slots into Wan cross-attention tokens.

    The projection is intentionally per-action-slot.  Future slots are passed
    directly to every DiT cross-attention layer, instead of being collapsed into
    an action average or a hidden policy state.
    """

    def __init__(
        self,
        text_dim: int,
        *,
        action_dim: int = ACTION_DIM,
        action_slots: int = TOTAL_ACTION_SLOTS,
        hidden_dim: int | None = None,
        trajectory_conditioned: bool = False,
        trajectory_hidden_dim: int | None = None,
        latent_action_conditioned: bool = False,
        latent_channels: int | None = None,
        latent_action_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if text_dim < 1 or action_dim < 1 or action_slots < 1:
            raise ValueError("conditioner dimensions must be positive")
        self.text_dim = int(text_dim)
        self.action_dim = int(action_dim)
        self.action_slots = int(action_slots)
        self.trajectory_conditioned = bool(trajectory_conditioned)
        self.latent_action_conditioned = bool(latent_action_conditioned)
        hidden_dim = int(hidden_dim or max(128, min(1024, text_dim)))
        # Inputs are already normalized per physical action dimension from the
        # training split. A per-slot LayerNorm would erase absolute pose shifts.
        self.input_norm = nn.Identity()
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.text_dim),
        )
        self.position = nn.Parameter(torch.empty(1, self.action_slots, self.text_dim))
        self.task_token = nn.Parameter(torch.empty(1, 1, self.text_dim))
        self.output_norm = nn.LayerNorm(self.text_dim)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.task_token, std=0.02)
        if self.trajectory_conditioned:
            self.trajectory_hidden_dim = int(trajectory_hidden_dim or max(128, min(512, text_dim)))
            self.trajectory_gru = nn.GRU(
                input_size=self.action_dim * 2,
                hidden_size=self.trajectory_hidden_dim,
                batch_first=True,
            )
            self.trajectory_projection = nn.Sequential(
                nn.LayerNorm(self.trajectory_hidden_dim),
                nn.Linear(self.trajectory_hidden_dim, self.text_dim),
            )
            # A v2 conditioner initialized from v1 must initially generate the
            # same tokens. Training first learns this residual readout, then the
            # GRU receives gradient as the readout becomes non-zero.
            nn.init.zeros_(self.trajectory_projection[-1].weight)
            nn.init.zeros_(self.trajectory_projection[-1].bias)
        else:
            self.trajectory_hidden_dim = None
        if self.latent_action_conditioned:
            if latent_channels is None or int(latent_channels) < 1:
                raise ValueError("latent_action_conditioned requires a positive latent_channels value")
            self.latent_channels = int(latent_channels)
            self.latent_action_hidden_dim = int(latent_action_hidden_dim or max(128, self.action_dim * 8))
            # Each compressed Wan slot receives its mean pose, local delta, and
            # displacement from the final observed pose. This preserves the
            # four-action grouping represented by one temporal VAE slot.
            self.latent_action_mlp = nn.Sequential(
                nn.Linear(self.action_dim * 3, self.latent_action_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.latent_action_hidden_dim, self.latent_channels),
            )
            nn.init.zeros_(self.latent_action_mlp[-1].weight)
            nn.init.zeros_(self.latent_action_mlp[-1].bias)
        else:
            self.latent_channels = None
            self.latent_action_hidden_dim = None

    def trajectory_inputs(self, action_slots: torch.Tensor) -> torch.Tensor:
        """Return per-slot velocity and displacement from the last observed pose.

        ``action_slots`` is already normalized in physical abs14 coordinates.
        Slot zero is an RGB-only anchor, so its two trajectory features remain
        zero. Slots one through twelve preserve both the immediate action delta
        and displacement relative to the final history action (slot four).
        """
        if action_slots.ndim != 3 or tuple(action_slots.shape[1:]) != (self.action_slots, self.action_dim):
            raise ValueError(
                f"action_slots must be [B,{self.action_slots},{self.action_dim}], got {tuple(action_slots.shape)}"
            )
        delta = torch.zeros_like(action_slots)
        # Slot zero is an RGB anchor rather than a physical action, so the
        # first actual action deliberately has no synthetic delta from zero.
        delta[:, 2:] = action_slots[:, 2:] - action_slots[:, 1:-1]
        relative = torch.zeros_like(action_slots)
        # Four action observations map to slots 1--4; slot four is the latest
        # physical pose known before the eight actions to be rolled out.
        history_end = min(CONTEXT_ACTIONS, self.action_slots - 1)
        if history_end > 0:
            relative[:, 1:] = action_slots[:, 1:] - action_slots[:, history_end : history_end + 1]
        return torch.cat((delta, relative), dim=-1)

    def latent_action_features(self, action_slots: torch.Tensor) -> torch.Tensor:
        """Summarize aligned raw actions for Wan's four causal VAE time slots."""
        if action_slots.ndim != 3 or tuple(action_slots.shape[1:]) != (self.action_slots, self.action_dim):
            raise ValueError(
                f"action_slots must be [B,{self.action_slots},{self.action_dim}], got {tuple(action_slots.shape)}"
            )
        if self.action_slots != TOTAL_ACTION_SLOTS:
            raise ValueError("latent action grouping requires the exact 13-slot Track 2 contract")
        groups = action_slots.new_zeros((action_slots.shape[0], TRACK2_LATENT_FRAMES, 4, self.action_dim))
        # Slot zero is the RGB anchor. Slots 1--4, 5--8, and 9--12 match the
        # three four-frame causal VAE groups exactly.
        groups[:, 1:] = action_slots[:, 1:].reshape(action_slots.shape[0], TRACK2_LATENT_FRAMES - 1, 4, self.action_dim)
        mean = groups.mean(dim=2)
        local_delta = groups[:, :, -1] - groups[:, :, 0]
        final_history = action_slots[:, CONTEXT_ACTIONS : CONTEXT_ACTIONS + 1]
        displacement = groups[:, :, -1] - final_history
        features = torch.cat((mean, local_delta, displacement), dim=-1)
        # The anchor has no physical action and must not create a latent bias.
        features[:, :1] = 0
        return features

    def latent_action_residual(self, action_slots: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Return a zero-initialized action bias aligned to Wan latent time slots."""
        if not self.latent_action_conditioned or self.latent_action_mlp is None or self.latent_channels is None:
            raise RuntimeError("latent action conditioning is disabled")
        if height < 1 or width < 1:
            raise ValueError("latent spatial dimensions must be positive")
        values = self.latent_action_mlp(self.latent_action_features(action_slots))
        values[:, :TRACK2_CONTEXT_LATENT_FRAMES] = 0
        return values.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, height, width)

    def forward(self, action_slots: torch.Tensor) -> torch.Tensor:
        if action_slots.ndim != 3 or tuple(action_slots.shape[1:]) != (self.action_slots, self.action_dim):
            raise ValueError(
                f"action_slots must be [B,{self.action_slots},{self.action_dim}], got {tuple(action_slots.shape)}"
            )
        if not torch.isfinite(action_slots).all():
            raise ValueError("action_slots contains a non-finite value")
        tokens = self.action_mlp(self.input_norm(action_slots)) + self.position.to(action_slots.dtype)
        if self.trajectory_conditioned:
            trajectory, _ = self.trajectory_gru(self.trajectory_inputs(action_slots))
            tokens = tokens + self.trajectory_projection(trajectory)
        tokens = self.output_norm(tokens)
        task = self.task_token.to(dtype=tokens.dtype, device=tokens.device).expand(tokens.shape[0], -1, -1)
        return torch.cat((task, tokens), dim=1)


@dataclass(frozen=True)
class Track2WanLayout:
    """Static temporal layout recorded with every formal Wan checkpoint."""

    raw_frames: int = TOTAL_FRAMES
    context_raw_frames: int = CONTEXT_FRAMES
    latent_frames: int = TRACK2_LATENT_FRAMES
    context_latent_frames: int = TRACK2_CONTEXT_LATENT_FRAMES
    temporal_compression: int = WAN_TEMPORAL_COMPRESSION
    action_slots: int = TOTAL_ACTION_SLOTS
    action_dim: int = ACTION_DIM

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_frames": self.raw_frames,
            "context_raw_frames": self.context_raw_frames,
            "latent_frames": self.latent_frames,
            "context_latent_frames": self.context_latent_frames,
            "temporal_compression": self.temporal_compression,
            "action_slots": self.action_slots,
            "action_dim": self.action_dim,
        }


class Track2WanWorldModel(nn.Module):
    """Wan denoiser that keeps context slots fixed and denoises only the future.

    ``transformer`` is a normal ``diffusers.WanTransformer3DModel``.  Wan2.2
    TI2V-5B natively receives the mixed latent video and identifies clean
    context through its per-token zero timestep.  The legacy 3x-latent
    conditional input layout remains supported for compatible Wan variants.
    """

    def __init__(self, transformer: nn.Module, action_conditioner: Track2ActionConditioner) -> None:
        super().__init__()
        self.transformer = transformer
        self.action_conditioner = action_conditioner
        config = getattr(transformer, "config", None)
        if config is None or not hasattr(config, "in_channels") or not hasattr(config, "out_channels"):
            raise ValueError("transformer must expose Wan in_channels and out_channels")
        self.transformer_in_channels = int(config.in_channels)
        self.transformer_out_channels = int(config.out_channels)
        self.text_dim = int(getattr(config, "text_dim"))
        if self.text_dim != action_conditioner.text_dim:
            raise ValueError("action conditioner text_dim must match the Wan transformer text_dim")

    @staticmethod
    def _require_clean_latents(clean_latents: torch.Tensor) -> None:
        if clean_latents.ndim != 5:
            raise ValueError("clean_latents must be [B,C,4,H,W]")
        if clean_latents.shape[2] != TRACK2_LATENT_FRAMES:
            raise ValueError(
                f"Track 2 requires {TRACK2_LATENT_FRAMES} Wan latent slots, got {clean_latents.shape[2]}"
            )
        if not torch.isfinite(clean_latents).all():
            raise ValueError("clean_latents contains a non-finite value")

    @staticmethod
    def context_mask(clean_latents: torch.Tensor) -> torch.Tensor:
        """Return [B,1,4,1,1] with slots 0--1 fixed as observed context."""
        Track2WanWorldModel._require_clean_latents(clean_latents)
        mask = clean_latents.new_zeros((clean_latents.shape[0], 1, TRACK2_LATENT_FRAMES, 1, 1))
        mask[:, :, :TRACK2_CONTEXT_LATENT_FRAMES] = 1
        return mask

    def _model_input(self, mixed_latents: torch.Tensor, clean_latents: torch.Tensor) -> torch.Tensor:
        self._require_clean_latents(mixed_latents)
        self._require_clean_latents(clean_latents)
        if mixed_latents.shape != clean_latents.shape:
            raise ValueError("mixed_latents and clean_latents must have the same shape")
        latent_channels = int(mixed_latents.shape[1])
        context_mask = self.context_mask(clean_latents)
        if self.transformer_in_channels == latent_channels:
            return mixed_latents
        if self.transformer_in_channels == latent_channels * 3:
            # Wan2.2 TI2V expects [noisy latent, clean condition latent, mask].
            # The mask is one channel per latent channel in the public model.
            clean_context = clean_latents * context_mask
            repeated_mask = context_mask.expand(-1, latent_channels, -1, mixed_latents.shape[3], mixed_latents.shape[4])
            return torch.cat((mixed_latents, clean_context, repeated_mask), dim=1)
        raise ValueError(
            "unsupported Wan input channel count: expected latent_channels or "
            f"3*latent_channels ({latent_channels} or {latent_channels * 3}), got {self.transformer_in_channels}"
        )

    def _timestep_for_model(self, model_input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Build Wan2.2 TI2V's per-patch time conditioning.

        TI2V is trained with clean observed-video tokens at timestep zero and
        noisy future-video tokens at the sampled flow timestep.  Diffusers
        accepts this as ``[B, token_count]`` for Wan2.2.  Passing one scalar
        timestep would incorrectly mark the fixed context as noisy.
        """
        if timestep.ndim == 2:
            if timestep.shape[0] != model_input.shape[0]:
                raise ValueError("token timestep batch size does not match model input")
            return timestep.to(device=model_input.device, dtype=model_input.dtype)
        if timestep.ndim != 1 or timestep.shape[0] != model_input.shape[0]:
            raise ValueError("timestep must be [B] or [B,token_count]")
        timestep = timestep.to(device=model_input.device, dtype=model_input.dtype)
        # Wan2.2 TI2V-5B is a 48-channel, high-compression model.  Unlike
        # concatenative I2V variants, it uses a token-level timestep mask to
        # distinguish its clean observed slots from noisy generated slots.
        latent_channels = int(model_input.shape[1] // 3)
        uses_token_timesteps = (
            self.transformer_in_channels == 48
            and self.transformer_out_channels == 48
        ) or self.transformer_in_channels == latent_channels * 3
        if not uses_token_timesteps:
            return timestep

        patch_t, patch_h, patch_w = tuple(getattr(self.transformer.config, "patch_size", (1, 2, 2)))
        if patch_t < 1 or patch_h < 1 or patch_w < 1:
            raise ValueError(f"invalid Wan patch size: {(patch_t, patch_h, patch_w)}")
        if any(size % patch for size, patch in zip(model_input.shape[2:], (patch_t, patch_h, patch_w))):
            raise ValueError("Track 2 latent shape must be divisible by the Wan patch size")
        temporal_tokens = model_input.shape[2] // patch_t
        spatial_tokens = (model_input.shape[3] // patch_h) * (model_input.shape[4] // patch_w)
        if temporal_tokens != TRACK2_LATENT_FRAMES:
            raise ValueError(f"expected {TRACK2_LATENT_FRAMES} temporal Wan tokens, got {temporal_tokens}")
        per_slot = torch.cat(
            (
                torch.zeros(
                    (timestep.shape[0], TRACK2_CONTEXT_LATENT_FRAMES),
                    dtype=timestep.dtype,
                    device=timestep.device,
                ),
                timestep[:, None].expand(-1, TRACK2_LATENT_FRAMES - TRACK2_CONTEXT_LATENT_FRAMES),
            ),
            dim=1,
        )
        return per_slot.repeat_interleave(spatial_tokens, dim=1)

    def predict_velocity(
        self,
        mixed_latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        action_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity for an already-mixed Track 2 latent video."""
        conditioned_latents = mixed_latents
        if self.action_conditioner.latent_action_conditioned:
            residual = self.action_conditioner.latent_action_residual(
                action_slots.to(dtype=next(self.action_conditioner.parameters()).dtype, device=mixed_latents.device),
                height=int(mixed_latents.shape[3]),
                width=int(mixed_latents.shape[4]),
            ).to(dtype=mixed_latents.dtype)
            if residual.shape != mixed_latents.shape:
                raise RuntimeError("latent action residual must match Wan latent shape")
            conditioned_latents = mixed_latents + residual * (1.0 - self.context_mask(clean_latents))
        model_input = self._model_input(conditioned_latents, clean_latents)
        conditioner_dtype = next(self.action_conditioner.parameters()).dtype
        action_tokens = self.action_conditioner(action_slots.to(
            dtype=conditioner_dtype, device=model_input.device
        )).to(
            dtype=model_input.dtype, device=model_input.device
        )
        timestep = self._timestep_for_model(model_input, timestep)
        prediction = self.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=action_tokens,
            return_dict=False,
        )[0]
        # TI2V 5B has 48 input/output channels: [velocity, context, mask].
        # Only its leading VAE-latent channels are the future flow prediction.
        if prediction.shape[1] == clean_latents.shape[1] * 3:
            prediction = prediction[:, : clean_latents.shape[1]]
        if prediction.shape != clean_latents.shape:
            raise RuntimeError(
                "Wan output must match VAE latent shape for future flow supervision; "
                f"got {tuple(prediction.shape)}, expected {tuple(clean_latents.shape)}"
            )
        return prediction

    def forward(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
        action_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a flow-matching sample and return velocity, target, and mixed input."""
        self._require_clean_latents(clean_latents)
        if noise.shape != clean_latents.shape:
            raise ValueError("noise must have the same shape as clean_latents")
        if sigma.ndim != 1 or sigma.shape[0] != clean_latents.shape[0]:
            raise ValueError("sigma must be [B]")
        if not torch.isfinite(noise).all() or not torch.isfinite(sigma).all():
            raise ValueError("noise and sigma must be finite")
        sigma_5d = sigma.view(-1, 1, 1, 1, 1).to(dtype=clean_latents.dtype, device=clean_latents.device)
        noisy = (1.0 - sigma_5d) * clean_latents + sigma_5d * noise
        context_mask = self.context_mask(clean_latents)
        mixed = clean_latents * context_mask + noisy * (1.0 - context_mask)
        timestep = sigma.to(dtype=clean_latents.dtype, device=clean_latents.device) * 1000.0
        velocity = self.predict_velocity(mixed, clean_latents, timestep, action_slots)
        return velocity, noise - clean_latents, mixed

    @torch.no_grad()
    def sample_future_latents(
        self,
        clean_latents: torch.Tensor,
        action_slots: torch.Tensor,
        *,
        num_inference_steps: int,
        generator: torch.Generator | None = None,
        solver: str = "euler",
    ) -> torch.Tensor:
        """Deterministically integrate the future-only flow from sigma=1 to 0.

        ``heun`` evaluates the learned velocity at both ends of every
        integration interval.  Context slots stay clamped in both calls, so
        switching solvers cannot alter the observed portion of a Track 2
        sample.
        """
        self._require_clean_latents(clean_latents)
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        noise = torch.randn(
            clean_latents.shape,
            generator=generator,
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        context_mask = self.context_mask(clean_latents)
        future_mask = 1.0 - context_mask
        latents = clean_latents * context_mask + noise * future_mask
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=clean_latents.device)
        for index in range(num_inference_steps):
            sigma = sigmas[index].expand(clean_latents.shape[0]).to(dtype=clean_latents.dtype)
            velocity = self.predict_velocity(latents, clean_latents, sigma * 1000.0, action_slots)
            delta = (sigmas[index + 1] - sigmas[index]).to(dtype=clean_latents.dtype)
            proposal = latents + delta * velocity * future_mask
            if solver == "heun":
                proposal = clean_latents * context_mask + proposal * future_mask
                next_sigma = sigmas[index + 1].expand(clean_latents.shape[0]).to(dtype=clean_latents.dtype)
                next_velocity = self.predict_velocity(
                    proposal, clean_latents, next_sigma * 1000.0, action_slots
                )
                latents = latents + delta * (velocity + next_velocity) * 0.5 * future_mask
            else:
                latents = proposal
            # These observed slots never drift during denoising.
            latents = clean_latents * context_mask + latents * future_mask
        return latents


def vae_latent_stats(vae: nn.Module, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Wan latent mean and inverse standard deviation tensors."""
    config = getattr(vae, "config", None)
    mean_values = getattr(config, "latents_mean", None)
    std_values = getattr(config, "latents_std", None)
    channels = int(getattr(config, "z_dim", 0))
    if not mean_values or not std_values or channels < 1:
        raise ValueError("Wan VAE must expose z_dim, latents_mean, and latents_std")
    if len(mean_values) != channels or len(std_values) != channels:
        raise ValueError("Wan VAE latent statistics do not match z_dim")
    mean = torch.tensor(mean_values, device=device, dtype=dtype).view(1, channels, 1, 1, 1)
    std = torch.tensor(std_values, device=device, dtype=dtype).view(1, channels, 1, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("Wan VAE latent standard deviations must be positive")
    return mean, std.reciprocal()


def normalize_vae_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply the Wan Diffusers affine normalization using inverse std."""
    return (latents - mean.to(device=latents.device, dtype=latents.dtype)) * std.to(
        device=latents.device, dtype=latents.dtype
    )


def denormalize_vae_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert :func:`normalize_vae_latents` with its inverse std argument."""
    return latents / std.to(device=latents.device, dtype=latents.dtype) + mean.to(
        device=latents.device, dtype=latents.dtype
    )


def extract_lora_state(transformer: nn.Module, adapter_name: str) -> dict[str, torch.Tensor]:
    """Extract only LoRA tensors so checkpoints never duplicate a Wan base model."""
    from peft import get_peft_model_state_dict

    return {
        name: value.detach().cpu()
        for name, value in get_peft_model_state_dict(transformer, adapter_name=adapter_name).items()
    }


def load_lora_state(transformer: nn.Module, adapter_name: str, state: dict[str, torch.Tensor]) -> None:
    """Load the compact LoRA portion saved by :func:`extract_lora_state`."""
    from peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(transformer, state, adapter_name=adapter_name)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA checkpoint keys: {result.unexpected_keys[:5]}")


def configure_wan_lora(
    transformer: nn.Module,
    *,
    rank: int,
    alpha: int | None = None,
    adapter_name: str = "track2",
    target_modules: Iterable[str] = ("to_q", "to_k", "to_v", "to_out.0"),
) -> list[str]:
    """Freeze the Wan base and attach LoRA to all self/cross-attention projections."""
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    from peft import LoraConfig

    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=int(alpha or rank),
            target_modules=list(target_modules),
            init_lora_weights=True,
        ),
        adapter_name=adapter_name,
    )
    names = [name for name, parameter in transformer.named_parameters() if parameter.requires_grad]
    if not names:
        raise RuntimeError("LoRA adapter attached no trainable parameters")
    return names
