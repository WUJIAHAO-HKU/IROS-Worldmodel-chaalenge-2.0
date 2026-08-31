"""Inference runtime for compact Track 2 action-conditioned Wan LoRA checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_ACTIONS, CONTEXT_FRAMES, PREDICTION_FRAMES
from .track2_wan import (
    TRACK2_LATENT_FRAMES,
    Track2ActionConditioner,
    Track2WanWorldModel,
    build_action_slots,
    denormalize_vae_latents,
    load_lora_state,
    normalize_action_slots,
    normalize_vae_latents,
    vae_latent_stats,
)


FORMAT_V1 = "track2-action-conditioned-wan-lora-v1"
FORMAT_V2 = "track2-action-conditioned-wan-trajectory-lora-v2"
FORMAT_V3 = "track2-action-conditioned-wan-latent-action-lora-v3"
FORMAT_V4 = "track2-action-conditioned-wan-trajectory-latent-lora-v4"
SUPPORTED_FORMATS = {FORMAT_V1, FORMAT_V2, FORMAT_V3, FORMAT_V4}


class Track2WanRuntime:
    """Load a local Wan base plus its compact Track 2 LoRA/action adapter."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        base_model: str | Path | None = None,
        device: str = "cuda",
        inference_steps: int = 30,
        inference_solver: str = "euler",
    ) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        self.torch = torch
        self.device = torch.device(device)
        self.inference_steps = int(inference_steps)
        if self.inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        if inference_solver not in {"euler", "heun"}:
            raise ValueError("inference_solver must be 'euler' or 'heun'")
        self.inference_solver = inference_solver
        root = Path(checkpoint_dir)
        state_path = root / "track2_wan_lora.pt"
        manifest_path = root / "training_manifest.json"
        if not state_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Track 2 Wan checkpoint requires track2_wan_lora.pt and training_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        checkpoint_format = manifest.get("format")
        if checkpoint_format not in SUPPORTED_FORMATS:
            raise RuntimeError("unsupported Track 2 Wan checkpoint format")
        base = Path(base_model) if base_model else Path(str(manifest.get("base_model", "")))
        if not (base / "transformer").is_dir() or not (base / "vae").is_dir():
            raise RuntimeError("Wan base model is unavailable; set WAM_WAN_BASE_MODEL to its local Diffusers directory")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        transformer = WanTransformer3DModel.from_pretrained(
            str(base), subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        vae = AutoencoderKLWan.from_pretrained(
            str(base), subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()
        lora_rank = int(manifest.get("lora_rank", 0))
        if lora_rank < 1:
            raise RuntimeError("Track 2 Wan manifest has no valid LoRA rank")
        from .track2_wan import configure_wan_lora

        configure_wan_lora(transformer, rank=lora_rank)
        trajectory_conditioned = checkpoint_format in {FORMAT_V2, FORMAT_V4}
        latent_action_conditioned = checkpoint_format in {FORMAT_V3, FORMAT_V4}
        conditioner = Track2ActionConditioner(
            int(transformer.config.text_dim),
            trajectory_conditioned=trajectory_conditioned,
            trajectory_hidden_dim=manifest.get("trajectory_hidden_dim"),
            latent_action_conditioned=latent_action_conditioned,
            latent_channels=int(transformer.config.out_channels) if latent_action_conditioned else None,
            latent_action_hidden_dim=manifest.get("latent_action_hidden_dim"),
        ).to(self.device, dtype=dtype)
        self.model = Track2WanWorldModel(transformer, conditioner).to(self.device).eval()
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("format") != checkpoint_format:
            raise RuntimeError("Track 2 Wan checkpoint format mismatch")
        self.model.action_conditioner.load_state_dict(state["action_conditioner"], strict=True)
        load_lora_state(self.model.transformer, "track2", state["transformer_lora"])
        self.model.eval()
        self.vae = vae
        self.latent_mean, self.latent_std = vae_latent_stats(vae, device=self.device, dtype=dtype)
        self.action_mean = state["action_mean"].to(self.device, dtype=torch.float32)
        self.action_std = state["action_std"].to(self.device, dtype=torch.float32)

    def _encode_context(self, frames: np.ndarray):
        torch = self.torch
        context = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        context = context.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        # The 13-frame VAE layout has two context latent slots.  Padding with
        # the final observed frame is used only to provide the missing future
        # pixels required by the causal VAE encoder; those slots are discarded.
        padded = torch.cat((context, context[-1:].expand(PREDICTION_FRAMES, -1, -1, -1)), dim=0)
        video = padded.unsqueeze(0).permute(0, 2, 1, 3, 4).to(dtype=next(self.vae.parameters()).dtype)
        with torch.inference_mode():
            latents = self.vae.encode(video).latent_dist.mode()
        latents = normalize_vae_latents(latents, self.latent_mean, self.latent_std)
        if latents.shape[2] != TRACK2_LATENT_FRAMES:
            raise RuntimeError(f"Wan VAE encoded {latents.shape[2]} latent slots, expected {TRACK2_LATENT_FRAMES}")
        return latents

    def predict(self, context_frames, history_actions, future_actions, seed: int, instruction: str | None):
        del instruction  # The formal action-conditioned Wan checkpoint has no text-task encoder input.
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context_frames must be [5,256,256,3] uint8")
        if history_actions.shape != (CONTEXT_ACTIONS, ACTION_DIM) or future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("actions must be [4,14] history and [8,14] future")
        torch = self.torch
        clean = self._encode_context(context_frames)
        history = torch.from_numpy(np.ascontiguousarray(history_actions)).to(self.device).unsqueeze(0)
        future = torch.from_numpy(np.ascontiguousarray(future_actions)).to(self.device).unsqueeze(0)
        slots = normalize_action_slots(build_action_slots(history, future), self.action_mean, self.action_std)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            sampled = self.model.sample_future_latents(
                clean,
                slots,
                num_inference_steps=self.inference_steps,
                generator=generator,
                solver=self.inference_solver,
            )
            raw = denormalize_vae_latents(sampled, self.latent_mean, self.latent_std)
            decoded = self.vae.decode(raw.to(dtype=next(self.vae.parameters()).dtype), return_dict=False)[0]
        # The VAE returns the full 13-slot RGB sequence.  The formal contract
        # exposes only the eight future positions after the five observations.
        rgb = decoded[:, :, CONTEXT_FRAMES:].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        prediction = rgb.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().copy()
        if prediction.shape != (PREDICTION_FRAMES, 256, 256, 3):
            raise RuntimeError(f"Wan decoder returned unexpected Track 2 shape: {prediction.shape}")
        return prediction
