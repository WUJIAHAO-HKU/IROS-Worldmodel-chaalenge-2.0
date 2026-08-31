"""Model backends behind the exact same 5-frame/8-action inference contract."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class ModelBackend(ABC):
    """A backend only predicts RGB frames; it never computes reward or actions."""

    @abstractmethod
    def predict(
        self,
        context_frames: np.ndarray,
        history_actions: np.ndarray,
        future_actions: np.ndarray,
        seed: int,
        instruction: str | None,
    ) -> np.ndarray:
        """Return [8, 256, 256, 3] uint8 future RGB frames."""


class SyntheticActionBackend(ModelBackend):
    """Deterministic protocol-test backend, deliberately not a contest model."""

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3):
            raise ValueError("unexpected context frame shape")
        if history_actions.shape != (CONTEXT_FRAMES - 1, ACTION_DIM):
            raise ValueError("unexpected history action shape")
        if future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("unexpected future action shape")
        # The seed-only dither makes different seeds valid stochastic samples while preserving repeatability.
        rng = np.random.default_rng(seed)
        output = np.empty((PREDICTION_FRAMES, 256, 256, 3), dtype=np.uint8)
        frame = context_frames[-1].astype(np.float32)
        action_scale = np.array([8.0, 5.0, 3.0], dtype=np.float32)
        for index, action in enumerate(future_actions):
            shift = np.rint(np.tanh(action[:2]) * 5).astype(int)
            frame = np.roll(frame, shift=(int(shift[1]), int(shift[0])), axis=(0, 1))
            tint = np.tanh(action[2:5]) * action_scale
            frame = np.clip(frame + tint.reshape(1, 1, 3), 0, 255)
            noise = rng.integers(-1, 2, size=(256, 256, 1), dtype=np.int16)
            output[index] = np.clip(frame + noise, 0, 255).astype(np.uint8)
        return output


class IVideoGPTBackend(ModelBackend):
    """Runtime adapter for a fine-tuned 64x64 iVideoGPT checkpoint.

    This class intentionally refuses to pretend that a BAIR 4-D checkpoint is
    compatible with Track 2. Supply a checkpoint fine-tuned with the provided
    5-context/14-action configuration before using this backend.
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime
        metadata_path = self.checkpoint_dir / "track2_ivideogpt_config.npz"
        if not metadata_path.exists():
            raise RuntimeError(
                "iVideoGPT checkpoint is not Track-2-adapted; run scripts/train_ivideogpt64.py first"
            )
        # The concrete loader is deferred so protocol tests do not require torch/transformers.
        try:
            from .ivideogpt_runtime import load_track2_ivideogpt
        except ImportError as exc:
            raise RuntimeError("iVideoGPT runtime dependencies are not installed") from exc
        self._runtime = load_track2_ivideogpt(self.checkpoint_dir, self.device)
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        runtime = self._load_runtime()
        return runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class ResidualUNetBackend(ModelBackend):
    """Track-2-specific residual baseline with the same external contract."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .residual_unet_runtime import Track2ResidualUNet

            self._runtime = Track2ResidualUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class FlowResidualUNetBackend(ModelBackend):
    """Appearance-preserving action-conditioned Track 2 predictor."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .flow_residual_unet_runtime import Track2FlowResidualUNet

            self._runtime = Track2FlowResidualUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class MultiSourceFlowUNetBackend(ModelBackend):
    """One-pass five-source flow student with native request batching."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .multisource_flow_unet_runtime import Track2MultiSourceFlowUNet

            self._runtime = Track2MultiSourceFlowUNet(
                self.checkpoint_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class TemporalUNetBackend(ModelBackend):
    """Per-step action-conditioned Track 2 predictor."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .temporal_unet_runtime import Track2TemporalUNet

            self._runtime = Track2TemporalUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class AutoregressiveUNetBackend(ModelBackend):
    """One-step model recursively rolled forward for the exact 8-frame contract."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .autoregressive_unet_runtime import Track2AutoregressiveUNet

            self._runtime = Track2AutoregressiveUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class ArmRoutedAutoregressiveUNetBackend(ModelBackend):
    """Frozen left/right autoregressive experts routed by declared task arm/actions."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import json

        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        manifest = json.loads(
            (self.checkpoint_dir / "arm_routed_autoregressive_manifest.json").read_text()
        )
        if manifest.get("format") != "track2-arm-routed-autoregressive-release-v1":
            raise RuntimeError("unsupported arm-routed autoregressive release")
        self.left_dir = self.checkpoint_dir / manifest["left_expert"]
        self.right_dir = self.checkpoint_dir / manifest["right_expert"]
        for relative, expected in manifest["model_sha256"].items():
            path = self.checkpoint_dir / relative / "model.pt"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"arm-routed expert hash mismatch: {path}")
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .arm_routed_autoregressive_runtime import (
                Track2ArmRoutedAutoregressiveUNet,
            )

            self._runtime = Track2ArmRoutedAutoregressiveUNet(
                self.left_dir, self.right_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V428V426RewardTraceBackend(ModelBackend):
    """Exact v426 backend with outcome-free v209/v426 final-frame telemetry."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        import json

        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        manifest = json.loads((self.checkpoint_dir / "motion_guarded_right_manifest.json").read_text())
        if manifest.get("format") != "track2-v426-motion-guarded-right-release-v1":
            raise RuntimeError("unsupported v426 motion-guarded release")
        expert_paths = {
            "left": self.checkpoint_dir / manifest["left_expert"] / "model.pt",
            "learned_right": self.checkpoint_dir / manifest["learned_right_expert"] / "model.pt",
            "frozen_right": self.checkpoint_dir / manifest["frozen_right_expert"] / "model.pt",
        }
        for name, path in expert_paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"][name]:
                raise RuntimeError(f"v426 expert hash mismatch: {path}")
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v428_v426_reward_trace_runtime import Track2V428V426RewardTrace

            self._runtime = Track2V428V426RewardTrace(self.checkpoint_dir, self.device)
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(context_frames, history_actions, future_actions, seeds, instructions)


class V431V169RewardTraceBackend(ModelBackend):
    """Bit-exact original v169 backend with outcome-free trace telemetry."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v431_v169_reward_trace_runtime import Track2V431V169RewardTrace

            self._runtime = Track2V431V169RewardTrace(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V436V432Step25TraceBackend(ModelBackend):
    """Passed v432-step25 output with v354-parent diagnostic telemetry."""
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir=Path(checkpoint_dir); self.device=device; self._runtime=None

    def _load_runtime(self):
        if self._runtime is None:
            from .v436_v432_step25_reward_trace_runtime import Track2V436V432Step25RewardTrace
            self._runtime=Track2V436V432Step25RewardTrace(self.checkpoint_dir,self.device)
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(context_frames,history_actions,future_actions,seed,instruction)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(context_frames,history_actions,future_actions,seeds,instructions)


class V439V169ActionCausalTraceBackend(ModelBackend):
    """v439 hybrid RGB with same-request v169 baseline telemetry."""
    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v439_v169_action_causal_reward_trace_runtime import (
                Track2V439V169ActionCausalRewardTrace,
            )
            self._runtime = Track2V439V169ActionCausalRewardTrace(
                self.checkpoint_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(context_frames, history_actions, future_actions, seeds, instructions)


class V216PublicKNNBlendBackend(ModelBackend):
    """Frozen arm-routed parent blended with public-only right-arm RGB retrieval."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_index: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_index = Path(library_index)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v216_public_knn_blend_runtime import Track2V216PublicKNNBlend

            self._runtime = Track2V216PublicKNNBlend(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V236CausalPublicKNNBlendBackend(V216PublicKNNBlendBackend):
    """Right-arm public retrieval gated by the requested gripper state."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v236_causal_public_knn_runtime import (
                Track2V236CausalPublicKNNBlend,
            )

            self._runtime = Track2V236CausalPublicKNNBlend(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V241AlignmentGatedTransportBackend(V216PublicKNNBlendBackend):
    """Post-grasp public retrieval gated by expert-aligned right-arm motion."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v241_alignment_gated_transport_runtime import (
                Track2V241AlignmentGatedTransport,
            )

            self._runtime = Track2V241AlignmentGatedTransport(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V245CleanProgressiveSuccessorBackend(V216PublicKNNBlendBackend):
    """Action-conditioned progressive successor from public clean demos."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v245_clean_progressive_successor_runtime import (
                Track2V245CleanProgressiveSuccessor,
            )

            self._runtime = Track2V245CleanProgressiveSuccessor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V247TerminalSuccessorBackend(V216PublicKNNBlendBackend):
    """Action-conditioned terminal successor from public clean demos."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v247_terminal_successor_runtime import Track2V247TerminalSuccessor

            self._runtime = Track2V247TerminalSuccessor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V375BoundedCartesianPhaseBackend(V216PublicKNNBlendBackend):
    """Locally refine a visual phase with frozen end-effector geometry."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v375_bounded_cartesian_phase_runtime import (
                Track2V375BoundedCartesianPhase,
            )

            self._runtime = Track2V375BoundedCartesianPhase(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V376OODRoutedCartesianPhaseBackend(V216PublicKNNBlendBackend):
    """Keep v355 on clean contexts and apply v375 only after visual drift."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v376_ood_routed_cartesian_phase_runtime import (
                Track2V376OODRoutedCartesianPhase,
            )

            self._runtime = Track2V376OODRoutedCartesianPhase(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V378SourceRoutedBlendedCartesianBackend(V216PublicKNNBlendBackend):
    """Blend Cartesian correction only for generated-looking contexts."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v378_source_routed_blended_cartesian_runtime import (
                Track2V378SourceRoutedBlendedCartesian,
            )
            self._runtime = Track2V378SourceRoutedBlendedCartesian(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V382SourceGateTerminalHoldBackend(V216PublicKNNBlendBackend):
    """Use a train-only source gate and Cartesian terminal-hold rendering."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v382_source_gate_terminal_hold_runtime import Track2V382SourceGateTerminalHold
            self._runtime = Track2V382SourceGateTerminalHold(self.checkpoint_dir, self.library_index, self.device)
        return self._runtime


class V384ActionPhaseCleanReanchorBackend(V216PublicKNNBlendBackend):
    """Use a source gate and action-phase-qualified clean terminal reanchor."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v384_action_phase_clean_reanchor_runtime import (
                Track2V384ActionPhaseCleanReanchor,
            )
            self._runtime = Track2V384ActionPhaseCleanReanchor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V385NativeBatchCleanReanchorBackend(V216PublicKNNBlendBackend):
    """Native-batch v326 plus action-phase clean terminal reanchor."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v385_native_batch_clean_reanchor_runtime import (
                Track2V385NativeBatchCleanReanchor,
            )
            self._runtime = Track2V385NativeBatchCleanReanchor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V387V385RouteTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v385 diagnostic backend with route telemetry."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v387_v385_route_trace_runtime import Track2V387V385RouteTrace
            self._runtime = Track2V387V385RouteTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V388NoPhaseCleanReanchorBackend(V216PublicKNNBlendBackend):
    """v385 clean reanchor with the diagnosed phase gate removed."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v388_no_phase_clean_reanchor_runtime import Track2V388NoPhaseCleanReanchor
            self._runtime = Track2V388NoPhaseCleanReanchor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V390ContinuousPhaseCleanReanchorBackend(V216PublicKNNBlendBackend):
    """v385 clean reanchor with a train-only continuous visual/action phase gate."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v390_continuous_phase_clean_reanchor_runtime import (
                Track2V390ContinuousPhaseCleanReanchor,
            )
            self._runtime = Track2V390ContinuousPhaseCleanReanchor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V391V390RouteTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v390 backend with preflight route telemetry."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v391_v390_route_trace_runtime import Track2V391V390RouteTrace
            self._runtime = Track2V391V390RouteTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V398V390RelativeActionTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v390 backend with relative-action phase telemetry."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v398_v390_relative_action_trace_runtime import (
                Track2V398V390RelativeActionTrace,
            )
            self._runtime = Track2V398V390RelativeActionTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V399PosteriorBlendCleanReanchorBackend(V216PublicKNNBlendBackend):
    """Posterior-weighted continuous clean terminal reanchor."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v399_posterior_blend_clean_reanchor_runtime import (
                Track2V399PosteriorBlendCleanReanchor,
            )
            self._runtime = Track2V399PosteriorBlendCleanReanchor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V400SupportedPosteriorBlendBackend(V216PublicKNNBlendBackend):
    """Posterior blend restricted to the classifier's declared support."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v400_supported_posterior_blend_runtime import Track2V400SupportedPosteriorBlend
            self._runtime = Track2V400SupportedPosteriorBlend(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V401V400RouteTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v400 backend with preflight telemetry."""
    def _load_runtime(self):
        if self._runtime is None:
            from .v401_v400_route_trace_runtime import Track2V401V400RouteTrace
            self._runtime = Track2V401V400RouteTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V404V400DenseDiagnosticBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v400 backend with dense right post-grasp telemetry."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v404_v400_dense_diagnostic_runtime import (
                Track2V404V400DenseDiagnostic,
            )

            self._runtime = Track2V404V400DenseDiagnostic(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V405V400ProgressiveDiagnosticBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v400 backend with progressive-action telemetry."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v405_v400_progressive_diagnostic_runtime import (
                Track2V405V400ProgressiveDiagnostic,
            )

            self._runtime = Track2V405V400ProgressiveDiagnostic(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V406ProgressiveThenTerminalBackend(V216PublicKNNBlendBackend):
    """Reachable public-expert progression with protected v400 terminals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v406_progressive_then_terminal_runtime import (
                Track2V406ProgressiveThenTerminal,
            )

            self._runtime = Track2V406ProgressiveThenTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V407OneChunkProgressiveBackend(V216PublicKNNBlendBackend):
    """One official action-chunk progression with protected v400 terminals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v407_one_chunk_progressive_runtime import (
                Track2V407OneChunkProgressive,
            )

            self._runtime = Track2V407OneChunkProgressive(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V408TemporalRampProgressiveBackend(V216PublicKNNBlendBackend):
    """Temporally ramped one-chunk progression with protected terminals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v408_temporal_ramp_progressive_runtime import (
                Track2V408TemporalRampProgressive,
            )

            self._runtime = Track2V408TemporalRampProgressive(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V409HalfContractedProgressiveBackend(V216PublicKNNBlendBackend):
    """Half-contracted one-chunk progression with protected terminals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v409_half_contracted_progressive_runtime import (
                Track2V409HalfContractedProgressive,
            )

            self._runtime = Track2V409HalfContractedProgressive(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V410V409RouteTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v409 backend with outcome-free route telemetry."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v410_v409_route_trace_runtime import Track2V410V409RouteTrace

            self._runtime = Track2V410V409RouteTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V414V409RewardTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v409 backend capturing outcome-free reward inputs."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v414_v409_reward_trace_runtime import Track2V414V409RewardTrace

            self._runtime = Track2V414V409RewardTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V421V420RewardTraceBackend(V216PublicKNNBlendBackend):
    """Output-equivalent v420 backend capturing outcome-free reward inputs."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v421_v420_reward_trace_runtime import Track2V421V420RewardTrace

            self._runtime = Track2V421V420RewardTrace(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V250SpecificTerminalSuccessorBackend(V216PublicKNNBlendBackend):
    """Terminal successor requiring expert-consistent right-arm direction."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v250_specific_terminal_successor_runtime import (
                Track2V250SpecificTerminalSuccessor,
            )
            self._runtime = Track2V250SpecificTerminalSuccessor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V254DeltaRegimeTerminalBackend(V216PublicKNNBlendBackend):
    """Terminal successor with offset-invariant delta-action regime gates."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v254_delta_regime_terminal_runtime import Track2V254DeltaRegimeTerminal

            self._runtime = Track2V254DeltaRegimeTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V271EndpointCalibratedTerminalBackend(V216PublicKNNBlendBackend):
    """v254 terminal successor with absolute endpoint calibration for OOD actions."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v271_endpoint_calibrated_terminal_runtime import (
                Track2V271EndpointCalibratedTerminal,
            )

            self._runtime = Track2V271EndpointCalibratedTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V290RightClosedMirrorBackend(V216PublicKNNBlendBackend):
    """Mirror-equivariant v271 dynamics for right-arm post-grasp requests."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v290_right_closed_mirror_runtime import (
                Track2V290RightClosedMirror,
            )

            self._runtime = Track2V290RightClosedMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V292TerminalPreservingMirrorBackend(V216PublicKNNBlendBackend):
    """Mirrored base dynamics with original-coordinate v271 terminal logic."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v292_terminal_preserving_mirror_runtime import (
                Track2V292TerminalPreservingMirror,
            )

            self._runtime = Track2V292TerminalPreservingMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V295TerminalFramePreservingMirrorBackend(V216PublicKNNBlendBackend):
    """Mirrored right dynamics with the direct v271 terminal frame."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v295_terminal_frame_preserving_mirror_runtime import (
                Track2V295TerminalFramePreservingMirror,
            )

            self._runtime = Track2V295TerminalFramePreservingMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V301BatchedTerminalFrameMirrorBackend(V216PublicKNNBlendBackend):
    """Native-batched output-equivalent implementation of v295."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v301_batched_terminal_frame_mirror_runtime import (
                Track2V301BatchedTerminalFrameMirror,
            )

            self._runtime = Track2V301BatchedTerminalFrameMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V312CausalTerminalMirrorBackend(V216PublicKNNBlendBackend):
    """Full right-terminal mirror guarded by a public-train action classifier."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v312_causal_terminal_mirror_runtime import (
                Track2V312CausalTerminalMirror,
            )

            self._runtime = Track2V312CausalTerminalMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V313SerialCausalTerminalMirrorBackend(V216PublicKNNBlendBackend):
    """Bit-exact serial-batch validation implementation of v312."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v313_serial_causal_terminal_mirror_runtime import (
                Track2V313SerialCausalTerminalMirror,
            )

            self._runtime = Track2V313SerialCausalTerminalMirror(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V314TransitionCausalTerminalBackend(V216PublicKNNBlendBackend):
    """Phase-aware successful-transition and rejected-action terminal dynamics."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v314_transition_causal_terminal_runtime import (
                Track2V314TransitionCausalTerminal,
            )

            self._runtime = Track2V314TransitionCausalTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V315SparseFailureTerminalBackend(V216PublicKNNBlendBackend):
    """v295-preserving dynamics with sparse action-causal failure terminals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v315_sparse_failure_terminal_runtime import (
                Track2V315SparseFailureTerminal,
            )

            self._runtime = Track2V315SparseFailureTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V317BatchedSparseFailureTerminalBackend(V216PublicKNNBlendBackend):
    """Native-batched deployment of v315 sparse failure semantics."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v317_batched_sparse_failure_terminal_runtime import (
                Track2V317BatchedSparseFailureTerminal,
            )

            self._runtime = Track2V317BatchedSparseFailureTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V322CausalAlphaTerminalBackend(V216PublicKNNBlendBackend):
    """Action-causal terminal selector and blend-strength calibration."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v322_causal_alpha_terminal_runtime import (
                Track2V322CausalAlphaTerminal,
            )

            self._runtime = Track2V322CausalAlphaTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V324PhaseGuardedTerminalBackend(V216PublicKNNBlendBackend):
    """Sparse public-phase repair on top of frozen v317 semantics."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v324_phase_guarded_terminal_runtime import (
                Track2V324PhaseGuardedTerminal,
            )

            self._runtime = Track2V324PhaseGuardedTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V325SameEpisodePhaseRepairBackend(V216PublicKNNBlendBackend):
    """Same-episode repair for terminal-degraded public retrievals."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v325_same_episode_phase_repair_runtime import (
                Track2V325SameEpisodePhaseRepair,
            )

            self._runtime = Track2V325SameEpisodePhaseRepair(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V326BlendedPhaseTerminalBackend(V216PublicKNNBlendBackend):
    """Strength-limited public-phase repair on frozen v317."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v326_blended_phase_terminal_runtime import (
                Track2V326BlendedPhaseTerminal,
            )

            self._runtime = Track2V326BlendedPhaseTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V366SerialConsistentV355V326Backend(V216PublicKNNBlendBackend):
    """Bit-exact batch wrapper for the frozen v326 terminal stack on v355."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v366_serial_consistent_v355_v326_runtime import (
                Track2V366SerialConsistentV355V326,
            )

            self._runtime = Track2V366SerialConsistentV355V326(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V367SparseTerminalOverlayBackend(V216PublicKNNBlendBackend):
    """Sparse phase-qualified terminal overlay on frozen v355 dynamics."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v367_sparse_terminal_overlay_runtime import (
                Track2V367SparseTerminalOverlay,
            )

            self._runtime = Track2V367SparseTerminalOverlay(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V328CoherentPhaseTrajectoryBackend(V216PublicKNNBlendBackend):
    """Keep the complete phase-qualified successor instead of a 4+1 splice."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v328_coherent_phase_trajectory_runtime import (
                Track2V328CoherentPhaseTrajectory,
            )

            self._runtime = Track2V328CoherentPhaseTrajectory(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V331PhaseOnsetTerminalBackend(V216PublicKNNBlendBackend):
    """Visible success-onset repair on frozen v326/v317 gates."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v331_phase_onset_terminal_runtime import (
                Track2V331PhaseOnsetTerminal,
            )

            self._runtime = Track2V331PhaseOnsetTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V332PhaseAlignedSuccessorBackend(V216PublicKNNBlendBackend):
    """Progress-aligned public success sequence on frozen v326 gates."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v332_phase_aligned_successor_runtime import (
                Track2V332PhaseAlignedSuccessor,
            )

            self._runtime = Track2V332PhaseAlignedSuccessor(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V333HybridOnsetTerminalBackend(V216PublicKNNBlendBackend):
    """Same-episode onset for ineligible tails, v326 terminal otherwise."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v333_hybrid_onset_terminal_runtime import (
                Track2V333HybridOnsetTerminal,
            )

            self._runtime = Track2V333HybridOnsetTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V334BoundedOnsetTerminalBackend(V216PublicKNNBlendBackend):
    """Immediate-onset v333 repair with v326 fallback outside offset two."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v334_bounded_onset_terminal_runtime import (
                Track2V334BoundedOnsetTerminal,
            )

            self._runtime = Track2V334BoundedOnsetTerminal(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class V335AllOffsetBoundedOnsetBackend(V216PublicKNNBlendBackend):
    """V334 semantics frozen under the corrected all-offset recursive gate."""

    def _load_runtime(self):
        if self._runtime is None:
            from .v335_all_offset_bounded_onset_runtime import (
                Track2V335AllOffsetBoundedOnset,
            )

            self._runtime = Track2V335AllOffsetBoundedOnset(
                self.checkpoint_dir, self.library_index, self.device
            )
        return self._runtime


class HybridUNetBackend(ModelBackend):
    """Horizon-specialized Track 2 predictor packaged as one checkpoint directory."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .hybrid_unet_runtime import Track2HorizonHybridUNet

            self._runtime = Track2HorizonHybridUNet(self.checkpoint_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class V15CompositeBackend(ModelBackend):
    """Validated v15 composite exposed through the online Track-2 contract."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .v15_runtime import Track2V15Runtime

            self._runtime = Track2V15Runtime(self.checkpoint_dir, self.library_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        if self._runtime is None:
            from .v15_runtime import Track2V15Runtime

            self._runtime = Track2V15Runtime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V15GatedCompositeBackend(ModelBackend):
    """Frozen V15 with a learned state router to an on-policy AR expert."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .v15_gated_runtime import Track2V15GatedRuntime

            self._runtime = Track2V15GatedRuntime(self.checkpoint_dir, self.library_dir, self.device)
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        if self._runtime is None:
            from .v15_gated_runtime import Track2V15GatedRuntime

            self._runtime = Track2V15GatedRuntime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime.predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V15ArmRoutedCompositeBackend(ModelBackend):
    """Frozen V15 with a same-domain parent selected only by request actions."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .v15_arm_routed_runtime import Track2V15ArmRoutedRuntime

            self._runtime = Track2V15ArmRoutedRuntime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime.predict(context_frames, history_actions, future_actions, seed, instruction)


class V15PostResidualCompositeBackend(ModelBackend):
    """Frozen complete V15 followed by the audited bounded residual."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        if self._runtime is None:
            from .v15_post_residual_runtime import Track2V15PostResidualRuntime

            self._runtime = Track2V15PostResidualRuntime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime.predict(
            context_frames, history_actions, future_actions, seed, instruction
        )


class V168TerminalProtectedBackend(ModelBackend):
    """Fixed right-arm middle-horizon blend with bit-exact V15.7 terminals."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        library_dir: str | Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v168_terminal_protected_runtime import Track2V168TerminalProtectedRuntime

            self._runtime = Track2V168TerminalProtectedRuntime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


class V169ArmRoutedBackend(ModelBackend):
    """V16.9 fixed blend with a deployable visual/action arm router."""

    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device="cuda") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.library_dir = Path(library_dir)
        self.device = device
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is None:
            from .v169_arm_routed_runtime import Track2V169ArmRoutedRuntime

            self._runtime = Track2V169ArmRoutedRuntime(
                self.checkpoint_dir, self.library_dir, self.device
            )
        return self._runtime

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        return self._load_runtime().predict(
            context_frames, history_actions, future_actions, seed, instruction
        )

    def predict_batch(self, context_frames, history_actions, future_actions, seeds, instructions):
        return self._load_runtime().predict_batch(
            context_frames, history_actions, future_actions, seeds, instructions
        )


def build_backend(
    name: str,
    checkpoint_dir: str | None = None,
    device: str = "cuda",
    *,
    v15_library_dir: str | Path | None = None,
    v216_library_index: str | Path | None = None,
) -> ModelBackend:
    if name == "synthetic":
        return SyntheticActionBackend()
    if name == "ivideogpt":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend ivideogpt")
        return IVideoGPTBackend(checkpoint_dir, device)
    if name == "residual-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend residual-unet")
        return ResidualUNetBackend(checkpoint_dir, device)
    if name == "flow-residual-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend flow-residual-unet")
        return FlowResidualUNetBackend(checkpoint_dir, device)
    if name == "multisource-flow-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend multisource-flow-unet")
        return MultiSourceFlowUNetBackend(checkpoint_dir, device)
    if name == "temporal-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend temporal-unet")
        return TemporalUNetBackend(checkpoint_dir, device)
    if name == "autoregressive-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend autoregressive-unet")
        return AutoregressiveUNetBackend(checkpoint_dir, device)
    if name == "arm-routed-autoregressive-unet":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend arm-routed-autoregressive-unet"
            )
        return ArmRoutedAutoregressiveUNetBackend(checkpoint_dir, device)
    if name == "v428-v426-reward-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v428-v426-reward-trace")
        return V428V426RewardTraceBackend(checkpoint_dir, device)
    if name == "v431-v169-reward-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v431-v169-reward-trace")
        if not v15_library_dir:
            raise ValueError(
                "WAM_V15_LIBRARY_DIR is required with --backend v431-v169-reward-trace"
            )
        return V431V169RewardTraceBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v436-v432-step25-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v436-v432-step25-trace")
        return V436V432Step25TraceBackend(checkpoint_dir, device)
    if name == "v439-v169-action-causal-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v439-v169-action-causal-trace")
        return V439V169ActionCausalTraceBackend(checkpoint_dir, device)
    if name == "v216-public-knn-blend":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v216-public-knn-blend")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v216-public-knn-blend")
        return V216PublicKNNBlendBackend(checkpoint_dir, v216_library_index, device)
    if name == "v236-causal-public-knn-blend":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v236-causal-public-knn-blend"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v236-causal-public-knn-blend"
            )
        return V236CausalPublicKNNBlendBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v241-alignment-gated-transport":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v241-alignment-gated-transport"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v241-alignment-gated-transport"
            )
        return V241AlignmentGatedTransportBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v245-clean-progressive-successor":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v245-clean-progressive-successor"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v245-clean-progressive-successor"
            )
        return V245CleanProgressiveSuccessorBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v247-terminal-successor":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend v247-terminal-successor"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend v247-terminal-successor"
            )
        return V247TerminalSuccessorBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v375-bounded-cartesian-phase":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend v375-bounded-cartesian-phase"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v375-bounded-cartesian-phase"
            )
        return V375BoundedCartesianPhaseBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v376-ood-routed-cartesian-phase":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v376-ood-routed-cartesian-phase"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v376-ood-routed-cartesian-phase"
            )
        return V376OODRoutedCartesianPhaseBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v378-source-routed-blended-cartesian":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v378-source-routed-blended-cartesian")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v378-source-routed-blended-cartesian")
        return V378SourceRoutedBlendedCartesianBackend(checkpoint_dir, v216_library_index, device)
    if name == "v382-source-gate-terminal-hold":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v382-source-gate-terminal-hold")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v382-source-gate-terminal-hold")
        return V382SourceGateTerminalHoldBackend(checkpoint_dir, v216_library_index, device)
    if name == "v384-action-phase-clean-reanchor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v384-action-phase-clean-reanchor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v384-action-phase-clean-reanchor")
        return V384ActionPhaseCleanReanchorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v385-native-batch-clean-reanchor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v385-native-batch-clean-reanchor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v385-native-batch-clean-reanchor")
        return V385NativeBatchCleanReanchorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v387-v385-route-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v387-v385-route-trace")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v387-v385-route-trace")
        return V387V385RouteTraceBackend(checkpoint_dir, v216_library_index, device)
    if name == "v388-no-phase-clean-reanchor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v388-no-phase-clean-reanchor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v388-no-phase-clean-reanchor")
        return V388NoPhaseCleanReanchorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v390-continuous-phase-clean-reanchor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v390-continuous-phase-clean-reanchor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v390-continuous-phase-clean-reanchor")
        return V390ContinuousPhaseCleanReanchorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v391-v390-route-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v391-v390-route-trace")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v391-v390-route-trace")
        return V391V390RouteTraceBackend(checkpoint_dir, v216_library_index, device)
    if name == "v398-v390-relative-action-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v398-v390-relative-action-trace")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v398-v390-relative-action-trace")
        return V398V390RelativeActionTraceBackend(checkpoint_dir, v216_library_index, device)
    if name == "v399-posterior-blend-clean-reanchor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v399-posterior-blend-clean-reanchor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v399-posterior-blend-clean-reanchor")
        return V399PosteriorBlendCleanReanchorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v400-supported-posterior-blend":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v400-supported-posterior-blend")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v400-supported-posterior-blend")
        return V400SupportedPosteriorBlendBackend(checkpoint_dir, v216_library_index, device)
    if name == "v401-v400-route-trace":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v401-v400-route-trace")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v401-v400-route-trace")
        return V401V400RouteTraceBackend(checkpoint_dir, v216_library_index, device)
    if name == "v404-v400-dense-diagnostic":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v404-v400-dense-diagnostic"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v404-v400-dense-diagnostic"
            )
        return V404V400DenseDiagnosticBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v405-v400-progressive-diagnostic":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v405-v400-progressive-diagnostic"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v405-v400-progressive-diagnostic"
            )
        return V405V400ProgressiveDiagnosticBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v406-progressive-then-terminal":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v406-progressive-then-terminal"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v406-progressive-then-terminal"
            )
        return V406ProgressiveThenTerminalBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v407-one-chunk-progressive":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v407-one-chunk-progressive"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v407-one-chunk-progressive"
            )
        return V407OneChunkProgressiveBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v408-temporal-ramp-progressive":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v408-temporal-ramp-progressive"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v408-temporal-ramp-progressive"
            )
        return V408TemporalRampProgressiveBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v409-half-contracted-progressive":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend "
                "v409-half-contracted-progressive"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend "
                "v409-half-contracted-progressive"
            )
        return V409HalfContractedProgressiveBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v410-v409-route-trace":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend v410-v409-route-trace"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend v410-v409-route-trace"
            )
        return V410V409RouteTraceBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v414-v409-reward-trace":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend v414-v409-reward-trace"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend v414-v409-reward-trace"
            )
        return V414V409RewardTraceBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v421-v420-reward-trace":
        if not checkpoint_dir:
            raise ValueError(
                "--checkpoint-dir is required with --backend v421-v420-reward-trace"
            )
        if not v216_library_index:
            raise ValueError(
                "WAM_V216_LIBRARY_INDEX is required with --backend v421-v420-reward-trace"
            )
        return V421V420RewardTraceBackend(
            checkpoint_dir, v216_library_index, device
        )
    if name == "v250-specific-terminal-successor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v250-specific-terminal-successor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v250-specific-terminal-successor")
        return V250SpecificTerminalSuccessorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v254-delta-regime-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v254-delta-regime-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v254-delta-regime-terminal")
        return V254DeltaRegimeTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v271-endpoint-calibrated-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v271-endpoint-calibrated-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v271-endpoint-calibrated-terminal")
        return V271EndpointCalibratedTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v290-right-closed-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v290-right-closed-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v290-right-closed-mirror")
        return V290RightClosedMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v292-terminal-preserving-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v292-terminal-preserving-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v292-terminal-preserving-mirror")
        return V292TerminalPreservingMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v295-terminal-frame-preserving-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v295-terminal-frame-preserving-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v295-terminal-frame-preserving-mirror")
        return V295TerminalFramePreservingMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v301-batched-terminal-frame-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v301-batched-terminal-frame-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v301-batched-terminal-frame-mirror")
        return V301BatchedTerminalFrameMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v312-causal-terminal-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v312-causal-terminal-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v312-causal-terminal-mirror")
        return V312CausalTerminalMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v313-serial-causal-terminal-mirror":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v313-serial-causal-terminal-mirror")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v313-serial-causal-terminal-mirror")
        return V313SerialCausalTerminalMirrorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v314-transition-causal-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v314-transition-causal-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v314-transition-causal-terminal")
        return V314TransitionCausalTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v315-sparse-failure-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v315-sparse-failure-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v315-sparse-failure-terminal")
        return V315SparseFailureTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v317-batched-sparse-failure-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v317-batched-sparse-failure-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v317-batched-sparse-failure-terminal")
        return V317BatchedSparseFailureTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v322-causal-alpha-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v322-causal-alpha-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v322-causal-alpha-terminal")
        return V322CausalAlphaTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v324-phase-guarded-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v324-phase-guarded-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v324-phase-guarded-terminal")
        return V324PhaseGuardedTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v325-same-episode-phase-repair":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v325-same-episode-phase-repair")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v325-same-episode-phase-repair")
        return V325SameEpisodePhaseRepairBackend(checkpoint_dir, v216_library_index, device)
    if name == "v326-blended-phase-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v326-blended-phase-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v326-blended-phase-terminal")
        return V326BlendedPhaseTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v366-serial-consistent-v355-v326":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v366-serial-consistent-v355-v326")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v366-serial-consistent-v355-v326")
        return V366SerialConsistentV355V326Backend(checkpoint_dir, v216_library_index, device)
    if name == "v367-sparse-terminal-overlay":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v367-sparse-terminal-overlay")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v367-sparse-terminal-overlay")
        return V367SparseTerminalOverlayBackend(checkpoint_dir, v216_library_index, device)
    if name == "v328-coherent-phase-trajectory":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v328-coherent-phase-trajectory")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v328-coherent-phase-trajectory")
        return V328CoherentPhaseTrajectoryBackend(checkpoint_dir, v216_library_index, device)
    if name == "v331-phase-onset-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v331-phase-onset-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v331-phase-onset-terminal")
        return V331PhaseOnsetTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v332-phase-aligned-successor":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v332-phase-aligned-successor")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v332-phase-aligned-successor")
        return V332PhaseAlignedSuccessorBackend(checkpoint_dir, v216_library_index, device)
    if name == "v333-hybrid-onset-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v333-hybrid-onset-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v333-hybrid-onset-terminal")
        return V333HybridOnsetTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v334-bounded-onset-terminal":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v334-bounded-onset-terminal")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v334-bounded-onset-terminal")
        return V334BoundedOnsetTerminalBackend(checkpoint_dir, v216_library_index, device)
    if name == "v335-all-offset-bounded-onset":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v335-all-offset-bounded-onset")
        if not v216_library_index:
            raise ValueError("WAM_V216_LIBRARY_INDEX is required with --backend v335-all-offset-bounded-onset")
        return V335AllOffsetBoundedOnsetBackend(checkpoint_dir, v216_library_index, device)
    if name == "hybrid-unet":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend hybrid-unet")
        return HybridUNetBackend(checkpoint_dir, device)
    if name == "v15-composite":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v15-composite")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v15-composite")
        return V15CompositeBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v15-gated-composite":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v15-gated-composite")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v15-gated-composite")
        return V15GatedCompositeBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v15-arm-routed-composite":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v15-arm-routed-composite")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v15-arm-routed-composite")
        return V15ArmRoutedCompositeBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v15-post-residual-composite":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v15-post-residual-composite")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v15-post-residual-composite")
        return V15PostResidualCompositeBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v168-terminal-protected":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v168-terminal-protected")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v168-terminal-protected")
        return V168TerminalProtectedBackend(checkpoint_dir, v15_library_dir, device)
    if name == "v169-arm-routed":
        if not checkpoint_dir:
            raise ValueError("--checkpoint-dir is required with --backend v169-arm-routed")
        if not v15_library_dir:
            raise ValueError("WAM_V15_LIBRARY_DIR is required with --backend v169-arm-routed")
        return V169ArmRoutedBackend(checkpoint_dir, v15_library_dir, device)
    raise ValueError(f"unknown backend {name!r}")
