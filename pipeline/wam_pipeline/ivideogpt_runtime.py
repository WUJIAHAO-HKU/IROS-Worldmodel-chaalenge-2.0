"""Real iVideoGPT-64 runtime adapted to the Track 2 5/14/8 contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ivideogpt_upstream import import_upstream, load_bair_llm, set_context_length
from .profile import ACTION_DIM, CONTEXT_FRAMES, PREDICTION_FRAMES


class Track2IVideoGPT:
    """Frozen public tokenizer/LLM plus a Track-2-trained action projection."""

    def __init__(self, checkpoint_dir: Path, device: str) -> None:
        import torch

        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device)
        self.torch = torch
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        metadata = np.load(checkpoint_dir / "track2_ivideogpt_config.npz", allow_pickle=False)
        expected = {
            "context_frames": CONTEXT_FRAMES,
            "action_dim": ACTION_DIM,
            "prediction_frames": PREDICTION_FRAMES,
            "working_resolution": 64,
        }
        for name, value in expected.items():
            if int(metadata[name]) != value:
                raise RuntimeError(f"checkpoint metadata {name} does not match Track 2")
        manifest_path = checkpoint_dir / "training_manifest.json"
        state_path = checkpoint_dir / "track2_ivideogpt_state.pt"
        if not manifest_path.is_file() or not state_path.is_file():
            raise RuntimeError("Track-2 iVideoGPT weights are absent; run scripts/train_ivideogpt64.py")
        manifest = json.loads(manifest_path.read_text())
        upstream_checkpoint = Path(manifest["upstream_checkpoint"])
        if not upstream_checkpoint.is_dir():
            raise RuntimeError(f"upstream iVideoGPT checkpoint is missing: {upstream_checkpoint}")

        CompressiveVQModel, HeadModelWithAction, self.upstream_source = import_upstream()
        self.tokenizer = CompressiveVQModel.from_pretrained(str(upstream_checkpoint / "tokenizer"))
        set_context_length(self.tokenizer, CONTEXT_FRAMES)
        self.tokenizer.to(self.device).eval()
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad_(False)

        llm = load_bair_llm(upstream_checkpoint / "transformer", self.dtype, self.device).eval()
        self.model = HeadModelWithAction(
            llm=llm,
            action_dim=ACTION_DIM,
            prelude_tokens_num=(256 + 1) * CONTEXT_FRAMES - 1,
            tokens_num_per_dyna=16,
            context=CONTEXT_FRAMES,
            segment_length=CONTEXT_FRAMES + PREDICTION_FRAMES,
        ).to(self.device, dtype=self.dtype)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("format") not in {"track2-action-head-v1", "track2-ivideogpt-v2"}:
            raise RuntimeError("unsupported Track-2 iVideoGPT checkpoint format")
        if state.get("llm_tuning", "frozen") == "lora":
            from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

            lora_config = state.get("lora_config")
            lora_state = state.get("llm_lora")
            if not isinstance(lora_config, dict) or not isinstance(lora_state, dict):
                raise RuntimeError("LoRA checkpoint is missing its adapter configuration or weights")
            llm_lora = get_peft_model(
                llm,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(lora_config["r"]),
                    lora_alpha=int(lora_config["lora_alpha"]),
                    lora_dropout=float(lora_config["lora_dropout"]),
                    target_modules=list(lora_config["target_modules"]),
                    bias="none",
                ),
            )
            set_peft_model_state_dict(llm_lora, lora_state)
            self.model.llm = llm_lora
        self.model.action_linear.load_state_dict(state["action_linear"], strict=True)
        self.model.eval()
        normalization_path = checkpoint_dir / "action_normalization.npz"
        if normalization_path.is_file():
            normalization = np.load(normalization_path, allow_pickle=False)
            self.action_mean = np.asarray(normalization["mean"], dtype=np.float32)
            self.action_std = np.asarray(normalization["std"], dtype=np.float32)
            if self.action_mean.shape != (ACTION_DIM,) or self.action_std.shape != (ACTION_DIM,) or np.any(self.action_std <= 0):
                raise RuntimeError("invalid action normalization metadata")
        else:
            # Backward-compatible behavior for the original one-step smoke checkpoint.
            self.action_mean = np.zeros(ACTION_DIM, dtype=np.float32)
            self.action_std = np.ones(ACTION_DIM, dtype=np.float32)

    def _to_model_frames(self, frames: np.ndarray):
        import torch.nn.functional as functional

        tensor = self.torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
        tensor = functional.interpolate(tensor, size=(64, 64), mode="bilinear", align_corners=False)
        # Upstream iVideoGPT trains on ToTensor output: RGB in [0, 1].
        return tensor.div(255.0)

    @staticmethod
    def _validate(context_frames, history_actions, future_actions) -> None:
        if context_frames.shape != (CONTEXT_FRAMES, 256, 256, 3) or context_frames.dtype != np.uint8:
            raise ValueError("context frames must be [5,256,256,3] uint8")
        if history_actions.shape != (CONTEXT_FRAMES - 1, ACTION_DIM):
            raise ValueError("history actions must be [4,14]")
        if future_actions.shape != (PREDICTION_FRAMES, ACTION_DIM):
            raise ValueError("future actions must be [8,14]")

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        """Generate eight deterministic 256 RGB frames for the supplied actions."""
        self._validate(context_frames, history_actions, future_actions)
        torch = self.torch
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        # Tokenize one throw-away dynamic frame only to obtain the context token prefix.
        # The generated continuation replaces that dynamic frame and all eight outputs.
        model_context = self._to_model_frames(context_frames).unsqueeze(0).to(self.device)
        token_input = torch.cat([model_context, model_context[:, -1:]], dim=1)
        actions = np.concatenate([history_actions, future_actions], axis=0)
        actions = (actions - self.action_mean) / self.action_std
        action_tensor = torch.from_numpy(np.ascontiguousarray(actions)).unsqueeze(0).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            context_tokens, _ = self.tokenizer.tokenize(token_input, context_length=CONTEXT_FRAMES)
            # ``HeadModelWithAction`` injects u0 at the initial SDF token.
            # Keep the dummy frame's SDF (one token past the context prefix),
            # then generation replaces its 16 dynamic tokens and appends seven
            # further SDF/dynamic groups.
            prefix_length = (256 + 1) * CONTEXT_FRAMES
            generated_tokens = self.model.generate(
                context_tokens[:, :prefix_length],
                do_sample=False,
                top_k=None,
                max_new_tokens=(17 * PREDICTION_FRAMES) - 1,
                pad_token_id=0,
                action=action_tensor,
            )
            decoded = self.tokenizer.detokenize(generated_tokens, context_length=CONTEXT_FRAMES)
            predicted = decoded[:, CONTEXT_FRAMES:]
            predicted = self.torch.nn.functional.interpolate(
                predicted.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False
            ).reshape(1, PREDICTION_FRAMES, 3, 256, 256)
            predicted = predicted.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        return predicted.squeeze(0).permute(0, 2, 3, 1).cpu().numpy().copy()


def load_track2_ivideogpt(checkpoint_dir: str | Path, device: str) -> Track2IVideoGPT:
    return Track2IVideoGPT(Path(checkpoint_dir), device)
