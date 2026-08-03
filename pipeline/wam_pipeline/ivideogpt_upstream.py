"""Small compatibility layer around the locally vendored iVideoGPT source."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def locate_upstream_source() -> Path:
    """Find the vendored core source required by the runtime adapter."""
    pipeline_root = Path(__file__).resolve().parents[1]
    workspace_root = pipeline_root.parent
    candidates = []
    configured = os.environ.get("WAM_IVIDEOGPT_SOURCE")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            # Use the complete upstream checkout by default. The older vendor
            # copies remain only as an offline fallback for existing artifacts.
            workspace_root / "third_party" / "iVideoGPT",
            pipeline_root / "vendor" / "iVideoGPT",
            pipeline_root / "vendor" / "iVideoGPT.incomplete-20260801",
        ]
    )
    required = Path("ivideogpt/vq_model/compressive_vq_model.py")
    for candidate in candidates:
        if (candidate / required).is_file() and (candidate / "ivideogpt/transformer/action_model.py").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "complete iVideoGPT source is unavailable; set WAM_IVIDEOGPT_SOURCE to a checkout containing ivideogpt/"
    )


def import_upstream():
    """Import the two upstream model classes only after the source path is known."""
    source = str(locate_upstream_source())
    if source not in sys.path:
        sys.path.insert(0, source)
        importlib.invalidate_caches()
    from ivideogpt.transformer import HeadModelWithAction
    from ivideogpt.vq_model import CompressiveVQModel

    return CompressiveVQModel, HeadModelWithAction, Path(source)


def load_bair_llm(checkpoint_dir: Path, dtype, device):
    """Load the upstream action-head safetensors into a plain HF Llama model."""
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    llm = LlamaForCausalLM(LlamaConfig.from_pretrained(str(checkpoint_dir), local_files_only=True))
    raw_state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    llm_state = {name.removeprefix("llm."): value for name, value in raw_state.items() if name.startswith("llm.")}
    missing, unexpected = llm.load_state_dict(llm_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"iVideoGPT LLM weights are incompatible; missing={missing}, unexpected={unexpected}")
    return llm.to(device=device, dtype=dtype)


def set_context_length(tokenizer, context_frames: int) -> None:
    """Expand the upstream one-frame cross-attention positions to Track 2 context.

    The public BAIR tokenizer was trained with one context frame.  Its upstream
    ``set_context_length`` changes a counter but cannot enlarge the learned
    key/value positional tensor.  Repeating that learned one-frame position is
    the explicit initialization used before Track 2 fine-tuning.
    """
    import torch

    for module in tokenizer.modules():
        if not hasattr(module, "kv_frames") or not hasattr(module, "kv_pos_emb"):
            continue
        old_frames = int(module.kv_frames)
        if old_frames == context_frames:
            continue
        position = module.kv_pos_emb.detach()
        if position.shape[0] % old_frames:
            raise RuntimeError("unexpected iVideoGPT cross-attention positional shape")
        per_frame = position.shape[0] // old_frames
        # Reuse the learned reference-frame positions for each added history frame.
        reference = position[-per_frame:]
        expanded = reference.repeat(context_frames, 1)
        module.kv_pos_emb = torch.nn.Parameter(expanded, requires_grad=module.kv_pos_emb.requires_grad)
        module.kv_frames = context_frames
    tokenizer.context_length = context_frames
    tokenizer.config["context_length"] = context_frames
