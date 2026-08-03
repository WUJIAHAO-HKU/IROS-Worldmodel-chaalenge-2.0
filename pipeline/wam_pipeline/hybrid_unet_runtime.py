"""A packaged Track 2 model selecting the best validated predictor by horizon."""

from __future__ import annotations

import json
from pathlib import Path

from .autoregressive_unet_runtime import Track2AutoregressiveUNet
from .residual_unet_runtime import Track2ResidualUNet


class Track2HorizonHybridUNet:
    """Use the autoregressive model for short horizon and direct model afterward."""

    def __init__(self, checkpoint_dir: str | Path, device: str = "cuda") -> None:
        root = Path(checkpoint_dir)
        config = json.loads((root / "track2_hybrid_config.json").read_text())
        if config.get("format") != "track2-horizon-hybrid-v1":
            raise RuntimeError("unsupported Track 2 hybrid checkpoint format")
        self.early_frames = int(config["autoregressive_frames"])
        if not 1 <= self.early_frames < 8:
            raise RuntimeError("hybrid autoregressive_frames must be in [1, 7]")
        self.autoregressive = Track2AutoregressiveUNet(root / "autoregressive", device)
        self.direct = Track2ResidualUNet(root / "direct", device)

    def predict(self, context_frames, history_actions, future_actions, seed, instruction):
        early = self.autoregressive.predict(context_frames, history_actions, future_actions, seed, instruction)
        late = self.direct.predict(context_frames, history_actions, future_actions, seed, instruction)
        return __import__("numpy").concatenate([early[: self.early_frames], late[self.early_frames :]], axis=0)
