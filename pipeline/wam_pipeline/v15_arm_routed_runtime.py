"""V15 composite whose parent is a direct request-action-routed expert."""

from __future__ import annotations

import json
from pathlib import Path

from .arm_routed_autoregressive_runtime import Track2ArmRoutedAutoregressiveUNet
from .v15_runtime import Track2V15Runtime, _sha256


FORMAT = "track2-v15-arm-routed-initial-window-v3"


class Track2V15ArmRoutedRuntime(Track2V15Runtime):
    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device: str = "cuda") -> None:
        super().__init__(checkpoint_dir, library_dir, device)
        adaptation = self.root / "onpolicy_adaptation"
        manifest = json.loads((adaptation / "adaptation_manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise RuntimeError("unsupported direct arm-routed V15 adaptation format")
        if manifest.get("base_release_manifest_sha256") != _sha256(self.root / "release_manifest.json"):
            raise RuntimeError("V15 adaptation was packaged against another base release")
        files = manifest.get("sha256", {})
        if not isinstance(files, dict) or not files:
            raise RuntimeError("V15 adaptation manifest has no artifacts")
        for relative, expected in files.items():
            path = adaptation / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"V15 adaptation artifact hash mismatch: {path}")
        self.arm_routed_autoregressive = Track2ArmRoutedAutoregressiveUNet(
            adaptation / "adapted_autoregressive",
            adaptation / "adapted_autoregressive_right",
            device,
        )
        self.parent.baseline.autoregressive = self.arm_routed_autoregressive
