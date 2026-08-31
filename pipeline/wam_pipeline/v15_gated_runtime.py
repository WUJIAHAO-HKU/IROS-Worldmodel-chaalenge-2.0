"""V15 composite with an audited visual router on its autoregressive parent."""

from __future__ import annotations

import json
from pathlib import Path

from .gated_autoregressive_runtime import Track2GatedAutoregressiveUNet
from .v15_runtime import Track2V15Runtime, _sha256


FORMATS = {
    "track2-v15-gated-onpolicy-adaptation-v1",
    "track2-v15-gated-arm-routed-adaptation-v2",
    "track2-v15-action-gated-onpolicy-adaptation-v1",
    "track2-v15-instruction-gated-arm-temporal-adaptation-v3",
}


class Track2V15GatedRuntime(Track2V15Runtime):
    def __init__(self, checkpoint_dir: str | Path, library_dir: str | Path, device: str = "cuda") -> None:
        super().__init__(checkpoint_dir, library_dir, device)
        adaptation = self.root / "onpolicy_adaptation"
        manifest_path = adaptation / "adaptation_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") not in FORMATS:
            raise RuntimeError("unsupported V15 on-policy adaptation format")
        if manifest.get("base_release_manifest_sha256") != _sha256(self.root / "release_manifest.json"):
            raise RuntimeError("V15 adaptation was packaged against another base release")
        files = manifest.get("sha256", {})
        if not isinstance(files, dict) or not files:
            raise RuntimeError("V15 adaptation manifest has no artifacts")
        for relative, expected in files.items():
            path = adaptation / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"V15 adaptation artifact hash mismatch: {path}")
        baseline = self.root / "v8" / "baseline" / "autoregressive"
        candidate = adaptation / "adapted_autoregressive"
        right_candidate = (
            adaptation / "adapted_autoregressive_right"
            if manifest.get("format") in {
                "track2-v15-gated-arm-routed-adaptation-v2",
                "track2-v15-instruction-gated-arm-temporal-adaptation-v3",
            }
            else None
        )
        gate = adaptation / "source_gate.pt"
        self.gated_autoregressive = Track2GatedAutoregressiveUNet(
            baseline,
            candidate,
            gate,
            device,
            right_candidate_dir=right_candidate,
            candidate_blend=float(manifest.get("candidate_blend", 1.0)),
            candidate_blend_schedule=manifest.get("left_candidate_blend_schedule"),
            right_candidate_blend_schedule=manifest.get("right_candidate_blend_schedule"),
            instruction_arm_router=bool(manifest.get("instruction_arm_router", False)),
        )
        self.parent.baseline.autoregressive = self.gated_autoregressive
