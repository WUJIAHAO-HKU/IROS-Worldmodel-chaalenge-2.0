"""Pure-v355 baseline adapter for inherited public gate audits."""

from __future__ import annotations

import json
from pathlib import Path

from wam_pipeline.arm_routed_autoregressive_runtime import (
    Track2ArmRoutedAutoregressiveUNet,
)


class PureV355Baseline:
    def __init__(self, checkpoint_dir, library_index=None, device="cuda", *args):
        root = Path(checkpoint_dir)
        manifest = json.loads(
            (root / "arm_routed_autoregressive_manifest.json").read_text()
        )
        self.parent = Track2ArmRoutedAutoregressiveUNet(
            root / manifest["left_expert"],
            root / manifest["right_expert"],
            device,
        )

    def predict(self, context, history, future, seed, instruction):
        return self.parent.predict(context, history, future, seed, instruction)

    def predict_batch(self, contexts, histories, futures, seeds, instructions):
        return self.parent.predict_batch(
            contexts, histories, futures, seeds, instructions
        )
