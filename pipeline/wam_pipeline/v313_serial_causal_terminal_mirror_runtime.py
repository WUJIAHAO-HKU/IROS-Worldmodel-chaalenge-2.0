"""Bit-exact serial-batch validation implementation of v312."""

from __future__ import annotations

import numpy as np

from .v312_causal_terminal_mirror_runtime import Track2V312CausalTerminalMirror


class Track2V313SerialCausalTerminalMirror(Track2V312CausalTerminalMirror):
    """Keep v312 single-request semantics and make batch execution exact."""

    def predict_batch(
        self,
        context_frames,
        history_actions,
        future_actions,
        seeds,
        instructions,
    ):
        return np.stack(
            [
                self.predict(context, history, future, int(seed), instruction)
                for context, history, future, seed, instruction in zip(
                    context_frames,
                    history_actions,
                    future_actions,
                    seeds,
                    instructions,
                    strict=True,
                )
            ],
            axis=0,
        )
