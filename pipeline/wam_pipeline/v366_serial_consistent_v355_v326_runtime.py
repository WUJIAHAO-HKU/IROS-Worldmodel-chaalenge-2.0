"""Serial-consistent deployment of the frozen v326 terminal stack on v355."""

from __future__ import annotations

import numpy as np

from .v326_blended_phase_terminal_runtime import Track2V326BlendedPhaseTerminal


class Track2V366SerialConsistentV355V326(Track2V326BlendedPhaseTerminal):
    """Preserve exact single-request semantics for every batch element."""

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
