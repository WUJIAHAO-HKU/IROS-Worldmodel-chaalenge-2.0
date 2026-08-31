"""Conservative 0.75-strength temporal blend using v342 threaded routing."""

from __future__ import annotations

from .v342_temporal_blended_public_reanchor_runtime import (
    Track2V342TemporalBlendedPublicReanchor,
)


CONSERVATIVE_REANCHOR_ALPHA = 0.75


class Track2V344ConservativeTemporalReanchor(Track2V342TemporalBlendedPublicReanchor):
    def __init__(self, *args, **kwargs):
        kwargs["reanchor_alpha"] = CONSERVATIVE_REANCHOR_ALPHA
        super().__init__(*args, **kwargs)
