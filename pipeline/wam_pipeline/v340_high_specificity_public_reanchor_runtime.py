"""Frozen v338 semantics paired with the train-selected v339 OOD gate."""

from __future__ import annotations

from .v338_recursive_public_reanchor_runtime import (
    PHASE_MAX_OFFSET,
    REANCHOR_ALPHA,
    Track2V338RecursivePublicReanchor,
)


class Track2V340HighSpecificityPublicReanchor(Track2V338RecursivePublicReanchor):
    """Release candidate: alpha 1.0 and public success-phase offset at most 16."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reanchor_alpha", REANCHOR_ALPHA)
        kwargs.setdefault("phase_max_offset", PHASE_MAX_OFFSET)
        super().__init__(*args, **kwargs)
