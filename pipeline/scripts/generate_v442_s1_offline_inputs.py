#!/usr/bin/env python3
"""Run the frozen correct-comparator S1 protocol with the v442 runtime."""

from __future__ import annotations

import generate_v440_s1_offline_inputs as protocol
from wam_pipeline.v442_v169_close_aligned_projection_runtime import (
    Track2V442V169CloseAlignedProjection,
)


def main() -> int:
    protocol.LINEAGE = "v442"
    protocol.INPUT_FORMAT = "strict-track2-v442-close-s1-offline-inputs-v1"
    protocol.RUNTIME_CLASS = Track2V442V169CloseAlignedProjection
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
