#!/usr/bin/env python3
"""Run the frozen correct-comparator S1 protocol with the v441 runtime."""

from __future__ import annotations

import generate_v440_s1_offline_inputs as protocol
from wam_pipeline.v441_v169_postclose_aligned_projection_runtime import (
    Track2V441V169PostcloseAlignedProjection,
)


def main() -> int:
    protocol.LINEAGE = "v441"
    protocol.INPUT_FORMAT = "strict-track2-v441-postclose-s1-offline-inputs-v1"
    protocol.RUNTIME_CLASS = Track2V441V169PostcloseAlignedProjection
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
