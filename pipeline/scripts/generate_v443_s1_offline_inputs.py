#!/usr/bin/env python3
"""Run the frozen correct-comparator S1 protocol with the v443 runtime."""

from __future__ import annotations

import generate_v440_s1_offline_inputs as protocol
from wam_pipeline.v443_v169_close_spatial_projection_runtime import (
    Track2V443V169CloseSpatialProjection,
)


def main() -> int:
    protocol.LINEAGE = "v443"
    protocol.INPUT_FORMAT = "strict-track2-v443-close-s1-offline-inputs-v1"
    protocol.RUNTIME_CLASS = Track2V443V169CloseSpatialProjection
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
