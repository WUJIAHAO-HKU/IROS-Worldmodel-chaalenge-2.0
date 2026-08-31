#!/usr/bin/env python3
"""Audit v443 with the frozen close-only and correct-comparator contracts."""

from __future__ import annotations

import audit_v440_s1_offline as protocol
from audit_v442_s1_offline import close_only_structure_checks


def main() -> int:
    protocol.LINEAGE = "v443"
    protocol.STATIC_FORMAT = "strict-track2-v443-close-s0-static-contract-v1"
    protocol.REPORT_FORMAT = "strict-track2-v443-close-s1-offline-gate-v1"
    protocol.STRUCTURAL_GATE_CONTRACT = (
        "action_only_preregistered_close_exact8_grasp_pattern1000_no_repeat"
    )
    protocol.intervention_structure_checks = close_only_structure_checks
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
