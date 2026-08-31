#!/usr/bin/env python3
"""Audit v441 using the frozen correct-comparator S1 protocol."""

from __future__ import annotations

import audit_v440_s1_offline as protocol


def main() -> int:
    protocol.LINEAGE = "v441"
    protocol.STATIC_FORMAT = "strict-track2-v441-postclose-s0-static-contract-v1"
    protocol.REPORT_FORMAT = "strict-track2-v441-postclose-s1-offline-gate-v1"
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
