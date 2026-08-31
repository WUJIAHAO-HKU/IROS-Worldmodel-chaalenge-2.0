#!/usr/bin/env python3
"""Audit v442 with its action-only preregistered close-gate coverage."""

from __future__ import annotations

import numpy as np

import audit_v440_s1_offline as protocol


def close_only_structure_checks(
    gate: np.ndarray, s1: np.ndarray, phase: np.ndarray
) -> dict[str, bool]:
    selected = gate[s1].astype(bool)
    selected_phase = phase[s1].astype("U")
    active_rows = selected.any(axis=1)
    expected_pattern = np.asarray([True, False, False, False], dtype=np.bool_)
    return {
        "close_gate_active_requests_exact_8": int(selected.sum()) == 8,
        "close_gate_distinct_active_samples_exact_8": int(active_rows.sum()) == 8,
        "close_gate_active_samples_all_grasp": bool(
            active_rows.any() and np.all(selected_phase[active_rows] == "grasp")
        ),
        "close_gate_each_active_pattern_1000": bool(
            active_rows.any()
            and np.all(selected[active_rows] == expected_pattern[None])
        ),
        "close_gate_other_phases_zero": bool(
            not np.any(selected[selected_phase != "grasp"])
        ),
        "close_gate_no_repeated_request_per_sample": bool(
            np.all(selected.sum(axis=1) <= 1)
        ),
    }


def main() -> int:
    protocol.LINEAGE = "v442"
    protocol.STATIC_FORMAT = "strict-track2-v442-close-s0-static-contract-v1"
    protocol.REPORT_FORMAT = "strict-track2-v442-close-s1-offline-gate-v1"
    protocol.STRUCTURAL_GATE_CONTRACT = (
        "action_only_preregistered_close_exact8_grasp_pattern1000_no_repeat"
    )
    protocol.intervention_structure_checks = close_only_structure_checks
    return protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
