#!/usr/bin/env python3
"""Run the complete v334 contract against the frozen v335 release alias."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v334_bounded_onset_terminal as inherited
from wam_pipeline.v335_all_offset_bounded_onset_runtime import Track2V335AllOffsetBoundedOnset


def main() -> int:
    inherited.Track2V334BoundedOnsetTerminal = Track2V335AllOffsetBoundedOnset
    inherited.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v335-all-offset-bounded-onset-contract-v1"
    report["candidate"] = "track2-v335-all-offset-bounded-onset-v334-v333-v326-v325-v317"
    report["inherited_contract"] = "strict-track2-v334-bounded-onset-terminal-contract-v1"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"v335_passed": report["passed"], "checks": report["checks"]}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
