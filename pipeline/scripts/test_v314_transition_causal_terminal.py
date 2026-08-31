#!/usr/bin/env python3
"""Run the exact v312 contract against v314."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v312_causal_terminal_mirror as base
from wam_pipeline.v314_transition_causal_terminal_runtime import (
    Track2V314TransitionCausalTerminal,
)


def main() -> int:
    base.Track2V312CausalTerminalMirror = Track2V314TransitionCausalTerminal
    code = base.main()
    path = Path(sys.argv[sys.argv.index("--output") + 1])
    report = json.loads(path.read_text())
    report["format"] = "strict-track2-v314-transition-causal-terminal-contract-v1"
    report["candidate"] = "track2-v314-transition-causal-terminal-v271"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
