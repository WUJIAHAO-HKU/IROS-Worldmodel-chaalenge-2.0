#!/usr/bin/env python3
"""Run the frozen v312 contract against the exact-batch v313 runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import test_v312_causal_terminal_mirror as base
from wam_pipeline.v313_serial_causal_terminal_mirror_runtime import (
    Track2V313SerialCausalTerminalMirror,
)


def output_path() -> Path:
    return Path(sys.argv[sys.argv.index("--output") + 1])


def main() -> int:
    base.Track2V312CausalTerminalMirror = Track2V313SerialCausalTerminalMirror
    code = base.main()
    path = output_path()
    report = json.loads(path.read_text())
    report["format"] = "strict-track2-v313-serial-causal-terminal-mirror-contract-v1"
    report["candidate"] = "track2-v313-serial-public-action-gated-full-terminal-mirror-v271"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
