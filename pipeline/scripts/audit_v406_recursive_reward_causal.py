#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import audit_v385_recursive_reward_causal as base
from wam_pipeline.v406_progressive_then_terminal_runtime import (
    Track2V406ProgressiveThenTerminal,
)


def main() -> int:
    argv = list(sys.argv)
    prereg_index = argv.index("--preregistration") + 1
    output_index = argv.index("--output") + 1
    source = Path(argv[prereg_index])
    output = Path(argv[output_index])
    payload = json.loads(source.read_text())
    if payload.get("format") != "strict-track2-v406-progressive-then-terminal-preregistration-v1":
        raise RuntimeError("wrong v406 preregistration")
    candidate = payload["model_version"]
    payload["format"] = "strict-track2-v385-native-batch-clean-reanchor-preregistration-v1"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
        temporary = stream.name
    argv[prereg_index] = temporary
    sys.argv = argv
    base.Track2V385NativeBatchCleanReanchor = Track2V406ProgressiveThenTerminal
    try:
        code = base.main()
    finally:
        Path(temporary).unlink(missing_ok=True)
    report = json.loads(output.read_text())
    report["format"] = "strict-track2-v406-recursive-reward-causal-gate-v1"
    report["candidate"] = candidate
    output.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
