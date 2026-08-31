#!/usr/bin/env python3
"""Check the isolated DiffSynth runtime required by official RLinf Wan."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from wam_pipeline.official_rlinf_wan import (
    OFFICIAL_RLINF_WAN_RUNTIME_REVISION,
    validate_official_rlinf_wan_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diffsynth-root",
        default="artifacts/upstream/diffsynth-studio-runtime-local",
    )
    args = parser.parse_args()
    runtime = validate_official_rlinf_wan_runtime(args.diffsynth_root)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    try:
        module = importlib.import_module("diffsynth.pipelines.wan_video_new")
    except Exception as exc:
        raise SystemExit(f"pinned DiffSynth exists but cannot be imported: {exc}") from exc
    result = {
        "status": "passed",
        "diffsynth_root": str(runtime),
        "revision": OFFICIAL_RLINF_WAN_RUNTIME_REVISION,
        "pipeline_class": module.WanVideoPipeline.__name__,
        "model_config_class": module.ModelConfig.__name__,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
