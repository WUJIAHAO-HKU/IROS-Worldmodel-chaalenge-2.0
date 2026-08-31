#!/usr/bin/env python3
"""Freeze a formal V15.6 release and prove behavior matches its evaluated shell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v15_gated_runtime import Track2V15GatedRuntime
from wam_pipeline.v15_runtime import Track2V15Runtime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample(path: Path):
    with np.load(path, allow_pickle=False) as values:
        return tuple(
            values[name].copy()
            for name in ("context_frames", "history_actions", "future_actions")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-release", required=True)
    parser.add_argument("--evaluated-release", required=True)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--onpolicy-sample", required=True)
    parser.add_argument("--official-sample", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--parent-screen", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    formal = Path(args.formal_release).resolve()
    evaluated = Path(args.evaluated_release).resolve()
    base = Path(args.base_release).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    release_lines = [
        f"{sha256(path)}  {path.relative_to(formal)}"
        for path in sorted(formal.rglob("*"))
        if path.is_file()
    ]
    release_hashes = output / "release_tree_sha256.txt"
    release_hashes.write_text("\n".join(release_lines) + "\n")
    runtime_root = Path(args.runtime_root).resolve()
    runtime_files = [
        runtime_root / "pipeline/wam_pipeline/gated_autoregressive_runtime.py",
        runtime_root / "pipeline/wam_pipeline/v15_gated_runtime.py",
        runtime_root / "pipeline/wam_pipeline/v15_runtime.py",
        runtime_root / "pipeline/wam_pipeline/backends.py",
        runtime_root / "pipeline/wam_pipeline/service.py",
    ]
    runtime_hashes = output / "runtime_code_sha256.txt"
    runtime_hashes.write_text(
        "\n".join(f"{sha256(path)}  {path}" for path in runtime_files) + "\n"
    )

    formal_runtime = Track2V15GatedRuntime(formal, args.library, args.device)
    evaluated_runtime = Track2V15GatedRuntime(evaluated, args.library, args.device)
    base_runtime = Track2V15Runtime(base, args.library, args.device)
    onpolicy = sample(Path(args.onpolicy_sample))
    official = sample(Path(args.official_sample))
    evaluated_onpolicy = evaluated_runtime.predict(*onpolicy, 0, None)
    formal_onpolicy = formal_runtime.predict(*onpolicy, 0, None)
    formal_onpolicy_route = formal_runtime.gated_autoregressive.last_route
    formal_onpolicy_probability = formal_runtime.gated_autoregressive.last_probability_synthetic
    formal_official = formal_runtime.predict(*official, 0, None)
    formal_official_route = formal_runtime.gated_autoregressive.last_route
    base_official = base_runtime.predict(*official, 0, None)
    replay = {
        "format": "strict-track2-v156-freeze-replay-v1",
        "onpolicy_eval_formal_bit_exact": bool(np.array_equal(evaluated_onpolicy, formal_onpolicy)),
        "official_formal_base_bit_exact": bool(np.array_equal(formal_official, base_official)),
        "formal_onpolicy_route": formal_onpolicy_route,
        "formal_onpolicy_probability": formal_onpolicy_probability,
        "formal_official_route": formal_official_route,
    }
    if not replay["onpolicy_eval_formal_bit_exact"] or not replay["official_formal_base_bit_exact"]:
        raise RuntimeError(f"freeze replay failed: {replay}")
    replay_path = output / "bitexact_replay_report.json"
    replay_path.write_text(json.dumps(replay, indent=2) + "\n")
    screen = Path(args.parent_screen).resolve()
    manifest = json.loads((formal / "onpolicy_adaptation/adaptation_manifest.json").read_text())
    freeze = {
        "format": "strict-track2-frozen-parent-v2",
        "status": "frozen-before-official-rl",
        "model_version": manifest["model_version"],
        "release": str(formal),
        "candidate_blend": manifest["candidate_blend"],
        "participant_action_selection": False,
        "mpc": False,
        "release_tree_sha256_file": str(release_hashes),
        "release_tree_sha256_file_sha256": sha256(release_hashes),
        "runtime_code_sha256_file": str(runtime_hashes),
        "runtime_code_sha256_file_sha256": sha256(runtime_hashes),
        "bitexact_replay_report": str(replay_path),
        "bitexact_replay_report_sha256": sha256(replay_path),
        "parent_screen": str(screen),
        "parent_screen_sha256": sha256(screen),
    }
    (output / "freeze_record.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
