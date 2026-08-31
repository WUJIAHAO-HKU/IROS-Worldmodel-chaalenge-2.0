#!/usr/bin/env python3
"""Static S0 audit for v445 full-coordinate mirror semantics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from wam_pipeline.v445_v169_full_mirror_runtime import (
    MIRROR_SIGN, explicit_right, mirror_joint14, mirror_prompt, mirror_rgb,
)


FORMAT = "strict-track2-v445-full-mirror-s0-static-contract-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path(values: np.ndarray, begin: int) -> float:
    return float(np.linalg.norm(np.diff(values[:, begin:begin + 7], axis=0), axis=1).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("runtime", "preregistration", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    source = args.runtime.read_text()
    tree = ast.parse(source, filename=str(args.runtime))
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    golden = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.float32)
    golden_expected = np.asarray([-8, 9, 10, 11, -12, -13, 14, -1, 2, 3, 4, -5, -6, 7], dtype=np.float32)
    rng = np.random.default_rng(445)
    actions = rng.normal(size=(3, 12, 14)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(3, 5, 17, 23, 3), dtype=np.uint8)
    causal = np.zeros((12, 14), dtype=np.float32)
    causal[:, 7] = np.arange(12, dtype=np.float32)
    mirrored_causal = mirror_joint14(causal)
    prompt = "Use the right arm to adjust the bottle."
    transformed_prompt = mirror_prompt(prompt)
    checks = {
        "preregistration_format": prereg.get("format") == "strict-track2-v445-full-mirror-preregistration-v1",
        "mirror_sign_exact": np.array_equal(MIRROR_SIGN, np.asarray([-1, 1, 1, 1, -1, -1, 1], dtype=np.float32)),
        "joint_golden": np.array_equal(mirror_joint14(golden), golden_expected),
        "joint_involution_bitexact": np.array_equal(mirror_joint14(mirror_joint14(actions)), actions),
        "rgb_involution_bitexact": np.array_equal(mirror_rgb(mirror_rgb(rgb)), rgb),
        "rgb_golden": np.array_equal(mirror_rgb(rgb)[..., 0, :], rgb[..., -1, :]),
        "prompt_golden": transformed_prompt == "Use the left arm to adjust the bottle.",
        "prompt_gate": explicit_right(prompt) and not explicit_right(transformed_prompt) and not explicit_right("Use the left arm."),
        "action_path_causal": path(causal, 7) > path(causal, 0) and path(mirrored_causal, 0) > path(mirrored_causal, 7),
        "runtime_no_reward_import": not any("reward" in value.lower() for value in imports),
        "runtime_no_compute_reward": "compute_reward" not in calls,
        "no_seed_selection": "stable_seed" not in source and "seed %" not in source and "hashlib.sha256(str(seed" not in source,
        "seed_forwarded": "np.asarray(seeds)[enabled]" in source,
        "no_action_generation": "sample_action" not in source and "generate_action" not in source and "policy.predict" not in source,
        "fallback_copy": "output = baseline.copy()" in source,
    }
    passed = all(checks.values())
    report = {
        "format": FORMAT, "passed": passed, "checks": checks,
        "golden": {"joint_input": golden.tolist(), "joint_output": mirror_joint14(golden).tolist(), "prompt_output": transformed_prompt},
        "causal_paths": {
            "original_left": path(causal, 0), "original_right": path(causal, 7),
            "mirrored_left": path(mirrored_causal, 0), "mirrored_right": path(mirrored_causal, 7),
        },
        "sha256": {"runtime": sha256(args.runtime), "preregistration": sha256(args.preregistration)},
        "guards": {"service_started": False, "development_runs": 0, "policy_updates": 0, "rl_authorized": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
