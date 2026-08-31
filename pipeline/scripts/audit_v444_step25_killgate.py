#!/usr/bin/env python3
"""S0 and fixed train-only kill3 audit for v444 step25."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wam_pipeline.v442_v169_close_aligned_projection_runtime import gate_decision
from wam_pipeline.v444_v169_direct_residual_runtime import DirectResidualHead, PROTECTED_FRAMES, apply_direct_residual


FORMAT = "strict-track2-v444-step25-s0-killgate-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "training-report", "preregistration", "runtime", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = json.loads(args.preregistration.read_text())
    report = json.loads(args.training_report.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tree = ast.parse(args.runtime.read_text(), filename=str(args.runtime))
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imports += [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    model = DirectResidualHead(channels=int(checkpoint.get("channels", 0)))
    model.load_state_dict(checkpoint["model"]); model.eval()
    baseline_t = torch.full((2, 8, 3, 16, 16), 0.5)
    context_t = torch.full((2, 3, 16, 16), 0.5)
    actions_t = torch.zeros((2, 12, 14)); tokens_t = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    with torch.inference_mode():
        residual_t = model(baseline_t, context_t, actions_t, tokens_t)
    residual = residual_t[0].permute(0, 2, 3, 1).numpy()
    baseline = np.full((8, 16, 16, 3), 128, dtype=np.uint8)
    enabled = apply_direct_residual(baseline, residual)
    disabled = apply_direct_residual(baseline, None)
    history = np.zeros((4, 14), dtype=np.float32); history[:, 13] = 1.0
    future = np.zeros((8, 14), dtype=np.float32); future[:4, 13] = 1.0; future[4:, 13] = 0.25
    close = gate_decision(history, future, "Use the right arm to adjust the bottle.")
    held_history = history.copy(); held_history[:, 13] = 0.25
    repeated = gate_decision(held_history, future, "Use the right arm to adjust the bottle.")
    left = gate_decision(history, future, "Use the left arm to adjust the bottle.")
    ratios = report.get("ratios", {})
    params = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "preregistration_format": prereg.get("format") == "strict-track2-v444-direct-residual-preregistration-v1",
        "training_report_format": report.get("format") == "strict-track2-v444-step25-kill-report-v1",
        "checkpoint_format_step25": checkpoint.get("format") == "strict-track2-v444-direct-residual-checkpoint-v1" and checkpoint.get("step") == 25,
        "checkpoint_report_hash_binding": report.get("sha256", {}).get("checkpoint") == sha256(args.checkpoint),
        "checkpoint_prereg_hash_binding": checkpoint.get("preregistration_sha256") == sha256(args.preregistration),
        "fit12_kill3": len(checkpoint.get("fit_episodes", [])) == 12 and len(checkpoint.get("kill_episodes", [])) == 3 and not bool(set(checkpoint.get("fit_episodes", [])) & set(checkpoint.get("kill_episodes", []))),
        "fit96_kill24": report.get("fit_windows") == 96 and report.get("kill_windows") == 24,
        "residual_mae_ratio": float(ratios.get("residual_mae", 2.0)) <= 0.99,
        "rgb_mae_ratio": float(ratios.get("rgb_mae", 2.0)) <= 0.99,
        "kill3_episode_improvement": report.get("kill_episode_rgb_improved") == 3,
        "recursive32_ratio": float(ratios.get("overall_rgb_mae", 2.0)) <= 1.0,
        "recursive32_chunk3_ratio": float(ratios.get("chunk3_rgb_mae", 2.0)) <= 1.0,
        "recursive32_chunk4_ratio": float(ratios.get("chunk4_rgb_mae", 2.0)) <= 1.0,
        "trainer_declared_pass": report.get("passed") is True,
        "small_model": 0 < params < 1_000_000,
        "finite_residual": bool(torch.isfinite(residual_t).all()),
        "protected_residual_exact_zero": bool(torch.all(residual_t[:, list(PROTECTED_FRAMES)] == 0.0)),
        "protected_output_bitexact": np.array_equal(enabled[list(PROTECTED_FRAMES)], baseline[list(PROTECTED_FRAMES)]),
        "g0_bitexact": np.array_equal(disabled, baseline),
        "residual_bound_8": int(np.abs(enabled.astype(np.int16) - baseline.astype(np.int16)).max(initial=0)) <= 8,
        "close_gate": close.get("gate") is True and close.get("first_close_index") == 4,
        "repeat_gate_disabled": repeated.get("gate") is False,
        "left_gate_disabled": left.get("gate") is False,
        "runtime_no_reward_import": not any("reward" in value.lower() for value in imports),
        "runtime_no_compute_reward": "compute_reward" not in calls,
        "no_dev_or_reward": report.get("guards", {}).get("development_or_final_used") is False and report.get("guards", {}).get("reward_or_outcome_used") is False,
    }
    passed = all(checks.values())
    audit = {
        "format": FORMAT, "passed": passed, "checks": checks,
        "metrics": {"parameter_count": params, "ratios": ratios, "kill_episode_rgb_improved": report.get("kill_episode_rgb_improved")},
        "sha256": {
            "checkpoint": sha256(args.checkpoint), "training_report": sha256(args.training_report),
            "preregistration": sha256(args.preregistration), "runtime": sha256(args.runtime),
        },
        "guards": {
            "service_started": False, "policy_updates": 0, "real_submission": False,
            "rl_authorized": False, "s1_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
