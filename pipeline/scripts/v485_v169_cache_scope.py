#!/usr/bin/env python3
"""Exact per-call deterministic-warning scope for v485 cache qualification only."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings

import numpy as np
import torch


ALLOWED_WARNING_CATEGORY = "UserWarning"
ALLOWED_WARNING_MESSAGE = (
    "median CUDA with indices output does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at "
    "https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support "
    "for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:93.)"
)


def raw_sha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def rng_state_sha256() -> dict:
    cpu = torch.random.get_rng_state().detach().cpu().contiguous().numpy()
    cuda = []
    if torch.cuda.is_available():
        for device, state in enumerate(torch.cuda.get_rng_state_all()):
            cuda.append({"device": device, "sha256": raw_sha(state.detach().cpu().contiguous().numpy())})
    return {"cpu_sha256": raw_sha(cpu), "cuda": cuda}


def strict_state() -> dict:
    return {
        "enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    }


def predict_one(runtime, context, history, future, seed, instruction) -> tuple[np.ndarray, dict]:
    expected_strict = {"enabled": True, "warn_only": False}
    if strict_state() != expected_strict:
        raise RuntimeError("v485 strict deterministic precondition")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    rng_before = rng_state_sha256()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        if strict_state() != {"enabled": True, "warn_only": True}:
            raise RuntimeError("v485 warn-only scope entry")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = np.asarray(runtime.predict(context, history, future, int(seed), str(instruction)))
            if torch.cuda.is_available(): torch.cuda.synchronize()
    finally:
        torch.use_deterministic_algorithms(True, warn_only=False)
    if strict_state() != expected_strict:
        raise RuntimeError("v485 strict deterministic restoration")
    rng_after = rng_state_sha256()
    warning_evidence = [
        {"category": item.category.__name__, "message": str(item.message)} for item in caught
    ]
    if warning_evidence != [{"category": ALLOWED_WARNING_CATEGORY, "message": ALLOWED_WARNING_MESSAGE}]:
        raise RuntimeError(f"v485 exact deterministic warning mismatch: {warning_evidence!r}")
    if rng_after != rng_before:
        raise RuntimeError("v485 v169 predict changed CPU/CUDA RNG state")
    if output.shape != (8, 256, 256, 3) or output.dtype != np.uint8 or not np.isfinite(output).all():
        raise RuntimeError("v485 v169 output schema")
    return np.ascontiguousarray(output), {
        "mode_before": expected_strict,
        "mode_during": {"enabled": True, "warn_only": True},
        "mode_after": expected_strict,
        "warning": warning_evidence[0],
        "rng_before": rng_before,
        "rng_after": rng_after,
        "rng_unchanged": True,
        "output_sha256": raw_sha(output),
    }


class _SyntheticRuntime:
    def predict(self, context, history, future, seed, instruction):
        warnings.warn(ALLOWED_WARNING_MESSAGE, UserWarning)
        value = np.empty((8, 256, 256, 3), np.uint8)
        value.fill(int(seed) % 251)
        return value


def self_test() -> dict:
    torch.use_deterministic_algorithms(True, warn_only=False)
    context = np.zeros((5, 256, 256, 3), np.uint8)
    history = np.zeros((4, 14), np.float32)
    future = np.zeros((8, 14), np.float32)
    value, evidence = predict_one(_SyntheticRuntime(), context, history, future, 1627, "synthetic")
    passed = value.shape == (8, 256, 256, 3) and evidence["rng_unchanged"] and strict_state() == {"enabled": True, "warn_only": False}
    return {"passed": bool(passed), "evidence": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test: raise SystemExit(78)
    result = self_test(); print(json.dumps(result, sort_keys=True)); raise SystemExit(0 if result["passed"] else 3)
