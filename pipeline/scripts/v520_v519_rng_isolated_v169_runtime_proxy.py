#!/usr/bin/env python3
"""Per-call RNG-isolation proxy for the v520 Phase-A qualification worker.

This module is a source-only component.  It never invokes a model by itself.
The frozen qualification scope calls :meth:`predict` exactly once per request;
the proxy delegates exactly once inside ``torch.random.fork_rng`` and returns
both the model output and a complete four-stage RNG evidence record.
"""
from __future__ import annotations

import hashlib
import json


FORMAT = "strict-track2-v520-v519-rng-isolation-four-stage-v1"
V519_ACTUAL_SAMPLE0 = {
    "evidence_sha256sum_lines_digest_sha256": "f386dec633821d8e85b7097eed5fcc6668c07958ee52aaab9327804286bfbd26",
    "evidence_canonical_json_digest_sha256": "e06c481df6865a157b6d34dab36fd99f867689a6eddad0309484f72885ab0538",
    "evidence_logical_file_bytes": 653273,
    "process_receipt_sha256": "06ce4ad84ad9bd3efdf35b429dbf5c0f1986c4d32ae495afcbc7a1e97060ed11",
    "process_receipt_logical_bytes": 369884,
    "child_stdout_sha256": "6585fbb99e063cb86287ca109ce7c4595dc0d07be1330f914ab53d83be06963d",
    "child_stdout_logical_bytes": 126129,
    "output_sha256": "119d47797f50b24b2050c91f99b1733b9f2c0ed20a7987f61d44ca42ef3dfce3",
    "raw_warning_count": 1,
    "cuda_device_count": 1,
    "cuda_device_indices": [0],
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record(raw: bytes) -> dict:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "logical_bytes": len(raw)}


def _python_state(random_module) -> dict:
    raw = json.dumps(random_module.getstate(), separators=(",", ":")).encode()
    return {"format": "python_random_getstate_canonical_json", **_record(raw)}


def _numpy_state(np) -> dict:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    value = {
        "bit_generator": name,
        "state_uint32": [int(item) for item in keys.tolist()],
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian_hex": float(cached_gaussian).hex(),
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"format": "numpy_random_get_state_canonical_json", **_record(raw)}


def _torch_state(torch) -> dict:
    cpu = torch.random.get_rng_state().detach().cpu().contiguous().numpy().tobytes(order="C")
    cuda = []
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            raw = state.detach().cpu().contiguous().numpy().tobytes(order="C")
            cuda.append({"device_index": index, **_record(raw)})
    return {"cpu": _record(cpu), "cuda": cuda}


def _all_state(torch, np, random_module) -> dict:
    return {
        "torch": _torch_state(torch),
        "python": _python_state(random_module),
        "numpy": _numpy_state(np),
    }


class RngIsolationRuntimeProxy:
    """A one-use delegate proxy with complete call accounting and restoration."""

    def __init__(self, delegate, exact_seed: int, torch, np, random_module):
        self.delegate = delegate
        self.exact_seed = int(exact_seed)
        self.torch = torch
        self.np = np
        self.random = random_module
        self.evidence = None
        self._used = False

    def predict(self, context, history, future, seed, instruction):
        if self._used:
            raise RuntimeError("rng proxy exactly one delegate call")
        self._used = True
        if type(seed) is not int or seed != self.exact_seed:
            raise RuntimeError("rng proxy exact seed")
        device_count = int(self.torch.cuda.device_count()) if self.torch.cuda.is_available() else 0
        device_indices = list(range(device_count))
        python_entry_state = self.random.getstate()
        numpy_entry_state = self.np.random.get_state()
        entry = _all_state(self.torch, self.np, self.random)
        inside_before = internal_after = exit_restored = None
        output = None
        delegate_error = None
        started = completed = 0
        try:
            with self.torch.random.fork_rng(devices=device_indices, enabled=True):
                self.torch.manual_seed(self.exact_seed)
                if device_count:
                    self.torch.cuda.manual_seed_all(self.exact_seed)
                inside_before = _all_state(self.torch, self.np, self.random)
                try:
                    started = 1
                    output = self.delegate.predict(context, history, future, seed, instruction)
                    completed = 1
                except BaseException as error:
                    delegate_error = error
                finally:
                    internal_after = _all_state(self.torch, self.np, self.random)
        finally:
            self.random.setstate(python_entry_state)
            self.np.random.set_state(numpy_entry_state)
            exit_restored = _all_state(self.torch, self.np, self.random)
            self.evidence = {
                "format": FORMAT,
                "exact_seed": self.exact_seed,
                "cuda_device_count": device_count,
                "cuda_device_indices": device_indices,
                "fork_rng_devices_exact_all_cuda_indices": True,
                "delegate_invocations_started": started,
                "delegate_invocations_completed": completed,
                "entry_external": entry,
                "inside_before_delegate": inside_before,
                "internal_after_delegate": internal_after,
                "exit_restored": exit_restored,
                "python_all_four_stages_equal": inside_before is not None and internal_after is not None and entry["python"] == inside_before["python"] == internal_after["python"] == exit_restored["python"],
                "numpy_all_four_stages_equal": inside_before is not None and internal_after is not None and entry["numpy"] == inside_before["numpy"] == internal_after["numpy"] == exit_restored["numpy"],
                "torch_internal_change_allowed": True,
                "python_exit_restored": exit_restored["python"] == entry["python"],
                "numpy_exit_restored": exit_restored["numpy"] == entry["numpy"],
                "torch_cpu_exit_restored": exit_restored["torch"]["cpu"] == entry["torch"]["cpu"],
                "torch_cuda_exit_restored": exit_restored["torch"]["cuda"] == entry["torch"]["cuda"],
                "exception_finally_restoration_complete": True,
            }
            self.evidence["four_stage_canonical_sha256"] = canonical_sha256({
                key: self.evidence[key]
                for key in ("entry_external", "inside_before_delegate", "internal_after_delegate", "exit_restored")
            })
        validate_evidence(self.evidence, require_completed=delegate_error is None)
        if delegate_error is not None:
            raise delegate_error
        return output


def validate_evidence(evidence: dict, *, require_completed: bool) -> None:
    required = {
        "format", "exact_seed", "cuda_device_count", "cuda_device_indices",
        "fork_rng_devices_exact_all_cuda_indices", "delegate_invocations_started",
        "delegate_invocations_completed", "entry_external", "inside_before_delegate",
        "internal_after_delegate", "exit_restored", "python_all_four_stages_equal",
        "numpy_all_four_stages_equal", "torch_internal_change_allowed",
        "python_exit_restored", "numpy_exit_restored", "torch_cpu_exit_restored",
        "torch_cuda_exit_restored", "exception_finally_restoration_complete",
        "four_stage_canonical_sha256",
    }
    if set(evidence) != required or evidence["format"] != FORMAT:
        raise RuntimeError("rng proxy evidence schema")
    if evidence["cuda_device_indices"] != list(range(evidence["cuda_device_count"])):
        raise RuntimeError("rng proxy CUDA device inventory")
    if evidence["delegate_invocations_started"] != 1:
        raise RuntimeError("rng proxy delegate start accounting")
    if require_completed and evidence["delegate_invocations_completed"] != 1:
        raise RuntimeError("rng proxy delegate completion accounting")
    if not all(evidence[key] is True for key in (
        "fork_rng_devices_exact_all_cuda_indices", "python_all_four_stages_equal",
        "numpy_all_four_stages_equal", "python_exit_restored", "numpy_exit_restored",
        "torch_cpu_exit_restored", "torch_cuda_exit_restored",
        "exception_finally_restoration_complete",
    )):
        raise RuntimeError("rng proxy restoration contract")
    expected = canonical_sha256({
        key: evidence[key]
        for key in ("entry_external", "inside_before_delegate", "internal_after_delegate", "exit_restored")
    })
    if evidence["four_stage_canonical_sha256"] != expected:
        raise RuntimeError("rng proxy four-stage digest")


if __name__ == "__main__":
    raise SystemExit("source module only")
