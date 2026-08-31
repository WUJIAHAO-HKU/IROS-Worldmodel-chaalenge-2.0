#!/usr/bin/env python3
"""Sole launcher for the v531 actual-OOF execution boundary."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

SEED = 1665


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, digest: str) -> None:
    if path.is_symlink() or not path.is_file() or sha(path) != digest:
        raise RuntimeError(f"source drift:{path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--authority-receipt", type=Path, required=True); parser.add_argument("--authority-receipt-sha", required=True)
    parser.add_argument("--executor-source", type=Path, required=True); parser.add_argument("--executor-sha", required=True)
    parser.add_argument("--auditor-source", type=Path, required=True); parser.add_argument("--auditor-sha", required=True)
    parser.add_argument("--launcher-sha", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def run(argv=None):
    literal = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(literal)
    if type(args.seed) is not int or args.seed != SEED:
        raise RuntimeError("seed strict int")
    launcher = Path(__file__).resolve()
    exact(launcher, args.launcher_sha); exact(args.executor_source, args.executor_sha); exact(args.auditor_source, args.auditor_sha)
    exact(args.preregistration, args.preregistration_sha); exact(args.manifest, args.manifest_sha)
    exact(args.contract, args.contract_sha); exact(args.authority_receipt, args.authority_receipt_sha)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("seed") != SEED or prereg.get("active_source_paths", {}).get("actual_oof_launcher") != str(launcher) or prereg["active_source_paths"].get("actual_oof_executor") != str(args.executor_source) or prereg["active_source_paths"].get("actual_oof_auditor") != str(args.auditor_source):
        raise RuntimeError("preregistered active paths")
    if args.output_root != Path(prereg["fresh_oof_root"]):
        raise RuntimeError("fresh output root")
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        spec = importlib.util.spec_from_file_location("v531_actual_oof_executor", args.executor_source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        inner = [
            "--preregistration", str(args.preregistration), "--preregistration-sha", args.preregistration_sha,
            "--manifest", str(args.manifest), "--manifest-sha", args.manifest_sha,
            "--contract", str(args.contract), "--contract-sha", args.contract_sha,
            "--authority-receipt", str(args.authority_receipt), "--authority-receipt-sha", args.authority_receipt_sha,
            "--executor-sha", args.executor_sha, "--auditor-source", str(args.auditor_source), "--auditor-sha", args.auditor_sha,
            "--launcher-source", str(launcher), "--launcher-sha", args.launcher_sha,
            "--launcher-argv-json", json.dumps(literal, separators=(",", ":")),
            "--launcher-pid", str(os.getpid()), "--launcher-ppid", str(os.getppid()),
            "--launcher-pgid", str(os.getpgid(0)), "--launcher-sid", str(os.getsid(0)),
            "--seed", str(args.seed), "--output-root", str(args.output_root),
        ]
        receipt = module.main(inner)
        if receipt.get("passed") is not True or receipt.get("actual_oof_execution_boundary_invocations") != 1 or receipt.get("launcher_invocations") != 1 or receipt.get("executor_invocations") != 1 or receipt.get("auditor_import_invocations") != 1:
            raise RuntimeError("actual OOF boundary result")
        return receipt
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def synthetic_self_test() -> bool:
    parser = build_parser()
    names = {action.dest for action in parser._actions}
    return names == {"help", "preregistration", "preregistration_sha", "manifest", "manifest_sha", "contract", "contract_sha", "authority_receipt", "authority_receipt_sha", "executor_source", "executor_sha", "auditor_source", "auditor_sha", "launcher_sha", "seed", "output_root"}


if __name__ == "__main__":
    if sys.argv[1:] == ["--synthetic-self-test"]:
        print(json.dumps({"passed": synthetic_self_test()}, sort_keys=True)); raise SystemExit(0)
    try:
        result = run(); print(json.dumps({"passed": True, "status": result["status"], "events": result["events"], "folds": result["folds"]}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"passed": False, "status": "failed_no_retry", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr); raise SystemExit(79)
