#!/usr/bin/env python3
"""Make the RoboTwin planner safe across legacy/new Curobo installations.

RoboTwin's dual-arm reset path performs an ``isinstance`` check against the
legacy ``CuroboPlanner`` class even when the selected backend is ``mplib``.
On machines with no legacy Curobo (or with the newer API), that class is
``None`` and the check raises before mplib can be used.  This small, idempotent
compatibility patch only guards that check; it does not change action,
reward, seed, or scoring code.

The original file is backed up once and a manifest records its SHA256 so the
runtime remains auditable and reversible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


OLD = "if not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):"
NEW = "if CuroboPlanner is not None and (not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner)):"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patch(root: Path) -> dict[str, object]:
    robot = root / "envs" / "robot" / "robot.py"
    if not robot.is_file():
        raise FileNotFoundError(robot)
    backup = robot.with_suffix(robot.suffix + ".compat.bak")
    if not backup.exists():
        backup.write_bytes(robot.read_bytes())
    baseline_sha = sha256(backup)

    source = robot.read_text()
    if OLD in source:
        source = source.replace(OLD, NEW)
        robot.write_text(source)
    elif NEW not in source:
        raise RuntimeError("RoboTwin reset guard changed; refusing an unverified patch")

    result = {
        "robotwin_root": str(root),
        "original_sha256": baseline_sha,
        "patched_sha256": sha256(robot),
        "backup": str(backup),
        "guard": "mplib-safe-when-curobo-is-missing-or-incompatible",
    }
    (root / ".curobo_compat_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("robotwin_root", type=Path)
    args = ap.parse_args()
    print(json.dumps(patch(args.robotwin_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
