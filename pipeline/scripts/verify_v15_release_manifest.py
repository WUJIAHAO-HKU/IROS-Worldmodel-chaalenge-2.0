#!/usr/bin/env python3
"""Verify every frozen V15 file declared by its release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.release_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["sha256"]
    rows = []
    for relative, expected_hash in sorted(expected.items()):
        path = args.release_root / relative
        actual_hash = digest(path) if path.is_file() else None
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "matches": actual_hash == expected_hash,
            }
        )
    result = {
        "format": "track2-v15-frozen-manifest-verification-v1",
        "release_root": str(args.release_root),
        "checked_file_count": len(rows),
        "passed": bool(rows) and all(row["matches"] for row in rows),
        "files": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
