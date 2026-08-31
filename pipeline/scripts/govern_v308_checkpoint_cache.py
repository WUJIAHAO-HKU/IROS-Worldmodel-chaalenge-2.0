#!/usr/bin/env python3
"""Evict clean page cache for stable v308 checkpoints without changing files."""

from __future__ import annotations

import datetime as dt
import os
import time
import zipfile
from pathlib import Path


CHECKPOINT_ROOT = Path(
    "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/"
    "strict_track2_official_20260810/runs/"
    "v308_v301_rtx5090_fresh_official_h200_r4_step10_lr2e5_beta001_seed1471_20260821/"
    "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints"
)
SCAN_SECONDS = 15
MIN_STABLE_AGE_SECONDS = 30


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    previous: dict[Path, tuple[int, int]] = {}
    evicted: set[tuple[Path, int, int]] = set()
    print(f"{utc_now()} checkpoint_cache_governor_started root={CHECKPOINT_ROOT}", flush=True)
    while True:
        current: dict[Path, tuple[int, int]] = {}
        now_ns = time.time_ns()
        for path in sorted(CHECKPOINT_ROOT.glob("global_step_*/actor/**/*.pt")):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            current[path] = stamp
            stable_age = (now_ns - stat.st_mtime_ns) / 1_000_000_000
            key = (path, *stamp)
            if (
                previous.get(path) != stamp
                or stable_age < MIN_STABLE_AGE_SECONDS
                or key in evicted
                or not zipfile.is_zipfile(path)
            ):
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            evicted.add(key)
            print(
                f"{utc_now()} evicted_clean_checkpoint_cache "
                f"size={stat.st_size} path={path}",
                flush=True,
            )
        previous = current
        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
