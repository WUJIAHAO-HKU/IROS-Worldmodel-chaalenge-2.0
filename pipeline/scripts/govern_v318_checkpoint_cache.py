#!/usr/bin/env python3
"""Evict only clean, stable v318 checkpoint page cache."""

from __future__ import annotations

import datetime as dt
import os
import time
import zipfile
from pathlib import Path


CHECKPOINT_ROOT = Path(
    "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_official_20260810/runs/"
    "v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822/"
    "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints"
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    previous = {}
    evicted = set()
    print(f"{now()} governor_started root={CHECKPOINT_ROOT}", flush=True)
    while True:
        current = {}
        now_ns = time.time_ns()
        for path in sorted(CHECKPOINT_ROOT.glob("global_step_*/actor/**/*.pt")):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            stamp = (stat.st_size, stat.st_mtime_ns)
            current[path] = stamp
            key = (path, *stamp)
            if previous.get(path) != stamp or (now_ns - stat.st_mtime_ns) < 30_000_000_000 or key in evicted or not zipfile.is_zipfile(path):
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            evicted.add(key)
            print(f"{now()} evicted_clean_cache size={stat.st_size} path={path}", flush=True)
        previous = current
        time.sleep(15)


if __name__ == "__main__":
    main()
