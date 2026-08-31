#!/usr/bin/env python3
"""Reliably fetch the large action-only Wan weights from a lossy mirror.

The regular Hub downloader is ideal under normal network conditions, but a
long-lived HTTP stream from this host is frequently interrupted.  This helper
uses short, independently retryable range requests.  Existing Hub
``.incomplete`` files are treated as an immutable contiguous prefix; the
completed file is SHA-256 verified and atomically moved into the model tree.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import requests


REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
VERIFY_MANIFEST = "track2_wan_weight_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_url(endpoint: str, relative_path: str) -> str:
    return f"{endpoint.rstrip('/')}/{REPO_ID}/resolve/{REVISION}/{relative_path}"


def fetch_range(
    *,
    url: str,
    start: int,
    end: int,
    destination: Path,
    retries: int,
) -> None:
    """Fill one byte range, retrying only the missing suffix after a reset."""
    expected_size = end - start + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    while destination.exists() and destination.stat().st_size > expected_size:
        raise RuntimeError(f"range cache is too large: {destination}")
    failures = 0
    while (destination.stat().st_size if destination.exists() else 0) < expected_size:
        existing = destination.stat().st_size if destination.exists() else 0
        next_start = start + existing
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={next_start}-{end}"},
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(f"range request returned HTTP {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {next_start}-"):
                raise RuntimeError(f"unexpected Content-Range {content_range!r}")
            with destination.open("ab") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
            failures = 0
        except (requests.RequestException, RuntimeError) as exc:
            failures += 1
            if failures > retries:
                raise RuntimeError(f"range {start}-{end} failed after {retries} retries") from exc
            time.sleep(min(30, 2**failures))
    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"wrong range size for {destination}")


def existing_hub_prefix(base: Path, relative_path: str, checksum: str, expected_size: int) -> Path | None:
    """Return the Hub resumable prefix if it belongs to this exact LFS object."""
    download_dir = base / ".cache" / "huggingface" / "download" / Path(relative_path).parent
    matches = list(download_dir.glob(f"*.{checksum}.incomplete"))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous Hub prefixes for {relative_path}")
    if not matches:
        return None
    prefix = matches[0]
    if prefix.stat().st_size > expected_size:
        raise RuntimeError(f"Hub prefix exceeds expected size: {prefix}")
    return prefix


def get_weight_specs(endpoint: str) -> list[dict]:
    response = requests.get(
        f"{endpoint.rstrip('/')}/api/models/{REPO_ID}/tree/{REVISION}?recursive=true&expand=true",
        timeout=(30, 60),
    )
    response.raise_for_status()
    values = response.json()
    wanted = []
    for value in values:
        path = value.get("path", "")
        lfs = value.get("lfs") or {}
        if path.startswith(("transformer/", "vae/")) and path.endswith(".safetensors"):
            wanted.append({"path": path, "size": int(lfs["size"]), "sha256": str(lfs["oid"])})
    # VAE comes first so latent caching can use an otherwise idle GPU while
    # the remaining DiT shards continue transferring.
    wanted.sort(key=lambda value: (not value["path"].startswith("vae/"), value["path"]))
    if len(wanted) != 6:
        raise RuntimeError(f"expected five transformer shards and one VAE, found {len(wanted)}")
    return wanted


def manifest_path(base: Path) -> Path:
    return base / VERIFY_MANIFEST


def write_verify_manifest(base: Path, specs: list[dict]) -> None:
    """Record the immutable hashes only after every requested object verifies."""
    manifest = {
        "format": "track2-wan-weight-manifest-v1",
        "repository": REPO_ID,
        "revision": REVISION,
        "weights": specs,
    }
    target = manifest_path(base)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, target)


def verify_local_manifest(base: Path) -> None:
    """Reject an incomplete or modified base before a multi-hour GPU run."""
    path = manifest_path(base)
    if not path.is_file():
        raise RuntimeError(f"missing verified Wan weight manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Wan weight manifest: {path}") from exc
    if (
        manifest.get("format") != "track2-wan-weight-manifest-v1"
        or manifest.get("repository") != REPO_ID
        or manifest.get("revision") != REVISION
    ):
        raise RuntimeError("Wan weight manifest identifies a different upstream revision")
    weights = manifest.get("weights")
    if not isinstance(weights, list) or len(weights) != 6:
        raise RuntimeError("Wan weight manifest must list five DiT shards and one VAE")
    expected_paths = {
        "vae/diffusion_pytorch_model.safetensors",
        *(f"transformer/diffusion_pytorch_model-0000{index}-of-00005.safetensors" for index in range(1, 6)),
    }
    if {item.get("path") for item in weights if isinstance(item, dict)} != expected_paths:
        raise RuntimeError("Wan weight manifest has an unexpected file set")
    for item in weights:
        if not isinstance(item, dict):
            raise RuntimeError("Wan weight manifest entry is malformed")
        relative_path = str(item.get("path"))
        target = base / relative_path
        expected_size = int(item.get("size", -1))
        expected_sha256 = str(item.get("sha256", ""))
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError(f"Wan weight is missing or has the wrong size: {target}")
        actual_sha256 = sha256(target)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Wan weight checksum mismatch: {target}")
        print(json.dumps({"status": "verified", "path": relative_path}), flush=True)


def download_weight(
    spec: dict,
    *,
    base: Path,
    endpoint: str,
    chunk_bytes: int,
    workers: int,
    retries: int,
) -> None:
    relative_path = spec["path"]
    expected_size = int(spec["size"])
    expected_sha256 = str(spec["sha256"])
    target = base / relative_path
    if target.is_file():
        if target.stat().st_size != expected_size or sha256(target) != expected_sha256:
            raise RuntimeError(f"existing weight does not match immutable upstream checksum: {target}")
        print(json.dumps({"status": "verified", "path": relative_path}), flush=True)
        return

    prefix = existing_hub_prefix(base, relative_path, expected_sha256, expected_size)
    prefix_size = 0 if prefix is None else prefix.stat().st_size
    work = base / ".ranges" / relative_path
    work.mkdir(parents=True, exist_ok=True)
    ranges = []
    for start in range(prefix_size, expected_size, chunk_bytes):
        end = min(start + chunk_bytes, expected_size) - 1
        ranges.append((start, end, work / f"{start:012d}-{end:012d}.part"))
    print(
        json.dumps(
            {
                "status": "downloading",
                "path": relative_path,
                "prefix_bytes": prefix_size,
                "remaining_bytes": expected_size - prefix_size,
                "range_count": len(ranges),
            }
        ),
        flush=True,
    )
    url = mirror_url(endpoint, relative_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(fetch_range, url=url, start=start, end=end, destination=path, retries=retries)
            for start, end, path in ranges
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if completed == len(ranges) or completed % max(1, workers) == 0:
                print(json.dumps({"path": relative_path, "ranges_complete": completed, "range_count": len(ranges)}), flush=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".assembled")
    with temporary.open("wb") as output:
        if prefix is not None:
            with prefix.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        for _, _, part in ranges:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != expected_size or sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"assembled checksum mismatch for {relative_path}")
    os.replace(temporary, target)
    shutil.rmtree(work)
    print(json.dumps({"status": "complete", "path": relative_path, "sha256": expected_sha256}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel range downloader for Track 2 Wan action-only weights.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.chunk_mib < 1 or args.workers < 1 or args.retries < 1:
        raise SystemExit("chunk size, workers, and retries must be positive")
    base = Path(args.output)
    if args.verify_only:
        verify_local_manifest(base)
        return
    specs = get_weight_specs(args.endpoint)
    for spec in specs:
        download_weight(
            spec,
            base=base,
            endpoint=args.endpoint,
            chunk_bytes=args.chunk_mib * 1024 * 1024,
            workers=args.workers,
            retries=args.retries,
        )
    write_verify_manifest(base, specs)
    verify_local_manifest(base)


if __name__ == "__main__":
    main()
