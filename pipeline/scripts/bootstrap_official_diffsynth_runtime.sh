#!/usr/bin/env bash
# Create an isolated full checkout for the exact DiffSynth revision used by RLinf WanEnv.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
target_dir="${1:-$root_dir/artifacts/upstream/diffsynth-studio-runtime-local}"
repo_url="${WAM_OFFICIAL_DIFFSYNTH_REPO:-git@github.com:RLinf/diffsynth-studio.git}"
revision="2a2e05fa1f724828b243f272540989b19a6e54f8"

if [[ -e "$target_dir" ]]; then
  [[ -d "$target_dir/.git" ]] || { echo "target exists but is not a Git checkout: $target_dir" >&2; exit 2; }
  current_revision="$(git -C "$target_dir" rev-parse HEAD)"
  [[ "$current_revision" == "$revision" ]] || {
    echo "target has unexpected DiffSynth revision: $current_revision" >&2
    exit 2
  }
  git -C "$target_dir" diff --quiet || {
    echo "target checkout is dirty: $target_dir" >&2
    exit 2
  }
  echo "pinned DiffSynth runtime already present: $target_dir"
  exit 0
fi

mkdir -p "$(dirname "$target_dir")"
git clone "$repo_url" "$target_dir"
git -C "$target_dir" checkout --detach "$revision"
git -C "$target_dir" fsck --full --no-dangling
git -C "$target_dir" diff --quiet
echo "pinned full DiffSynth runtime created: $target_dir"
