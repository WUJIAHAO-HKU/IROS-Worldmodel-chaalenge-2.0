#!/usr/bin/env bash
# Apply the Track 2 probability-audit hook to the official RLinf dependency.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
rlinf_root=${RLINF_ROOT:-$root_dir/third_party/WorldArena-2.0/RL_env_benchmark}
patch_file=$root_dir/pipeline/patches/rlinf_probability_consistency.patch
target=$rlinf_root/rlinf/workers/actor/fsdp_actor_worker.py

if [[ ! -f "$target" ]]; then
  echo "missing RLinf actor worker: $target" >&2
  exit 2
fi

if patch --batch --forward --dry-run -p2 -d "$rlinf_root" <"$patch_file" >/dev/null 2>&1; then
  patch --batch --forward -p2 -d "$rlinf_root" <"$patch_file"
  echo "applied probability consistency patch to $target"
elif patch --batch --reverse --dry-run -p2 -d "$rlinf_root" <"$patch_file" >/dev/null 2>&1; then
  echo "probability consistency patch is already applied to $target"
else
  echo "RLinf source does not match the audited upstream file; refusing to patch" >&2
  exit 3
fi

python -m py_compile "$target"
