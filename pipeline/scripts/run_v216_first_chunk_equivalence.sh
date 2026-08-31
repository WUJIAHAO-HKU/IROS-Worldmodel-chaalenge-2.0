#!/usr/bin/env bash
set -euo pipefail

BASE='/root/autodl-tmp/IROS_WAM_2.0 challenge'
cd "$BASE"

restart_parent() {
  bash pipeline/scripts/restart_v209_services.sh start \
    >/tmp/restart_v209_after_v216_equivalence.log 2>&1 || true
}
trap restart_parent EXIT

screen -S wm_v209_bridge -X quit >/dev/null 2>&1 || true
screen -S wm_v209_gpu -X quit >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if ! ss -ltn | grep -qE ':(8004|18083) '; then
    break
  fi
  sleep 1
done
if ss -ltn | grep -qE ':(8004|18083) '; then
  echo 'v209 ports did not stop cleanly' >&2
  exit 3
fi

PYTHONPATH=pipeline /root/miniconda3/envs/go1/bin/python \
  pipeline/scripts/test_v216_first_chunk_equivalence.py
