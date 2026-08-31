#!/usr/bin/env bash
# Four-context public-train paired simulator technical pilot. No model/policy/RL work.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v451_paired_intervention_pilot_seed1604_20260823"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
DATASET="$ROOT/artifacts/datasets/aloha-agilex_clean_50"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
test ! -e "$REG"
test "$(readlink -f "$("$RLPY" -c 'import sys; print(sys.executable)')")" = "$(readlink -f "$RLPY")"
"$RLPY" -c 'import numpy, h5py, yaml, open3d, toppra, mplib, sapien'
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v451_paired_intervention_pilot.py" | awk '{print $1}')" = '764c3335e6e3d6e43721a049faa02a5f82e9a8bd19ec5fd93b372a50244b2e9f'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v451_paired_intervention_pilot.py" | awk '{print $1}')" = 'a9b47f9230be719d0854a6ab7ece2284fa80f875abc69e2f6b7dada8fe08f038'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v451_paired_intervention_pilot.py" | awk '{print $1}')" = '2816f0e717aa6397ec7371ae2c9e67ca4537bd2527ce5411c6a5b5e9a43ceee2'
mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$RLPY" pipeline/scripts/prepare_v451_paired_intervention_pilot.py \
 --split "$SPLIT" --dataset "$DATASET" --windows "$WINDOWS" --seed-file "$DATASET/seed.txt" \
 --collector "$ROOT/pipeline/scripts/generate_v451_paired_intervention_pilot.py" --auditor "$ROOT/pipeline/scripts/audit_v451_paired_intervention_pilot.py" \
 --gate-runtime "$ROOT/pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$SUPPORT"
PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" timeout --signal=TERM --kill-after=30s 20m nice -n 10 taskset -c 0-7 "$RLPY" "$ROOT/pipeline/scripts/generate_v451_paired_intervention_pilot.py" \
 --preregistration "$REG/preregistration.json" --support-root "$SUPPORT" --task-config "$SUPPORT/task_config/demo_clean.yml" --output-dir "$REG/dataset" >"$REG/generation.log" 2>&1
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$RLPY" pipeline/scripts/audit_v451_paired_intervention_pilot.py \
 --preregistration "$REG/preregistration.json" --dataset-dir "$REG/dataset" --generator "$ROOT/pipeline/scripts/generate_v451_paired_intervention_pilot.py" --output "$REG/audit.json" >"$REG/audit.log" 2>&1
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
trap - EXIT
echo V451_TECHNICAL_PILOT_COMPLETE
