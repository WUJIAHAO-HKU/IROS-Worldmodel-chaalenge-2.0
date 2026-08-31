#!/usr/bin/env bash
# Four-context public-train paired simulator technical pilot. No model/policy/RL work.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v452_paired_intervention_pilot_seed1605_20260823"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
DATASET="$ROOT/artifacts/datasets/aloha-agilex_clean_50"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
test ! -e "$REG"
test "$(readlink -f "$("$RLPY" -c 'import sys; print(sys.executable)')")" = "$(readlink -f "$RLPY")"
"$RLPY" -c 'import numpy, h5py, yaml, open3d, toppra, mplib, sapien'
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v452_paired_intervention_pilot.py" | awk '{print $1}')" = 'adf9c52d16a40676e562e6d8cc8a5fc0430f7aff0d6a822108d24270f23f15d5'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v452_paired_intervention_pilot.py" | awk '{print $1}')" = '934cbd1df3cd0dd6503514fd743622df366b604d3c29fdc3e30418920697b069'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v452_paired_intervention_pilot.py" | awk '{print $1}')" = '59e38caab660cd3e1b7767d73e3ac6f244073e4f283bac3bb8ca42933f53dc65'
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" "$RLPY" -c 'from generate_v452_paired_intervention_pilot import ensure_open3d_importable; assert ensure_open3d_importable() is False'
mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$RLPY" pipeline/scripts/prepare_v452_paired_intervention_pilot.py \
 --split "$SPLIT" --dataset "$DATASET" --windows "$WINDOWS" --seed-file "$DATASET/seed.txt" \
 --collector "$ROOT/pipeline/scripts/generate_v452_paired_intervention_pilot.py" --auditor "$ROOT/pipeline/scripts/audit_v452_paired_intervention_pilot.py" \
 --gate-runtime "$ROOT/pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$SUPPORT"
PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" timeout --signal=TERM --kill-after=30s 20m nice -n 10 taskset -c 0-7 "$RLPY" "$ROOT/pipeline/scripts/generate_v452_paired_intervention_pilot.py" \
 --preregistration "$REG/preregistration.json" --support-root "$SUPPORT" --task-config "$SUPPORT/task_config/demo_clean.yml" --output-dir "$REG/dataset" >"$REG/generation.log" 2>&1
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$RLPY" pipeline/scripts/audit_v452_paired_intervention_pilot.py \
 --preregistration "$REG/preregistration.json" --dataset-dir "$REG/dataset" --generator "$ROOT/pipeline/scripts/generate_v452_paired_intervention_pilot.py" --output "$REG/audit.json" >"$REG/audit.log" 2>&1
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
trap - EXIT
echo V452_TECHNICAL_PILOT_COMPLETE
