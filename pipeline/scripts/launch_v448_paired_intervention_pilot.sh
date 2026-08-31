#!/usr/bin/env bash
# Four-context public-train paired simulator technical pilot. No model/policy/RL work.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v448_paired_intervention_pilot_seed1601_20260823"
SUPPORT="$ROOT/artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
DATASET="$ROOT/artifacts/datasets/aloha-agilex_clean_50"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
GO1='/root/miniconda3/envs/go1/bin/python'
test ! -e "$REG"
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v448_paired_intervention_pilot.py" | awk '{print $1}')" = 'e624840e56b246faed296f74bbac66a02623c74f2084e07618e0e62be2c35435'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v448_paired_intervention_pilot.py" | awk '{print $1}')" = 'eedc7cd090c5ac285361b99629486227690bc7c28f953f6e196c743efb5549d2'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v448_paired_intervention_pilot.py" | awk '{print $1}')" = '31afd2985e644497be726a5206aeb9f7d26123cf5065c5e767696cb9c95d9693'
mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" pipeline/scripts/prepare_v448_paired_intervention_pilot.py \
 --split "$SPLIT" --dataset "$DATASET" --windows "$WINDOWS" --seed-file "$DATASET/seed.txt" \
 --collector "$ROOT/pipeline/scripts/generate_v448_paired_intervention_pilot.py" --auditor "$ROOT/pipeline/scripts/audit_v448_paired_intervention_pilot.py" \
 --gate-runtime "$ROOT/pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py" --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1
cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" stop
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$SUPPORT"
PYTHONPATH="$SUPPORT:$ROOT/pipeline:$ROOT/pipeline/scripts" timeout --signal=TERM --kill-after=30s 20m nice -n 10 taskset -c 0-7 "$GO1" "$ROOT/pipeline/scripts/generate_v448_paired_intervention_pilot.py" \
 --preregistration "$REG/preregistration.json" --support-root "$SUPPORT" --task-config "$SUPPORT/task_config/demo_clean.yml" --output-dir "$REG/dataset" >"$REG/generation.log" 2>&1
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" pipeline/scripts/audit_v448_paired_intervention_pilot.py \
 --preregistration "$REG/preregistration.json" --dataset-dir "$REG/dataset" --generator "$ROOT/pipeline/scripts/generate_v448_paired_intervention_pilot.py" --output "$REG/audit.json" >"$REG/audit.log" 2>&1
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
trap - EXIT
echo V448_TECHNICAL_PILOT_COMPLETE
