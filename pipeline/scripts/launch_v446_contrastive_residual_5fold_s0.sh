#!/usr/bin/env bash
# V446 train-only five-fold S0. Never starts S1, service promotion, or RL.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v446_contrastive_residual_5fold_seed1597_20260823"
RELEASE="$J/v446_v169_contrastive_residual_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
V169="$J/v169_instruction_arm_routed_release"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
V169_LIBRARY_MANIFEST="$V169/v168_release/base_release/release_manifest.json"
V169_LIBRARY_SPLIT="$ROOT/artifacts/splits/adjust_bottle_50episodes_full.json"
GO1='/root/miniconda3/envs/go1/bin/python'

test ! -e "$REG"
test ! -e "$RELEASE"
test "$(sha256sum "$ROOT/pipeline/scripts/prepare_v446_contrastive_residual_5fold.py" | awk '{print $1}')" = '4478b298d2d4820cb057a4838e82057f955b0b8828ad69c87759c9972075d348'
test "$(sha256sum "$ROOT/pipeline/scripts/train_v446_contrastive_residual_5fold.py" | awk '{print $1}')" = '94a3c815c9c742a57aef1dd240b1632e4deea9a2ba4ab1527acdeb762d1249d1'
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v446_v169_contrastive_residual_unet_runtime.py" | awk '{print $1}')" = '808d5797ce6cad6a01a084a50b344f94a93bf5dd8276c4b2c7e4b7f39f796f63'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v446_s0_contract.py" | awk '{print $1}')" = '6f71adf8d96cb2990ecdad5aa1b73794b1cb7e388de05242f9955016af30a2d7'
test "$(sha256sum "$ROOT/pipeline/scripts/package_v446_contrastive_residual_release.py" | awk '{print $1}')" = '8bcf2864e18a71fcf991d518349c7687c067e38c31dde74c11936dc8b0a8864c'
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v169_arm_routed_runtime.py" | awk '{print $1}')" = '0044d49ae2a3083f0b638b701fb398b407d7c436c3bb0ee1798a3233e076c7c1'
test "$(sha256sum "$ROOT/pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py" | awk '{print $1}')" = '922cf4e22dd1c2fd0d88c9717db609d8b180cdbf8d0c03f8eb15d17c8b0b24a4'

mkdir -p "$REG"
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/prepare_v446_contrastive_residual_5fold.py \
  --split "$SPLIT" --windows "$WINDOWS" \
  --v169-release-manifest "$V169/v169_arm_routed_manifest.json" \
  --v169-library-manifest "$V169_LIBRARY_MANIFEST" --v169-library-split-manifest "$V169_LIBRARY_SPLIT" \
  --reward-checkpoint "$REWARD" --t5-config "$T5/config.json" \
  --output "$REG/preregistration.json" >"$REG/prepare.log" 2>&1

cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(ss -ltn | grep -E ':(8005|18084) ' || true)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export PYTHONHASHSEED=1597 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
set +e
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/train_v446_contrastive_residual_5fold.py \
  --windows "$WINDOWS" --split "$SPLIT" --preregistration "$REG/preregistration.json" \
  --v169-release "$V169" --v169-library "$ROOT/artifacts" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --output-dir "$REG/fold_run" \
  --device cuda >"$REG/training.log" 2>&1
TRAIN_STATUS=$?
set -e
test -s "$REG/fold_run/training_report.json"
if ! "$GO1" - "$REG/fold_run/training_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['format']=='strict-track2-v446-5fold-s0-training-report-v1'
assert len(r['folds'])==5 and r['guards']['all_five_folds_completed'] is True
assert r['passed'] is True and r['final_training']['performed'] is True
PY
then
  exit 2
fi
test "$TRAIN_STATUS" -eq 0
test -s "$REG/fold_run/final_all15_step50.pt"

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/audit_v446_s0_contract.py \
  --training-report "$REG/fold_run/training_report.json" --preregistration "$REG/preregistration.json" \
  --runtime "$ROOT/pipeline/wam_pipeline/v446_v169_contrastive_residual_unet_runtime.py" \
  --trainer "$ROOT/pipeline/scripts/train_v446_contrastive_residual_5fold.py" \
  --v169-runtime "$ROOT/pipeline/wam_pipeline/v169_arm_routed_runtime.py" \
  --close-gate-runtime "$ROOT/pipeline/wam_pipeline/v442_v169_close_aligned_projection_runtime.py" \
  --final-checkpoint "$REG/fold_run/final_all15_step50.pt" --output "$REG/s0_audit.json" \
  >"$REG/s0_audit.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/package_v446_contrastive_residual_release.py \
  --checkpoint "$REG/fold_run/final_all15_step50.pt" --training-report "$REG/fold_run/training_report.json" \
  --preregistration "$REG/preregistration.json" --s0-audit "$REG/s0_audit.json" \
  --output "$RELEASE" >"$REG/package.log" 2>&1
bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1
curl -fsS 'http://127.0.0.1:8005/v1/health' | grep -q 'track2-v218-public-knn-blend-alpha070-route-aware'
curl -fsS 'http://127.0.0.1:18084/health' | grep -q '"status":"ready"'
trap - EXIT
echo V446_S0_COMPLETE
