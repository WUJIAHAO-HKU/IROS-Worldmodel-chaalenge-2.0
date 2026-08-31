#!/usr/bin/env bash
# Manually authorized one-shot public-development S1 for frozen v445. No service/RL.
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
REG="$J/v445_full_mirror_seed1596_20260823"
WORK='/dev/shm/v445_full_mirror_s1_seed1596_20260823'
RELEASE="$J/v445_v169_full_mirror_release"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
WINDOWS="$ROOT/artifacts/adjust_bottle_windows_full"
SOURCE="$ROOT/artifacts/datasets/aloha-agilex_clean_50/data"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
GO1='/root/miniconda3/envs/go1/bin/python'

test -s "$REG/preregistration.json"
test -s "$REG/s0_audit.json"
test -s "$RELEASE/v445_full_mirror_manifest.json"
test ! -e "$REG/s1_gate.json"
test ! -e "$WORK"
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v445_s1_offline.py" | awk '{print $1}')" = '30f96ab9c27e958c70917e0e248066abf1d246383e524532b01f01721382932f'
test "$(sha256sum "$ROOT/pipeline/scripts/audit_v445_s1_offline.py" | awk '{print $1}')" = 'e8558d2052cfa13ead3959ba26820163978411fa136acd7083564077c65817a7'
test "$(sha256sum "$ROOT/pipeline/scripts/generate_v439_s1_offline_inputs.py" | awk '{print $1}')" = '2fcd3cb9ddb83ece52494101c2a6c372972572720a27bca94eabf403925cb0e6'
test "$(sha256sum "$ROOT/pipeline/scripts/train_v423_mirror_augmented_autoregressive_unet.py" | awk '{print $1}')" = '3cf356de1116386b3916068af68f69a1e590d721b9f0510343d40dca04ccb389'
"$GO1" - "$REG/preregistration.json" "$REG/s0_audit.json" <<'PY'
import json,sys
prereg=json.load(open(sys.argv[1])); s0=json.load(open(sys.argv[2]))
assert prereg['format']=='strict-track2-v445-full-mirror-preregistration-v1'
assert prereg['guards']['development_runs']==0
assert s0['format']=='strict-track2-v445-full-mirror-s0-static-contract-v1' and s0['passed'] is True
PY
mkdir -p "$WORK"

cleanup() { bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/s1_restore_v218.log" 2>&1 || true; }
trap cleanup EXIT
for name in wm_v218_bridge wm_v218_gpu; do screen -S "$name" -X quit >/dev/null 2>&1 || true; done
for _ in $(seq 1 60); do ! ss -ltn | grep -qE ':(8005|18084) ' && break; sleep 1; done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6 RAYON_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-11 "$GO1" \
  pipeline/scripts/generate_v445_s1_offline.py --release "$RELEASE" \
  --windows "$WINDOWS" --source-data "$SOURCE" --split "$SPLIT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --output "$WORK/s1_inputs.npz" \
  --device cuda --inference-batch-size 4 --reward-batch-size 32 >"$REG/s1_generator.log" 2>&1

PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" taskset -c 0-5 "$GO1" \
  pipeline/scripts/audit_v445_s1_offline.py --inputs "$WORK/s1_inputs.npz" \
  --preregistration "$REG/preregistration.json" --s0-audit "$REG/s0_audit.json" \
  --output "$REG/s1_gate.json" >"$REG/s1_auditor.log" 2>&1
echo V445_S1_COMPLETE
