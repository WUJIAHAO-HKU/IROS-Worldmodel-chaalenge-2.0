#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
O="$ROOT/artifacts/strict_track2_official_20260810"
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
P="$ROOT/pipeline/scripts"
RLINF="$ROOT/third_party/WorldArena-2.0/WorldArena-2.0-main/RL_env_benchmark"
DIFFSYNTH="$O/official_deps/diffsynth_2a2e05f"
PY='/root/miniconda3/envs/go1/bin/python'
NAME='v432_public_mirror_prompt_terminal_seed1582_20260823'
REG="$O/run_registry/$NAME"
WORK="/dev/shm/$NAME"
SPLIT="$J/v205_v202_public_right_terminal_multichunk_seed1405/public_demo_split.json"
PARENT="$J/v354_v353_parametric_right_dynamics_extension_seed1523_20260822/model/checkpoints/checkpoint_step_000150"
REWARD="$ROOT/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt"
T5="$ROOT/artifacts/official_resources/reward_model/t5-base"
RESET="$ROOT/artifacts/rlinf_public_reset_adjust_bottle_train40/manifest.json"

cd "$ROOT"
test ! -e "$REG"
test ! -e "$WORK"
mkdir -p "$REG" "$WORK"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export PYTHONPATH="$ROOT/pipeline:$P:$RLINF:$DIFFSYNTH"

"$PY" "$P/audit_v432_public_mirror_contract.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split "$SPLIT" --output "$REG/data_contract.json"
"$PY" "$P/prepare_v432_public_mirror_prompt_terminal.py"

# This launcher deliberately stops at the preregistered 25-step short gate.
# Extension to step50 and any policy update require separate passed audits.
taskset -c 0-11 "$PY" "$P/train_v432_public_mirror_prompt_terminal.py" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --split-manifest "$SPLIT" \
  --init-checkpoint "$PARENT" \
  --reward-checkpoint "$REWARD" --t5-model "$T5" --reset-manifest "$RESET" \
  --output "$WORK/model" \
  --steps 25 --batch-size 1 --chunks 4 --chunk-stride 8 --arm-filter right \
  --learning-rate 1e-7 --reward-loss-weight 0.1 --reward-objective probability \
  --reward-probability-scale-floor 0.01 --reward-delta-weight 2.0 \
  --reward-terminal-weight 10.0 --right-weight 1.0 --success-weight 1.0 \
  --late-weight 1.0 --late-start 64 --terminal-visual-weight 2.0 \
  --prompts-per-arm 4 --checkpoint-interval 25 --max-grad-norm 1.0 \
  --seed 1582 --device cuda \
  >"$REG/training.log" 2>&1

CANDIDATE="$WORK/model/checkpoint_step_000025"
test -s "$CANDIDATE/model.pt"

# audit_v432_step25_shortgate.py always writes its JSON before returning 2 on
# rejection.  With set -e this preserves the candidate/report and stops here;
# no step50 continuation and no policy update are reachable from this script.
taskset -c 0-11 "$PY" "$P/audit_v432_step25_shortgate.py" \
  --parent "$PARENT" --candidate "$CANDIDATE" \
  --windows "$ROOT/artifacts/adjust_bottle_windows_full" \
  --source-data "$ROOT/artifacts/datasets/aloha-agilex_clean_50/data" \
  --split "$SPLIT" --reward-checkpoint "$REWARD" --t5-model "$T5" \
  --preregistration "$REG/preregistration.json" \
  --output "$REG/step25_shortgate.json" \
  --device cuda --reward-batch-size 32 \
  >"$REG/step25_shortgate.log" 2>&1
