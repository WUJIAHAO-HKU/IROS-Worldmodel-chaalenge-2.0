#!/usr/bin/env bash
set -euo pipefail

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
P="$ROOT/pipeline/scripts"
NAME='v318_v317_authorized_official_h200_r4_step10_lr2e5_beta001_seed1471_20260822'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
PY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
GO1='/root/miniconda3/envs/go1/bin/python'
RECOVERY="$REG/global_step6_cuda_reset_recovery_preregistration.json"

test -s "$RECOVERY"
test ! -e "$RUN/audit/V318_TRAINING_ACCEPTED"
test ! -e "$RUN/audit/v318_training_prefix_step6.json"

for python in "$GO1" "$PY"; do
  CUDA_VISIBLE_DEVICES=0 "$python" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.arange(1024, device="cuda", dtype=torch.float32)
assert x.sum().item() == 523776.0
torch.cuda.synchronize()
print(torch.cuda.get_device_name(0))
PY
done

has_screen() {
  screen -ls 2>/dev/null | grep -qE "[.]$1[[:space:]]"
}

"$PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
for stale in v318_resume_from_step3 v318_resume_from_step6_cuda_reset wm_v271_v274_gpu wm_v271_v274_bridge wm_v317_gpu wm_v317_bridge; do
  screen -S "$stale" -X quit >/dev/null 2>&1 || true
done

if ! has_screen v318_checkpoint_cache_governor; then
  screen -L -Logfile "$REG/step6_cuda_recovery_governor.screen.log" \
    -dmS v318_checkpoint_cache_governor bash "$P/start_v318_checkpoint_cache_governor.sh"
fi

screen -L -Logfile "$REG/step6_cuda_reset_resume.screen.log" \
  -dmS v318_resume_from_step6_cuda_reset \
  bash "$P/launch_v318_resume_from_step6_after_cuda_reset.sh"
sleep 8
has_screen v318_resume_from_step6_cuda_reset

if ! has_screen v318_resumed_prefix_watcher; then
  screen -L -Logfile "$REG/step6_cuda_recovery_prefix_watcher.screen.log" \
    -dmS v318_resumed_prefix_watcher bash "$P/watch_v318_resumed_training_prefixes.sh"
fi
if ! has_screen v318_to_posttraining; then
  screen -L -Logfile "$REG/step6_cuda_recovery_posttraining_watcher.screen.log" \
    -dmS v318_to_posttraining bash "$P/watch_v318_resumed_to_posttraining.sh"
fi

printf 'V318_STEP6_CUDA_RECOVERY_STARTED utc=%s\n' "$(date -u +%FT%TZ)"
screen -ls
