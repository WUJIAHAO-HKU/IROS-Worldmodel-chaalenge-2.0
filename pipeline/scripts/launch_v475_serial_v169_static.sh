#!/usr/bin/env bash
# v475 real-v169 interface S0 only; no S1/reward/policy/RL.
set -euo pipefail
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge';J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810";S="$ROOT/pipeline/scripts";W="$ROOT/pipeline/wam_pipeline";REG="$J/v475_serial_v169_interface_s0_seed1619_20260824";RELEASE="$J/v475_v474_serial_v169_release_seed1619_20260824";RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RUNTIME="$W/v475_v474_serial_v169_runtime.py";PACKAGE="$S/package_v475_serial_v169_release.py";AUDIT="$S/audit_v475_serial_v169_release.py";CONTRACT="$S/v475_serial_interface_s1_migration_contract.json"
exec 9>/var/lock/v475_serial_v169_static.lock;flock -n 9||exit 73;test ! -e "$REG";test ! -e "$RELEASE";mkdir -p "$REG"
test "$(sha256sum "$RUNTIME"|awk '{print $1}')" = 'c37dd46ab5268549345ba9684167fb8bae32e0bb54fadb0423a3cdda5c6dcd70';test "$(sha256sum "$PACKAGE"|awk '{print $1}')" = '4e01342a0d5b4ebf459c7fb5677f463bcdf47dfc951a01d10e085a8d1fe4a56e';test "$(sha256sum "$AUDIT"|awk '{print $1}')" = 'd7535608a3758a08b0a2faae3a9de0996a754b6838744eaac96345137c9ea044';test "$(sha256sum "$CONTRACT"|awk '{print $1}')" = 'e4562d466bf9c82234eeabad2397e5531653a23d6fd393246bca27af6cd5eaa9'
"$RLPY" -m py_compile "$RUNTIME" "$PACKAGE" "$AUDIT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/pipeline/scripts" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
taskset -c 0-11 timeout 120s "$RLPY" "$PACKAGE" --runtime "$RUNTIME" --contract "$CONTRACT" --output "$RELEASE" >"$REG/package.log" 2>&1
restore(){ bash "$ROOT/pipeline/scripts/restart_v218_services.sh" start >"$REG/restore_v218.log" 2>&1||true;for _ in $(seq 1 120);do curl -fsS http://127.0.0.1:8005/v1/health >/dev/null 2>&1&&curl -fsS http://127.0.0.1:18084/health >/dev/null 2>&1&&return;sleep 1;done;return 1;};trap restore EXIT INT TERM
for n in wm_v218_bridge wm_v218_gpu;do screen -S "$n" -X quit >/dev/null 2>&1||true;done;for _ in $(seq 1 90);do ! ss -ltn|grep -qE ':(8005|18084) '&&break;sleep 1;done;! ss -ltn|grep -qE ':(8005|18084) '
USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null|awk 'NF{s+=$1}END{if(NR==0)print 0;else print s}');[[ "$USED" =~ ^[0-9]+$ ]]&&((USED<=1024))
cd "$ROOT";taskset -c 0-11 timeout 600s "$RLPY" "$AUDIT" --release "$RELEASE" --runtime "$RUNTIME" --packager "$PACKAGE" --output "$REG/static_audit_receipt.json" >"$REG/audit.log" 2>&1
restore;trap - EXIT INT TERM;echo V475_SERIAL_V169_STATIC_PASS
