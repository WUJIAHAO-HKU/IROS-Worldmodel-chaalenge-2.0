#!/usr/bin/env bash
set -u
ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
OFF="$ROOT/artifacts/strict_track2_official_20260810"
NAME='v278_v271_fresh_fullbudget_h200_r4_step10_lr2e5_beta001_seed1471_retry2_20260820'
RUN="$OFF/runs/$NAME"
REG="$OFF/run_registry/$NAME"
quota_cores=12
cap_cores=unknown
cap_affinity=unknown
previous_usage=$(awk '$1=="usage_usec"{print $2}' /sys/fs/cgroup/cpu.stat)
previous_ns=$(date +%s%N)
previous_high=$(awk '$1=="high"{print $2}' /sys/fs/cgroup/memory.events)
previous_oom=$(awk '$1=="oom"{print $2}' /sys/fs/cgroup/memory.events)
previous_oom_kill=$(awk '$1=="oom_kill"{print $2}' /sys/fs/cgroup/memory.events)
while screen -ls | grep -q '[.]v278_retry2_rl'; do
  now=$(date -u +%FT%TZ)
  gpu=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  tb=$(find "$RUN/tensorboard" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  chunks=$(find "$RUN/video/train" -type f 2>/dev/null | wc -l)
  ckpts=$(find "$RUN" -path '*/checkpoints/global_step_*' -maxdepth 6 -type d -printf '%f\n' 2>/dev/null | sort -V | paste -sd,)
  avail=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
  current_usage=$(awk '$1=="usage_usec"{print $2}' /sys/fs/cgroup/cpu.stat)
  current_ns=$(date +%s%N)
  current_high=$(awk '$1=="high"{print $2}' /sys/fs/cgroup/memory.events)
  current_oom=$(awk '$1=="oom"{print $2}' /sys/fs/cgroup/memory.events)
  current_oom_kill=$(awk '$1=="oom_kill"{print $2}' /sys/fs/cgroup/memory.events)
  memory_bytes=$(cat /sys/fs/cgroup/memory.current)
  memory_anon_bytes=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
  memory_file_bytes=$(awk '$1=="file"{print $2}' /sys/fs/cgroup/memory.stat)
  memory_shmem_bytes=$(awk '$1=="shmem"{print $2}' /sys/fs/cgroup/memory.stat)
  main_pid=$(pgrep -f "train_embodied_agent.py.*$NAME" | head -1 || true)
  if [[ -n "$main_pid" ]]; then
    cap_affinity=$(taskset -pc "$main_pid" 2>/dev/null | awk -F': ' 'END{print $2}')
    cap_cores=$(awk -v list="$cap_affinity" 'BEGIN {
      n=split(list, parts, ","); total=0;
      for (i=1; i<=n; i++) {
        if (parts[i] ~ /-/) { split(parts[i], bounds, "-"); total += bounds[2]-bounds[1]+1; }
        else if (parts[i] != "") total++;
      }
      print (total > 0 ? total : "unknown");
    }')
  else
    cap_affinity=unknown
    cap_cores=unknown
  fi
  cpu_pct=$(awk -v du="$((current_usage-previous_usage))" -v dn="$((current_ns-previous_ns))" -v q="$quota_cores" 'BEGIN{if(dn>0)printf "%.2f",100.0*du*1000.0/dn/q;else print "0.00"}')
  memory_gib=$(awk -v b="$memory_bytes" 'BEGIN{printf "%.2f",b/1073741824.0}')
  memory_anon_gib=$(awk -v b="$memory_anon_bytes" 'BEGIN{printf "%.2f",b/1073741824.0}')
  memory_file_gib=$(awk -v b="$memory_file_bytes" 'BEGIN{printf "%.2f",b/1073741824.0}')
  memory_shmem_gib=$(awk -v b="$memory_shmem_bytes" 'BEGIN{printf "%.2f",b/1073741824.0}')
  high_delta=$((current_high-previous_high))
  oom_delta=$((current_oom-previous_oom))
  oom_kill_delta=$((current_oom_kill-previous_oom_kill))
  previous_usage=$current_usage
  previous_ns=$current_ns
  previous_high=$current_high
  previous_oom=$current_oom
  previous_oom_kill=$current_oom_kill
  printf '%s cpu_pct_of_12=%s cpu_cap_cores=%s cpu_affinity=%s memory_gib=%s memory_anon_gib=%s memory_file_gib=%s memory_shmem_gib=%s memory_high_delta=%s memory_oom_delta=%s memory_oom_kill_delta=%s gpu_mib=%s tensorboard_bytes=%s video_files=%s checkpoints=%s disk_avail_gb=%s\n' \
    "$now" "$cpu_pct" "$cap_cores" "$cap_affinity" "$memory_gib" "$memory_anon_gib" "$memory_file_gib" "$memory_shmem_gib" "$high_delta" "$oom_delta" "$oom_kill_delta" "$gpu" "$tb" "$chunks" "${ckpts:-none}" "$avail" >>"$REG/health.log"
  sleep 60
done
echo V278_RETRY2_HEALTH_WATCH_STOPPED >>"$REG/health.log"
