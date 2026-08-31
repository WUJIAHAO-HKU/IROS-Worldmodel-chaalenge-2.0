#!/usr/bin/env bash
# Run a CUDA command only while all other compute processes are idle.
set -euo pipefail

usage() {
    echo "usage: $0 --log FILE [--idle-seconds N] [--poll-seconds N] [--busy-sm N] [--max-other-memory-mib N] [--min-free-mib N] [--yield-on-other] -- COMMAND [ARGS...]" >&2
    exit 2
}

log_file=""
idle_seconds=30
poll_seconds=2
busy_sm=10
max_other_memory_mib=512
min_free_mib=28000
yield_on_other=0
while (($#)); do
    case "$1" in
        --log) log_file=${2:-}; shift 2 ;;
        --idle-seconds) idle_seconds=${2:-}; shift 2 ;;
        --poll-seconds) poll_seconds=${2:-}; shift 2 ;;
        --busy-sm) busy_sm=${2:-}; shift 2 ;;
        --max-other-memory-mib) max_other_memory_mib=${2:-}; shift 2 ;;
        --min-free-mib) min_free_mib=${2:-}; shift 2 ;;
        --yield-on-other) yield_on_other=1; shift ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ -n "$log_file" && $# -gt 0 ]] || usage
[[ "$idle_seconds" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[1-9][0-9]*$ && "$busy_sm" =~ ^[0-9]+$ && "$max_other_memory_mib" =~ ^[0-9]+$ && "$min_free_mib" =~ ^[0-9]+$ ]] || usage

mkdir -p "$(dirname "$log_file")"
touch "$log_file"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"; }

# ``setsid`` makes the launched command and all descendants one process group.
# Matching that group is more reliable than recursively walking a changing
# ``conda run`` process tree while CUDA worker processes are being spawned.
process_group_pids() {
    local group=$1
    ps -eo pid=,pgid= 2>/dev/null | awk -v group="$group" '$2 == group { printf "%s ", $1 }'
}

other_sm() {
    local own_pid=${1:-0}
    # Driver sampling can transiently fail while another CUDA job reconfigures.
    # Treat an unreadable sample as busy instead of terminating or stealing GPU.
    local pmon
    pmon=$(nvidia-smi pmon -c 1 2>/dev/null || true)
    if [[ -z "$pmon" ]]; then
        printf '999\n'
        return
    fi
    local own_pids=""
    if ((own_pid > 0)); then
        own_pids=" $(process_group_pids "$own_pid")"
    fi
    awk -v own="${own_pids} " '
        $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && index(own, " " $2 " ") == 0 && $4 ~ /^[0-9]+$/ { total += $4 }
        END { print total + 0 }
    ' <<<"$pmon"
}

other_compute_memory_mib() {
    local own_pid=${1:-0}
    local applications
    applications=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    [[ -n "$applications" ]] || { printf '0\n'; return; }
    local own_pids=""
    if ((own_pid > 0)); then
        own_pids=" $(process_group_pids "$own_pid")"
    fi
    awk -F, -v own="${own_pids} " '
        $1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            pid = $1; gsub(/[[:space:]]/, "", pid)
            memory = $2; gsub(/[^0-9]/, "", memory)
            if (index(own, " " pid " ") == 0) total += memory
        }
        END { print total + 0 }
    ' <<<"$applications"
}

gpu_free_mib() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    free=${free//[^0-9]/}
    [[ -n "$free" ]] && printf '%s\n' "$free" || printf '0\n'
}

idle_for=0
while ((idle_for < idle_seconds)); do
    sm=$(other_sm 0)
    other_memory=$(other_compute_memory_mib 0)
    free_memory=$(gpu_free_mib)
    if ((sm < busy_sm && other_memory <= max_other_memory_mib && free_memory >= min_free_mib)); then
        idle_for=$((idle_for + poll_seconds))
    else
        idle_for=0
    fi
    log "waiting other_sm=${sm}% other_compute_memory=${other_memory}MiB free_memory=${free_memory}MiB idle_for=${idle_for}/${idle_seconds}s"
    sleep "$poll_seconds"
done

log "starting low-priority GPU command: $*"
# Create a dedicated session so yielding this worker reliably stops every
# descendant of conda/Python, rather than leaving a GPU child orphaned.
setsid nice -n 19 ionice -c 3 "$@" >>"$log_file" 2>&1 &
child=$!
paused=0
idle_for=0
trap 'kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT INT TERM

while kill -0 "$child" 2>/dev/null; do
    sm=$(other_sm "$child")
    other_memory=$(other_compute_memory_mib "$child")
    if ((sm >= busy_sm || other_memory > max_other_memory_mib)); then
        idle_for=0
        if ((yield_on_other == 1)); then
            log "yielding child=${child}; other_sm=${sm}% other_compute_memory=${other_memory}MiB"
            kill -TERM -- "-${child}" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
            wait "$child" 2>/dev/null || true
            trap - EXIT INT TERM
            exec "$0" --log "$log_file" --idle-seconds "$idle_seconds" --poll-seconds "$poll_seconds" \
                --busy-sm "$busy_sm" --max-other-memory-mib "$max_other_memory_mib" --min-free-mib "$min_free_mib" \
                --yield-on-other -- "$@"
        elif ((paused == 0)); then
            kill -STOP "$child"
            paused=1
            log "paused child=${child}; other_sm=${sm}% other_compute_memory=${other_memory}MiB"
        fi
    else
        idle_for=$((idle_for + poll_seconds))
        if ((paused == 1 && idle_for >= idle_seconds)); then
            kill -CONT "$child"
            paused=0
            log "resumed child=${child}; other_sm=${sm}% after ${idle_for}s idle"
        fi
    fi
    sleep "$poll_seconds"
done

wait "$child"
status=$?
trap - EXIT INT TERM
log "child=${child} exited status=${status}"
exit "$status"
