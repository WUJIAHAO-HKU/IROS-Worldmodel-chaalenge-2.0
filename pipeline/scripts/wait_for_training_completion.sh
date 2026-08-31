#!/usr/bin/env bash
# Wait for a successful fixed-step train run, then use an idle GPU for evaluation.
set -euo pipefail

usage() {
    echo "usage: $0 --completion FILE --expected-steps N --log FILE -- COMMAND [ARGS...]" >&2
    exit 2
}

completion=""
expected_steps=""
log_file=""
while (($#)); do
    case "$1" in
        --completion) completion=${2:-}; shift 2 ;;
        --expected-steps) expected_steps=${2:-}; shift 2 ;;
        --log) log_file=${2:-}; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ -n "$completion" && "$expected_steps" =~ ^[1-9][0-9]*$ && -n "$log_file" && $# -gt 0 ]] || usage

mkdir -p "$(dirname "$log_file")"
mkdir -p "$(dirname "$completion")"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"; }
while :; do
    if [[ -f "$completion" ]]; then
        completed=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_steps"])' "$completion" 2>/dev/null || true)
        if [[ "$completed" == "$expected_steps" ]]; then
            log "training completion verified at step ${completed}; delegating evaluation to idle-GPU worker"
            exec "$(dirname "$0")/opportunistic_gpu_worker.sh" \
                --log "$log_file" --idle-seconds 30 --poll-seconds 2 --busy-sm 10 -- "$@"
        fi
        log "completion file has unexpected step ${completed:-missing}; waiting"
    else
        log "waiting for training completion file"
    fi
    sleep 60
done
