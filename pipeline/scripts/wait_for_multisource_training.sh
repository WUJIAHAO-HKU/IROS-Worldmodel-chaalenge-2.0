#!/usr/bin/env bash
# Start formal training only after train-only labels are complete and GPU is idle.
set -euo pipefail

usage() {
    echo "usage: $0 --flow-targets DIR --log FILE -- COMMAND [ARGS...]" >&2
    exit 2
}

flow_targets=""
log_file=""
while (($#)); do
    case "$1" in
        --flow-targets) flow_targets=${2:-}; shift 2 ;;
        --log) log_file=${2:-}; shift 2 ;;
        --) shift; break ;;
        *) usage ;;
    esac
done
[[ -n "$flow_targets" && -n "$log_file" && $# -gt 0 ]] || usage

mkdir -p "$(dirname "$log_file")"
mkdir -p "$flow_targets"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$log_file"; }
argument_value() {
    local needle=$1
    shift
    while (($#)); do
        if [[ "$1" == "$needle" ]]; then
            [[ $# -ge 2 ]] || return 1
            printf '%s\n' "$2"
            return 0
        fi
        shift
    done
    return 1
}
manifest="$flow_targets/manifest.json"
while [[ ! -f "$manifest" ]]; do
    count=$(find "$flow_targets" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)
    log "waiting for complete train-only labels: ${count} windows"
    sleep 60
done

runner=$1
windows=$(argument_value --windows "$@") || { log "training command is missing --windows"; exit 2; }
split_manifest=$(argument_value --split-manifest "$@") || { log "training command is missing --split-manifest"; exit 2; }
command_targets=$(argument_value --flow-targets "$@") || { log "training command is missing --flow-targets"; exit 2; }
output=$(argument_value --output "$@") || { log "training command is missing --output"; exit 2; }
flow_resolution=$(argument_value --flow-resolution "$@" || true)
flow_resolution=${flow_resolution:-128}
if [[ "$(realpath "$command_targets")" != "$(realpath "$flow_targets")" ]]; then
    log "training command --flow-targets does not match the waited directory"
    exit 2
fi
root_dir=$(cd "$(dirname "$0")/../.." && pwd)
log "labels complete; validating every train label before GPU training"
"$runner" "$root_dir/pipeline/scripts/validate_multisource_training_inputs.py" \
    --windows "$windows" --split-manifest "$split_manifest" --flow-targets "$flow_targets" \
    --flow-resolution "$flow_resolution" --report "$output/training_input_validation.json" >>"$log_file" 2>&1
log "all labels validated; delegating training to idle-GPU worker"
exec "$(dirname "$0")/opportunistic_gpu_worker.sh" \
    --log "$log_file" --idle-seconds 30 --poll-seconds 2 --busy-sm 10 -- "$@"
