#!/usr/bin/env bash
set -euo pipefail
B='/root/autodl-tmp/IROS_WAM_2.0 challenge';O="$B/artifacts/strict_track2_official_20260810";N='v257_v254_conservativekl_h200_r2_step8_lr5e6_beta005_seed1459_20260819';test ! -e "$O/run_registry/$N/preregistration.json";test ! -e "$O/runs/$N";bash -n "$B/pipeline/scripts/launch_v257_pipeline.sh";screen -dmS v257_pipeline bash -lc "bash '$B/pipeline/scripts/launch_v257_pipeline.sh' >> '$O/run_registry/$N.console.log' 2>&1";sleep 2;screen -ls|grep '[.]v257_pipeline';echo V257_PIPELINE_STARTED
