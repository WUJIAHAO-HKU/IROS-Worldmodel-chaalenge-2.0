# Track 2 status — 2026-08-13 (final RL evaluation)

## Sealed 128-seed evaluation

| Policy | Success | Rate | Grasp completion | Rate |
|---|---:|---:|---:|---:|
| Official Pi0.5 baseline | 41 / 128 | 32.031% | 113 / 128 | 88.281% |
| Official RL policy, global_step_4 | 49 / 128 | 38.281% | 114 / 128 | 89.063% |
| Difference | +8 | +6.250 pp (+19.512% relative) | +1 | +0.781 pp |

Arm success: left 41/62 (66.129%) → 49/62 (79.032%), +12.903 pp; right 0/66 → 0/66. The overall success and grasp non-regression gates are passed on the paired 128-seed evaluation.

Primary result JSON: `artifacts/strict_track2_official_20260810/real_robotwin_eval/v169_four_step_retry2_summary.json`.

## Current experiment

The remote server ran the unchanged official Pi0.5/reward/RL pipeline with the V16.9 parent release, seed 1243, `runner.max_steps=4`, `algorithm.rollout_epoch=8`, and offload/cache resource settings only. No MPC action selection was used. The final checkpoint is written under:

`artifacts/strict_track2_official_20260810/runs/probability_consistent_four_step_v169_seed1243_retry2/wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/global_step_4/actor/model_state_dict/full_weights.pt`

The 128 fixed-seed RoboTwin evaluation is complete and its batch logs are retained alongside the summary JSON.

## Service

`pipeline/scripts/serve.py` accepts `WAM_SSL_CERTFILE` and `WAM_SSL_KEYFILE` together, preserving HTTP compatibility when unset. Final checkpoint HTTPS acceptance passed 20/20 tests; evidence is `artifacts/strict_track2_official_20260810/https_tls/final_checkpoint_tls_acceptance.json`. Restart reproducibility is recorded in `final_checkpoint_before_restart.json` and `final_checkpoint_after_restart.json` (identical capability and pixel hashes).
