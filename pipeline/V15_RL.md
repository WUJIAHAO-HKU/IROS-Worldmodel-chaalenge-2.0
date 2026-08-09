# Track 2 v15 online model and RLinf

Current policy-training status and paired evaluation results are recorded in
[`RL_TRAINING_STATUS.md`](RL_TRAINING_STATUS.md).

The published `track2_v15_best` directory contains every learned weight in the
accepted v15 chain. The retrieval stage uses only the training episodes from
the official public 50-episode dataset; data files are not duplicated in Git.

Start the prediction API:

```bash
export WAM_MODEL_VERSION=track2-v15.0-best
export WAM_BACKEND=v15-composite
export WAM_CHECKPOINT_DIR="$PWD/artifacts/releases/track2_v15_best"
export WAM_V15_LIBRARY_DIR="$PWD/artifacts"
export WAM_BEARER_TOKEN=local-dev-token
PYTHONPATH=pipeline python pipeline/scripts/serve.py
```

Start the RLinf bridge in a second process:

```bash
PYTHONPATH=pipeline python -m wam_pipeline.rlinf_bridge.server \
  --world-model-url http://127.0.0.1:8000 \
  --model-version track2-v15.0-best
```

The low-level policy launcher is `pipeline/scripts/run_public_rlinf_track2.sh`.
For a formal run, use the conservative wrapper:

```bash
export RLINF_ROOT="$PWD/third_party/WorldArena-2.0/RL_env_benchmark"
export OFFICIAL_RESOURCES="$PWD/artifacts/official_resources"
export RLINF_RESET_DATASET="$PWD/artifacts/rlinf_public_reset_adjust_bottle"
export RLINF_BRIDGE_URL=http://127.0.0.1:18080
bash pipeline/scripts/run_v15_grpo_training.sh
```

Its defaults are 20 GRPO updates, `5e-7` actor learning rate, `0.1` PPO
clipping, `0.5` gradient clipping, and one checkpoint at the final update.
These intentionally replace RLinf's much more aggressive example learning
rate. Set `RLINF_TRAIN_STEPS`, `RLINF_ACTOR_LR`, `RLINF_CLIP_RATIO`,
`RLINF_CLIP_GRAD`, `RLINF_SAVE_INTERVAL`, `RLINF_LOG_PATH`, or
`RLINF_RESUME_DIR` to override them.

Run the included world-model smoke test and strict acceptance check before a
long RL job. The reported scores are public-data diagnostics, not organizer
hidden-set results. RLinf's current token-level `approx_kl` and
`clip_fraction` diagnostics sum over the 14 action dimensions while dividing
by the unexpanded token mask; compare them consistently across runs (or divide
by 14 for a per-action diagnostic) rather than treating the displayed values
as scalar-action PPO metrics.
