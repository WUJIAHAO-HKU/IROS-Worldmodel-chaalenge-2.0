# Track 2 v15 online model and RLinf

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

The formal policy launcher is `pipeline/scripts/run_public_rlinf_track2.sh`.
Run the included world-model smoke test and strict acceptance check before a
long RL job. The reported scores are public-data diagnostics, not organizer
hidden-set results.
