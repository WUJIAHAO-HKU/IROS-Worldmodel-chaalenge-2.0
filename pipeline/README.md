# WorldArena Track 2 Pipeline

This directory runs the path required before a WorldArena Track 2 submission:

```text
RoboTwin HDF5 -> 5-frame/4-history-action/8-future-action windows
             -> action-conditioned model -> official /v1 API
             -> closed-loop rollout -> local MBRL smoke test
```

`synthetic` is a deterministic protocol-test backend. It proves pipeline wiring only and is never a submission model. `ivideogpt` is reserved for a Track-2-adapted iVideoGPT-64 checkpoint; the public BAIR checkpoint is a starting point, not a compatible final model.

The full pinned upstream checkout is `third_party/iVideoGPT` at
`d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210` (65 commits; verified non-shallow,
non-partial with `git fsck --full`). The runtime resolves this checkout before
any legacy vendor fallback. Its official `inference/predict.py` has been run
locally against the BAIR checkpoint; output is in
`artifacts/official_ivideogpt_bair_smoke/`. The full-checkout backend also
passed the two-round rollout, local MBRL smoke, and HTTP contract test.

## 1. Environment

Use the existing GPU-enabled environment:

```bash
conda run -n go1 python pipeline/scripts/check_environment.py
conda run -n go1 pip install -r pipeline/requirements-go1.txt
```

For real iVideoGPT training/inference, install `pipeline/requirements-ivideogpt.txt` too. The current machine has a 32 GB RTX 5090 in `go1`.

## 2. First complete smoke run

This run has no external model or dataset download dependency:

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/make_toy_data.py --output artifacts/toy_windows
conda run -n go1 python pipeline/scripts/run_mbrl_smoke.py \
  --window artifacts/toy_windows/toy_00000.npz --backend synthetic --rounds 2

WAM_BACKEND=synthetic WAM_BEARER_TOKEN=local-dev-token \
  conda run -n go1 python pipeline/scripts/serve.py
```

In a second shell:

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/contract_test.py
conda run -n go1 python -m pytest -q pipeline/tests
```

The contract test covers health, capabilities, one-item and eight-item batches, deterministic retry, request-ID conflict, invalid action shape, invalid profile, and authentication.

## 3. Official data adapter

After downloading and unpacking the official `aloha-agilex_clean_50.zip`, verify its actual fields first:

```bash
conda run -n go1 python pipeline/scripts/verify_robotwin_data.py \
  --input artifacts/datasets/aloha-agilex_clean_50/data
```

Then generate all valid windows (omit `--limit-per-episode`; the checked data has 6,638 windows):

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/adapt_robotwin.py \
  --input artifacts/datasets/aloha-agilex_clean_50/data --output artifacts/adjust_bottle_windows_full
```

Each output NPZ contains:

```text
context_frames  [5, 256, 256, 3] uint8
history_actions [4, 14] float32
future_actions  [8, 14] float32
target_frames   [8, 256, 256, 3] uint8
```

Alignment is exact: future action `u0` transforms `o4 -> p0`; the next request uses `p3..p7` plus action history `u4..u7`.

## 4. iVideoGPT-64 path

1. Download the public, MIT-licensed starting checkpoint:

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/download_ivideogpt.py \
  --output artifacts/upstream/ivideogpt-bair-64-act-cond
```

2. Create an episode-disjoint split, then train the real minimal 5-context / 14-action adapter. The default `--steps 4` is a smoke run; begin development with 200 steps:

```bash
conda run -n go1 python pipeline/scripts/make_episode_split.py \
  --windows artifacts/adjust_bottle_windows_full \
  --output artifacts/splits/adjust_bottle_50episodes_full.json
conda run -n go1 python pipeline/scripts/train_ivideogpt64.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --upstream-checkpoint artifacts/upstream/ivideogpt-bair-64-act-cond \
  --output artifacts/checkpoints/ivideogpt64-track2-dev \
  --steps 200 --batch-size 1 --learning-rate 1e-3 \
  --validation-interval 25 --validation-batches 64
```

This freezes the public BAIR tokenizer/LLM and trains the 14D action projection using real token cross-entropy. It writes a runnable root checkpoint plus `checkpoints/checkpoint_step_*`; the root and `best/` are the lowest validation-loss candidate. Validation is episode-disjoint and sampled evenly across held-out episodes. The original BAIR model has one context frame and four action dimensions, so its public checkpoint alone remains incompatible.

Export an Adjust Bottle comparison GIF after any run:

```bash
conda run -n go1 python pipeline/scripts/export_prediction_gif.py \
  --window artifacts/adjust_bottle_windows_full/episode5_00000.npz \
  --checkpoint-dir artifacts/checkpoints/ivideogpt64-track2-dev/best \
  --output artifacts/visualizations/adjust_bottle_dev_validation_episode5.gif
```

4. The iVideoGPT reference service remains available:

```bash
WAM_BACKEND=ivideogpt WAM_CHECKPOINT_DIR=artifacts/checkpoints/ivideogpt64-track2-dev/best \
  WAM_BEARER_TOKEN=local-dev-token conda run -n go1 python pipeline/scripts/serve.py
```

The 64x64 model is a development baseline. Before submission, train/evaluate at 256x256 and rerun the same API tests with `model_version` frozen.

## 5. Native 256px candidate

The current frozen native-resolution candidate is:

```text
backend:       autoregressive-unet
model version: autoregressive-unet-track2-native256-rollout8
checkpoint:    artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1/best
```

It predicts one action-conditioned frame at a time, recycles predictions into
the next five-frame context, and returns the required eight frames at `256x256`.
It passed the HTTP contract test. The fixed episode-held-out open-loop checks
are development evidence, not an organizer score:

| split | model MAE | copy-last MAE |
| --- | ---: | ---: |
| validation, all 682 windows | 7.57 | 15.33 |
| local-test, all 667 windows | 6.27 | 15.11 |

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend autoregressive-unet --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1/best \
  --split validation --samples 682 \
  --output artifacts/evaluations/autoregressive_unet_rollout8_track2_native256_validation_all.json
```

The held-out comparison GIFs are
`artifacts/visualizations/autoregressive_unet_rollout8_track2_native256_validation_episode16_00025.gif`
and
`artifacts/visualizations/autoregressive_unet_rollout8_track2_native256_localtest_episode2_00025.gif`.

## 6. Public RLinf closed loop

The local `track2_http` environment uses the published pickle transport at
`/reset` and `/chunk_step`, but does not import Wan or DiffSynth. A local bridge
converts each RLinf action chunk into one strict Track 2 `/v1/predict` request;
the environment loads the official T5 reward checkpoint and returns the
per-frame reward differences to GRPO.

1. Download only the published policy and reward directories, using the mirror
   when direct Hugging Face access is unavailable:

```bash
conda run -n go1 bash pipeline/scripts/download_official_rlinf_resources.sh
```

2. Generate public reset states. `DiffSynth-Studio`, Wan weights, and the
   RoboTwin simulator are not used: frame generation stays in this project's
   Track-2 API. `ROBOTWIN_REWARD_MODEL_PATH` points to the official checkpoint.

```bash
export T5_MODEL_PATH=/path/to/t5-base
export ROBOTWIN_REWARD_MODEL_PATH=/path/to/reward_model_checkpoint
export WAM_API_URL=http://127.0.0.1:8002
export WAM_BEARER_TOKEN=local-dev-token
export WAM_MODEL_VERSION=autoregressive-unet-track2-native256-rollout8
```

```bash
PYTHONPATH="$PWD/pipeline" conda run -n go1 python \
  pipeline/scripts/make_public_rlinf_reset_dataset.py \
  --input artifacts/datasets/aloha-agilex_clean_50 \
  --output artifacts/rlinf_public_reset_adjust_bottle
export RLINF_RESET_DATASET=$PWD/artifacts/rlinf_public_reset_adjust_bottle
```

3. Start the frozen candidate API and bridge, then launch the public RLinf
   recipe. The script checks the published policy/reward, RLinf source, reset
   data, and bridge health.

```bash
WAM_BACKEND=autoregressive-unet \
WAM_CHECKPOINT_DIR=artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1/best \
WAM_MODEL_VERSION=$WAM_MODEL_VERSION WAM_BEARER_TOKEN=$WAM_BEARER_TOKEN \
WAM_PORT=8002 conda run -n go1 python pipeline/scripts/serve.py

PYTHONPATH="$PWD/pipeline" conda run -n go1 python -m wam_pipeline.rlinf_bridge.server \
  --world-model-url=$WAM_API_URL --token=$WAM_BEARER_TOKEN \
  --model-version=$WAM_MODEL_VERSION --port=18081

RLINF_BRIDGE_URL=http://127.0.0.1:18081 \
RLINF_RESET_DATASET=$PWD/artifacts/rlinf_public_reset_adjust_bottle \
conda run -n go1 bash pipeline/scripts/run_public_rlinf_track2.sh runner.max_steps=1
```

The bridge and the actual published components have been tested as:

```text
RLinf-format reset -> 8 x 14D pi05 action chunk -> Track 2 /v1/predict
-> [1,3,1,13,256,256] updated world state
```

Run the two-round component smoke before a GRPO job:

```bash
PYTHONPATH="$PWD/pipeline:$PWD/third_party/WorldArena-2.0/RL_env_benchmark:$PWD/third_party/openpi-rlinf-full/src" \
conda run -n go1 python pipeline/scripts/run_real_track2_closed_loop.py \
  --reset artifacts/rlinf_public_reset_adjust_bottle/episode0.npy \
  --bridge-url http://127.0.0.1:18081 --rounds 2
```

It runs `official pi05 -> Track-2 API -> official reward` for two chunks and
reports finite per-chunk reward statistics. It remains a public-data local
smoke test, not the organizer’s hidden held-out result.

This produces a *public local RLinf/RoboTwin reproduction*.  It is not the
organizer's hidden held-out final score, which cannot be computed locally.

The included configuration enables CPU offload and CPU weight transport for a
single 32-GB GPU. The current autoregressive candidate passed the API contract,
the two-round `pi05 -> Track-2 API -> official T5 reward` check, and one actual
GRPO update. Its `global_step_1` checkpoint is under
`artifacts/rlinf_track2_autoregressive_unet_rollout8_grpo_step1/autoregressive_unet_rollout8_grpo_step1/checkpoints/global_step_1/`.
This only verifies the execution path. The one-step relative reward was very
small, so it is not evidence of useful policy improvement; longer public
closed-loop runs are required before comparing candidate quality.

## 7. Earlier 128px development baseline

The iVideoGPT path remains useful as a complete upstream-compatible reference, but its 64px public BAIR visual codebook is not the strongest choice for this small single-task data set. Use the Track-2 residual U-Net candidate for current development:

```bash
conda run -n go1 python pipeline/scripts/train_residual_unet.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --output artifacts/checkpoints/residual-unet-track2-dev128 \
  --steps 5000 --batch-size 8 --resolution 128 \
  --learning-rate 2e-4 --validation-interval 500 --validation-batches 32

conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend residual-unet --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/residual-unet-track2-dev128/best \
  --split validation --samples 64 \
  --output artifacts/evaluations/residual_unet_track2_dev128_validation.json
```

The supplied 5,000-step candidate reaches a 256px held-out open-loop MAE of `8.61`, versus `15.02` for copying the last observation. This is evidence of useful prediction, not an official score: the model still requires official closed-loop MBRL evaluation and a native 256px training candidate before submission.

## 8. Boundary of the MBRL smoke test

`run_mbrl_smoke.py` verifies the integration loop: policy -> 8 actions -> world model -> local finite proxy reward -> policy update. Its image-distance reward is not the organizer's reward checkpoint. Use the official policy/reward/RLinf stack only after all its matching components are available; the official final held-out evaluator is not public.
