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

## 5. Current native 256px structure-aware local-fusion candidate

The current best reproducible backend is:

```text
backend:       local-motion-texture-fusion
model version: autoregressive-direct-flow-local-motion-texture-structure-v5
checkpoint:    artifacts/checkpoints/autoregressive-direct-flow-local-motion-texture-structure-v5
```

It runs the horizon-refined autoregressive and Direct Flow parents, then uses a
small action-conditioned spatial network to predict per-pixel blend weights and
a bounded residual. It returns the required eight frames at `256x256`.
It passed the HTTP contract test. The fixed episode-held-out open-loop checks
are development evidence, not an organizer score or an RL admission result:

| split | model MAE | copy-last MAE |
| --- | ---: | ---: |
| validation, all 682 windows | 5.950 | 15.334 |
| local-test, all 667 windows | 5.075 | 15.108 |

The validation high-motion MAE is `10.434`; local-test high-motion MAE is
`8.815`. Relative to v4, overall RGB improves by 1.07%/1.32%, moving-region RGB
by 0.62%/0.86%, and moving-region Laplacian error by 1.44%/1.56% on
validation/local-test. The structure-aware loss explicitly protects dark robot
silhouettes, moving edges, and high-frequency markings. Paired bootstrap
improvement probability is 1.0 on both splits. The candidate still fails the strict `1/255` gate: only
1 of 682 validation windows keeps all eight predicted frames below the threshold.

A cross-validated blend with Direct Flow v2 reduced holdout overall MAE from
`6.670` to `6.554`, but increased high-motion MAE from `10.974` to `11.081`.
It is rejected as the default fixed backend.

The promoted `local-motion-texture-fusion` uses only the observable context,
both parent predictions, and the supplied future actions. The self-contained,
SHA256-verified package is
`artifacts/checkpoints/autoregressive-direct-flow-local-motion-texture-structure-v5`
with artifact hash
`a73d707a286a0e9403d4421a627fe77b81c54333f374f023e1c3f7c8bf664336`.
Its packaged prediction path is bit-exact with the audited cache on the
64-window replay, and its HTTP contract test passes, so it is the active
open-loop candidate. The previous v3 package remains the threshold-count
reference because it passes two rather than one easy validation window.

Rebuild both resume-safe training stages and automatically promote a successful
64-window pilot to the full validation evaluation:

```bash
bash pipeline/scripts/run_autoregressive_rebuild.sh
bash pipeline/scripts/run_autoregressive_motion_rebuild_pilot.sh
bash pipeline/scripts/run_autoregressive_horizon_pilot.sh
bash pipeline/scripts/run_autoregressive_horizon_largebatch_refine.sh
```

Package the learned local fusion model:

```bash
conda run -n go1 python pipeline/scripts/package_local_motion_texture_fusion.py \
  --autoregressive artifacts/checkpoints/autoregressive-unet-track2-rollout8-horizon-largebatch-refine-v1/best \
  --direct-flow artifacts/checkpoints/direct-flow-unet-track2-formal-v2/best \
  --fusion artifacts/checkpoints/local-motion-texture-fusion-structure-v2/best \
  --validation-report artifacts/evaluations/local_motion_texture_fusion_structure_v2_vs_v1_validation682.json \
  --local-report artifacts/evaluations/local_motion_texture_fusion_structure_v2_vs_v1_local667.json \
  --output artifacts/checkpoints/autoregressive-direct-flow-local-motion-texture-structure-v5
```

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend local-motion-texture-fusion --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/autoregressive-direct-flow-local-motion-texture-structure-v5 \
  --split validation --samples 682 \
  --output artifacts/evaluations/local_motion_texture_structure_v5_runtime_validation_all.json
```

The current held-out MP4/GIF comparisons and per-frame metrics are grouped in
`artifacts/evaluation_videos/local_motion_texture_structure_v5/`; it contains one
high-motion and one low-motion example.

No candidate may enter RL until it passes `<1/255` MAE for every one of the
`682 x 8` validation frames. The learned fusion package is therefore open-loop
development evidence and a deployable API candidate, not RL admission.

## 6. Formal native Wan path

The formal replacement is a native `256x256` action-conditioned Wan LoRA. It
uses the exact organizer transition rather than the public DiffSynth
`RLinfNpyDataset` loader (that public loader hard-codes a seven-dimensional
first action):

```text
5 observed RGB + 4 history abs14 + 8 future abs14
    -> [zero anchor, 4 history, 8 future] = 13 action slots
    -> Wan VAE: 13 RGB frames = 4 latent slots
    -> slots 0--1 fixed context, slots 2--3 future-only flow loss
    -> 8 predicted RGB frames
```

The encoded data is already checked at
`artifacts/wan/adjust_bottle_rlinf_npy_v1`: 5,289 train windows and 682
episode-held-out validation windows. Validate the contract and action path:

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/validate_wan_track2_windows.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1
conda run -n go1 python pipeline/scripts/test_track2_wan_action_conditioning.py --device cuda
```

Download the action-only base (transformer + VAE; no text encoder is needed):

```bash
conda run -n go1 bash pipeline/scripts/download_track2_wan_base.sh
```

Before a long run, execute the real-model smoke test. It uses one official
window and verifies the 48-channel Wan2.2 VAE encode/decode, token-level
context timestep mask, action-conditioned LoRA backward pass, and checkpoint
round-trip:

```bash
conda run -n go1 python pipeline/scripts/smoke_track2_wan_real.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1 \
  --base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/smoke/track2-wan-real
```

Cache the frozen VAE latents once. This covers every 5,289 training and 682
held-out validation window, checks both dataset and VAE hashes, and lets the
formal training process load only the DiT:

```bash
conda run -n go1 python pipeline/scripts/cache_track2_wan_latents.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1 \
  --base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/wan/adjust_bottle_wan22_latents_v1 --batch-size 4
```

Then start resumable single-GPU LoRA training:

```bash
conda run -n go1 python pipeline/scripts/train_track2_wan.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1 \
  --base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/checkpoints/track2-wan-adjust-bottle-v1 \
  --latent-cache artifacts/wan/adjust_bottle_wan22_latents_v1 \
  --steps 30000 --batch-size 1 --gradient-accumulation 4 --lora-rank 32
```

### Motion-following Wan v2

The v1 Wan adapter keeps the arm and scene visually stable, but its difficult
rollouts can lag behind fast arm trajectories. The v2 continuation retains the
same official `5 RGB + 4 history abs14 + 8 future abs14 -> 8 RGB` contract and
loads every v1 LoRA/absolute-action weight. It adds a zero-initialized residual
trajectory branch: each of the 13 slots receives its physical action delta and
its displacement from the fourth (latest) history action. This lets v2 learn
motion direction without discarding v1's absolute-pose representation.

Run the tiny action-path test, then the real-model smoke test before formal
training:

```bash
export PYTHONPATH="$PWD/pipeline"
conda run -n go1 python pipeline/scripts/test_track2_wan_action_conditioning.py \
  --trajectory-conditioned
conda run -n go1 python pipeline/scripts/smoke_track2_wan_real.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1 \
  --base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/smoke/track2-wan-trajectory-v2-real \
  --trajectory-conditioned --lora-rank 32
```

Start v2 in a separate output directory. The first 1,000 AdamW updates train
only the new trajectory residual; the inherited v1 visual/absolute-action
parameters then unfreeze at a ten-times lower learning rate. High-motion
training windows receive extra sampling weight, while all validation windows
use fixed flow noise for stable checkpoint selection:

```bash
conda run -n go1 python pipeline/scripts/train_track2_wan.py \
  --dataset-root artifacts/wan/adjust_bottle_rlinf_npy_v1 \
  --base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/checkpoints/track2-wan-adjust-bottle-trajectory-v2 \
  --latent-cache artifacts/wan/adjust_bottle_wan22_latents_v1 \
  --init-checkpoint artifacts/checkpoints/track2-wan-adjust-bottle-v1/track2_wan_lora.pt \
  --trajectory-conditioned --trajectory-hidden-dim 256 \
  --trajectory-warmup-steps 1000 --high-motion-oversample 1.0 \
  --learning-rate 2e-4 --inherited-learning-rate 2e-5 \
  --steps 30000 --batch-size 1 --gradient-accumulation 4 --lora-rank 32
```

`--init-checkpoint` deliberately starts a fresh optimizer, whereas `--resume`
continues an interrupted v2 run and restores its optimizer. Do not use both.
As with v1, only the 682 episode-held-out windows decide whether the model is
eligible for RL/RLinf; run full pixel evaluation before serving it.

The compact checkpoint is `track2_wan_lora.pt`; it references the immutable
base model and contains only the action conditioner, LoRA weights, optimizer,
and action normalization. Evaluate every one of the 682 held-out windows and
export a comparison GIF:

```bash
conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend track2-wan --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/track2-wan-adjust-bottle-v1 \
  --wan-base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --split validation --samples 682 --require-pass \
  --output artifacts/evaluations/track2_wan_adjust_bottle_validation_all.json

conda run -n go1 python pipeline/scripts/export_prediction_gif.py \
  --backend track2-wan --window artifacts/adjust_bottle_windows_full/episode16_00025.npz \
  --checkpoint-dir artifacts/checkpoints/track2-wan-adjust-bottle-v1 \
  --wan-base-model artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
  --output artifacts/visualizations/track2_wan_adjust_bottle_validation.gif
```

The native Wan backend is available through the same API after it passes the
full strict MAE gate:

```bash
WAM_BACKEND=track2-wan \
WAM_CHECKPOINT_DIR=artifacts/checkpoints/track2-wan-adjust-bottle-v1 \
WAM_WAN_BASE_MODEL=artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
WAM_BEARER_TOKEN=local-dev-token \
conda run -n go1 python pipeline/scripts/serve.py
```

### Wan + Direct Flow candidate

The Wan trajectory v2 model preserves arm and scene appearance, while the
direct-flow U-Net follows short-horizon pixel motion more accurately. A fixed
RGB ensemble selected on a 64-window pilot uses `25%` Wan and `75%` direct
flow; it reduced pilot MAE from `9.428` (Wan-45) to `7.910` and high-motion
MAE from `16.834` to `14.045`. The weight is now fixed: the full validation
does not search it again.

Create the self-contained, SHA256-verified package once, then use the
resume-safe evaluator. Each completed rollout is persisted before the next
one, so an opportunistic GPU worker can yield without losing completed work:

```bash
conda run -n go1 bash pipeline/scripts/run_track2_wan_flow_ensemble_full.sh

# Read durable progress without loading either model or using GPU.
conda run -n go1 python pipeline/scripts/track2_wan_flow_ensemble_status.py
```

The evaluation uses all 682 held-out windows, writes the exact package digest,
exports the three worst GIFs, and keeps a resumable cache under
`artifacts/evaluations/`. After the final report is written, the entrypoint
automatically verifies full coverage, package/base identity, and every-frame
MAE before writing a strict acceptance record. It remains ineligible for RLinf
unless that gate passes. The pilot is still far above the requested `1.0 / 255`
threshold, so this is validation evidence rather than permission to begin RLinf.

Serve the packaged ensemble through the same Track 2 API:

```bash
WAM_BACKEND=wan-flow-ensemble \
WAM_CHECKPOINT_DIR=artifacts/checkpoints/track2-wan-direct-flow-ensemble-v2 \
WAM_WAN_BASE_MODEL=artifacts/upstream/Wan2.2-TI2V-5B-Diffusers-action-only \
WAM_BEARER_TOKEN=local-dev-token \
conda run -n go1 python pipeline/scripts/serve.py
```

### Recursive Flow rollout candidate

The direct-flow model predicts all eight horizons from the last observed
frame. Its held-out error grows sharply with horizon on fast arm and bottle
motion. The recursive candidate reuses the exact Direct Flow parameter layout
for a strict warm start, but predicts one frame per future action and feeds its
own prediction back into the five-frame context. Training backpropagates each
horizon independently to fit a 32-GB GPU while preserving the accumulated
self-rollout state seen by later horizons. The direct RAFT targets are not used
here because they describe last-observation-to-future flow, not the required
previous-frame-to-next-frame recursive flow.

Run the paired 64-window pilot through the idle-GPU worker:

```bash
nohup bash pipeline/scripts/opportunistic_gpu_worker.sh \
  --log artifacts/logs/recursive_flow_pilot_v1_worker.log \
  --idle-seconds 15 --poll-seconds 2 --busy-sm 10 \
  --max-other-memory-mib 512 --min-free-mib 28000 --yield-on-other -- \
  bash pipeline/scripts/run_recursive_flow_pilot.sh >/dev/null 2>&1 &
```

The trainer atomically checkpoints model, optimizer, scheduler, and completed
step every 25 updates. If another CUDA process appears, the worker terminates
the pilot only after this state is written, releases all GPU memory, and later
restarts with `--resume`. The pilot first measures the frozen Direct Flow warm
start on the same windows, then trains 500 updates, evaluates the best recursive
checkpoint, and writes a paired bootstrap comparison. Do not run a full sweep
or RLinf unless that comparison improves both overall and high-motion MAE.

## 7. Official RLinf Wan RobotWin baseline

This is separate from the local Diffusers LoRA. It accepts only the
organizer/RLinf-published RobotWin checkpoint layout:

```text
RLinf-Wan-RobotWin-AdjustBottle/
  dit_model.safetensors
  Wan2.2_VAE.pth
  dataset/*.npy
```

The backend mirrors native `WanEnv`: `input_image=o0`, `input_image4=o1..o4`,
and `action=[zero anchor, 4 history abs14, 8 future abs14]`. The 13-frame
native output is sliced to frames 5--12. Baseline sampling is fixed to 5
denoising steps and `cfg_scale=1.0`.

The Track 2 package and public RLinf configuration do not give a checkpoint
URL. Do not substitute a generic Wan or local LoRA checkpoint. Once the exact
organizer source is available, run:

```bash
export WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT=/absolute/path/RLinf-Wan-RobotWin-AdjustBottle
export WAN_ROBOTWIN_ADJUST_BOTTLE_SOURCE='exact organizer model URL or revision'
export PYTHONPATH="$PWD/pipeline"

conda run -n go1 bash pipeline/scripts/bootstrap_official_diffsynth_runtime.sh
conda run -n go1 python pipeline/scripts/check_official_rlinf_wan_runtime.py \
  --diffsynth-root artifacts/upstream/diffsynth-studio-runtime-local
conda run -n go1 bash pipeline/scripts/run_official_rlinf_wan_baseline.sh
```

This writes a comparison GIF, API self-test, and two-round local MBRL
data-contract smoke to `artifacts/evaluations/official_rlinf_wan_adjust_bottle/`.
It is neither RL training nor an organizer hidden-test score. Use the same
backend for held-out evaluation:

```bash
conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend official-rlinf-wan \
  --checkpoint-dir "$WAN_ROBOTWIN_ADJUST_BOTTLE_CKPT" \
  --official-diffsynth-root artifacts/upstream/diffsynth-studio-runtime-local \
  --official-wan-inference-steps 5 \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --split validation --samples 682 \
  --output artifacts/evaluations/official_rlinf_wan_validation_all.json
```

## 8. Public RLinf closed loop

Do not run this section for a candidate that has not passed the strict
episode-held-out world-model acceptance test above. The autoregressive result
described below is retained only as an integration reference.

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
export WAM_MODEL_VERSION=multisource-flow-unet-track2-formal-v1
export WAM_BACKEND=multisource-flow-unet
export WAM_CHECKPOINT_DIR=$PWD/artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best
export WAM_STRICT_EVALUATION=$PWD/artifacts/evaluations/multisource_flow_unet_formal_v1_validation_all.json
```

```bash
PYTHONPATH="$PWD/pipeline" conda run -n go1 python \
  pipeline/scripts/make_public_rlinf_reset_dataset.py \
  --input artifacts/datasets/aloha-agilex_clean_50 \
  --output artifacts/rlinf_public_reset_adjust_bottle
export RLINF_RESET_DATASET=$PWD/artifacts/rlinf_public_reset_adjust_bottle
```

3. Start the accepted candidate API and bridge, then launch the public RLinf
   recipe. The script checks the full validation report, matching checkpoint
   hash, published policy/reward, RLinf source, reset data, and bridge health.

```bash
WAM_BACKEND=multisource-flow-unet \
WAM_CHECKPOINT_DIR=artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best \
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
  --bridge-url http://127.0.0.1:18081 --rounds 2 \
  --strict-evaluation $WAM_STRICT_EVALUATION \
  --world-model-checkpoint $WAM_CHECKPOINT_DIR --world-model-backend $WAM_BACKEND
```

It runs `official pi05 -> Track-2 API -> official reward` for two chunks and
reports finite per-chunk reward statistics. It remains a public-data local
smoke test, not the organizer’s hidden held-out result.

This produces a *public local RLinf/RoboTwin reproduction*.  It is not the
organizer's hidden held-out final score, which cannot be computed locally.

The included configuration enables CPU offload and CPU weight transport for a
single 32-GB GPU. The historical autoregressive baseline passed the API contract,
the two-round `pi05 -> Track-2 API -> official T5 reward` check, and one actual
GRPO update. Its `global_step_1` checkpoint is under
`artifacts/rlinf_track2_autoregressive_unet_rollout8_grpo_step1/autoregressive_unet_rollout8_grpo_step1/checkpoints/global_step_1/`.
This only verifies the execution path. The one-step relative reward was very
small, so it is not evidence of useful policy improvement; longer public
closed-loop runs are required before comparing candidate quality.

## 9. Earlier 128px development baseline

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

## 10. Boundary of the MBRL smoke test

`run_mbrl_smoke.py` verifies the integration loop: policy -> 8 actions -> world model -> local finite proxy reward -> policy update. Its image-distance reward is not the organizer's reward checkpoint. Synthetic protocol testing is unrestricted; every non-synthetic run requires the exact full-validation report and matching checkpoint hash. The official final held-out evaluator is not public.
