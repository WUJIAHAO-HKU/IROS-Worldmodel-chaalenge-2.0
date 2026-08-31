# 赛道 2 本地 Pipeline SOP

## 0. 固定契约

一次请求固定为：`5` 帧 RGB + `4` 个历史动作 + **策略给出的 `8` 个未来动作**，世界模型返回 `8` 帧预测 RGB。

```text
o0 o1 o2 o3 o4 + a0 a1 a2 a3 + u0 ... u7  ->  p0 ... p7
                     a3: o3 -> o4             u0: o4 -> p0
```

模型不生成动作、奖励或 policy。比赛端拥有 policy、奖励模型与 MBRL 更新；本地 MBRL 仅做接口闭环自测。

本 SOP 的多轮自测采用不重叠 `8` 步：下一轮输入为 `p3..p7` 和 `u4..u7`，策略再给下一组 `8` 个动作。官方公开规则没有规定 policy 必须逐步还是逐 chunk 重规划，勿将这个本地选择视为官方限制。

## 1. 资源与模型

| 项目 | 本地位置 | 用途 |
| --- | --- | --- |
| 官方 Adjust Bottle 数据 | `artifacts/datasets/aloha-agilex_clean_50/data/` | `50` 个 HDF5，`6638` 个可用 5+8 窗口 |
| RGB 字段 | `observation/head_camera/rgb` | JPEG RGB 帧 |
| 动作字段 | `joint_action/vector` | 对齐的 `[T,14]` 浮点动作 |
| 起点权重 | `artifacts/upstream/ivideogpt-bair-64-act-cond/` | MIT 的 BAIR 64x64 action-conditioned iVideoGPT |
| iVideoGPT 基线 | `iVideoGPT-64 + 14D action adapter/LoRA` | 已完成真实数据适配；作为 token 世界模型对照 |
| 旧基线 | `native-256px autoregressive U-Net` | 已完成接口/闭环验证，但未通过当前严格预测门槛 |
| 正式候选 | `multi-source flow U-Net` | 5 路 RAFT warp 融合；训练集标签和正式训练已排队 |

公开 BAIR 权重原始配置是 `1` 帧上下文、`4D` 动作，不能直接提交。本实现把条件长度扩至 `5`，并重新训练其 `14 -> 768` 动作投影层；图像输入严格使用上游约定的 `[0,1]` RGB。iVideoGPT 仍是对照基线：即使 LoRA 训练 `5000` step，公开验证上的 256px MAE 仍为 `22.00`，差于复制最后帧的 `15.02`，不得提交。

官方完整代码固定在 `third_party/iVideoGPT`，提交为 `d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210`。该副本已经核验为非 shallow、非 partial clone（`65` commits、`290` blobs、`141` trees，`git fsck --full` 通过）。已直接使用官方 `inference/predict.py` 加载本地 BAIR 权重并生成 `artifacts/official_ivideogpt_bair_smoke/pred-samples-0.gif`。本项目训练和服务默认优先引用该完整副本。

## 2. 执行顺序

所有命令在项目根目录执行：

```bash
export PYTHONPATH="$PWD/pipeline"
```

### 当前有效候选（不再训练退化分支）

`Wan trajectory v2` 的画面结构较稳，`Direct Flow U-Net` 的短期运动跟随较好。
固定采用 `Wan 25% + Direct Flow 75%` RGB 融合：同一 64 个留出窗口上，平均 MAE
`9.428 -> 7.910 / 255`，高运动 MAE `16.834 -> 14.045 / 255`。权重已经固定，禁止在全量
验证集继续调参。`v3/v4 latent-action residual` 已经退化，停止训练和评估。

全量验证使用可恢复缓存；每完成一个窗口立即落盘，因此空闲 GPU 被其他任务占用后可以安全让出：

```bash
conda run -n go1 bash pipeline/scripts/run_track2_wan_flow_ensemble_full.sh

# 不加载模型、不占 GPU，读取断点进度。
conda run -n go1 python pipeline/scripts/track2_wan_flow_ensemble_status.py
```

它评估全部 `682` 个隔离验证窗口，并导出 3 个最差 GIF。评估完成后会自动复核窗口覆盖、权重/
Wan 基座身份及逐帧 MAE，只有每个窗口、每一张未来帧均低于 `1.0 / 255` 才写出有效验收记录并进入
RLinf；当前候选尚未通过该硬门槛，不能提前声称通过。

### 2.1 验证官方数据

```bash
conda run -n go1 python pipeline/scripts/verify_robotwin_data.py \
  --input artifacts/datasets/aloha-agilex_clean_50/data
```

必须输出 `episodes=50`、`joint_action/vector[T,14]`；失败时停止，不训练。

### 2.2 生成全量训练窗口

```bash
conda run -n go1 python pipeline/scripts/adapt_robotwin.py \
  --input artifacts/datasets/aloha-agilex_clean_50/data \
  --output artifacts/adjust_bottle_windows_full
```

每个 NPZ 为：

```text
context_frames  [5,256,256,3] uint8
history_actions [4,14] float32
future_actions  [8,14] float32
target_frames   [8,256,256,3] uint8
```

### 2.3 先做真实模型 smoke（已跑通的最小配置）

```bash
conda run -n go1 python pipeline/scripts/train_ivideogpt64.py \
  --windows artifacts/adjust_bottle_windows \
  --upstream-checkpoint artifacts/upstream/ivideogpt-bair-64-act-cond \
  --output artifacts/checkpoints/ivideogpt64-track2-smoke \
  --steps 1 --batch-size 1
```

成功条件：产生 `track2_ivideogpt_state.pt`，并打印有限的 loss。此权重已经实测可对官方窗口生成 `[8,256,256,3] uint8`，仅用于管线验证。

### 2.3a 导出并检查预测 GIF

```bash
conda run -n go1 python pipeline/scripts/export_prediction_gif.py \
  --window artifacts/adjust_bottle_windows/episode0_00000.npz \
  --checkpoint-dir artifacts/checkpoints/ivideogpt64-track2-smoke \
  --output artifacts/visualizations/adjust_bottle_smoke.gif
```

GIF 前 `5` 帧是输入上下文；后 `8` 帧并列展示未来动作编号、预测帧和对应真值帧。1-step smoke 的画质只用于确认数据和时序，没有评测价值。

### 2.4 训练开发权重

先按完整 episode 固定划分，禁止随机按窗口划分：

```bash
conda run -n go1 python pipeline/scripts/make_episode_split.py \
  --windows artifacts/adjust_bottle_windows_full \
  --output artifacts/splits/adjust_bottle_50episodes_full.json
```

固定结果为训练 `40` 个 episode / 验证 `5` 个 episode / 本地测试 `5` 个 episode，窗口不会跨集合泄漏。先训练 `200` step 检查 loss 和 GIF：

```bash
conda run -n go1 python pipeline/scripts/train_ivideogpt64.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --upstream-checkpoint artifacts/upstream/ivideogpt-bair-64-act-cond \
  --output artifacts/checkpoints/ivideogpt64-track2-dev \
  --steps 200 --batch-size 1 --learning-rate 1e-3 \
  --validation-interval 25 --validation-batches 64
```

每次验证均从 `5` 个验证 episode 均匀抽样，保存 `checkpoints/checkpoint_step_*`；`best/` 和输出目录根部始终是最低 validation token loss 的可运行权重。token loss 只用于选择开发权重，不是视频质量或官方得分。完成 200 step 后，用 `best/` 导出验证 GIF；确认趋势后再增加步数或进入微调阶段。

### 2.5 历史 256px 基线

```bash
conda run -n go1 python pipeline/scripts/train_autoregressive_unet.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --output artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1 \
  --init-autoregressive-checkpoint artifacts/checkpoints/autoregressive-unet-track2-native256-v2/best \
  --steps 5000 --batch-size 2 --learning-rate 1e-5 --train-rollout-steps 8 \
  --validation-interval 500 --validation-batches 32
```

冻结权重：`artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1/best`。
它只使用训练 episode `0,1,3,...,49` 的 `5289` 个窗口；验证 `5,7,16,18,39`
的 `682` 个窗口和 local-test `2,6,9,22,48` 的 `667` 个窗口均不参与训练。

全量 8 帧开放环 MAE（0--255 RGB）：验证 `7.57`，local-test `6.27`；复制最后一帧
基线分别为 `15.33`、`15.11`。这是公开数据上的本地指标，不是官方隐藏集得分。

评估视频（每个未来帧均为“预测 | 真值”）：

```text
验证 episode 16: artifacts/visualizations/autoregressive_unet_rollout8_track2_native256_validation_episode16_00025.gif
本地测试 episode 2: artifacts/visualizations/autoregressive_unet_rollout8_track2_native256_localtest_episode2_00025.gif
```

### 2.5a 正式世界模型训练门槛（当前阶段）

在进入任何 RLinf 任务前，最终世界模型必须在隔离验证集 `682` 个窗口中，**每个窗口的
每一张未来帧**都满足 RGB MAE `< 2.5 / 255`。均值达标、少量样本达标或训练集效果均不算通过。

当前选择是多源光流融合模型：每个未来帧对 `5` 个上下文帧分别预测反向 RAFT 光流，重采样后
由可学习权重融合，并用 RGB 残差修正新显露区域。融合权重由训练集 RAFT 重建误差监督，能在遮挡
处选择实际可见的历史来源。它保持官方固定输入输出契约，不使用验证或本地测试 episode 的教师标签。
此前单源光流 v2 的验证平均 MAE 为 `8.03 / 255`，未通过，不能提交。

先仅用训练 episode 生成标签（约 `13GB`）：

```bash
conda run -n go1 python pipeline/scripts/precompute_raft_flow_targets.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --output artifacts/flow_targets/raft_small_train_multisource_v1_128 \
  --source-mode all-context --flow-resolution 128 --batch-size 1 --pair-batch-size 8
```

再开始正式训练。默认每 `1000` step 全量评估 `682` 个验证窗口；候选按“最差未来帧
MAE，再平均 MAE”保存，不能以平均值掩盖任一失败帧（默认只保留该最佳 checkpoint）：

```bash
conda run -n go1 python pipeline/scripts/train_multisource_flow_unet.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --flow-targets artifacts/flow_targets/raft_small_train_multisource_v1_128 \
  --flow-resolution 128 --output artifacts/checkpoints/multisource-flow-unet-track2-formal-v1 \
  --steps 60000 --batch-size 1 --base-channels 64
```

若 GPU 同时被其他任务使用，标签工作器完成后可自动等待空档启动同一训练命令：

```bash
nohup pipeline/scripts/wait_for_multisource_training.sh \
  --flow-targets artifacts/flow_targets/raft_small_train_multisource_v1_128 \
  --log artifacts/logs/multisource_train_opportunistic.log -- \
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/train_multisource_flow_unet.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --flow-targets artifacts/flow_targets/raft_small_train_multisource_v1_128 \
  --flow-resolution 128 --output artifacts/checkpoints/multisource-flow-unet-track2-formal-v1 \
  --steps 60000 --batch-size 1 --base-channels 64 --resume \
  > artifacts/logs/multisource_train_opportunistic.watchdog.log 2>&1 &
```

训练结束后必须全量验收，失败则只记录结果并继续模型迭代，不导出为候选或启动 RL：

```bash
conda run -n go1 python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend multisource-flow-unet --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best \
  --split validation --samples 682 --accept-mae 2.5 --require-pass \
  --output artifacts/evaluations/multisource_flow_unet_formal_v1_validation_all.json
```

若未通过，JSON 同时记录最差窗口/预测帧，并自动导出前三个失败窗口的对比 GIF；未通过不得进入 RL。

也可让验收只在完整训练结束并且 GPU 空闲时自动运行：

```bash
nohup pipeline/scripts/wait_for_training_completion.sh \
  --completion artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/training_complete.json \
  --expected-steps 60000 --log artifacts/logs/multisource_eval_opportunistic.log -- \
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/evaluate_ivideogpt64.py \
  --backend multisource-flow-unet --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --checkpoint-dir artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best \
  --split validation --samples 682 --accept-mae 2.5 --require-pass \
  --output artifacts/evaluations/multisource_flow_unet_formal_v1_validation_all.json \
  > artifacts/logs/multisource_eval_opportunistic.watchdog.log 2>&1 &
```

GPU 被其他任务使用时，可用机会式工作器在其持续空闲时生成标签，并在检测到其他 GPU 计算恢复后
暂停自身；它不会用无效张量占用显存：

```bash
nohup pipeline/scripts/opportunistic_gpu_worker.sh \
  --log artifacts/logs/multisource_raft_opportunistic.log --idle-seconds 30 --busy-sm 10 -- \
  /root/miniconda3/envs/go1/bin/python pipeline/scripts/precompute_raft_flow_targets.py \
  --windows artifacts/adjust_bottle_windows_full \
  --split-manifest artifacts/splits/adjust_bottle_50episodes_full.json \
  --output artifacts/flow_targets/raft_small_train_multisource_v1_128 \
  --source-mode all-context --flow-resolution 128 --batch-size 1 --pair-batch-size 8 \
  > artifacts/logs/multisource_raft_opportunistic.watchdog.log 2>&1 &
```

### 2.6 模型与多轮 rollout 自测

```bash
conda run -n go1 python pipeline/scripts/rollout_smoke.py \
  --window artifacts/adjust_bottle_windows_full/episode5_00000.npz \
  --backend autoregressive-unet \
  --checkpoint-dir artifacts/checkpoints/autoregressive-unet-track2-rollout8-v1/best
```

此处只保留 rollout 契约检查。非 synthetic 的 MBRL/RLinf 均由严格验收门禁拦截，必须在 2.5a 通过后执行。

### 2.7 官方 HTTP API 自测

终端 A：

```bash
WAM_BACKEND=multisource-flow-unet \
WAM_CHECKPOINT_DIR=artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best \
WAM_MODEL_VERSION=multisource-flow-unet-track2-formal-v1 \
WAM_BEARER_TOKEN=local-dev-token \
WAM_PORT=8001 \
conda run -n go1 python pipeline/scripts/serve.py
```

终端 B：

```bash
conda run -n go1 python pipeline/scripts/contract_test.py \
  --base-url http://127.0.0.1:8001 --token local-dev-token \
  --model-version multisource-flow-unet-track2-formal-v1
```

通过标准：`health`、`capabilities`、单条/8 条 batch、同请求确定性重试、request ID 冲突、错误尺寸、错误 profile、鉴权全部通过。完整副本上的 smoke 权重已经通过该测试；提交前固定 `model_version`，再以最终权重完整重跑。

### 2.8 官方 policy / reward / RLinf 闭环（严格验收通过后）

在 2.5a 的 `682 x 8` 严格验证通过前，禁止对任何候选启动 RLinf。本地正式闭环使用 `track2_http`：它不导入 Wan/DiffSynth，也不需要 RoboTwin
模拟器。`track2_http` 读取公开 reset，调用本地 bridge；bridge 将每组 `8x14`
动作转换为一次严格的 `/v1/predict`；环境用官方 T5 reward 计算每帧差分奖励。

执行顺序：

```text
公开 reset(5 帧/4 历史动作) -> pi05(8x14 动作) -> /v1/predict(8 帧)
-> 官方 T5 reward(8 个标量) -> GRPO update
```

```bash
conda run -n go1 python pipeline/scripts/make_public_rlinf_reset_dataset.py \
  --input artifacts/datasets/aloha-agilex_clean_50 \
  --output artifacts/rlinf_public_reset_adjust_bottle

export RLINF_RESET_DATASET=$PWD/artifacts/rlinf_public_reset_adjust_bottle
export WAM_API_URL=http://127.0.0.1:8001
export WAM_BEARER_TOKEN=local-dev-token
export WAM_MODEL_VERSION=multisource-flow-unet-track2-formal-v1
export WAM_BACKEND=multisource-flow-unet
export WAM_CHECKPOINT_DIR=$PWD/artifacts/checkpoints/multisource-flow-unet-track2-formal-v1/best
export WAM_STRICT_EVALUATION=$PWD/artifacts/evaluations/multisource_flow_unet_formal_v1_validation_all.json
export ROBOTWIN_REWARD_MODEL_PATH=$PWD/artifacts/official_resources/reward_model/adjust_bottle/full_weights.pt
export T5_MODEL_PATH=$PWD/artifacts/official_resources/reward_model/t5-base
```

先启动 API 与 bridge（已启动可跳过）：

```bash
WAM_BACKEND=$WAM_BACKEND \
WAM_CHECKPOINT_DIR=$WAM_CHECKPOINT_DIR \
WAM_MODEL_VERSION=$WAM_MODEL_VERSION WAM_BEARER_TOKEN=$WAM_BEARER_TOKEN WAM_PORT=8001 \
conda run -n go1 python pipeline/scripts/serve.py

PYTHONPATH="$PWD/pipeline" conda run -n go1 python -m wam_pipeline.rlinf_bridge.server \
  --world-model-url=$WAM_API_URL --token=$WAM_BEARER_TOKEN \
  --model-version=$WAM_MODEL_VERSION --port=18080
```

先做两轮组件检查，再运行 GRPO：

```bash
PYTHONPATH="$PWD/pipeline:$PWD/third_party/WorldArena-2.0/RL_env_benchmark:$PWD/third_party/openpi-rlinf-full/src" \
conda run -n go1 python pipeline/scripts/run_real_track2_closed_loop.py \
  --reset artifacts/rlinf_public_reset_adjust_bottle/episode0.npy \
  --bridge-url http://127.0.0.1:18080 --rounds 2 \
  --strict-evaluation $WAM_STRICT_EVALUATION \
  --world-model-checkpoint $WAM_CHECKPOINT_DIR --world-model-backend $WAM_BACKEND

conda run -n go1 bash pipeline/scripts/run_public_rlinf_track2.sh runner.max_steps=1
```

单张 32GB GPU 使用 CPU offload 与 CPU weight transport，避免 actor、rollout、T5
同时占用显存，并避开本容器禁止 CUDA IPC 的限制。已通过一轮真实 GRPO：生成、官方
reward、advantage、actor update 与 `global_step_1` checkpoint 均完成。结果只能称为
公开数据本地复现，不能宣称隐藏官方最终得分。

当前自回归候选已用独立 `8002/18081` 端口完成 API contract、两轮
`pi05 -> /v1/predict -> 官方 T5 reward`，以及 `runner.max_steps=1` 的真实
GRPO 更新；产物在
`artifacts/rlinf_track2_autoregressive_unet_rollout8_grpo_step1/autoregressive_unet_rollout8_grpo_step1/checkpoints/global_step_1/`。
本次单步相对 reward 很低，只说明当前权重能接入正式公开闭环，不能说明策略效果达标。

## 3. 不可省略的提交前检查

1. 最终模型输出严格为 PNG Base64 的 `8 x 256 x 256 x 3 uint8`，动作严格为 `float32[14]`。
2. 同一输入、模型、profile、seed 的解码 RGB 必须逐像素一致。
3. 先以最终真实模型后端通过 API 自测；`synthetic` 仅验证服务协议，不能提交。
4. 仅在 `682 x 8` 严格验证通过后冻结最终多源权重和 model version；之后才以官方 policy、reward、RLinf 的公开本地闭环 return 验收。
