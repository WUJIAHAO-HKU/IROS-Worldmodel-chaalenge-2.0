# Track 2 RL training status

## Released world model

- Git branch: `jiahao_master`
- v15 release commit: `6c8a770ce9e384a0587474f638484dc198ce7ecb`
- Packaged model: `artifacts/releases/track2_v15_best`
- Package size: 205,368,963 bytes (35 files, each covered by the release manifest)
- Runtime chain: v8 parent -> v10.1 flow/risk routing -> v14.1 retrieval -> v15 texture/gripper experts

## First conservative GRPO run

The first 20-update policy run completed successfully against the v15 HTTP
world-model environment and the published RoboTwin reward model. It started
from the official Pi0.5 `adjust_bottle` checkpoint, not from the discarded
one-step high-learning-rate trial.

| Setting | Value |
|---|---:|
| Updates | 20 |
| Trajectories/update | 2 |
| Predicted frames/trajectory | 8 |
| Actor learning rate | 5e-7 |
| PPO low/high clip | 0.1 / 0.1 |
| Gradient clip | 0.5 |
| Runtime | 5m 59s including final checkpoint write |
| Mean warm step time | 15.45 s (median) |

Server checkpoint:

```text
/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/rl_runs/v15_grpo_conservative20/track2_robotwin_adjust_bottle_http_grpo_openpi_pi05/checkpoints/global_step_20
```

| Checkpoint file | Bytes | SHA-256 |
|---|---:|---|
| `actor/model_state_dict/full_weights.pt` | 8,529,316,588 | `38cb95f9a18538c26ae6687b6e4ccb9a3589df649997ee20ea9eba59c18b9e9b` |
| `actor/dcp_checkpoint/__0_0.distcp` | 11,785,292,153 | recorded by DCP metadata |

Training diagnostics over all 20 updates:

| Metric | First | Last | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| env reward | 1.037e-6 | 1.677e-6 | 2.037e-6 | 1.579e-6 | 9.010e-7 | 8.891e-6 |
| env return | 8.293e-6 | 1.342e-5 | 1.630e-5 | 1.263e-5 | 7.208e-6 | 7.113e-5 |
| actor policy loss | 0.00456 | 0.06192 | 0.03385 | 0.02207 | 0.00059 | 0.10173 |
| actor grad norm | 0.1607 | 0.9386 | 0.7413 | 0.4142 | 0.0237 | 3.2503 |
| actor ratio | 0.9574 | 0.7636 | 0.8811 | 0.9475 | 0.5456 | 1.1659 |
| displayed approx KL | 2.144 | 17.043 | 27.092 | 5.560 | 1.346 | 142.229 |

The current RLinf token-level KL and clip diagnostics aggregate 14 action
dimensions against an unexpanded token-mask count, so the displayed values are
not directly comparable to conventional scalar-action PPO KL thresholds. They
also contain isolated logprob-recomputation outliers. The lower learning rate
prevents those batches from causing a large optimizer step; nevertheless this
is a short integration run, not a converged policy.

## Paired closed-loop evaluation

The official base policy and step-20 policy were evaluated with identical
seeds on six public resets: three explicit right-arm tasks (`episode0`,
`episode20`, `episode37`) and three explicit left-arm tasks (`episode1`,
`episode5`, `episode16`). Each episode used two consecutive 8-frame closed-loop
rounds. Scores are published-reward/public-data diagnostics, not organizer
hidden-set results.

| Slice | Official Pi0.5 | Step-20 | Relative change |
|---|---:|---:|---:|
| All, 12 round means | 0.11728757 | 0.11740078 | +0.0965% |
| Right arm, 6 round means | 0.13783399 | 0.13818259 | +0.2529% |
| Left arm, 6 round means | 0.09674114 | 0.09661896 | -0.1263% |
| First rounds | 0.09343507 | 0.09331985 | -0.1233% |
| Second rounds | 0.14114006 | 0.14148171 | +0.2421% |

The candidate wins 6 of 12 paired rounds. This is essentially neutral and is
not enough evidence to promote step-20 over the official Pi0.5 baseline. The
checkpoint is retained as a verified resume point; official Pi0.5 remains the
deployment policy baseline until a longer, validation-gated run improves both
arms consistently.
