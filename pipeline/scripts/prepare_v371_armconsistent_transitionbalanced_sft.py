#!/usr/bin/env python3
"""Pre-register the sole v371 SFT candidate from the frozen v169 policy."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from datetime import datetime, timezone


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def mentions_arm(task: str, arm: str) -> bool:
    return re.search(rf"\b{re.escape(arm)}\s+arm\b", task, flags=re.IGNORECASE) is not None


(
    output, checkpoint, dataset_audit, conversion, historical_evidence_path,
    actor_source, runner,
) = map(pathlib.Path, sys.argv[1:8])
candidate = sys.argv[8]
if output.exists():
    raise FileExistsError(output)

dataset = json.loads(dataset_audit.read_text())
converted = json.loads(conversion.read_text())
historical = json.loads(historical_evidence_path.read_text())
split_path = pathlib.Path(converted["split"])
split = json.loads(split_path.read_text())
v370_registration_path = dataset_audit.parent / "preregistration.json"
v370_registration = json.loads(v370_registration_path.read_text())
original_summary_path = pathlib.Path(
    "/root/autodl-tmp/iros_v15_rl_probability_audit_v2/artifacts/lerobot/"
    "adjust_bottle_train40/conversion_summary.json"
)
original = json.loads(original_summary_path.read_text())
openpi_source = actor_source.parents[2] / "models" / "embodiment" / "openpi" / "openpi_action_model.py"

assert dataset["accepted"] and dataset["source_public_only"]
assert not dataset["reserved_or_hidden_evaluation_access"]
assert not dataset["real_competition_submission"]
assert dataset["dataset_episodes"] == converted["dataset_episodes"] == 85
assert dataset["frames"] == converted["total_frames"] == 6818
assert dataset["left_frames"] == converted["left_frames"] == 3638
assert dataset["right_frames"] == converted["right_frames"] == 3180
assert converted["left_dataset_episodes"] == 25
assert converted["right_dataset_episodes"] == 60
assert converted["forbidden_episodes_read"] == []
assert v370_registration["dataset"] == str(conversion.parent.resolve())
assert "build_mixed_track2_sft_action_loss_weights" in actor_source.read_text()
assert "reduce_sft_action_loss" in openpi_source.read_text()

arm_by_episode = {int(key): value for key, value in split["arm_by_episode"].items()}
old_prompt_status = {"explicit_correct": 0, "explicit_conflict": 0, "implicit": 0}
old_prompt_records = []
for item in original["episode_summaries"]:
    episode = int(item["source_episode"])
    arm = arm_by_episode[episode]
    other = "left" if arm == "right" else "right"
    task = str(item["task"])
    if mentions_arm(task, other):
        status = "explicit_conflict"
    elif mentions_arm(task, arm):
        status = "explicit_correct"
    else:
        status = "implicit"
    old_prompt_status[status] += 1
    old_prompt_records.append({"episode": episode, "action_arm": arm, "task": task, "status": status})
assert old_prompt_status == {"explicit_correct": 17, "explicit_conflict": 1, "implicit": 22}

historical_variant = historical["candidate_variant"]
historical_metrics = historical["auxiliary_read_only_metrics"][historical_variant]
assert historical_metrics["count"] == 128
assert historical_metrics["right"]["count"] == 66
assert historical_metrics["right"]["grasp_completions"] == 60
assert historical_metrics["right"]["successes"] == 0

ordinary_batches = 8
extra_batches = 1697
batch_size = 4
record = {
    "format": "strict-track2-v371-armconsistent-transitionbalanced-sft-preregistration-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "candidate": candidate,
    "hypothesis": (
        "The frozen v169 policy already grasps 60/66 right-side scenes but completes none. "
        "Prior SFT reused noisy arm language (only 17/40 prompts explicit-correct, one conflict) "
        "or over-weighted full/terminal frames. One clean epoch on explicit action-arm-consistent "
        "prompts, with real right close/lift/post-grasp transitions balanced against full left "
        "rehearsal, targets the missing transport behavior while preserving left performance."
    ),
    "rules": {
        "real_submission": False,
        "reserved_or_hidden_evaluation_access": False,
        "training_data": "frozen official public train40 only",
        "selection_data": "public batch00, then batch00+01 only if preregistered R0 gate passes",
        "single_candidate": True,
    },
    "initial_policy": {
        "role": "frozen v169 step5 selected baseline; no failed SFT/RL continuation",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
    },
    "dataset": {
        "path": str(conversion.parent.resolve()),
        "audit": str(dataset_audit.resolve()),
        "audit_sha256": sha256(dataset_audit),
        "conversion": str(conversion.resolve()),
        "conversion_sha256": sha256(conversion),
        "v370_preregistration": str(v370_registration_path.resolve()),
        "v370_preregistration_sha256": sha256(v370_registration_path),
        "split": str(split_path.resolve()),
        "split_sha256": sha256(split_path),
        "episodes": 85,
        "frames": 6818,
        "unique_left_sources": 25,
        "unique_right_sources": 15,
        "left_frames": 3638,
        "right_transition_frames": 3180,
        "right_source_repeats": 4,
        "source_public_only": True,
    },
    "corrected_prompt_evidence": {
        "original_conversion": str(original_summary_path),
        "original_conversion_sha256": sha256(original_summary_path),
        "counts": old_prompt_status,
        "records": old_prompt_records,
    },
    "historical_public_diagnostic": {
        "path": str(historical_evidence_path),
        "sha256": sha256(historical_evidence_path),
        "variant": historical_variant,
        "right_episodes": 66,
        "right_grasps": 60,
        "right_successes": 0,
        "use": "hypothesis motivation only; not a v371 selection measurement",
    },
    "training": {
        "rl_global_steps": 1,
        "episode_steps": 8,
        "group_size": 4,
        "total_envs": 8,
        "actor_seed": 1535,
        "actor_lr": 1e-6,
        "kl_beta_to_initial_policy": 0.2,
        "sft_batch_size": batch_size,
        "ordinary_sft_batches": ordinary_batches,
        "extra_sft_batches": extra_batches,
        "total_sft_batches": ordinary_batches + extra_batches,
        "sampled_sft_frames": (ordinary_batches + extra_batches) * batch_size,
        "nominal_dataset_epochs": (ordinary_batches + extra_batches) * batch_size / 6818,
        "use_action_chunk_loss": True,
        "supervised_temporal_horizon": 8,
        "model_prediction_horizon": 50,
        "ordinary_sft_loss_weight": 0.3,
        "extra_sft_loss_weight": 1.0,
        "mixed_arm_structured_action_loss": True,
        "active_joint_weight": 1.0,
        "active_gripper_weight": 3.0,
        "inactive_keep_weight": 2.0,
        "gripper_activity_weight": 0.25,
        "physical_action_dimensions": 14,
        "padded_model_action_dimensions_ignored": True,
    },
    "public_gate": {
        "stage_r0_batch": "batch_00 (4 left, 12 right)",
        "thresholds": {"right_success": 4, "left_success": 3, "total_success": 7, "right_grasp": 10},
        "stage_r1_batches": ["batch_00", "batch_01"],
        "stage_r1_thresholds": {"right_success": 10, "left_success": 8, "total_success": 18, "grasp_once": 27},
        "on_reject": "stop; do not access reserved/final 128",
    },
    "resource_guard": {
        "cpu_affinity": "0-21 of 25 logical CPUs",
        "reserved_control_cpus": "22-24",
        "dataloader_workers": 0,
        "omp_threads": 8,
        "terminal_model_only_checkpoint": True,
    },
    "implementation": {
        "actor_source": str(actor_source),
        "actor_source_sha256": sha256(actor_source),
        "openpi_source": str(openpi_source),
        "openpi_source_sha256": sha256(openpi_source),
        "runner": str(runner),
        "runner_sha256": sha256(runner),
    },
    "reserved_final128_access": False,
    "real_competition_submission": False,
}
if candidate.startswith("v372r_"):
    failed_run = output.parents[2] / "runs" / (
        "v371_v169_armconsistent_transitionbalanced_chunk8_sft1697_b4_"
        "lr1e6_seed1535_20260822"
    )
    failed_log = failed_run / "launcher.log"
    failed_partial = failed_run / (
        "wan_robotwin_adjust_bottle_http_full_grpo_openpi_pi05/checkpoints/"
        "global_step_1/actor/model_state_dict/full_weights.pt"
    )
    failed_reference = failed_run / "reference_policy_state/rank_0.pt"
    failed_text = failed_log.read_text(errors="replace")
    assert "PytorchStreamWriter failed writing file" in failed_text
    assert "[EXTRA_SFT] update=1697/1697" in failed_text
    assert not failed_partial.exists() and not failed_reference.exists()
    save_source = actor_source.read_text()
    assert save_source.index("os.remove(self.reference_state_path)") < save_source.index(
        "model_state_dict = self.get_model_state_dict"
    )
    record["checkpoint_recovery"] = {
        "prior_failed_run": str(failed_run),
        "prior_failed_log": str(failed_log),
        "prior_failed_log_sha256": sha256(failed_log),
        "failure": "training completed 1697/1697; terminal torch.save filled disk",
        "cleanup": "removed only the incomplete checkpoint and actor-owned immutable reference copy",
        "fix": "delete actor-owned disk reference before terminal state-dict materialisation",
        "training_recipe_changed": False,
        "same_actor_seed": 1535,
        "same_data_and_hyperparameters": True,
        "new_model_route": False,
    }
output.parent.mkdir(parents=True, exist_ok=False)
output.write_text(json.dumps(record, indent=2) + "\n")
print("V371_ARMCONSISTENT_TRANSITIONBALANCED_SFT_PREREGISTERED")
