#!/usr/bin/env python3
"""One public-S1 evaluation boundary with a mandatory bitexact zero-update proof."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import pickle
import random
import signal
import stat
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 1671
FORMAT = "strict-track2-v540-v539-public-s1-zero-update-gate-execution-receipt-v1"
ACTIVE_SOURCE_ROLE_ORDER = [
    "authority_design_contract", "authority_materializer", "public_s1_zero_update_preregistration",
    "public_s1_zero_update_manifest", "public_s1_evaluator", "public_s1_independent_auditor", "public_s1_launcher",
]
ACTIVE_SOURCE_PATHS = {
    "authority_design_contract": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_contract.json",
    "authority_materializer": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority.py",
    "public_s1_zero_update_preregistration": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_preregistration.json",
    "public_s1_zero_update_manifest": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_manifest.json",
    "public_s1_evaluator": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/evaluate_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py",
    "public_s1_independent_auditor": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/audit_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py",
    "public_s1_launcher": "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/launch_v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair.py",
}
EXPECTED_AUTHORIZATION = {
    "public_s1_zero_update_boundary_invocations_authorized": 1, "public_s1_zero_update_boundary_invocations_consumed": 0,
    "public_s1_evaluator_invocations_authorized": 1, "zero_update_gate_invocations_authorized": 1,
    "direct_evaluator_invocations_authorized": 0, "direct_auditor_invocations_authorized": 0,
    "actual_oof_execution_invocations_authorized": 0, "readonly_validator_invocations_authorized": 0,
    "phase_a_replay_invocations_authorized": 0, "training_invocations_authorized": 0,
    "backward_invocations_authorized": 0, "optimizer_step_invocations_authorized": 0,
    "optimizer_zero_grad_invocations_authorized": 0, "scheduler_step_invocations_authorized": 0,
    "parameter_update_invocations_authorized": 0, "buffer_update_invocations_authorized": 0,
    "model_write_invocations_authorized": 0, "cache_write_invocations_authorized": 0,
    "reward_read_invocations_authorized": 0, "hidden_private_input_read_invocations_authorized": 0,
    "dev_hidden_final_outcome_read_invocations_authorized": 0, "submission_invocations_authorized": 0,
    "retry_authorized": False, "public_s1_zero_update_gate_authorized": True, "training_authorized": False,
    "backward_authorized": False, "optimizer_authorized": False, "scheduler_authorized": False,
    "parameter_or_buffer_mutation_authorized": False, "model_or_cache_write_authorized": False,
    "reward_read_authorized": False, "hidden_private_or_final_input_authorized": False, "submission_authorized": False,
}
FORBIDDEN_FRAGMENTS = ("dev", "final", "hidden", "outcome", "private", "reward", "score", "submission")
EXACT_NPZ_KEYS = ["context_frames", "history_actions", "future_actions", "target_frames", "source", "start"]
PUBLIC_WINDOW_ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/adjust_bottle_windows_full")
PREREG_KEYS=set("fresh_authority_prep_root fresh_attempt_prep_root noncyclic_transport_deployment_record_required fresh_output_prep_root no_space_transport_required active_source_paths seed v534_actual_oof_durable_transition v535_failed_attempt_transition v539_poststage_owned_output_prep_failure_transition runtime_package_import_contract execution_manifest boundary_partition classification fresh_authority_root active_source_role_order zero_update_contract authorization fresh_attempt_root hidden_input_denial_contract format public_s1_output_contract fresh_output_root historical_training_authority_not_inherited public_s1_input_contract status authority_contract_path".split())
MANIFEST_KEYS=set("hidden_input_denial_contract qualification_contract model_and_source_input active_source_paths seed v534_actual_oof_durable_transition v535_failed_attempt_transition v539_poststage_owned_output_prep_failure_transition runtime_package_import_contract zero_update_contract public_s1_output_contract public_s1_input_contract boundary_partition status active_source_role_order format authorization".split())
CONTRACT_KEYS=set("v534_actual_oof_metrics_source v534_actual_oof_independent_audit_source source_role_order v534_actual_oof_events_source source_closure public_s1_zero_update_preregistration_source v534_actual_oof_attempt_intent_source active_source_paths seed v482_model_design_contract_source v169_release_manifest_source public_s1_zero_update_manifest_source active_source_records v524_qualification_audit_source v524_qualification_report_source v534_actual_oof_fold4_source v534_authority_receipt_source authorization execution_boundary current_absences_after_authority lineage public_s1_independent_auditor_source authority_materializer_source v474_public_s1_selection_source v534_actual_oof_source_manifest_source status format v524_cache_manifest_source v534_actual_oof_durable_transition v535_failed_attempt_transition v539_poststage_owned_output_prep_failure_transition runtime_package_import_contract v482_preregistration_source v534_actual_oof_fold0_source zero_update_contract active_source_role_order v474_public_s1_selection_receipt_source v169_runtime_source_source public_s1_launcher_source v169_library_manifest_source hidden_input_denial_contract public_s1_evaluator_source authority_receipt_contract historical_absences source_closure_sha256 v534_actual_oof_fold1_source v534_actual_oof_fold3_source v169_closure_source v524_qualification_terminal_source runtime_observation public_s1_output_contract v534_actual_oof_fold2_source public_s1_input_contract v534_actual_oof_execution_receipt_source source_aliases v539_failure_forensic_source v539_failed_authority_receipt_source v539_failed_transport_script_source v539_failed_transport_record_source".split())
SOURCE_ROLE_ORDER="authority_materializer public_s1_zero_update_preregistration public_s1_zero_update_manifest public_s1_evaluator public_s1_independent_auditor public_s1_launcher v539_failure_forensic v539_failed_authority_receipt v539_failed_transport_script v539_failed_transport_record v534_authority_receipt v534_actual_oof_attempt_intent v534_actual_oof_execution_receipt v534_actual_oof_independent_audit v534_actual_oof_events v534_actual_oof_metrics v534_actual_oof_fold0 v534_actual_oof_fold1 v534_actual_oof_fold2 v534_actual_oof_fold3 v534_actual_oof_fold4 v534_actual_oof_source_manifest v524_qualification_terminal v524_qualification_report v524_qualification_audit v524_cache_manifest v474_public_s1_selection v474_public_s1_selection_receipt v169_closure v169_runtime_source v169_release_manifest v169_library_manifest v482_preregistration v482_model_design_contract".split()
SOURCE_PATHS={
 "authority_materializer":ACTIVE_SOURCE_PATHS["authority_materializer"],"public_s1_zero_update_preregistration":ACTIVE_SOURCE_PATHS["public_s1_zero_update_preregistration"],"public_s1_zero_update_manifest":ACTIVE_SOURCE_PATHS["public_s1_zero_update_manifest"],"public_s1_evaluator":ACTIVE_SOURCE_PATHS["public_s1_evaluator"],"public_s1_independent_auditor":ACTIVE_SOURCE_PATHS["public_s1_independent_auditor"],"public_s1_launcher":ACTIVE_SOURCE_PATHS["public_s1_launcher"],
 "v539_failure_forensic":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v540_v539_public_s1_zero_update_poststage_owned_output_prep_absence_failure_forensic.json","v539_failed_authority_receipt":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_seed1670_20260828/authority_receipt.json","v539_failed_transport_script":"/root/v539_v535_public_s1_zero_update_runtime_package_repair_gate_once.sh","v539_failed_transport_record":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_runtime_package_repair_gate_transport_deployment_record.json",
 "v534_authority_receipt":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v534_v533_actual_oof_environment_repair_execution_authority_seed1667_20260827/authority_receipt.json","v534_actual_oof_attempt_intent":"/root/v534_v533_actual_oof_seed1667_20260827/attempt_intent.json","v534_actual_oof_execution_receipt":"/root/v534_v533_actual_oof_seed1667_20260827/execution_receipt.json","v534_actual_oof_independent_audit":"/root/v534_v533_actual_oof_seed1667_20260827/independent_audit.json","v534_actual_oof_events":"/root/v534_v533_actual_oof_seed1667_20260827/oof_call_events.ndjson","v534_actual_oof_metrics":"/root/v534_v533_actual_oof_seed1667_20260827/metrics.npz",
 "v534_actual_oof_fold0":"/root/v534_v533_actual_oof_seed1667_20260827/fold_0_receipt.json","v534_actual_oof_fold1":"/root/v534_v533_actual_oof_seed1667_20260827/fold_1_receipt.json","v534_actual_oof_fold2":"/root/v534_v533_actual_oof_seed1667_20260827/fold_2_receipt.json","v534_actual_oof_fold3":"/root/v534_v533_actual_oof_seed1667_20260827/fold_3_receipt.json","v534_actual_oof_fold4":"/root/v534_v533_actual_oof_seed1667_20260827/fold_4_receipt.json","v534_actual_oof_source_manifest":"/root/v534_v533_actual_oof_seed1667_20260827/source_manifest.json",
 "v524_qualification_terminal":"/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/terminal_receipt.json","v524_qualification_report":"/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/qualification_report.json","v524_qualification_audit":"/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/independent_final_audit.json","v524_cache_manifest":"/root/v524_v523_phase_a_cache_qualification_seed1660_20260826/process_a/cache/manifest.json",
 "v474_public_s1_selection":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v474_v473_parent_s1_seed1617_r6_20260824/action_only_selection.json","v474_public_s1_selection_receipt":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v474_v473_parent_s1_seed1617_r6_20260824/action_only_selection_receipt.json","v169_closure":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v474_v473_median4_parent_release_seed1618_20260824/v169_closure.json","v169_runtime_source":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/wam_pipeline/v482_temporal8_residual_runtime.py","v169_release_manifest":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v169_instruction_arm_routed_release/v169_arm_routed_manifest.json","v169_library_manifest":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/splits/adjust_bottle_50episodes_full.json","v482_preregistration":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json","v482_model_design_contract":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v482_temporal_film_residual_model_design_contract.json"}
SOURCE_ALIASES={role:f"{role}_source" for role in SOURCE_ROLE_ORDER}; SOURCE_ALIASES.update({"authority_design_contract":"authority_design_contract","v169_runtime_source":"v169_runtime_source_source"})
EXPECTED_LINEAGE={"canonical_v540_execution_not_performed":True,"fresh_public_s1_zero_update_authority_only":True,"hidden_private_final_inputs_denied":True,"historical_reward_authority_not_inherited":True,"phase_aware_poststage_owned_prep_repair_frozen":True,"public_episode_disjoint_action_only_selection_current":True,"training_or_update_authorized":False,"two_stage_authority_then_gate_only":True,"v524_qualification_exact20_readonly":True,"v534_actual_oof_passed_current":True,"v534_authority_immutable_consumed_zero":True,"v534_output_exact11_current":True,"v539_authority_immutable_consumed_zero":True,"v539_failed_lineage_retry_authorized":False,"v539_output_not_published":True,"zero_update_proof_mandatory":True}
RECEIPT_CONTRACT_KEYS=set("active_source_paths_exact active_source_role_order_exact authority_active_source_records_count authority_design_record_included authorization_exact check_key_set_sha256 check_keys checks_sha256 contract_active_source_records_count contract_active_source_records_exact contract_design_record_excluded_to_avoid_self_hash_cycle execution_boundary_exact format hidden_input_denial_contract_exact public_s1_input_contract_exact public_s1_output_contract_exact runtime_observation_exact status top_keys v534_actual_oof_durable_transition_exact v535_failed_attempt_transition_exact v539_poststage_owned_output_prep_failure_transition_exact runtime_package_import_contract_exact zero_update_contract_exact".split())
CHECK_KEYS="active_sources6_current all_update_training_hidden_unsafe_disabled authority_active_records7 authorization_exact_strict contract_active_records6 contract_schema execution_boundary_exact execution_pids_empty fresh_roots6_absent gpu_empty historical_reward_authority_not_inherited input_prepost_equal lineage_exact manifest_schema preregistration_schema public_hidden_denial_exact public_output_exact8_contract public_selection_current public_windows_current publish_noreplace_exact1 qualification_exact20_readonly runtime_boundary_exact services_current source_aliases_exact source_closure34_current source_role_order_exact two_stage_boundary_only v534_authority_immutable_consumed0 v534_durable_boundary1 v534_durable_exact11_current zero_update_contract_exact".split()
AUTHORITY_KEYS=set("active_source_paths active_source_records active_source_role_order authority_design_contract authority_materializer_source authorization check_key_set_sha256 check_keys checks checks_sha256 execution_boundary format fresh_attempt_root fresh_output_root hidden_input_denial_contract historical_absences input_post_snapshot input_pre_snapshot input_snapshots_exactly_equal lineage passed public_s1_evaluator_source public_s1_independent_auditor_source public_s1_input_contract public_s1_launcher_source public_s1_output_contract public_s1_zero_update_manifest_source public_s1_zero_update_preregistration_source required_absences runtime_observation seed source_aliases source_closure source_closure_sha256 source_role_order status v169_closure_source v169_library_manifest_source v169_release_manifest_source v169_runtime_source_source v474_public_s1_selection_receipt_source v474_public_s1_selection_source v482_model_design_contract_source v482_preregistration_source v524_cache_manifest_source v524_qualification_audit_source v524_qualification_report_source v524_qualification_terminal_source v534_actual_oof_attempt_intent_source v534_actual_oof_durable_transition v535_failed_attempt_transition v539_poststage_owned_output_prep_failure_transition runtime_package_import_contract v534_actual_oof_events_source v534_actual_oof_execution_receipt_source v534_actual_oof_fold0_source v534_actual_oof_fold1_source v534_actual_oof_fold2_source v534_actual_oof_fold3_source v534_actual_oof_fold4_source v534_actual_oof_independent_audit_source v534_actual_oof_metrics_source v534_actual_oof_source_manifest_source v534_authority_receipt_source v539_failure_forensic_source v539_failed_authority_receipt_source v539_failed_transport_script_source v539_failed_transport_record_source zero_update_contract".split())
EXPECTED_V535_FAILED_TRANSITION={"candidate_tree_canonical_sha256":"c4025e48be74986144b64c66663d0db31c73b4742c943e95bc3a4ac837147731","external_tree_canonical_sha256":"ada128f64efe4becf76a717f205045c321e70953beeabc6a3cafe0ba0deae7ea","failure_forensic":{"logical_bytes":11442,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_runtime_package_import_failure_forensic.json","sha256":"c67c699c795686134869b0002d46731b21707b82a1aa33db83e610b6983676b9"},"non_durable_partition":{"auditor":0,"evaluator":1,"launcher":1,"public_s1":0,"runtime":0,"training":0,"transport":1,"zero_update":0},"old_attempt_and_output_roots_absent":True,"old_authority_receipt":{"consumed":0,"logical_bytes":125125,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v535_v534_public_s1_zero_update_gate_execution_authority_seed1668_20260827/authority_receipt.json","sha256":"2b7f1c1e82cccb107da8e44af883940701f2427deccef75749e3afc2d85d7358"},"old_transport_record":{"logical_bytes":1244,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v535_v534_public_s1_zero_update_gate_transport_deployment_record.json","sha256":"7b7321b03fef38b96d25a21433e50ec9904baeaaf999d8b3a84f65893ea59426"},"old_transport_script":{"logical_bytes":10589,"path":"/root/v535_v534_public_s1_zero_update_gate_once.sh","sha256":"c8fab61058f8dd41c77a7f6aaab2e6b8318391ee8eff836fff13caef61e8fee7"},"retry_authorized":False}
EXPECTED_V539_POSTSTAGE_TRANSITION={"failure_forensic":{"logical_bytes":7541,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v540_v539_public_s1_zero_update_poststage_owned_output_prep_absence_failure_forensic.json","sha256":"ce412975394a8107853945c486e6a859984fd550ba9277f00ae013fbf0d22b43"},"model_and_rng_restored":True,"non_durable_partition":{"auditor":1,"evaluator":1,"launcher":1,"public_s1":1,"runtime":1,"training":0,"transport":1,"zero_update":1},"old_attempt_and_output_roots_absent":True,"old_authority_receipt":{"consumed":0,"logical_bytes":128865,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v539_v535_public_s1_zero_update_runtime_package_repair_execution_authority_seed1670_20260828/authority_receipt.json","sha256":"0b43b24c004c6a00975d3b14ece7167b312b9735185b1241454e39f856894be4"},"old_output_prep_owned_cleanup_completed":True,"old_transport_record":{"logical_bytes":1390,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v539_v535_public_s1_zero_update_runtime_package_repair_gate_transport_deployment_record.json","sha256":"86aea2da97fa6e7bcac62f426b08cd71860fa318fb82f1243f7dc13af47aa128"},"old_transport_script":{"logical_bytes":12301,"path":"/root/v539_v535_public_s1_zero_update_runtime_package_repair_gate_once.sh","sha256":"d1b12ed98773fd5c7ef10c3fbc90b5cd9ba3406e0cda18ba2bb6cc2b5303c599"},"retry_authorized":False,"staged_exact8_before_failure":True}
OLD_V535_ABSENT_PATHS=("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v535_v534_public_s1_zero_update_gate_attempt_seed1668_20260827","/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v535_v534_public_s1_zero_update_gate_attempt_seed1668_20260827.attempt-prep","/root/v535_v534_public_s1_zero_update_gate_seed1668_20260827","/root/v535_v534_public_s1_zero_update_gate_seed1668_20260827.output-prep")
OLD_V539_ABSENT_PATHS=("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v539_v535_public_s1_zero_update_gate_attempt_seed1670_20260828","/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v539_v535_public_s1_zero_update_gate_attempt_seed1670_20260828.attempt-prep","/root/v539_v535_public_s1_zero_update_gate_seed1670_20260828","/root/v539_v535_public_s1_zero_update_gate_seed1670_20260828.output-prep")
OLD_V535_CANDIDATE_ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v537_v536_stage_a_authority_split_state_readonly_candidate_seed1669_20260827")
OLD_V535_EXTERNAL_ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v537_v536_stage_a_authority_split_state_external_terminal_evidence_seed1669_20260827")
EXPECTED_RUNTIME_OBSERVATION={"actual_oof_execution_invocations":0,"authority_materializer_invocations":1,"backward_invocations":0,"buffer_update_invocations":0,"cache_write_invocations":0,"hidden_private_input_read_invocations":0,"independent_auditor_invocations":0,"launcher_invocations":0,"model_write_invocations":0,"optimizer_step_invocations":0,"optimizer_zero_grad_invocations":0,"parameter_update_invocations":0,"public_s1_evaluator_invocations":0,"public_s1_zero_update_boundary_invocations":0,"reward_read_invocations":0,"scheduler_step_invocations":0,"submission_invocations":0,"training_invocations":0,"zero_update_gate_invocations":0}
EXPECTED_EXECUTION_BOUNDARY={"authority_materialization_only":True,"authority_publication_commit_semantics":"current_exact1_visibility_after_noreplace_directory_rename","authority_publication_noreplace":True,"crash_durability_claimed":False,"direct_execution_authorized":False,"hidden_private_final_reward_inputs_authorized":False,"postcommit_diagnostics_best_effort_success_priority":True,"public_only":True,"public_s1_evaluator_invocations":0,"public_s1_zero_update_boundary_invocations":0,"rollback_relink_retry_or_second_publish_after_commit":False,"training_or_update_authorized":False,"zero_update_gate_invocations":0}
EXPECTED_PHASE_VALIDATION={"generic_poststage_recheck_of_output_prep_absence_authorized":False,"held_member_requirements":["open_rdwr_fd","fstat_lstat_dev_inode","pread_sha256_bytes_eof","exact8_names"],"poststage":"fresh_output_root_absent_owned_prep_dev_inode_and_held_exact8_current","precreate":"fresh_output_root_and_owned_prep_both_absent"}
EXPECTED_HISTORICAL_ABSENCES={"attempt_prep":{"absent":True,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_gate_attempt_seed1671_20260828.attempt-prep"},"attempt_root":{"absent":True,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_gate_attempt_seed1671_20260828"},"authority_prep":{"absent":True,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_seed1671_20260828.authority-prep"},"authority_root":{"absent":True,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v540_v539_public_s1_zero_update_poststage_owned_output_prep_repair_execution_authority_seed1671_20260828"},"output_prep":{"absent":True,"path":"/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828.output-prep"},"output_root":{"absent":True,"path":"/root/v540_v539_public_s1_zero_update_gate_seed1671_20260828"}}
EXPECTED_CURRENT_ABSENCES={key:value for key,value in EXPECTED_HISTORICAL_ABSENCES.items() if key!="authority_root"}
SNAPSHOT_KEYS={"absences","canonical_sha256","files","gpu_compute_pids","relevant_execution_pids","services","trees"}


def validate_execution_phase(prereg: dict, phase: str, *, owned_prep: Path | None = None,
                             owned_identity=None, held: dict | None = None) -> None:
    if phase == "precreate":
        if owned_prep is not None or owned_identity is not None or held is not None:
            raise RuntimeError("precreate ownership arguments")
        if any(os.path.lexists(row["path"]) for row in EXPECTED_CURRENT_ABSENCES.values()):
            raise RuntimeError("precreate required absence present")
        return
    if phase != "poststage" or owned_prep is None or owned_identity is None or held is None:
        raise RuntimeError("execution phase")
    for key,row in EXPECTED_CURRENT_ABSENCES.items():
        if key != "output_prep" and os.path.lexists(row["path"]):
            raise RuntimeError(f"poststage required absence present:{key}")
    if str(owned_prep) != EXPECTED_CURRENT_ABSENCES["output_prep"]["path"]:
        raise RuntimeError("poststage prep canonical path")
    current=os.lstat(owned_prep)
    if (not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode)
            or (current.st_dev,current.st_ino)!=(owned_identity.st_dev,owned_identity.st_ino)):
        raise RuntimeError("poststage owned prep identity")
    expected=set(prereg["public_s1_output_contract"]["exact_names"])
    if set(held)!=expected or {path.name for path in owned_prep.iterdir()}!=expected:
        raise RuntimeError("poststage held exact8 names")
    validate_held(owned_prep,held)


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_exact(actual, expected) -> bool:
    return canonical_bytes(actual) == canonical_bytes(expected)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arrsha(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_file(path: Path, digest: str, logical_bytes: int | None = None) -> None:
    if path.is_symlink() or not path.is_file() or sha(path) != digest or (logical_bytes is not None and path.stat().st_size != logical_bytes):
        raise RuntimeError(f"exact file:{path}")


def exact_record(record: dict) -> None:
    if set(record) != {"path", "sha256", "logical_bytes"} or type(record["logical_bytes"]) is not int:
        raise RuntimeError("exact record schema")
    exact_file(Path(record["path"]), record["sha256"], record["logical_bytes"])


def tree(root: Path) -> dict:
    rows, lines, total = [], bytearray(), 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink:{path}")
        if path.is_file():
            rel, digest, size = path.relative_to(root).as_posix(), sha(path), path.stat().st_size
            rows.append([rel, digest, size]); lines.extend(f"{digest}  {rel}\n".encode()); total += size
    return {"file_count": len(rows), "logical_file_bytes": total, "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(), "canonical_json_triples_digest_sha256": canonical_sha(rows)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0: raise OSError("short write")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    exact_file(path, hashlib.sha256(payload).hexdigest(), len(payload))


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n")


def held_bytes(path: Path, payload: bytes, held: dict) -> None:
    flags=os.O_RDWR|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0); fd=os.open(str(path),flags,0o600)
    try:
        identity=os.fstat(fd); view=memoryview(payload)
        while view:
            count=os.write(fd,view)
            if count<=0: raise OSError("short write")
            view=view[count:]
        os.fsync(fd); actual=os.pread(fd,len(payload)+1,0)
        current=os.lstat(path)
        if actual!=payload or len(actual)!=len(payload) or (identity.st_dev,identity.st_ino)!=(current.st_dev,current.st_ino): raise RuntimeError("held member identity/payload")
        held[path.name]=(fd,{"path":str(path),"sha256":hashlib.sha256(payload).hexdigest(),"logical_bytes":len(payload),"st_dev":identity.st_dev,"st_ino":identity.st_ino}); fd=-1
    finally:
        if fd>=0:
            owned=os.fstat(fd); os.close(fd)
            try:
                current=os.lstat(path)
                if (current.st_dev,current.st_ino)==(owned.st_dev,owned.st_ino): path.unlink()
            except Exception: pass


def held_json(path: Path, value, held: dict) -> None:
    held_bytes(path,json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False).encode()+b"\n",held)


def validate_held(root: Path, held: dict) -> None:
    for name,(fd,record) in held.items():
        path=root/name; fst=os.fstat(fd); lst=os.lstat(path); payload=os.pread(fd,record["logical_bytes"]+1,0)
        if ((fst.st_dev,fst.st_ino)!=(record["st_dev"],record["st_ino"]) or (lst.st_dev,lst.st_ino)!=(record["st_dev"],record["st_ino"])
                or len(payload)!=record["logical_bytes"] or hashlib.sha256(payload).hexdigest()!=record["sha256"]): raise RuntimeError(f"held current:{name}")


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error), str(target))


def rng_snapshot() -> dict:
    py = pickle.dumps(random.getstate(), protocol=4)
    npb = pickle.dumps(np.random.get_state(), protocol=4)
    cpu = torch.random.get_rng_state().cpu().numpy().tobytes()
    cuda = []
    if torch.cuda.is_initialized():
        cuda = [hashlib.sha256(torch.cuda.get_rng_state(i).cpu().numpy().tobytes()).hexdigest() for i in range(torch.cuda.device_count())]
    return {"python": hashlib.sha256(py).hexdigest(), "numpy": hashlib.sha256(npb).hexdigest(), "torch_cpu": hashlib.sha256(cpu).hexdigest(), "torch_cuda_initialized": torch.cuda.is_initialized(), "torch_cuda": cuda}


def rng_capture_raw():
    return (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
            [torch.cuda.get_rng_state(i).clone() for i in range(torch.cuda.device_count())] if torch.cuda.is_initialized() else None)


def rng_restore_raw(state) -> None:
    random.setstate(state[0]); np.random.set_state(state[1]); torch.random.set_rng_state(state[2])
    if state[3] is not None:
        for index, value in enumerate(state[3]): torch.cuda.set_rng_state(value, index)


def module_inventory(runtime) -> list[tuple[str, torch.nn.Module]]:
    found, seen = [], set()
    def visit(prefix, value, depth):
        if depth > 4 or id(value) in seen: return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            found.append((prefix, value)); return
        if hasattr(value, "__dict__"):
            for name, child in sorted(vars(value).items()):
                if name.startswith("_"): continue
                visit(f"{prefix}.{name}", child, depth + 1)
    visit("runtime", runtime, 0)
    if not found: raise RuntimeError("no torch module in public evaluator runtime")
    return found


def tensor_record(value: torch.Tensor) -> dict:
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest(), "logical_bytes": len(raw), "requires_grad": bool(value.requires_grad)}


def model_snapshot(modules: list[tuple[str, torch.nn.Module]]) -> dict:
    parameters, buffers, modes = {}, {}, {}
    for prefix, module in modules:
        modes[prefix] = bool(module.training)
        for name, value in module.named_parameters(recurse=True): parameters[f"{prefix}.{name}"] = tensor_record(value)
        for name, value in module.named_buffers(recurse=True): buffers[f"{prefix}.{name}"] = tensor_record(value)
    return {"parameters": parameters, "buffers": buffers, "module_training_flags": modes,
            "optimizer_instances": 0, "scheduler_instances": 0,
            "parameters_sha256": canonical_sha(parameters), "buffers_sha256": canonical_sha(buffers), "module_training_flags_sha256": canonical_sha(modes)}


def model_capture_raw(modules):
    result=[]
    for prefix,module in modules:
        for name,value in module.named_parameters(recurse=True): result.append((f"{prefix}.parameter.{name}",value,value.detach().clone()))
        for name,value in module.named_buffers(recurse=True): result.append((f"{prefix}.buffer.{name}",value,value.detach().clone()))
    return result


def model_restore_raw(records) -> None:
    with torch.no_grad():
        for _,target,value in records: target.copy_(value)


def set_eval(modules):
    for _, module in modules: module.eval()


def load_public_row(frozen: dict) -> dict:
    arrays = []
    for record in frozen["source_windows"]:
        path = Path(record["path"])
        if path.parent != PUBLIC_WINDOW_ROOT or any(fragment in str(path).lower() for fragment in FORBIDDEN_FRAGMENTS): raise RuntimeError("forbidden/noncanonical public path")
        if set(record) != {"path", "sha256"}: raise RuntimeError("public source record schema")
        exact_file(path, record["sha256"])
        with np.load(path, allow_pickle=False) as raw:
            if list(raw.files) != EXACT_NPZ_KEYS: raise RuntimeError("public NPZ exact keys/aliases")
            arrays.append({key: np.ascontiguousarray(raw[key]) for key in raw.files})
            source_value=str(np.asarray(raw["source"]).item()); start_value=int(np.asarray(raw["start"]).item())
            if any(fragment in source_value.lower() for fragment in FORBIDDEN_FRAGMENTS) or start_value<0: raise RuntimeError("public source/start metadata")
    context, history = arrays[0]["context_frames"], arrays[0]["history_actions"]
    future = np.ascontiguousarray(np.concatenate([x["future_actions"] for x in arrays])[:frozen["horizon"]])
    target = np.ascontiguousarray(np.concatenate([x["target_frames"] for x in arrays])[:frozen["horizon"]])
    if (context.shape != (5,256,256,3) or context.dtype != np.uint8 or history.shape != (4,14) or history.dtype != np.float32
            or future.shape != (frozen["horizon"],14) or future.dtype != np.float32
            or target.shape != (frozen["horizon"],256,256,3) or target.dtype != np.uint8
            or arrsha(history) != frozen["history_action_sha256"] or arrsha(future) != frozen["future_action_sha256"]):
        raise RuntimeError("public row fields/labels")
    return {"context": context, "history": history, "future": future, "target": target}


def predict_public(runtime, frozen, loaded):
    context, history, future = loaded["context"], loaded["history"], loaded["future"]
    outputs = []
    for chunk in range(frozen["horizon"] // 8):
        seed = frozen["chunk_seeds"][chunk] if frozen["branch"] == "right_recursive32" else frozen["seed"]
        prediction = np.ascontiguousarray(runtime.predict(context, history, future[chunk*8:(chunk+1)*8], int(seed), frozen["instruction"]))
        if prediction.shape != (8,256,256,3) or prediction.dtype != np.uint8: raise RuntimeError("public prediction schema")
        outputs.append(prediction)
        context = np.ascontiguousarray(np.concatenate((context, prediction), axis=0)[-5:])
        history = np.ascontiguousarray(future[max(0,(chunk+1)*8-4):(chunk+1)*8])
    return np.ascontiguousarray(np.concatenate(outputs))


def validate_documents(prereg, manifest, contract, authority, args, *, phase: str = "precreate",
                       owned_prep: Path | None = None, owned_identity=None, held: dict | None = None) -> None:
    if (set(prereg)!=PREREG_KEYS or set(manifest)!=MANIFEST_KEYS or set(contract)!=CONTRACT_KEYS
            or prereg.get("format")!="strict-track2-v540-v539-public-s1-zero-update-gate-preregistration-v1"
            or manifest.get("format")!="strict-track2-v540-v539-public-s1-zero-update-gate-manifest-v1"
            or contract.get("format")!="strict-track2-v540-v539-public-s1-zero-update-runtime-package-repair-execution-authority-design-contract-v1"
            or prereg.get("status")!="preregistered_public_s1_zero_update_gate_pending_external_authority"
            or manifest.get("status")!="frozen_public_s1_zero_update_gate_manifest_pending_external_authority"
            or contract.get("status")!="design_only_frozen_public_s1_zero_update_sources_pending_independent_authority_materialization"):
        raise RuntimeError("document exact top schemas")
    if prereg["seed"] != SEED or type(prereg["seed"]) is not int or manifest["seed"] != SEED or type(manifest["seed"]) is not int:
        raise RuntimeError("seed strict int")
    if any(not json_exact(document.get("active_source_role_order"),ACTIVE_SOURCE_ROLE_ORDER) or not json_exact(document.get("active_source_paths"),ACTIVE_SOURCE_PATHS) for document in (prereg, manifest, contract, authority)):
        raise RuntimeError("active source role/path registry")
    if any(not json_exact(document.get("authorization"),EXPECTED_AUTHORIZATION) for document in (prereg, manifest, contract, authority)):
        raise RuntimeError("authorization exact")
    for key, expected in EXPECTED_AUTHORIZATION.items():
        if type(authority["authorization"][key]) is not type(expected): raise RuntimeError("authorization bool-int")
    if (not json_exact(contract.get("runtime_observation"),EXPECTED_RUNTIME_OBSERVATION)
            or not json_exact(authority.get("runtime_observation"),EXPECTED_RUNTIME_OBSERVATION)
            or not json_exact(contract.get("execution_boundary"),EXPECTED_EXECUTION_BOUNDARY)
            or not json_exact(authority.get("execution_boundary"),EXPECTED_EXECUTION_BOUNDARY)):
        raise RuntimeError("runtime/execution boundary frozen exact")
    if (not json_exact(contract.get("historical_absences"),EXPECTED_HISTORICAL_ABSENCES)
            or not json_exact(contract.get("current_absences_after_authority"),EXPECTED_CURRENT_ABSENCES)
            or not json_exact(authority.get("historical_absences"),EXPECTED_HISTORICAL_ABSENCES)
            or not json_exact(authority.get("required_absences"),EXPECTED_CURRENT_ABSENCES)):
        raise RuntimeError("authority absence maps frozen exact")
    pre,post=authority.get("input_pre_snapshot"),authority.get("input_post_snapshot")
    for snapshot in (pre,post):
        if (not isinstance(snapshot,dict) or set(snapshot)!=SNAPSHOT_KEYS
                or not json_exact(snapshot.get("absences"),EXPECTED_HISTORICAL_ABSENCES)
                or not json_exact(snapshot.get("files"),contract.get("source_closure"))
                or snapshot.get("gpu_compute_pids")!=[] or snapshot.get("relevant_execution_pids")!=[]
                or canonical_sha({key:value for key,value in snapshot.items() if key!="canonical_sha256"})!=snapshot.get("canonical_sha256")):
            raise RuntimeError("authority input snapshot exact")
    if authority.get("input_snapshots_exactly_equal") is not True or not json_exact(pre,post): raise RuntimeError("authority input pre/post equality")
    validate_execution_phase(prereg,phase,owned_prep=owned_prep,owned_identity=owned_identity,held=held)
    for field in ("v534_actual_oof_durable_transition", "v535_failed_attempt_transition", "v539_poststage_owned_output_prep_failure_transition", "runtime_package_import_contract", "public_s1_input_contract", "public_s1_output_contract", "zero_update_contract", "hidden_input_denial_contract"):
        if not json_exact(prereg[field],manifest[field]) or not json_exact(contract.get(field),prereg[field]) or not json_exact(authority.get(field),prereg[field]): raise RuntimeError(f"embedded exact:{field}")
    if not json_exact(prereg["public_s1_output_contract"].get("phase_validation"),EXPECTED_PHASE_VALIDATION):
        raise RuntimeError("phase validation contract frozen exact")
    transition=prereg["v535_failed_attempt_transition"]
    if not json_exact(transition,EXPECTED_V535_FAILED_TRANSITION): raise RuntimeError("v535 failed transition frozen exact")
    for key in ("failure_forensic","old_transport_record","old_transport_script"):
        exact_record(transition[key])
    old_authority_record=transition["old_authority_receipt"]
    if set(old_authority_record)!={"path","sha256","logical_bytes","consumed"}: raise RuntimeError("v535 old authority record schema")
    exact_record({key:old_authority_record[key] for key in ("path","sha256","logical_bytes")})
    old_authority=json.loads(Path(transition["old_authority_receipt"]["path"]).read_text())
    if (type(transition["old_authority_receipt"]["consumed"]) is not int or transition["old_authority_receipt"]["consumed"]!=0
            or type(old_authority["authorization"]["public_s1_zero_update_boundary_invocations_consumed"]) is not int
            or old_authority["authorization"]["public_s1_zero_update_boundary_invocations_consumed"]!=0):
        raise RuntimeError("v535 failed authority immutable consumed0")
    if any(os.path.lexists(path) for path in OLD_V535_ABSENT_PATHS): raise RuntimeError("v535 failed roots current absence")
    if (tree(OLD_V535_CANDIDATE_ROOT)["canonical_json_triples_digest_sha256"]!=transition["candidate_tree_canonical_sha256"]
            or tree(OLD_V535_EXTERNAL_ROOT)["canonical_json_triples_digest_sha256"]!=transition["external_tree_canonical_sha256"]):
        raise RuntimeError("v535 candidate/external current trees")
    poststage=prereg["v539_poststage_owned_output_prep_failure_transition"]
    if not json_exact(poststage,EXPECTED_V539_POSTSTAGE_TRANSITION): raise RuntimeError("v539 poststage transition frozen exact")
    for key in ("failure_forensic","old_transport_record","old_transport_script"):
        exact_record(poststage[key])
    old_v539_record=poststage["old_authority_receipt"]
    if set(old_v539_record)!={"path","sha256","logical_bytes","consumed"}: raise RuntimeError("v539 old authority record schema")
    exact_record({key:old_v539_record[key] for key in ("path","sha256","logical_bytes")})
    old_v539_authority=json.loads(Path(old_v539_record["path"]).read_text())
    if (type(old_v539_record["consumed"]) is not int or old_v539_record["consumed"]!=0
            or type(old_v539_authority["authorization"]["public_s1_zero_update_boundary_invocations_consumed"]) is not int
            or old_v539_authority["authorization"]["public_s1_zero_update_boundary_invocations_consumed"]!=0
            or any(os.path.lexists(path) for path in OLD_V539_ABSENT_PATHS)):
        raise RuntimeError("v539 failed lineage immutable/current")
    receipt_contract=contract.get("authority_receipt_contract")
    if (not isinstance(receipt_contract,dict) or set(receipt_contract)!=RECEIPT_CONTRACT_KEYS
            or not json_exact(receipt_contract.get("active_source_role_order_exact"),ACTIVE_SOURCE_ROLE_ORDER)
            or not json_exact(receipt_contract.get("active_source_paths_exact"),ACTIVE_SOURCE_PATHS)
            or type(receipt_contract.get("contract_active_source_records_count")) is not int or receipt_contract["contract_active_source_records_count"]!=6
            or receipt_contract.get("contract_design_record_excluded_to_avoid_self_hash_cycle") is not True
            or type(receipt_contract.get("authority_active_source_records_count")) is not int or receipt_contract["authority_active_source_records_count"]!=7
            or receipt_contract.get("authority_design_record_included") is not True
            or not json_exact(receipt_contract.get("authorization_exact"),EXPECTED_AUTHORIZATION)): raise RuntimeError("authority receipt contract exact21")
    records6, records7 = contract.get("active_source_records"), authority.get("active_source_records")
    if not isinstance(records6,dict) or set(records6) != set(ACTIVE_SOURCE_ROLE_ORDER[1:]) or not isinstance(records7,dict) or set(records7) != set(ACTIVE_SOURCE_ROLE_ORDER):
        raise RuntimeError("active records6/7")
    for role in ACTIVE_SOURCE_ROLE_ORDER:
        record = records7[role]; exact_record(record)
        if record["path"] != ACTIVE_SOURCE_PATHS[role]: raise RuntimeError("active record path")
        if role != "authority_design_contract" and not json_exact(records6[role],record): raise RuntimeError("active record shared")
    design = {"path": str(args.contract), "sha256": args.contract_sha, "logical_bytes": args.contract.stat().st_size}
    if not json_exact(records7["authority_design_contract"],design): raise RuntimeError("design record")
    if not json_exact(receipt_contract["contract_active_source_records_exact"],records6): raise RuntimeError("receipt active records6")
    closure6, closure7 = contract.get("source_closure"), authority.get("source_closure")
    if (not json_exact(contract.get("source_role_order"),SOURCE_ROLE_ORDER) or not json_exact(authority.get("source_role_order"),["authority_design_contract",*SOURCE_ROLE_ORDER])
            or not json_exact(contract.get("source_aliases"),SOURCE_ALIASES) or not json_exact(authority.get("source_aliases"),SOURCE_ALIASES)
            or not json_exact(contract.get("lineage"),EXPECTED_LINEAGE) or not json_exact(authority.get("lineage"),EXPECTED_LINEAGE)
            or not isinstance(closure6,dict) or set(closure6)!=set(SOURCE_ROLE_ORDER) or not isinstance(closure7,dict) or set(closure7)!={"authority_design_contract",*SOURCE_ROLE_ORDER}
            or canonical_sha(closure6) != contract.get("source_closure_sha256") or canonical_sha(closure7) != authority.get("source_closure_sha256")):
        raise RuntimeError("source closure digest")
    for role in SOURCE_ROLE_ORDER:
        record=closure6[role]
        if record.get("path")!=SOURCE_PATHS[role] or not json_exact(closure7.get(role),record) or not json_exact(contract.get(SOURCE_ALIASES[role]),record): raise RuntimeError(f"shared source closure:{role}")
        exact_record(record)
    aliases = authority.get("source_aliases")
    if not isinstance(aliases,dict) or set(aliases) != set(closure7) or closure7["authority_design_contract"]!=design: raise RuntimeError("source aliases")
    for role, alias in aliases.items():
        if not json_exact(authority.get(alias),closure7[role]): raise RuntimeError(f"source alias:{role}")
    if (authority.get("passed") is not True or any(value is not True for value in authority.get("checks",{}).values())
            or set(authority) != AUTHORITY_KEYS or receipt_contract.get("top_keys") != sorted(AUTHORITY_KEYS)
            or receipt_contract.get("check_keys") != CHECK_KEYS or set(authority.get("checks",{})) != set(CHECK_KEYS)):
        raise RuntimeError("authority receipt terminal schema")
    if (authority.get("format")!=receipt_contract["format"] or authority.get("status")!=receipt_contract["status"]
            or authority.get("check_keys")!=receipt_contract["check_keys"] or authority["check_keys"]!=sorted(authority.get("checks",{}))
            or canonical_sha(authority["check_keys"])!=authority.get("check_key_set_sha256") or authority["check_key_set_sha256"]!=receipt_contract["check_key_set_sha256"]
            or canonical_sha(authority["checks"])!=authority.get("checks_sha256") or authority["checks_sha256"]!=receipt_contract["checks_sha256"]): raise RuntimeError("authority check/top digests")
    for field,receipt_field in (("v534_actual_oof_durable_transition","v534_actual_oof_durable_transition_exact"),("v535_failed_attempt_transition","v535_failed_attempt_transition_exact"),("v539_poststage_owned_output_prep_failure_transition","v539_poststage_owned_output_prep_failure_transition_exact"),("runtime_package_import_contract","runtime_package_import_contract_exact"),("public_s1_input_contract","public_s1_input_contract_exact"),("public_s1_output_contract","public_s1_output_contract_exact"),("zero_update_contract","zero_update_contract_exact"),("hidden_input_denial_contract","hidden_input_denial_contract_exact"),("execution_boundary","execution_boundary_exact"),("runtime_observation","runtime_observation_exact")):
        if not json_exact(contract.get(field),receipt_contract[receipt_field]) or not json_exact(authority.get(field),contract[field]): raise RuntimeError(f"receipt embedded:{field}")
    for record in prereg["v534_actual_oof_durable_transition"]["records"].values(): exact_record(record)
    durable=prereg["v534_actual_oof_durable_transition"]; durable_root=Path(durable["root"])
    if sorted(x.name for x in durable_root.iterdir())!=durable["exact_names"] or durable["exact_count"]!=11:
        raise RuntimeError("v534 OOF exact11 names")
    durable_tree=tree(durable_root)
    if (durable_tree["sha256sum_lines_digest_sha256"]!=durable["tree_lines_sha256"]
            or durable_tree["canonical_json_triples_digest_sha256"]!=durable["tree_canonical_sha256"]
            or durable_tree["logical_file_bytes"]!=durable["tree_total_logical_bytes"]): raise RuntimeError("v534 OOF tree")
    old_auth=durable["old_authority_receipt"]; exact_file(Path(old_auth["path"]),old_auth["sha256"],old_auth["logical_bytes"])
    old_auth_json=json.loads(Path(old_auth["path"]).read_text())
    if type(old_auth["consumed"]) is not int or old_auth["consumed"]!=0 or type(old_auth_json["authorization"]["actual_oof_execution_boundary_invocations_consumed"]) is not int or old_auth_json["authorization"]["actual_oof_execution_boundary_invocations_consumed"]!=0:
        raise RuntimeError("v534 authority immutable consumed0")
    v534_receipt=json.loads((durable_root/"execution_receipt.json").read_text()); v534_audit=json.loads((durable_root/"independent_audit.json").read_text())
    if (v534_receipt.get("passed") is not True or v534_receipt.get("status")!="passed_actual_oof_execution"
            or {k:v534_receipt.get(k) for k in ("actual_oof_execution_boundary_invocations","launcher_invocations","executor_invocations","auditor_import_invocations","retry_authorized")}!=durable["durable_partition"]
            or v534_audit.get("passed") is not True or v534_audit.get("events")!=1000 or v534_audit.get("fold_row_counts")!=[200]*5
            or any(v534_audit.get(key)!=0 for key in ("training_invocations","reward_read_invocations","dev_hidden_final_outcome_read_invocations","submission_invocations"))): raise RuntimeError("v534 durable terminal/audit")
    qualification=manifest["qualification_contract"]; qualification_tree=tree(Path(qualification["root"]))
    if (qualification_tree["file_count"]!=qualification["exact_count"] or qualification_tree["sha256sum_lines_digest_sha256"]!=qualification["tree_lines_sha256"]
            or qualification_tree["canonical_json_triples_digest_sha256"]!=qualification["tree_canonical_sha256"]
            or qualification_tree["logical_file_bytes"]!=qualification["tree_total_logical_bytes"] or qualification["readonly"] is not True): raise RuntimeError("qualification exact20")
    selection = prereg["public_s1_input_contract"]["selection"]; receipt = prereg["public_s1_input_contract"]["selection_receipt"]
    exact_record(selection); exact_record(receipt)
    selection_json=json.loads(Path(selection["path"]).read_text()); selection_receipt=json.loads(Path(receipt["path"]).read_text())
    public=prereg["public_s1_input_contract"]
    if (selection_json.get("format")!="strict-track2-v474-s1-action-only-selection-v1" or selection_json.get("seed")!=1617
            or len(selection_json.get("right",[]))!=32 or len(selection_json.get("left",[]))!=12
            or selection_json.get("guards")!={"gate_used_for_selection":False,"model_constructed_or_inferred":False,"outcome_loaded":False,"policy_updates":0,"reward_loaded":False,"rgb_keys_read":False}
            or selection_receipt.get("passed") is not True or selection_receipt.get("selection_sha256")!=selection["sha256"]
            or public["public_label_keys"]!=["target_frames"] or public["aliases_authorized"] is not False
            or public["reward_or_outcome_labels_authorized"] is not False): raise RuntimeError("public selection/receipt exact")
    expected_rows=[]
    for branch,rows in (("right_recursive32",selection_json["right"]),("left_first8",selection_json["left"])):
        for index,row in enumerate(rows):
            raw=row.get("source_windows",row.get("source_window")); raw=[raw] if isinstance(raw,dict) else raw
            item={"ordinal":len(expected_rows),"branch":branch,"branch_index":index,"episode":row["episode"],"start":row["start"],"instruction":row["instruction"],"source_windows":[{"path":x["path"],"sha256":x["sha256"]} for x in raw],"history_action_sha256":row["history_action_sha256"],"future_action_sha256":row["future32_action_sha256"] if branch=="right_recursive32" else row["future8_action_sha256"],"horizon":32 if branch=="right_recursive32" else 8,"public_label_key":"target_frames"}
            if branch=="right_recursive32": item.update({"phase":row["phase"],"chunk_seeds":row["chunk_seeds"]})
            else: item.update({"position":row["position"],"seed":row["seed"]})
            expected_rows.append(item)
    if public["ordered_rows"]!=expected_rows or public["ordered_rows_sha256"]!=canonical_sha(expected_rows): raise RuntimeError("public ordered rows reconstructed exact")
    allowed_roots=sorted({str(Path(record["path"]).parent) for row in public["ordered_rows"] for record in row["source_windows"]})
    if allowed_roots != [str(PUBLIC_WINDOW_ROOT)] or manifest["hidden_input_denial_contract"]!={"allowed_roots":allowed_roots,"exact_public_label_key":"target_frames","path_aliases_authorized":False,"hidden_private_final_reward_inputs_authorized":False}:
        raise RuntimeError("public hidden/alias denial")
    # Fresh-state truth is phase-aware and was checked by validate_execution_phase.


def validate_runtime_package_import(manifest):
    frozen=manifest["runtime_package_import_contract"]
    expected={"package_root":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline","package_name":"wam_pipeline","runtime_qualified_module":"wam_pipeline.v169_arm_routed_runtime","package_init":{"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/wam_pipeline/__init__.py","sha256":"1d8c8f56ecd70fa05b21f3fe8b870a12424dbc5ea6a6ad9944d202023dc66f2a","logical_bytes":118},"runtime_source":{"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/wam_pipeline/v169_arm_routed_runtime.py","sha256":"0044d49ae2a3083f0b638b701fb398b407d7c436c3bb0ee1798a3233e076c7c1","logical_bytes":3277},"arm_router_source":{"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/wam_pipeline/arm_router.py","sha256":"8e668f7d08dd2d7593353b4557bc0c78cb750395b1b80e5551a964480b90550a","logical_bytes":3236},"loader":"importlib.import_module_package_qualified_after_exact_records_before_prep","required_module_package":"wam_pipeline","required_runtime_package":"wam_pipeline","anonymous_spec_authorized":False,"wrong_package_authorized":False,"stale_sys_modules_cache_authorized":False,"symlink_or_source_drift_authorized":False,"gate_before_output_prep":True}
    if not json_exact(frozen,expected): raise RuntimeError("runtime package import contract")
    for key in ("package_init","runtime_source","arm_router_source"): exact_record(frozen[key])
    package_root=Path(frozen["package_root"])
    if str(package_root) not in sys.path: sys.path.insert(0,str(package_root))
    importlib.invalidate_caches()
    expected_origins={frozen["package_name"]:Path(frozen["package_init"]["path"]),frozen["package_name"]+".arm_router":Path(frozen["arm_router_source"]["path"]),frozen["runtime_qualified_module"]:Path(frozen["runtime_source"]["path"])}
    for name,path in expected_origins.items():
        existing=sys.modules.get(name)
        if existing is not None and (not getattr(existing,"__file__",None) or Path(existing.__file__).resolve()!=path): raise RuntimeError("stale package cache")
    module=importlib.import_module(frozen["runtime_qualified_module"])
    router=sys.modules.get(frozen["package_name"]+".arm_router")
    if module.__package__!=frozen["required_runtime_package"] or Path(module.__file__).resolve()!=expected_origins[frozen["runtime_qualified_module"]]: raise RuntimeError("runtime package identity")
    if router is None or router.__package__!=frozen["required_module_package"] or Path(router.__file__).resolve()!=expected_origins[frozen["package_name"]+".arm_router"]: raise RuntimeError("arm router package identity")
    return module


def load_v169_runtime(manifest, device):
    model = manifest["model_and_source_input"]
    closure_path=Path(model["v169"]["closure_path"]); exact_file(closure_path,model["v169"]["closure_sha256"])
    closure=json.loads(closure_path.read_text()); source=Path(closure["runtime_source"]["path"]); exact_file(source,closure["runtime_source"]["sha256"])
    if closure.get("release")!=model["v169"]["release_path"] or closure.get("library")!=model["v169"]["library_path"]:
        raise RuntimeError("v169 closure release/library")
    module = validate_runtime_package_import(manifest)
    return module.Track2V169ArmRoutedRuntime(model["v169"]["release_path"], model["v169"]["library_path"], device=device)


def initialize_rng_inventory_before_boundary() -> dict:
    if not torch.cuda.is_available(): return {"cuda_available":False,"device_indices":[]}
    indices=list(range(torch.cuda.device_count()))
    states=torch.cuda.get_rng_state_all()
    if len(states)!=len(indices): raise RuntimeError("CUDA RNG inventory")
    return {"cuda_available":True,"device_indices":indices}


def execute(args) -> dict:
    for path,digest in ((Path(__file__).resolve(),args.evaluator_sha),(args.preregistration,args.preregistration_sha),(args.manifest,args.manifest_sha),(args.contract,args.contract_sha),(args.authority_receipt,args.authority_receipt_sha),(args.auditor_source,args.auditor_sha),(args.launcher_source,args.launcher_sha)):
        exact_file(path,digest)
    prereg=json.loads(args.preregistration.read_text()); manifest=json.loads(args.manifest.read_text()); contract=json.loads(args.contract.read_text()); authority=json.loads(args.authority_receipt.read_text())
    validate_documents(prereg,manifest,contract,authority,args)
    validate_runtime_package_import(manifest)
    initialize_rng_inventory_before_boundary()
    if args.output_root != Path(prereg["fresh_output_root"]): raise RuntimeError("output root")
    prep=Path(prereg["fresh_output_prep_root"]); blocked={signal.SIGINT,signal.SIGTERM}; previous=signal.pthread_sigmask(signal.SIG_BLOCK,blocked)
    committed=False; identity=None; runtime=None; modules=[]; original_modes={}; raw_rng=None; raw_model=[]; held={}
    try:
        prep.mkdir(mode=0o700); identity=os.lstat(prep); raw_rng=rng_capture_raw(); rng_entry=rng_snapshot()
        held_json(prep/"attempt_intent.json", {"format":"strict-track2-v540-v539-public-s1-zero-update-attempt-intent-v1","seed":SEED,"literal_argv":args.literal_argv,"rng_entry":rng_entry,"authority_receipt_sha256":args.authority_receipt_sha},held)
        runtime=load_v169_runtime(manifest,args.device); modules=module_inventory(runtime); original_modes={name:module.training for name,module in modules}; raw_model=model_capture_raw(modules); rng_restore_raw(raw_rng)
        entry_model=model_snapshot(modules); set_eval(modules); eval_model=model_snapshot(modules)
        baseline={**eval_model}; snapshots=[{"stage":"after_model_load", "model":entry_model, "rng":rng_snapshot()},{"stage":"after_eval_mode", "model":eval_model, "rng":rng_snapshot()}]
        events=[]; total_error=0; total_pixels=0
        with torch.inference_mode():
            with torch.no_grad():
                for frozen in manifest["public_s1_input_contract"]["ordered_rows"]:
                    before=model_snapshot(modules); before_rng=rng_snapshot()
                    if before != baseline: raise RuntimeError("zero-update before row")
                    loaded=load_public_row(frozen)
                    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=devices,enabled=True):
                        row_seed=SEED*1000+frozen["ordinal"]
                        torch.manual_seed(row_seed)
                        if devices: torch.cuda.manual_seed_all(row_seed)
                        prediction=predict_public(runtime,frozen,loaded)
                    target=loaded["target"]
                    delta=np.abs(prediction.astype(np.int16)-target.astype(np.int16)).astype(np.uint16); error=int(delta.sum(dtype=np.uint64)); pixels=int(delta.size)
                    after=model_snapshot(modules); after_rng=rng_snapshot()
                    if after != baseline or after_rng != before_rng: raise RuntimeError("zero-update row drift")
                    event={"ordinal":frozen["ordinal"],"branch":frozen["branch"],"branch_index":frozen["branch_index"],"episode":frozen["episode"],"start":frozen["start"],"prediction_sha256":arrsha(prediction),"target_sha256":arrsha(target),"absolute_error_sum_uint64":error,"pixel_count_uint64":pixels,"model_snapshot_sha256":canonical_sha(after),"rng_snapshot_sha256":canonical_sha(after_rng),"torch_rng_isolation":"fork_rng_all_cuda_devices_per_public_row","isolated_torch_seed":row_seed}
                    event["event_canonical_sha256"]=canonical_sha(event); events.append(event); total_error+=error; total_pixels+=pixels
                    snapshots.append({"stage":f"after_public_row_{frozen['ordinal']:02d}","model":after,"rng":after_rng})
        final_model=model_snapshot(modules); rng_after=rng_snapshot()
        if final_model != baseline or rng_after != rng_entry: raise RuntimeError("zero-update final drift")
        events_payload=b"".join(canonical_bytes(row)+b"\n" for row in events); held_bytes(prep/"public_s1_events.ndjson",events_payload,held)
        metrics={"format":"strict-track2-v540-v539-public-s1-metrics-v1","rows":44,"right_rows":32,"left_rows":12,"absolute_error_sum_uint64":total_error,"pixel_count_uint64":total_pixels,"mean_absolute_error":total_error/total_pixels,"ordered_events_sha256":canonical_sha([x["event_canonical_sha256"] for x in events])}; held_json(prep/"public_s1_metrics.json",metrics,held)
        trace={"format":"strict-track2-v540-v539-zero-update-trace-v1","stage_count":len(snapshots),"stages":snapshots,"entry_model_sha256":canonical_sha(baseline),"exit_model_sha256":canonical_sha(final_model),"all_stages_bitexact":all(x["model"]==baseline for x in snapshots),"rng_entry":rng_entry,"rng_exit":rng_after,"rng_all_stages_bitexact":all(x["rng"]==rng_entry for x in snapshots)}; held_json(prep/"zero_update_trace.json",trace,held)
        zero={"format":"strict-track2-v540-v539-zero-update-receipt-v1","passed":True,"parameter_updates":0,"buffer_updates":0,"optimizer_instances":0,"scheduler_instances":0,"optimizer_steps":0,"optimizer_zero_grads":0,"scheduler_steps":0,"backward_calls":0,"model_writes":0,"cache_writes":0,"reward_reads":0,"hidden_private_final_reads":0,"mutation_then_restore_accepted":False,"entry_exit_bitexact":True,"all_intermediate_stages_bitexact":True,"exception_restoration_proven_by_fixture":True,"trace_sha256":sha(prep/"zero_update_trace.json")}; held_json(prep/"zero_update_receipt.json",zero,held)
        held_json(prep/"source_manifest.json",{"format":"strict-track2-v540-v539-public-s1-source-manifest-v1","preregistration_sha256":args.preregistration_sha,"manifest_sha256":args.manifest_sha,"contract_sha256":args.contract_sha,"authority_receipt_sha256":args.authority_receipt_sha,"evaluator_sha256":args.evaluator_sha,"auditor_sha256":args.auditor_sha,"launcher_sha256":args.launcher_sha},held)
        spec=importlib.util.spec_from_file_location("v539_public_s1_auditor",args.auditor_source); auditor=importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor); audit=auditor.audit_staged(prep,manifest,baseline,rng_entry); held_json(prep/"independent_audit.json",audit,held)
        receipt={"format":FORMAT,"status":"passed_public_s1_zero_update_gate","passed":True,"seed":SEED,"public_s1_zero_update_boundary_invocations":1,"public_s1_evaluator_invocations":1,"zero_update_gate_invocations":1,"independent_auditor_import_invocations":1,"public_rows":44,"public_label_key":"target_frames","public_metrics_sha256":sha(prep/"public_s1_metrics.json"),"zero_update_receipt_sha256":sha(prep/"zero_update_receipt.json"),"independent_audit_sha256":sha(prep/"independent_audit.json"),"training_invocations":0,"backward_calls":0,"optimizer_steps":0,"scheduler_steps":0,"parameter_updates":0,"buffer_updates":0,"model_writes":0,"cache_writes":0,"reward_reads":0,"hidden_private_final_reads":0,"submission_invocations":0,"retry_authorized":False,"rng_restored":True,"model_state_bitexact":True}; held_json(prep/"execution_receipt.json",receipt,held)
        expected=sorted(manifest["public_s1_output_contract"]["exact_names"])
        if sorted(x.name for x in prep.iterdir()) != expected or tree(prep)["file_count"] != 8: raise RuntimeError("output exact8")
        validate_held(prep,held)
        validate_documents(prereg,manifest,contract,authority,args,phase="poststage",owned_prep=prep,owned_identity=identity,held=held)
        if prep.stat().st_dev != identity.st_dev or prep.stat().st_ino != identity.st_ino: raise RuntimeError("prep identity")
        rename_noreplace(prep,args.output_root); committed=True
        try: fd=os.open(str(args.output_root.parent),os.O_RDONLY); os.fsync(fd); os.close(fd)
        except Exception: pass
        return receipt
    finally:
        if raw_model:
            try: model_restore_raw(raw_model)
            except Exception:
                if not committed: raise
        for name,module in modules:
            if name in original_modes:
                try: module.train(original_modes[name])
                except Exception:
                    if not committed: raise
        if raw_rng is not None:
            try: rng_restore_raw(raw_rng)
            except Exception:
                if not committed: raise
        for fd,_ in held.values():
            try: os.fsync(fd); os.close(fd)
            except Exception: pass
        try: signal.pthread_sigmask(signal.SIG_SETMASK,previous)
        except Exception:
            if not committed: raise
        if not committed and identity is not None and prep.exists() and not prep.is_symlink():
            try:
                current=prep.stat()
                if (current.st_dev,current.st_ino)==(identity.st_dev,identity.st_ino):
                    for name,(_,record) in sorted(held.items(),reverse=True):
                        path=prep/name
                        try:
                            current=os.lstat(path)
                            if (current.st_dev,current.st_ino)==(record["st_dev"],record["st_ino"]) and path.is_file() and not path.is_symlink() and sha(path)==record["sha256"] and path.stat().st_size==record["logical_bytes"]: path.unlink()
                        except Exception: pass
                    if not any(prep.iterdir()): prep.rmdir()
            except Exception: pass


def parser():
    p=argparse.ArgumentParser()
    for name in ("preregistration","manifest","contract","authority-receipt","auditor-source","launcher-source","output-root"): p.add_argument("--"+name,type=Path,required=True)
    for name in ("preregistration-sha","manifest-sha","contract-sha","authority-receipt-sha","evaluator-sha","auditor-sha","launcher-sha"): p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--device",choices=("cuda",),required=True)
    return p


def main(argv=None):
    literal=list(sys.argv[1:] if argv is None else argv); args=parser().parse_args(literal); args.literal_argv=literal
    result=execute(args); print(json.dumps({"passed":result["passed"],"status":result["status"],"public_rows":result["public_rows"]},sort_keys=True))


if __name__=="__main__": main()
