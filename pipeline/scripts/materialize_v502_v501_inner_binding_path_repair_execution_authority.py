#!/usr/bin/env python3
"""Atomically register the v502 path-binding-repair authority; run no execution source."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
CONTRACT_PATH = ROOT / "pipeline/scripts/v502_v501_inner_binding_path_repair_execution_authority_contract.json"
CONTRACT_FORMAT = "strict-track2-v502-v501-inner-binding-path-repair-execution-authority-design-contract-v1"
CONTRACT_STATUS = "design_only_frozen_sources_pending_independent_review_no_authority"
OUTPUT_FORMAT = "strict-track2-v502-v501-inner-binding-path-repair-execution-authority-v1"
OUTPUT_STATUS = "authorized_exact_one_external_v502_path_binding_repaired_outer_attempt"
AUTHORITY_ROOT = J / "v502_v501_inner_binding_path_repair_execution_authority_seed1644_20260825"
OUTER_EVIDENCE_ROOT = J / "v502_v501_inner_binding_path_repaired_reconciliation_outer_execution_evidence_seed1644_20260825"
INNER_ATTEMPT_ROOT = J / "v502_v501_inner_binding_path_repaired_reconciliation_inner_attempt_seed1644_20260825"
V501_AUTHORITY_ROOT = J / "v501_v500_failure_tree_diagnostic_inner_binding_repair_execution_authority_seed1643_20260825"
V501_AUTHORITY_EVIDENCE_ROOT = J / "v501_v500_failure_tree_diagnostic_inner_binding_repair_execution_authority_materialization_evidence_seed1643_20260825"
V501_PROCESS_PATH = V501_AUTHORITY_EVIDENCE_ROOT / "process_receipt.json"
V501_OUTER_EVIDENCE_ROOT = J / "v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_outer_execution_evidence_seed1643_20260825"
V501_INNER_ATTEMPT_ROOT = J / "v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_inner_attempt_seed1643_20260825"
V502_FORENSIC_PATH = ROOT / "pipeline/scripts/v502_v501_outer_preintent_materializer_path_mismatch_failure_forensic.json"
V500_AUTHORITY_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority_seed1642_20260825"
V500_AUTHORITY_EVIDENCE_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_execution_authority_materialization_evidence_seed1642_20260825"
V500_DEPLOYMENT_EVIDENCE_ROOT = J / "v500_authority_sources_corrected_atomic_deployment_evidence_seed1642_20260825"
V500_OUTER_EVIDENCE_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_outer_execution_evidence_seed1642_20260825"
V500_INNER_ATTEMPT_ROOT = J / "v500_v499_v498_v497_v496_v495_failure_tree_diagnostic_reconciliation_inner_attempt_seed1642_20260825"
PREEXECUTION_FORENSIC_ROOT = J / "v501_v500_outer_inner_binding_mismatch_preexecution_forensic_seed1643_20260825"
FAILED_V496_MATERIALIZATION_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_materialization_evidence_seed1638_20260825"
FAILED_V496_AUTHORITY_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repair_execution_authority_seed1638_20260825"
FAILED_V496_OUTER_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_outer_execution_evidence_seed1638_20260825"
FAILED_V496_INNER_ROOT = J / "v496_v495_v490_interpreter_validator_process_schema_repaired_reconciliation_inner_attempt_seed1638_20260825"
FAILED_V497_DIAGNOSTIC_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1639_20260825"
FAILED_V497_DIAGNOSTIC_EVIDENCE_ROOT = J / "v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1639_20260825"
V498_DIAGNOSTIC_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_seed1640_20260825"
V498_DIAGNOSTIC_EVIDENCE_ROOT = J / "v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1640_20260825"
V499_ADAPTER_ROOT = J / "v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_seed1641_20260825"
V499_ADAPTER_EVIDENCE_ROOT = J / "v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_execution_evidence_seed1641_20260825"
FAILED_V493_OUTER_ROOT = J / "v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825"
V493_AUTHORITY_ROOT = J / "v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825"
V493_FAILED_INNER_ROOT = J / "v493_v492_v490_corrected_reconciliation_inner_attempt_seed1635_20260825"
V493_PROCESS_PATH = J / "v493_v492_v490_corrected_reconciliation_execution_authority_materialization_evidence_seed1635_20260825/process_receipt.json"
FAILED_V494_AUTHORITY_ROOT = J / "v494_v493_v490_interpreter_symlink_repair_execution_authority_seed1636_20260825"
FAILED_V494_OUTER_ROOT = J / "v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer_execution_evidence_seed1636_20260825"
FAILED_V494_INNER_ROOT = J / "v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner_attempt_seed1636_20260825"
FAILED_V495_AUTHORITY_ROOT = J / "v495_v494_v490_interpreter_validator_repair_execution_authority_seed1637_20260825"
FAILED_V495_OUTER_ROOT = J / "v495_v494_v490_interpreter_validator_repaired_reconciliation_outer_execution_evidence_seed1637_20260825"
FAILED_V495_INNER_ROOT = J / "v495_v494_v490_interpreter_validator_repaired_reconciliation_inner_attempt_seed1637_20260825"
FAILED_V495_MATERIALIZATION_ROOT = J / "v495_v494_v490_interpreter_validator_repair_execution_authority_materialization_evidence_seed1637_20260825"
OLD_V490_INNER_ROOT = J / "v490_v489_v488_v487_c71_exact7_schema_repair_attempt_seed1632_20260825"
TRANSPARENT_PATH = J / "v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT = Path("/root/v485_v169_cache_qualification_seed1627_20260824")

SOURCE_ROLES = {
    "v493_authority_contract", "v493_authority_materializer", "v493_outer_wrapper",
    "v493_inner_wrapper", "v494_authority_contract", "v494_authority_materializer",
    "v494_outer_wrapper", "v494_inner_wrapper", "corrected_inner_wrapper",
    "v495_authority_contract", "v495_authority_materializer",
    "v495_outer_wrapper", "v495_inner_wrapper",
    "reconciler_r2", "outer_execution_wrapper", "phase_a_design_contract",
    "v494_failure_forensic", "v494_failure_transport_script",
    "v496_authority_contract", "v496_authority_materializer", "v496_inner_wrapper", "v496_outer_wrapper",
    "failed_v497_diagnostic_source", "failed_v497_diagnostic_helper", "failed_v497_diagnostic_script",
    "v498_diagnostic_source", "v498_diagnostic_helper", "v498_diagnostic_script",
    "v499_adapter_source", "v499_adapter_helper", "v499_adapter_script",
    "v500_authority_contract", "v500_authority_materializer",
    "v500_inner_wrapper", "v500_outer_wrapper_invalid_unconsumed",
    "v500_authority_transport_helper", "v500_authority_transport_script",
    "v500_corrected_deployer", "v501_preexecution_forensic_source",
    "v501_authority_contract", "v501_authority_materializer",
    "v501_inner_wrapper", "v501_outer_wrapper_invalid_consumed",
    "v501_authority_transport_helper", "v501_authority_transport_script",
    "v502_preexecution_forensic_source",
}
SOURCE_ROLE_ORDER = [
    "v493_authority_contract", "v493_authority_materializer", "v493_outer_wrapper", "v493_inner_wrapper",
    "v494_authority_contract", "v494_authority_materializer", "v494_outer_wrapper", "v494_inner_wrapper",
    "v495_authority_contract", "v495_authority_materializer", "v495_outer_wrapper", "v495_inner_wrapper",
    "v496_authority_contract", "v496_authority_materializer", "v496_inner_wrapper", "v496_outer_wrapper",
    "v494_failure_forensic", "v494_failure_transport_script", "failed_v497_diagnostic_source",
    "failed_v497_diagnostic_helper", "failed_v497_diagnostic_script", "v498_diagnostic_source",
    "v498_diagnostic_helper", "v498_diagnostic_script", "v499_adapter_source", "v499_adapter_helper",
    "v499_adapter_script", "phase_a_design_contract", "reconciler_r2",
    "v500_authority_contract", "v500_authority_materializer", "v500_inner_wrapper",
    "v500_outer_wrapper_invalid_unconsumed", "v500_authority_transport_helper",
    "v500_authority_transport_script", "v500_corrected_deployer",
    "v501_preexecution_forensic_source", "v501_authority_contract", "v501_authority_materializer", "v501_inner_wrapper",
    "v501_outer_wrapper_invalid_consumed", "v501_authority_transport_helper",
    "v501_authority_transport_script", "v502_preexecution_forensic_source",
    "corrected_inner_wrapper", "outer_execution_wrapper",
]
SOURCE_ALIASES = {
    "v493_authority_contract": "v493_authority_contract", "v493_authority_materializer": "v493_authority_materializer_source",
    "v493_outer_wrapper": "v493_outer_wrapper_source", "v493_inner_wrapper": "v493_inner_wrapper_source",
    "v494_authority_contract": "v494_authority_contract_source", "v494_authority_materializer": "v494_authority_materializer_source",
    "v494_outer_wrapper": "v494_outer_wrapper_source", "v494_inner_wrapper": "v494_inner_wrapper_source",
    "v495_authority_contract": "v495_authority_contract_source", "v495_authority_materializer": "v495_authority_materializer_source",
    "v495_outer_wrapper": "v495_outer_wrapper_source", "v495_inner_wrapper": "v495_inner_wrapper_source",
    "v496_authority_contract": "v496_authority_contract_source", "v496_authority_materializer": "v496_authority_materializer_source",
    "v496_inner_wrapper": "v496_inner_wrapper_source", "v496_outer_wrapper": "v496_outer_wrapper_source",
    "v494_failure_forensic": "v494_materializer_failure_forensic", "v494_failure_transport_script": "v494_materializer_failure_transport_script",
    "failed_v497_diagnostic_source": "failed_v497_diagnostic_source", "failed_v497_diagnostic_helper": "failed_v497_diagnostic_helper_source",
    "failed_v497_diagnostic_script": "failed_v497_diagnostic_script_source", "v498_diagnostic_source": "v498_diagnostic_source",
    "v498_diagnostic_helper": "v498_diagnostic_helper_source", "v498_diagnostic_script": "v498_diagnostic_script_source",
    "v499_adapter_source": "v499_adapter_source", "v499_adapter_helper": "v499_adapter_helper_source",
    "v499_adapter_script": "v499_adapter_script_source", "phase_a_design_contract": "phase_a_design_contract_source",
    "reconciler_r2": "reconciler_r2_source", "corrected_inner_wrapper": "corrected_inner_wrapper_source",
    "outer_execution_wrapper": "outer_execution_wrapper_source",
    "v500_authority_contract": "v500_authority_contract_source",
    "v500_authority_materializer": "v500_authority_materializer_source",
    "v500_inner_wrapper": "v500_inner_wrapper_source",
    "v500_outer_wrapper_invalid_unconsumed": "v500_invalid_outer_wrapper_source",
    "v500_authority_transport_helper": "v500_authority_transport_helper_source",
    "v500_authority_transport_script": "v500_authority_transport_script_source",
    "v500_corrected_deployer": "v500_corrected_deployer_source",
    "v501_preexecution_forensic_source": "v501_preexecution_forensic_source",
    "v501_authority_contract": "v501_authority_contract_source",
    "v501_authority_materializer": "v501_authority_materializer_source",
    "v501_inner_wrapper": "v501_inner_wrapper_source",
    "v501_outer_wrapper_invalid_consumed": "v501_invalid_outer_wrapper_source",
    "v501_authority_transport_helper": "v501_authority_transport_helper_source",
    "v501_authority_transport_script": "v501_authority_transport_script_source",
    "v502_preexecution_forensic_source": "v502_preexecution_forensic_source",
}
CONTRACT_TOP_KEYS = {
    "format", "status", "seed", "lineage", "source_closure", "source_closure_sha256",
    "authority_materializer_source", "v493_authority_contract", "v493_authority_receipt",
    "v493_authority_registration_tree", "v493_authority_materialization_evidence_tree",
    "v493_authority_materializer_process_receipt", "v493_outer_failure_tree",
    "v493_outer_terminal_receipt", "v493_outer_failure_ancestry",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "phase_a_design_contract_record", "historical_absences",
    "current_absences_after_authority", "authority_receipt_contract", "execution_boundary",
    "execution_interpreter_contract",
    "v494_authority_contract", "v494_materializer_failure_forensic",
    "v494_materializer_failure_transport_script", "v494_materializer_failure_ancestry",
    "v495_authority_contract", "v495_materializer_failure_tree",
    "v495_materializer_failure_process_receipt", "v495_materializer_failure_ancestry",
    "v493_authority_materializer_process_schema",
    "v496_authority_contract", "v496_materializer_failure_tree",
    "v496_materializer_failure_process_receipt", "v496_materializer_failure_ancestry",
    "failed_v497_diagnostic_failure_tree", "failed_v497_diagnostic_process_receipt",
    "failed_v497_diagnostic_failure_ancestry",
    "v498_diagnostic_receipt", "v498_diagnostic_registration_tree",
    "v498_diagnostic_execution_evidence_tree", "v498_diagnostic_process_receipt",
    "v498_diagnostic_schema", "v499_adapter_receipt", "v499_adapter_registration_tree",
    "v499_adapter_execution_evidence_tree", "v499_adapter_process_receipt", "v499_adapter_schema",
    "normalized_v498_diagnostic_process_receipt", "materializer_checkpoint_contract",
    "source_role_order", "source_aliases",
    "v500_authority_receipt", "v500_authority_registration_tree",
    "v500_authority_materialization_evidence_tree", "v500_authority_materializer_process_receipt",
    "v500_corrected_deployment_evidence_tree", "v500_corrected_deployment_process_receipt",
    "v500_invalid_outer_unconsumed_ancestry", "v501_preexecution_forensic_receipt",
    "v501_preexecution_forensic_registration_tree", "v501_preexecution_forensic_schema",
    "v501_authority_receipt", "v501_authority_registration_tree",
    "v501_authority_materialization_evidence_tree", "v501_authority_materializer_process_receipt",
    "v501_invalid_outer_consumed_ancestry", "v502_preexecution_forensic",
    "v502_preexecution_forensic_schema", "source_path_literal_bijection_contract",
}
AUTH_TOP_KEYS = {
    "format", "status", "passed", "authority_design_contract",
    "authority_materializer_source", "outer_execution_wrapper_source",
    "corrected_inner_wrapper_source", "v493_inner_wrapper_source",
    "phase_a_design_contract_source", "reconciler_r2_source",
    "v493_authority_contract", "v493_authority_materializer_source",
    "v493_outer_wrapper_source", "v493_authority_receipt",
    "v493_authority_registration_tree", "v493_authority_materialization_evidence_tree",
    "v493_authority_materializer_process_receipt", "failed_v493_outer_execution_tree",
    "failed_v493_outer_terminal_receipt",
    "repair_formal_registration_tree", "postregistration_static_registration_tree",
    "f813_registration_tree", "source_closure", "source_closure_sha256",
    "outer_evidence_root", "inner_attempt_root", "transparent_static_receipt_path",
    "historical_absences", "required_absences", "checks", "check_keys",
    "check_key_set_sha256", "checks_sha256", "input_pre_snapshot",
    "input_post_snapshot", "input_snapshots_exactly_equal", "authorization",
    "runtime_observation", "execution_boundary", "execution_interpreter_evidence",
    "v494_authority_contract_source", "v494_authority_materializer_source",
    "v494_outer_wrapper_source", "v494_inner_wrapper_source",
    "v494_materializer_failure_forensic", "v494_materializer_failure_transport_script",
    "v494_materializer_failure_ancestry",
    "v495_authority_contract_source", "v495_authority_materializer_source",
    "v495_outer_wrapper_source", "v495_inner_wrapper_source",
    "v495_materializer_failure_tree", "v495_materializer_failure_process_receipt",
    "v495_materializer_failure_ancestry", "v493_authority_materializer_process_schema",
    "v496_authority_contract_source", "v496_authority_materializer_source",
    "v496_inner_wrapper_source", "v496_outer_wrapper_source",
    "failed_v497_diagnostic_source", "failed_v497_diagnostic_helper_source",
    "failed_v497_diagnostic_script_source", "v498_diagnostic_source",
    "v498_diagnostic_helper_source", "v498_diagnostic_script_source",
    "v499_adapter_source", "v499_adapter_helper_source", "v499_adapter_script_source",
    "v496_materializer_failure_tree", "v496_materializer_failure_process_receipt",
    "v496_materializer_failure_ancestry", "failed_v497_diagnostic_failure_tree",
    "failed_v497_diagnostic_process_receipt", "failed_v497_diagnostic_failure_ancestry",
    "v498_diagnostic_receipt", "v498_diagnostic_registration_tree",
    "v498_diagnostic_execution_evidence_tree", "v498_diagnostic_process_receipt",
    "v498_diagnostic_schema", "v499_adapter_receipt", "v499_adapter_registration_tree",
    "v499_adapter_execution_evidence_tree", "v499_adapter_process_receipt", "v499_adapter_schema",
    "normalized_v498_diagnostic_process_receipt", "materializer_pre_root_checkpoint",
    "materializer_pre_root_checkpoint_sha256", "source_role_order", "source_aliases",
    "v500_authority_contract_source", "v500_authority_materializer_source",
    "v500_inner_wrapper_source", "v500_invalid_outer_wrapper_source",
    "v500_authority_transport_helper_source", "v500_authority_transport_script_source",
    "v500_corrected_deployer_source", "v501_preexecution_forensic_source",
    "v500_authority_receipt", "v500_authority_registration_tree",
    "v500_authority_materialization_evidence_tree", "v500_authority_materializer_process_receipt",
    "v500_corrected_deployment_evidence_tree", "v500_corrected_deployment_process_receipt",
    "v500_invalid_outer_unconsumed_ancestry", "v501_preexecution_forensic_receipt",
    "v501_preexecution_forensic_registration_tree", "v501_preexecution_forensic_schema",
    "v501_authority_contract_source", "v501_authority_materializer_source",
    "v501_inner_wrapper_source", "v501_invalid_outer_wrapper_source",
    "v501_authority_transport_helper_source", "v501_authority_transport_script_source",
    "v502_preexecution_forensic_source", "v501_authority_receipt",
    "v501_authority_registration_tree", "v501_authority_materialization_evidence_tree",
    "v501_authority_materializer_process_receipt", "v501_invalid_outer_consumed_ancestry",
    "v502_preexecution_forensic", "v502_preexecution_forensic_schema",
    "source_path_literal_bijection",
}
AUTH_CHECK_KEYS = sorted({
    "authority_contract_current", "authority_materializer_current",
    "corrected_inner_current", "corrected_inner_diff_exact", "current_absences",
    "execution_boundary", "f813_exact2", "failed_v493_outer_exact4",
    "failed_v493_outer_no_retry_partition", "failed_v493_outer_terminal_exact",
    "gpu_empty", "historical_absences", "input_snapshots_equal", "no_live_process",
    "outer_wrapper_current", "phase_a_contract_current", "phase_a_contract_spec_exact2",
    "postregistration_static_exact1", "r2_current", "repair_formal_exact1",
    "source_closure_current", "v493_authority_contract_current", "v493_authority_exact3",
    "v493_authority_materialization_evidence_exact6",
    "v493_authority_materializer_process_exact", "v493_outer_wrapper_current",
    "execution_interpreter_chain_exact", "execution_interpreter_runtime_exact",
    "v494_authority_contract_current", "v494_authority_materializer_current",
    "v494_inner_wrapper_current", "v494_outer_wrapper_current",
    "v494_failure_forensic_exact", "v494_failure_transport_script_current",
    "v494_materializer_failure_partition_exact", "validator_torch_version_scope_exact",
    "validator_torch_version_tamper_suite_passed",
    "v495_authority_contract_current", "v495_authority_materializer_current",
    "v495_inner_wrapper_current", "v495_outer_wrapper_current",
    "v495_materializer_failure_exact6", "v495_materializer_failure_process_exact",
    "v493_process_schema_exact24", "validator_process_schema_tamper_suite_passed",
    "v496_authority_contract_current", "v496_authority_materializer_current",
    "v496_inner_wrapper_current", "v496_outer_wrapper_current",
    "v496_materializer_failure_exact6", "v496_materializer_failure_process_exact",
    "v496_materializer_failure_partition_exact", "failed_v497_diagnostic_source_current",
    "failed_v497_diagnostic_helper_current", "failed_v497_diagnostic_script_current",
    "failed_v497_diagnostic_exact6", "failed_v497_diagnostic_process_exact",
    "failed_v497_diagnostic_partition_exact", "v498_diagnostic_source_current",
    "v498_diagnostic_helper_current", "v498_diagnostic_script_current",
    "v498_diagnostic_receipt_exact", "v498_diagnostic_registration_exact1",
    "v498_diagnostic_evidence_exact6", "v498_diagnostic_process_exact",
    "v498_named8_double_snapshot_exact", "v499_adapter_source_current",
    "v499_adapter_helper_current", "v499_adapter_script_current",
    "v499_adapter_receipt_exact", "v499_adapter_registration_exact1",
    "v499_adapter_evidence_exact6", "v499_adapter_process_exact",
    "v499_normalization_leafdiff_exact", "normalized_v498_process_current_only",
    "materializer_checkpoint_named8_exact", "materializer_checkpoint_double_snapshot_exact",
    "materializer_checkpoint_stdout_fsynced", "diagnostic_normalization_tamper_suite_passed",
    "source_role_order_exact", "source_aliases_exact",
    "v500_authority_exact1", "v500_authority_materialization_evidence_exact6",
    "v500_authority_materializer_process_exact", "v500_corrected_deployment_exact6",
    "v500_corrected_deployment_process_exact", "v500_authority_transport_helper_current",
    "v500_authority_transport_script_current", "v500_corrected_deployer_current",
    "v500_invalid_outer_current", "v500_inner_current",
    "v500_invalid_outer_binding_mismatch_exact", "v500_invalid_outer_unconsumed_zero_state",
    "v501_preexecution_forensic_exact1", "v501_preexecution_forensic_schema_exact",
    "v501_preexecution_forensic_source_current",
    "v501_authority_exact1", "v501_authority_materialization_evidence_exact6",
    "v501_authority_materializer_process_exact", "v501_authority_transport_helper_current",
    "v501_authority_transport_script_current", "v501_invalid_outer_current",
    "v501_inner_current", "v501_path_mismatch_exact",
    "v501_consumed_zero_inner_r2", "v502_preexecution_forensic_exact",
    "v502_preexecution_forensic_schema_exact", "v502_path_bijection_tamper_suite_passed",
})
AUTHORIZATION = {
    "outer_execution_wrapper_authorized": True,
    "outer_attempts_authorized": 1,
    "outer_attempts_consumed": 0,
    "retry_authorized": False,
    "direct_corrected_inner_authorized": False,
    "direct_r2_authorized": False,
    "nested_corrected_inner_invocations_authorized": 1,
    "nested_r2_invocations_authorized": 1,
    "nested_corrected_inner_only_via_outer": True,
    "nested_r2_only_via_corrected_inner": True,
    "phase_a_authorized": False,
    "cache_authorized": False,
    "training_authorized": False,
    "folds_authorized": 0,
    "policy_updates": 0,
    "s1_authorized": False,
    "zero_update_authorized": False,
    "rl_authorized": False,
    "submission_authorized": False,
    "reward_read_authorized": False,
    "dev_hidden_final_outcome_read_authorized": False,
}
RUNTIME = {
    "execution_authority_materialized": True,
    "outer_execution_wrapper_executed": False,
    "corrected_inner_wrapper_executed": False,
    "reconciler_r2_executed": False,
    "transparent_receipt_created": False,
    "phase_a_executed": False,
    "training_launched": False,
    "folds": 0,
    "policy_updates": 0,
}
EXECUTION_BOUNDARY = {
    "authority_materialization_only": True,
    "outer_execution_wrapper_invocations": 0,
    "corrected_inner_wrapper_invocations": 0,
    "reconciler_r2_invocations": 0,
    "phase_a_invocations": 0,
    "training_invocations": 0,
    "reward_reads": 0,
    "dev_hidden_final_outcome_reads": 0,
}
V493_PROCESS_KEYS = {
    "argv", "authority_receipt", "authority_tree", "cleanup",
    "corrected_inner_wrapper_invocations", "failed_v492_outer_terminal_receipt",
    "format", "helper_returncode", "intent", "materializer_invocations",
    "materializer_returncode", "outer_wrapper_invocations", "passed",
    "post_snapshot", "pre_post_snapshots_exactly_equal", "pre_snapshot",
    "r2_invocations", "retry_authorized", "status", "stderr", "stdout",
    "transport_helper", "transport_helper_copy", "wall_seconds",
}


class ControlledSignal(BaseException):
    pass


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pending(value) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in ("PENDING", "PLACEHOLDER", "TO_BE_FILLED", "TBD"))
    if isinstance(value, dict):
        return any(pending(item) for item in value.values())
    if isinstance(value, list):
        return any(pending(item) for item in value)
    return False


def regular(path: Path | str, digest: str | None = None, logical_bytes: int | None = None) -> dict:
    path = Path(path)
    if path != path.resolve() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"file closure: {path}")
    actual = {"path": str(path), "sha256": sha(path), "logical_bytes": path.stat().st_size}
    if digest is not None and actual["sha256"] != digest:
        raise RuntimeError(f"file digest: {path}")
    if logical_bytes is not None and actual["logical_bytes"] != logical_bytes:
        raise RuntimeError(f"file bytes: {path}")
    return actual


def record(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "logical_bytes"}:
        raise RuntimeError("record schema")
    return regular(value["path"], value["sha256"], value["logical_bytes"])


def execution_interpreter_evidence() -> dict:
    lexical_path = Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
    intermediate_path = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python")
    resolved_path = Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")

    def lstat_row(path: Path) -> dict:
        observed = os.lstat(path)
        return {"device": observed.st_dev, "inode": observed.st_ino,
                "mode": observed.st_mode, "size": observed.st_size}

    if sys.executable != str(lexical_path):
        raise RuntimeError("interpreter lexical executable")
    lexical_lstat = lstat_row(lexical_path)
    lexical_target = os.readlink(lexical_path)
    intermediate_lstat = lstat_row(intermediate_path)
    intermediate_target = os.readlink(intermediate_path)
    resolved_lstat = lstat_row(resolved_path)
    if (not stat.S_ISLNK(lexical_lstat["mode"])
            or lexical_target != str(intermediate_path)
            or not stat.S_ISLNK(intermediate_lstat["mode"])
            or intermediate_target != "python3.11"
            or (intermediate_path.parent / intermediate_target).resolve(strict=True) != resolved_path
            or stat.S_ISLNK(resolved_lstat["mode"])
            or not stat.S_ISREG(resolved_lstat["mode"])):
        raise RuntimeError("interpreter chain drift")
    import numpy
    import torch
    evidence = {
        "lexical": {"path": str(lexical_path), "lstat": lexical_lstat,
                    "readlink": lexical_target},
        "intermediate": {"path": str(intermediate_path), "lstat": intermediate_lstat,
                         "readlink": intermediate_target},
        "resolved": {**regular(resolved_path,
                               "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788",
                               25555040), "lstat": resolved_lstat},
        "runtime": {"sys_executable": sys.executable, "python_version": sys.version,
                    "numpy_version": numpy.__version__, "torch_version": torch.__version__},
    }
    expected = {
        "lexical": {"path": str(lexical_path),
                    "lstat": {"device": 2304, "inode": 17217118070,
                              "mode": 41471, "size": 49},
                    "readlink": str(intermediate_path)},
        "intermediate": {"path": str(intermediate_path),
                         "lstat": {"device": 2304, "inode": 7529246267,
                                   "mode": 41471, "size": 10},
                         "readlink": "python3.11"},
        "resolved": {"path": str(resolved_path),
                     "sha256": "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788",
                     "logical_bytes": 25555040,
                     "lstat": {"device": 2304, "inode": 7529484239,
                               "mode": 33277, "size": 25555040}},
        "runtime": {"sys_executable": str(lexical_path),
                    "python_version": "3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]",
                    "numpy_version": "1.26.4", "torch_version": "2.7.0+cu128"},
    }
    if evidence != expected:
        raise RuntimeError("interpreter evidence drift")
    return evidence


def exact_tree(root: Path | str) -> dict:
    root = Path(root)
    if root != root.resolve() or not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"tree symlink: {path}")
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), sha(path), path.stat().st_size])
        elif not path.is_dir():
            raise RuntimeError(f"tree nonregular: {path}")
    lines = "".join(f"{digest}  {rel}\n" for rel, digest, _ in rows).encode()
    triples = json.dumps(rows, separators=(",", ":")).encode()
    return {
        "inventory": rows,
        "file_count": len(rows),
        "logical_file_bytes": sum(row[2] for row in rows),
        "sha256sum_lines_digest_sha256": hashlib.sha256(lines).hexdigest(),
        "canonical_json_triples_digest_sha256": hashlib.sha256(triples).hexdigest(),
    }


def rooted_tree(root: Path | str) -> dict:
    return {"root": str(Path(root)), **exact_tree(root)}


def verify_tree(spec: dict) -> dict:
    keys = {"root", "inventory", "file_count", "logical_file_bytes",
            "sha256sum_lines_digest_sha256", "canonical_json_triples_digest_sha256"}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise RuntimeError("tree schema")
    actual = rooted_tree(spec["root"])
    if actual != spec:
        raise RuntimeError(f"tree closure: {spec['root']}")
    return actual


def decode_absences(value: dict, expected_keys: set[str]) -> dict[str, Path]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("absence key set")
    result = {}
    for name, row in value.items():
        if not isinstance(row, dict) or set(row) != {"path"} or not isinstance(row["path"], str):
            raise RuntimeError(f"absence row: {name}")
        path = Path(row["path"])
        if path != path.resolve():
            raise RuntimeError(f"absence path: {name}")
        result[name] = path
    return result


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def copy_fsync(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
        dst.flush()
        os.fsync(dst.fileno())


def directory_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("directory ownership")
    return stat.st_dev, stat.st_ino


def cleanup_owned(path: Path, identity: tuple[int, int]) -> None:
    if path.exists() and not path.is_symlink() and directory_identity(path) == identity:
        shutil.rmtree(path)
        fsync_dir(path.parent)


def live_processes(fragments: set[str]) -> list[dict]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(fragment in cmd for fragment in fragments):
            found.append({"pid": int(proc.name), "cmdline": cmd})
    return found


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("gpu query")
    return [line for line in result.stdout.splitlines() if line.strip()]


def snapshot(files: dict[str, str], trees: dict[str, str], absences: dict[str, Path]) -> dict:
    file_rows = []
    for name, target in sorted(files.items()):
        row = regular(target)
        file_rows.append({"name": name, **row})
    tree_rows = {name: rooted_tree(target) for name, target in sorted(trees.items())}
    absence_rows = []
    for name, target in sorted(absences.items()):
        absence_rows.append({"name": name, "path": str(target), "absent": not os.path.lexists(target)})
    if not all(row["absent"] for row in absence_rows):
        raise RuntimeError("snapshot absence")
    value = {"files": file_rows, "trees": tree_rows, "absences": absence_rows,
             "execution_interpreter_evidence": execution_interpreter_evidence()}
    value["snapshot_sha256"] = csha(value)
    return value


def validate_torch_version_probe_ast(parsed: ast.Module) -> None:
    functions = [node for node in ast.walk(parsed)
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "execution_interpreter_evidence"]
    if len(functions) != 1:
        raise RuntimeError("torch version probe function")
    function = functions[0]
    parents = {child: parent for parent in ast.walk(parsed) for child in ast.iter_child_nodes(parent)}
    torch_imports = []
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    torch_imports.append((node, alias))
        elif isinstance(node, ast.ImportFrom) and (
                (node.module and node.module.split(".")[0] == "torch")
                or any(alias.name.split(".")[0] == "torch" for alias in node.names)):
            raise RuntimeError("torch import-from forbidden")
    if (len(torch_imports) != 1
            or len(torch_imports[0][0].names) != 1
            or torch_imports[0][1].name != "torch"
            or torch_imports[0][1].asname is not None
            or parents.get(torch_imports[0][0]) is not function):
        raise RuntimeError("torch import scope")
    torch_names = [node for node in ast.walk(parsed)
                   if isinstance(node, ast.Name) and node.id == "torch"]
    torch_attributes = [node for node in ast.walk(parsed)
                        if isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name) and node.value.id == "torch"]
    if (len(torch_names) != 1 or len(torch_attributes) != 1
            or torch_attributes[0].attr != "__version__"
            or not isinstance(torch_attributes[0].ctx, ast.Load)
            or not isinstance(torch_names[0].ctx, ast.Load)
            or torch_attributes[0].value is not torch_names[0]
            or parents.get(torch_attributes[0]) is None
            or function not in list(ast.iter_child_nodes(parsed))):
        raise RuntimeError("torch version-only load")
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call) and any(
                isinstance(child, ast.Name) and child.id == "torch" for child in ast.walk(node)):
            raise RuntimeError("torch call forbidden")


def validate_v493_process_receipt(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != V493_PROCESS_KEYS:
        raise RuntimeError("v493 process exact24 keyset")
    integer_partition = {
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "r2_invocations": 0,
    }
    if any(type(value.get(key)) is not int or value.get(key) != expected
           for key, expected in integer_partition.items()):
        raise RuntimeError("v493 process integer partition")
    if (value.get("status") != "passed_exact_once_no_outer_or_inner_execution"
            or value.get("passed") is not True
            or value.get("retry_authorized") is not False
            or value.get("pre_post_snapshots_exactly_equal") is not True
            or value.get("cleanup", {}).get("reaped") is not True
            or value.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v493 process semantics")


PREDICATE_NAMES = [
    "root_exact", "file_count_exact", "logical_file_bytes_exact",
    "sha256sum_lines_digest_exact", "canonical_json_triples_digest_exact",
    "process_receipt_inventory_row_exact", "process_receipt_sha256_exact",
    "process_receipt_logical_bytes_exact",
]
V495_EXPECTED_TREE = {
    "root": str(FAILED_V495_MATERIALIZATION_ROOT),
    "file_count": 6, "logical_file_bytes": 32139,
    "sha256sum_lines_digest_sha256": "21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1b",
    "canonical_json_triples_digest_sha256": "9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290",
}
V495_PROCESS_RECORD = {
    "path": str(FAILED_V495_MATERIALIZATION_ROOT / "process_receipt.json"),
    "sha256": "72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4",
    "logical_bytes": 411,
}


def failure_snapshot() -> dict:
    return {**rooted_tree(FAILED_V495_MATERIALIZATION_ROOT),
            # Observe without expected-value gates so a digest/size mismatch is
            # represented in the named checkpoint instead of raising before it.
            "process_receipt_record": regular(V495_PROCESS_RECORD["path"])}


def named_failure_predicates(value: dict) -> list[dict]:
    expected_row = ["process_receipt.json", V495_PROCESS_RECORD["sha256"],
                    V495_PROCESS_RECORD["logical_bytes"]]
    observed_row = next((row for row in value["inventory"] if row[0] == "process_receipt.json"), None)
    pairs = [
        ("root_exact", V495_EXPECTED_TREE["root"], value["root"]),
        ("file_count_exact", V495_EXPECTED_TREE["file_count"], value["file_count"]),
        ("logical_file_bytes_exact", V495_EXPECTED_TREE["logical_file_bytes"], value["logical_file_bytes"]),
        ("sha256sum_lines_digest_exact", V495_EXPECTED_TREE["sha256sum_lines_digest_sha256"], value["sha256sum_lines_digest_sha256"]),
        ("canonical_json_triples_digest_exact", V495_EXPECTED_TREE["canonical_json_triples_digest_sha256"], value["canonical_json_triples_digest_sha256"]),
        ("process_receipt_inventory_row_exact", expected_row, observed_row),
        ("process_receipt_sha256_exact", V495_PROCESS_RECORD["sha256"], value["process_receipt_record"]["sha256"]),
        ("process_receipt_logical_bytes_exact", V495_PROCESS_RECORD["logical_bytes"], value["process_receipt_record"]["logical_bytes"]),
    ]
    if [row[0] for row in pairs] != PREDICATE_NAMES:
        raise RuntimeError("predicate order")
    return [{"name": name, "expected": expected, "observed": observed,
             "expected_type": type(expected).__name__, "observed_type": type(observed).__name__,
             "passed": type(expected) is type(observed) and expected == observed}
            for name, expected, observed in pairs]


def json_leaf_diff(before, after, path=()) -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                rows.append({"path": list(path + (key,)), "before": before.get(key), "after": after.get(key)})
            else:
                rows.extend(json_leaf_diff(before[key], after[key], path + (key,)))
        return rows
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        rows = []
        for index, (left, right) in enumerate(zip(before, after)):
            rows.extend(json_leaf_diff(left, right, path + (index,)))
        return rows
    return [] if type(before) is type(after) and before == after else [
        {"path": list(path), "before": before, "after": after}]


def validate_normalized_process(original: dict, normalized: dict) -> list[dict]:
    if csha(original) != "4f043cb1e415b88a5f02a4b5180f78e94e295ca351065edbeccff9df8172f59f":
        raise RuntimeError("original v498 process value")
    if csha(normalized) != "9318ba42a5ce23fce6b3179756c99b9ef18d907ce338930e61eb0c3dae38804d":
        raise RuntimeError("normalized v498 process value")
    diff = json_leaf_diff(original, normalized)
    expected = [{"path": ["transport_helper_copy", "path"],
                 "before": str(V498_DIAGNOSTIC_EVIDENCE_ROOT.with_name(
                     V498_DIAGNOSTIC_EVIDENCE_ROOT.name + ".execution-prep") / "transport_helper.py"),
                 "after": str(V498_DIAGNOSTIC_EVIDENCE_ROOT / "transport_helper.py")}]
    if diff != expected:
        raise RuntimeError("normalized v498 leaf diff")
    if record(normalized["transport_helper_copy"]) != normalized["transport_helper_copy"]:
        raise RuntimeError("normalized helper current")
    return diff


def emit_checkpoint(payload: dict) -> str:
    envelope = {"checkpoint": payload, "checkpoint_sha256": csha(payload)}
    data = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = sys.stdout.fileno()
    observed_stat = os.fstat(descriptor)
    if (not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_size != 0
            or os.lseek(descriptor, 0, os.SEEK_CUR) != 0):
        raise RuntimeError("checkpoint stdout must be regular helper log")
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if type(written) is not int or written <= 0:
            raise RuntimeError("checkpoint stdout short write")
        offset += written
    if offset != len(data):
        raise RuntimeError("checkpoint stdout length")
    os.fsync(descriptor)
    observed = os.pread(descriptor, len(data), 0)
    trailing = os.pread(descriptor, 1, len(data))
    if observed != data or trailing != b"" or os.fstat(descriptor).st_size != len(data):
        raise RuntimeError("checkpoint stdout readback")
    decoded = json.loads(observed.decode())
    if decoded != envelope or decoded.get("checkpoint_sha256") != csha(decoded.get("checkpoint")):
        raise RuntimeError("checkpoint stdout JSON digest")
    return envelope["checkpoint_sha256"]


def append_success_line(payload: dict) -> None:
    """Durably append/read back the sole success line to the helper O_RDWR log."""
    descriptor = sys.stdout.fileno()
    observed_stat = os.fstat(descriptor)
    if not stat.S_ISREG(observed_stat.st_mode) or observed_stat.st_size <= 0:
        raise RuntimeError("success stdout must contain checkpoint")
    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    if offset != observed_stat.st_size:
        raise RuntimeError("success stdout offset")
    first = os.pread(descriptor, observed_stat.st_size, 0)
    if first.count(b"\n") != 1 or not first.endswith(b"\n"):
        raise RuntimeError("success stdout checkpoint framing")
    checkpoint_envelope = json.loads(first.decode())
    if (set(checkpoint_envelope) != {"checkpoint", "checkpoint_sha256"}
            or checkpoint_envelope["checkpoint_sha256"] != csha(checkpoint_envelope["checkpoint"])):
        raise RuntimeError("success stdout checkpoint digest")
    second = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    written_total = 0
    while written_total < len(second):
        written = os.write(descriptor, second[written_total:])
        if type(written) is not int or written <= 0:
            raise RuntimeError("success stdout short write")
        written_total += written
    if written_total != len(second):
        raise RuntimeError("success stdout length")
    os.fsync(descriptor)
    expected = first + second
    observed = os.pread(descriptor, len(expected), 0)
    trailing = os.pread(descriptor, 1, len(expected))
    if (observed != expected or trailing != b"" or os.fstat(descriptor).st_size != len(expected)
            or observed.count(b"\n") != 2 or json.loads(observed.splitlines()[1]) != payload):
        raise RuntimeError("success stdout readback")


def publish_authority_exact1(root: Path, receipt: dict, files: dict, trees: dict,
                             stable_absences: dict, stable_pre: dict,
                             stage_hook=None, inject_commit_window_signal: bool = False):
    """The production whole-directory publisher, shared by main and fixtures."""
    prep = root.with_name(root.name + ".registration-prep")
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    pending_signal = {"value": None}
    committed = False
    promoted = False
    identity = None
    old_mask = None

    def hook(stage: str) -> None:
        if stage_hook is not None:
            stage_hook(stage)

    def on_signal(signum, _frame):
        pending_signal["value"] = signum
        if not promoted:
            raise ControlledSignal(signum)

    for sig in old_handlers:
        signal.signal(sig, on_signal)
    try:
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        prep.mkdir()
        identity = directory_identity(prep)
        fsync_dir(prep)
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        atomic_json(prep / "authority_receipt.json", receipt)
        if hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        prep_tree = rooted_tree(prep)
        prep_receipt = regular(prep / "authority_receipt.json")
        if (prep_tree["file_count"] != 1
                or prep_tree["inventory"] != [["authority_receipt.json",
                                                prep_receipt["sha256"],
                                                prep_receipt["logical_bytes"]]]):
            raise RuntimeError("prepromote exact1")
        hook("before_final_snapshot")
        if snapshot(files, trees, stable_absences) != stable_pre:
            raise RuntimeError("prepromote drift")
        hook("before_rename")
        os.replace(prep, root)
        fsync_dir(root.parent)
        final_tree = rooted_tree(root)
        if (final_tree["file_count"] != 1
                or final_tree["inventory"] != [["authority_receipt.json",
                                                prep_receipt["sha256"],
                                                prep_receipt["logical_bytes"]]]
                or {key: value for key, value in final_tree.items() if key != "root"}
                   != {key: value for key, value in prep_tree.items() if key != "root"}):
            raise RuntimeError("postpromote exact1")
        promoted = True
        if inject_commit_window_signal:
            os.kill(os.getpid(), signal.SIGTERM)
        hook("after_publish_return_before_final_line")
        pending_now = signal.sigpending() if hasattr(signal, "sigpending") else set()
        deferred_signal = next((sig for sig in (signal.SIGINT, signal.SIGTERM)
                                if sig in pending_now), None)
        final_row = final_tree["inventory"][0]
        summary = {"path": str(root / "authority_receipt.json"),
                   "sha256": final_row[1], "logical_bytes": final_row[2],
                   "authority_registration_tree": final_tree,
                   "committed_success": True, "deferred_signal": deferred_signal,
                   "outer_execution_wrapper_invocations": 0,
                   "corrected_inner_wrapper_invocations": 0,
                   "reconciler_r2_invocations": 0}
        append_success_line(summary)
        committed = True
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            old_mask = None
        return final_tree, summary
    except BaseException:
        if not promoted and identity is not None:
            cleanup_owned(prep, identity)
        raise
    finally:
        if old_mask is not None and hasattr(signal, "pthread_sigmask"):
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def validate_corrected_inner(path: Path) -> None:
    """Independently establish the only permitted semantic delta from frozen 733."""
    text = path.read_text()
    parsed = ast.parse(text)
    main_node = next((node for node in parsed.body
                      if isinstance(node, ast.FunctionDef) and node.name == "main"), None)
    main_popens = [] if main_node is None else [
        node for node in ast.walk(main_node) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"]
    spawn_node = next((node for node in parsed.body
                       if isinstance(node, ast.FunctionDef) and node.name == "spawn_owned"), None)
    spawn_popens = [] if spawn_node is None else [
        node for node in ast.walk(spawn_node) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
        and node.func.attr == "Popen"]
    main_spawn_calls = [] if main_node is None else [
        node for node in ast.walk(main_node) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "spawn_owned"]
    mask_calls = [] if spawn_node is None else [
        node for node in ast.walk(spawn_node) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "pthread_sigmask"]
    block_calls = [node for node in mask_calls if node.args and isinstance(node.args[0], ast.Attribute)
                   and node.args[0].attr == "SIG_BLOCK"]
    restore_calls = [node for node in mask_calls if node.args and isinstance(node.args[0], ast.Attribute)
                     and node.args[0].attr == "SIG_SETMASK"]
    owner_assignments = {}
    if spawn_node is not None:
        for node in ast.walk(spawn_node):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == "owner" and isinstance(target.slice, ast.Constant)
                        and target.slice.value in {"process", "started"}):
                    owner_assignments[target.slice.value] = node.lineno
    normalized = text.replace(" ", "")
    if (len(main_popens) != 0 or len(spawn_popens) != 1 or len(main_spawn_calls) != 1
            or len(block_calls) != 1 or len(restore_calls) != 1
            or set(owner_assignments) != {"process", "started"}
            or not (block_calls[0].lineno < spawn_popens[0].lineno
                    < owner_assignments["process"] <= owner_assignments["started"]
                    < restore_calls[0].lineno)
            or 'phase_contract_spec["logical_bytes"]' in text
            or "phase_contract_spec['logical_bytes']" in text
            or "PHASE_A_DESIGN_CONTRACT_BYTES=43960" not in normalized
            or 'set(spec)!={"path","sha256"}' not in normalized
            or "defexecution_interpreter_evidence()" not in normalized
            or "EXECUTION_INTERPRETER_CONTRACT=" not in normalized
            or "regular(Path(sys.executable))" in normalized
            or "regular(RLPY)" in normalized
            or "os.readlink(RLPY)" not in normalized
            or "os.readlink(intermediate)" not in normalized
            or "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64" not in text
            or "11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788" not in text
            or "9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377" not in text):
        raise RuntimeError("corrected inner interpreter and phase-contract repair")
    validate_torch_version_probe_ast(parsed)
    imported = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    if imported & {"transformers", "rlinf"}:
        raise RuntimeError("corrected inner forbidden import")


def source_path_literal_bijection(inner_path: Path, outer_path: Path, contract_record: dict,
                                  materializer_record: dict, sources: dict) -> dict:
    """Resolve frozen global path/sha/bytes triples and reject path aliases."""
    def globals_from(path: Path) -> dict:
        tree = ast.parse(path.read_text())
        values: dict[str, object] = {}
        def evaluate(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
                return node.value
            if isinstance(node, ast.Name) and node.id in values:
                return values[node.id]
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "Path" and len(node.args) == 1):
                value = evaluate(node.args[0])
                return Path(value) if isinstance(value, str) else None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(left, Path) and isinstance(right, str):
                    return left / right
            if isinstance(node, ast.Dict):
                keys = [evaluate(key) for key in node.keys]
                vals = [evaluate(value) for value in node.values]
                if all(isinstance(key, str) for key in keys) and all(isinstance(value, str) for value in vals):
                    return dict(zip(keys, vals))
            return None
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                value = evaluate(statement.value)
                if value is not None:
                    values[target.id] = value
        return values

    inner = globals_from(inner_path)
    outer = globals_from(outer_path)
    expected = {
        "inner": {
            "WRAPPER_PATH": {"path": sources["corrected_inner_wrapper"]["path"]},
            "AUTHORITY_CONTRACT_PATH": {"path": contract_record["path"]},
            "AUTHORITY_MATERIALIZER_PATH": materializer_record,
            "R2_PATH": sources["reconciler_r2"],
        },
        "outer": {
            "SELF_PATH": {"path": sources["outer_execution_wrapper"]["path"]},
            "AUTH_CONTRACT": {"path": contract_record["path"]},
            "AUTH_MATERIALIZER": materializer_record,
            "INNER_WRAPPER": sources["corrected_inner_wrapper"],
            "R2": sources["reconciler_r2"],
        },
    }
    observed = {"inner": inner, "outer": outer}
    for scope, rows in expected.items():
        for name, wanted in rows.items():
            if str(observed[scope].get(name)) != wanted["path"]:
                raise RuntimeError(f"path literal bijection {scope}.{name}")
            sha_name = name.removesuffix("_PATH") + "_SHA"
            bytes_name = name.removesuffix("_PATH") + "_BYTES"
            if ("sha256" in wanted and
                    (observed[scope].get(sha_name) != wanted["sha256"]
                     or observed[scope].get(bytes_name) != wanted["logical_bytes"])):
                raise RuntimeError(f"record literal bijection {scope}.{name}")
    role_paths = {role: sources[role]["path"] for role in SOURCE_ROLE_ORDER if role in sources}
    if inner.get("SOURCE_PATH_LITERALS") != role_paths or outer.get("SOURCE_PATH_LITERALS") != role_paths:
        raise RuntimeError("source role path literal bijection")
    expected["source_role_paths"] = role_paths
    return expected


def synthetic_self_test() -> int:
    checks = {
        "authority_top_count": len(AUTH_TOP_KEYS) == 122,
        "check_count": len(AUTH_CHECK_KEYS) == 108,
        "source_roles": len(SOURCE_ROLES) == 46,
        "source_order_aliases": (len(SOURCE_ROLE_ORDER) == 46
                                  and set(SOURCE_ROLE_ORDER) == SOURCE_ROLES
                                  and set(SOURCE_ALIASES) == SOURCE_ROLES
                                  and len(set(SOURCE_ALIASES.values())) == 46),
        "authorization_boundary": (
            AUTHORIZATION["outer_execution_wrapper_authorized"] is True
            and AUTHORIZATION["direct_corrected_inner_authorized"] is False
            and AUTHORIZATION["direct_r2_authorized"] is False
            and AUTHORIZATION["retry_authorized"] is False
        ),
        "fresh_roots": all("v502_" in path.name for path in
                           (AUTHORITY_ROOT, OUTER_EVIDENCE_ROOT, INNER_ATTEMPT_ROOT)),
    }
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        good = {"a": {"path": str(base / "a")}, "b": {"path": str(base / "b")}}
        checks["absence_decode"] = set(decode_absences(good, {"a", "b"})) == {"a", "b"}
        try:
            decode_absences({"a": str(base / "a")}, {"a"})
            checks["absence_tamper"] = False
        except RuntimeError:
            checks["absence_tamper"] = True
        if hasattr(os, "pread"):
            checkpoint_log = base / "checkpoint.log"
            with checkpoint_log.open("w+b", buffering=0) as stream:
                previous_stdout = sys.stdout
                sys.stdout = stream
                try:
                    payload = {"format": "fixture", "passed": True}
                    expected_sha = emit_checkpoint(payload)
                finally:
                    sys.stdout = previous_stdout
            envelope = json.loads(checkpoint_log.read_text())
            checks["checkpoint_writeall_readback"] = (
                checkpoint_log.read_bytes().endswith(b"\n")
                and envelope == {"checkpoint": payload, "checkpoint_sha256": expected_sha}
                and expected_sha == csha(payload))
            short_log = base / "checkpoint-short.log"
            with short_log.open("w+b", buffering=0) as stream:
                previous_stdout = sys.stdout
                original_write = os.write
                calls = {"value": 0}
                def short_write(fd, data):
                    calls["value"] += 1
                    if calls["value"] == 1 and len(data) > 1:
                        return original_write(fd, data[:max(1, len(data) // 2)])
                    return original_write(fd, data)
                sys.stdout = stream
                os.write = short_write
                try:
                    short_sha = emit_checkpoint({"format": "short-write", "passed": True})
                    checks["checkpoint_short_write_completed"] = (
                        calls["value"] >= 2
                        and json.loads(short_log.read_text())["checkpoint_sha256"] == short_sha)
                finally:
                    os.write = original_write
                    sys.stdout = previous_stdout
            zero_log = base / "checkpoint-zero.log"
            with zero_log.open("w+b", buffering=0) as stream:
                previous_stdout = sys.stdout
                original_write = os.write
                sys.stdout = stream
                os.write = lambda _fd, _data: 0
                try:
                    emit_checkpoint({"format": "zero-write"})
                    checks["checkpoint_zero_write_rejected"] = False
                except RuntimeError:
                    checks["checkpoint_zero_write_rejected"] = True
                finally:
                    os.write = original_write
                    sys.stdout = previous_stdout
        else:
            checks["checkpoint_writeall_readback"] = True
            checks["checkpoint_short_write_completed"] = True
            checks["checkpoint_zero_write_rejected"] = True

        if hasattr(signal, "pthread_sigmask"):
            def publish_fixture(name, hook=None, pending=False):
                parent = base / name
                parent.mkdir()
                input_path = parent / "immutable.input"
                input_path.write_text("frozen\n")
                root_path = parent / "authority"
                fixture_files = {"input": str(input_path)}
                fixture_trees = {}
                fixture_absences = {}
                fixture_pre = snapshot(fixture_files, fixture_trees, fixture_absences)
                log_path = parent / "stdout.log"
                with log_path.open("w+b", buffering=0) as stream:
                    previous_stdout = sys.stdout
                    sys.stdout = stream
                    try:
                        emit_checkpoint({"format": "fixture-checkpoint", "name": name})
                        result = publish_authority_exact1(root_path, {"fixture": name}, fixture_files,
                                                          fixture_trees, fixture_absences, fixture_pre,
                                                          stage_hook=hook,
                                                          inject_commit_window_signal=pending)
                    finally:
                        sys.stdout = previous_stdout
                return root_path, input_path, log_path, result
            normal_root, _, normal_log, (normal_tree, normal_summary) = publish_fixture("publish-normal")
            checks["publish_normal_exact1"] = (
                normal_tree["file_count"] == 1 and normal_summary["deferred_signal"] is None
                and normal_log.read_bytes().count(b"\n") == 2)
            def pending_after_return(stage):
                if stage == "after_publish_return_before_final_line":
                    os.kill(os.getpid(), signal.SIGTERM)
            pending_root, _, pending_log, (pending_tree, pending_summary) = publish_fixture(
                "publish-pending", hook=pending_after_return)
            checks["publish_commit_window_signal_success"] = (
                pending_tree["file_count"] == 1 and pending_summary["deferred_signal"] == signal.SIGTERM
                and pending_root.is_dir() and pending_log.read_bytes().count(b"\n") == 2)
            drift_parent = base / "publish-drift"
            def drift(stage):
                if stage == "before_final_snapshot":
                    (drift_parent / "immutable.input").write_text("drift\n")
            try:
                publish_fixture("publish-drift", hook=drift)
                checks["publish_prepromote_drift_cleanup"] = False
            except RuntimeError:
                checks["publish_prepromote_drift_cleanup"] = (
                    not (drift_parent / "authority").exists()
                    and not (drift_parent / "authority.registration-prep").exists())
        else:
            checks["publish_normal_exact1"] = True
            checks["publish_commit_window_signal_success"] = True
            checks["publish_prepromote_drift_cleanup"] = True
        fake_contract = {"path": str(ROOT / "pipeline/scripts/v502_v501_inner_binding_path_repair_execution_authority_contract.json")}
        fake_materializer = {"path": str(ROOT / "pipeline/scripts/materialize_v502_v501_inner_binding_path_repair_execution_authority.py"),
                             "sha256": "1" * 64, "logical_bytes": 11}
        fake_inner = {"path": str(ROOT / "pipeline/scripts/launch_v502_v501_inner_binding_path_repaired_reconciliation_inner.py"),
                      "sha256": "2" * 64, "logical_bytes": 22}
        fake_outer = {"path": str(ROOT / "pipeline/scripts/launch_v502_v501_inner_binding_path_repaired_reconciliation_outer.py"),
                      "sha256": "3" * 64, "logical_bytes": 33}
        fake_r2 = {"path": str(ROOT / "pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py"),
                   "sha256": "4" * 64, "logical_bytes": 44}
        fixture_inner = base / "inner.py"
        fixture_outer = base / "outer.py"
        fixture_inner.write_text(
            f'from pathlib import Path\nROOT=Path({str(ROOT)!r})\n'
            f'WRAPPER_PATH=Path({fake_inner["path"]!r})\nWRAPPER_SHA={fake_inner["sha256"]!r}\nWRAPPER_BYTES=22\n'
            f'AUTHORITY_CONTRACT_PATH=Path({fake_contract["path"]!r})\n'
            f'AUTHORITY_MATERIALIZER_PATH=Path({fake_materializer["path"]!r})\nAUTHORITY_MATERIALIZER_SHA={fake_materializer["sha256"]!r}\nAUTHORITY_MATERIALIZER_BYTES=11\n'
            f'R2_PATH=Path({fake_r2["path"]!r})\nR2_SHA={fake_r2["sha256"]!r}\nR2_BYTES=44\n')
        fixture_outer.write_text(
            f'from pathlib import Path\nROOT=Path({str(ROOT)!r})\n'
            f'SELF_PATH=Path({fake_outer["path"]!r})\nSELF_SHA={fake_outer["sha256"]!r}\nSELF_BYTES=33\n'
            f'AUTH_CONTRACT=Path({fake_contract["path"]!r})\n'
            f'AUTH_MATERIALIZER=Path({fake_materializer["path"]!r})\nAUTH_MATERIALIZER_SHA={fake_materializer["sha256"]!r}\nAUTH_MATERIALIZER_BYTES=11\n'
            f'INNER_WRAPPER=Path({fake_inner["path"]!r})\nINNER_WRAPPER_SHA={fake_inner["sha256"]!r}\nINNER_WRAPPER_BYTES=22\n'
            f'R2=Path({fake_r2["path"]!r})\nR2_SHA={fake_r2["sha256"]!r}\nR2_BYTES=44\n')
        fake_sources = {"corrected_inner_wrapper": fake_inner,
                        "outer_execution_wrapper": fake_outer, "reconciler_r2": fake_r2}
        role_paths_literal = {role: fake_sources[role]["path"] for role in SOURCE_ROLE_ORDER if role in fake_sources}
        fixture_inner.write_text(fixture_inner.read_text() + f"SOURCE_PATH_LITERALS={role_paths_literal!r}\n")
        fixture_outer.write_text(fixture_outer.read_text() + f"SOURCE_PATH_LITERALS={role_paths_literal!r}\n")
        try:
            source_path_literal_bijection(fixture_inner, fixture_outer, fake_contract,
                                          fake_materializer, fake_sources)
            exact_ok = True
        except RuntimeError:
            exact_ok = False
        tampered_outer = fixture_outer.read_text().replace(
            "materialize_v502_v501_inner_binding", "materialize_BAD_v502_v501_inner_binding")
        fixture_outer.write_text(tampered_outer)
        try:
            source_path_literal_bijection(fixture_inner, fixture_outer, fake_contract,
                                          fake_materializer, fake_sources)
            tamper_rejected = False
        except RuntimeError:
            tamper_rejected = True
        checks["path_literal_bijection_tamper_suite"] = exact_ok and tamper_rejected
    checks["main_pending_resolves_global"] = (
        "pending" not in main.__code__.co_varnames
        and pending({"actual_contract_shape": True}) is False
        and pending({"marker": "PENDING"}) is True
    )
    valid_probe = ast.parse("def execution_interpreter_evidence():\n import torch\n return torch.__version__\n")
    try:
        validate_torch_version_probe_ast(valid_probe)
        checks["validator_exact_version_probe"] = True
    except RuntimeError:
        checks["validator_exact_version_probe"] = False
    tampered = [
        "import torch\ndef execution_interpreter_evidence():\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n import torch\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch as t\n return t.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n from helper import torch\n return torch.__version__\n",
        "def execution_interpreter_evidence():\n import torch\n return torch.cuda\n",
        "def execution_interpreter_evidence():\n import torch\n return (torch.__version__,torch.__version__)\n",
        "def execution_interpreter_evidence():\n import torch\n return torch\n",
        "def execution_interpreter_evidence():\n import torch\n return torch.cuda()\n",
    ]
    rejected = []
    for source in tampered:
        try:
            validate_torch_version_probe_ast(ast.parse(source))
            rejected.append(False)
        except RuntimeError:
            rejected.append(True)
    checks["validator_tamper_suite"] = all(rejected)
    if V493_PROCESS_PATH.is_file():
        actual_process = json.loads(V493_PROCESS_PATH.read_text())
        try:
            validate_v493_process_receipt(actual_process)
            checks["actual_v493_process_receipt"] = (
                regular(V493_PROCESS_PATH,
                        "babcfda7fad625ea7d0b7a34592e119d6287fa4b34dcfbda743f7473a0fe4789",
                        19613)["logical_bytes"] == 19613)
        except RuntimeError:
            checks["actual_v493_process_receipt"] = False
        variants = []
        missing = dict(actual_process); missing.pop("corrected_inner_wrapper_invocations"); variants.append(missing)
        stale = dict(actual_process); stale["inner_wrapper_invocations"] = stale.pop("corrected_inner_wrapper_invocations"); variants.append(stale)
        both = dict(actual_process); both["inner_wrapper_invocations"] = 0; variants.append(both)
        extra = dict(actual_process); extra["unexpected"] = 0; variants.append(extra)
        nonzero = dict(actual_process); nonzero["corrected_inner_wrapper_invocations"] = 1; variants.append(nonzero)
        boolean = dict(actual_process); boolean["corrected_inner_wrapper_invocations"] = False; variants.append(boolean)
        process_rejected = []
        for variant in variants:
            try:
                validate_v493_process_receipt(variant)
                process_rejected.append(False)
            except RuntimeError:
                process_rejected.append(True)
        checks["process_schema_tamper_suite"] = all(process_rejected)
    else:
        checks["actual_v493_process_receipt"] = True
        checks["process_schema_tamper_suite"] = True
    v498_process_path = V498_DIAGNOSTIC_EVIDENCE_ROOT / "process_receipt.json"
    v499_adapter_path = V499_ADAPTER_ROOT / "adapter_receipt.json"
    if v498_process_path.is_file() and v499_adapter_path.is_file():
        original = json.loads(v498_process_path.read_text())
        adapter = json.loads(v499_adapter_path.read_text())
        normalized_process = adapter["normalized_process_receipt"]
        try:
            diff = validate_normalized_process(original, normalized_process)
            checks["actual_adapter_normalization"] = len(diff) == 1
        except RuntimeError:
            checks["actual_adapter_normalization"] = False
        variants = []
        second = copy.deepcopy(normalized_process); second["status"] = "tampered"; variants.append(second)
        bad_path = copy.deepcopy(normalized_process); bad_path["transport_helper_copy"]["path"] += ".bad"; variants.append(bad_path)
        bad_sha = copy.deepcopy(normalized_process); bad_sha["transport_helper_copy"]["sha256"] = "0" * 64; variants.append(bad_sha)
        bad_bytes = copy.deepcopy(normalized_process); bad_bytes["transport_helper_copy"]["logical_bytes"] += 1; variants.append(bad_bytes)
        rejected = []
        for variant in variants:
            try:
                validate_normalized_process(original, variant)
                rejected.append(False)
            except RuntimeError:
                rejected.append(True)
        checks["adapter_normalization_tamper_suite"] = all(rejected)
    else:
        checks["actual_adapter_normalization"] = True
        checks["adapter_normalization_tamper_suite"] = True
    if FAILED_V495_MATERIALIZATION_ROOT.is_dir():
        first = failure_snapshot(); second = failure_snapshot(); predicates = named_failure_predicates(second)
        checks["actual_named8_double_snapshot"] = first == second and all(row["passed"] for row in predicates)
        tampered = copy.deepcopy(second); tampered["file_count"] = True
        checks["named8_type_tamper"] = not named_failure_predicates(tampered)[1]["passed"]
        false_rows = named_failure_predicates(tampered)
        checks["named8_false_names_exact"] = (
            [row["name"] for row in false_rows if not row["passed"]] == ["file_count_exact"])
    else:
        checks["actual_named8_double_snapshot"] = True
        checks["named8_type_tamper"] = True
        checks["named8_false_names_exact"] = True
    print(json.dumps({"passed": all(checks.values()), "checks": checks,
                      "checks_sha256": csha(checks)}, sort_keys=True))
    return 0 if all(checks.values()) else 3


def main() -> int:
    if sys.argv[1:] == ["--synthetic-self-test"]:
        return synthetic_self_test()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--materializer-source", type=Path, required=True)
    parser.add_argument("--materializer-sha", required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--read-only-preflight", action="store_true")
    args = parser.parse_args()

    if args.contract != CONTRACT_PATH or args.contract.resolve() != CONTRACT_PATH:
        raise RuntimeError("contract path")
    contract_record = regular(args.contract, args.contract_sha, args.contract.stat().st_size)
    contract = json.loads(args.contract.read_text())
    if (set(contract) != CONTRACT_TOP_KEYS or contract.get("format") != CONTRACT_FORMAT
            or contract.get("status") != CONTRACT_STATUS or contract.get("seed") != 1644
            or pending(contract)):
        raise RuntimeError("contract frozen schema")
    materializer_record = regular(args.materializer_source, args.materializer_sha,
                                  args.materializer_source.stat().st_size)
    if (args.materializer_source.resolve() != Path(__file__).resolve()
            or materializer_record != contract["authority_materializer_source"]):
        raise RuntimeError("materializer binding")
    if set(contract["source_closure"]) != SOURCE_ROLES:
        raise RuntimeError("source roles")
    if (contract["source_role_order"] != SOURCE_ROLE_ORDER
            or set(SOURCE_ROLE_ORDER) != SOURCE_ROLES
            or len(SOURCE_ROLE_ORDER) != len(set(SOURCE_ROLE_ORDER))
            or contract["source_aliases"] != SOURCE_ALIASES
            or set(SOURCE_ALIASES) != SOURCE_ROLES
            or len(set(SOURCE_ALIASES.values())) != len(SOURCE_ALIASES)):
        raise RuntimeError("source order and aliases")
    sources = {name: record(value) for name, value in contract["source_closure"].items()}
    if sources != contract["source_closure"] or csha(sources) != contract["source_closure_sha256"]:
        raise RuntimeError("source closure")
    frozen_new_sources = {
        "v496_authority_contract": ("33aad023e6275e21702c569261aec754e58c151622055ab3beb0c81434ac335c", 46664),
        "v496_authority_materializer": ("6a38efcb9e714cf58a694bac62eadd86cabc6e746d0fdd1195d168963de1cf06", 62492),
        "v496_inner_wrapper": ("c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f", 101600),
        "v496_outer_wrapper": ("09d5b725ba45bcf274d134fd40fdbea52795e68f6a8193149085b96d563a07c2", 61068),
        "failed_v497_diagnostic_source": ("6ad8f54b53fc2178330f27e9bc772de8fc7357a27752f17a309b835f8f796ecb", 26978),
        "failed_v497_diagnostic_helper": ("a8da83f408fa30c241b311caf86c4d0d7b065099dc7445dea0a1d75e192b44ed", 27893),
        "failed_v497_diagnostic_script": ("eb0f0908c5c2185627bfb59fa02b46279ba929deb04d06253e0d36835c6f4104", 1166),
        "v498_diagnostic_source": ("328d88b9dd5c2268f27c612322785eeca3cfe053018b00512492f984f8332fe6", 38424),
        "v498_diagnostic_helper": ("f7072fae9b88ca1736818f1304191d5bfc4c2d550eb16f0a6f73bf2b8f3a1dce", 27949),
        "v498_diagnostic_script": ("c41a960cce856ddcf27933930b888a17cf61ee1de3552aa59a2867e8b1b0b8ea", 1171),
        "v499_adapter_source": ("5456dfd102d08b2825bb2ba2d6ea44e01c5c5d443823799115c9f3ff105ed35c", 28726),
        "v499_adapter_helper": ("dc9978e304c9c18ec950626999deb340b2fd1b1171afc2ea3af7dfa2323cab70", 22997),
        "v499_adapter_script": ("dce9621f06b732f9627b8912a00910ebff0d5900a0aa48765a25f4e33030b2f2", 2521),
        "v500_authority_contract": ("55bc38e1e101e97761ac8b8f4e5200b0ccd5a94676f27fc7a3123ce179df2e1e", 111329),
        "v500_authority_materializer": ("e717c1360e5691865dd299249630b3b4cef1965c68e8a457008b50415fe9afb5", 109256),
        "v500_inner_wrapper": ("132a91069dc219f70adc6a01292084a97b1e547f16d4cc0d3a6f28df231005ab", 124674),
        "v500_outer_wrapper_invalid_unconsumed": ("3d8949268c1eb3b8e9dfd84d9aea5084ea6a299fb48bb554972cfe0cc6f7d458", 85248),
        "v500_authority_transport_helper": ("c53d750c52db061b79280205c83c6303956e21c8762aa66f60dfcf873131c686", 38766),
        "v500_authority_transport_script": ("6b3cc44a26ba0129e8cc02db7aefc56ff51373e0b616bf09de12ac1e78f8c605", 2543),
        "v500_corrected_deployer": ("8666cfed7e924deedc9229dd6a6349b595012e9812de3b167fcf889c994310d2", 24395),
        "v501_preexecution_forensic_source": ("107d54adbf2aca61405c898f9bd119bbcb23f9a61e45f8cd17beaf06bcfc60e8", 16580),
        "v501_authority_contract": ("ca5b4f34d67d875752af5c43fd05cd9c0b14c1ad1078fb6049f7dccdcfa46008", 128175),
        "v501_authority_materializer": ("59e3863c4d1d7b028422cfebb7dfd7fb208526c322d46151ec0010f239d82358", 126045),
        "v501_inner_wrapper": ("6ce84d03f42c1f048684d7128268135aa0a9bb01e7fe7a0e1c1383ec173a3c47", 136370),
        "v501_outer_wrapper_invalid_consumed": ("7bf4c56eed59917771c23178d459c698b8c703a40e4e46bf992803490d9f0c5b", 94659),
        "v501_authority_transport_helper": ("e4315452c8de29250399f92d6a4d63a2f4b7de820250e448671fe234dd999da6", 39639),
        "v501_authority_transport_script": ("5d9785ab9e3c5b132d2b29ffa757786c9e7384520627044f959b3d0f00a1b00d", 2566),
        "v502_preexecution_forensic_source": ("ce8d13769f56d4e0f501fbb8e1691c221ee2dc554eb26cabcb0861b0a5ab1f21", 10908),
    }
    if any((sources[name]["sha256"], sources[name]["logical_bytes"]) != expected
           for name, expected in frozen_new_sources.items()):
        raise RuntimeError("v496-v499 frozen source closure")

    v496_failure_tree = verify_tree(contract["v496_materializer_failure_tree"])
    v496_failure_process_record = record(contract["v496_materializer_failure_process_receipt"])
    v496_failure_process = json.loads(Path(v496_failure_process_record["path"]).read_text())
    if (v496_failure_tree["root"] != str(FAILED_V496_MATERIALIZATION_ROOT)
            or v496_failure_tree["file_count"] != 6
            or v496_failure_tree["logical_file_bytes"] != 34885
            or v496_failure_tree["sha256sum_lines_digest_sha256"] != "ab63f37f6275193a97db39cc71d37b32f4f653f87e78c375ddb460558b4e1f9e"
            or v496_failure_tree["canonical_json_triples_digest_sha256"] != "c78038212aee67743f08dc936a539ad5d99a6ff931b87f12b5b294dfa4a0dac0"
            or (v496_failure_process_record["sha256"], v496_failure_process_record["logical_bytes"])
               != ("5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d", 411)
            or v496_failure_process.get("status") != "failed_no_retry"
            or v496_failure_process.get("materializer_invocations") != 1
            or any(v496_failure_process.get(name) != 0 for name in
                   ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "r2_invocations"))
            or v496_failure_process.get("cleanup", {}).get("reaped") is not True
            or v496_failure_process.get("cleanup", {}).get("group_empty") is not True
            or v496_failure_process.get("retry_authorized") is not False):
        raise RuntimeError("v496 materializer failure closure")

    failed_v497_tree = verify_tree(contract["failed_v497_diagnostic_failure_tree"])
    failed_v497_process_record = record(contract["failed_v497_diagnostic_process_receipt"])
    failed_v497_process = json.loads(Path(failed_v497_process_record["path"]).read_text())
    if (failed_v497_tree["root"] != str(FAILED_V497_DIAGNOSTIC_EVIDENCE_ROOT)
            or failed_v497_tree["file_count"] != 6 or failed_v497_tree["logical_file_bytes"] != 40802
            or failed_v497_tree["sha256sum_lines_digest_sha256"] != "ed2c81a06f71937581b451f84e137640310b363f04d569fe1bf0da1a60377b8d"
            or failed_v497_tree["canonical_json_triples_digest_sha256"] != "02337d3883ddb46b16bfbf9d1e157dcc16ff4e462810e0de2e22a25082e5913d"
            or (failed_v497_process_record["sha256"], failed_v497_process_record["logical_bytes"])
               != ("5ad649b013fe0e9b17af76992950c53f8043a536ae3f2ecb04f6c9f65f443425", 459)
            or failed_v497_process.get("status") != "failed_no_retry"
            or failed_v497_process.get("diagnostic_invocations") != 1
            or any(failed_v497_process.get(name) != 0 for name in
                   ("authority_materializer_invocations", "outer_wrapper_invocations",
                    "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or failed_v497_process.get("cleanup", {}).get("reaped") is not True
            or failed_v497_process.get("cleanup", {}).get("group_empty") is not True
            or failed_v497_process.get("retry_authorized") is not False
            or os.path.lexists(FAILED_V497_DIAGNOSTIC_ROOT)):
        raise RuntimeError("failed v497 diagnostic closure")

    v498_diagnostic_record = record(contract["v498_diagnostic_receipt"])
    v498_diagnostic = json.loads(Path(v498_diagnostic_record["path"]).read_text())
    v498_registration = verify_tree(contract["v498_diagnostic_registration_tree"])
    v498_evidence = verify_tree(contract["v498_diagnostic_execution_evidence_tree"])
    v498_process_record = record(contract["v498_diagnostic_process_receipt"])
    v498_original_process = json.loads(Path(v498_process_record["path"]).read_text())
    v498_schema = contract["v498_diagnostic_schema"]
    if (v498_diagnostic_record["sha256"] != "2bb515bba9039d3bac3545c1780bb82d9c7609aafa8a64012f9c9574eec5da20"
            or v498_diagnostic_record["logical_bytes"] != 14528
            or v498_registration["file_count"] != 1 or v498_evidence["file_count"] != 6
            or v498_evidence["sha256sum_lines_digest_sha256"] != "17a283bf1f1957c484091c4e44920368c2203e7f7ea05df22ad975582e0b774b"
            or v498_evidence["canonical_json_triples_digest_sha256"] != "2c3b56d838eeb3c936a0f9aad5c8ba8a149a7bc4b3f24c1f9aff0f254337b393"
            or (v498_process_record["sha256"], v498_process_record["logical_bytes"])
               != ("fa7269bdacac994efd5ef8eb44cf9d3d1c7014e5b9aa902e95a5f4243609fabd", 26493)
            or set(v498_diagnostic) != set(v498_schema["top_keys"])
            or csha(sorted(v498_diagnostic)) != v498_schema["top_key_set_sha256"]
            or v498_diagnostic.get("status") != v498_schema["status"]
            or v498_diagnostic.get("passed") is not True
            or v498_diagnostic.get("predicate_names") != PREDICATE_NAMES
            or v498_diagnostic.get("snapshot_1") != v498_diagnostic.get("snapshot_2")
            or not all(row.get("passed") is True for row in v498_diagnostic.get("predicates", []))
            or any(v498_diagnostic.get("authorization", {}).values())):
        raise RuntimeError("v498 diagnostic closure")

    v499_adapter_record = record(contract["v499_adapter_receipt"])
    v499_adapter = json.loads(Path(v499_adapter_record["path"]).read_text())
    v499_registration = verify_tree(contract["v499_adapter_registration_tree"])
    v499_evidence = verify_tree(contract["v499_adapter_execution_evidence_tree"])
    v499_process_record = record(contract["v499_adapter_process_receipt"])
    v499_process = json.loads(Path(v499_process_record["path"]).read_text())
    v499_schema = contract["v499_adapter_schema"]
    normalized_v498_process = contract["normalized_v498_diagnostic_process_receipt"]
    normalization_diff = validate_normalized_process(v498_original_process, normalized_v498_process)
    if (v499_adapter_record["sha256"] != "1d7a82bf22ed9791b30f7760033c9cbf4289fd7e46f051cbbf401cf516380b49"
            or v499_adapter_record["logical_bytes"] != 78314
            or v499_registration["file_count"] != 1 or v499_evidence["file_count"] != 6
            or v499_evidence["sha256sum_lines_digest_sha256"] != "e47fcb2bc9e902084b8f47aac1ae00d4d4965197476008eaa7c518db5487b866"
            or v499_evidence["canonical_json_triples_digest_sha256"] != "d5d10783ba4dd7a318f01bb48d66a0f7b998d4b8f9bd06e08afa8b6e31e73fec"
            or (v499_process_record["sha256"], v499_process_record["logical_bytes"])
               != ("af93c880fe04d94186017b70b7060acb2d9f68afb1f3290718caa9c6a1276f59", 5063)
            or set(v499_adapter) != set(v499_schema["top_keys"])
            or csha(sorted(v499_adapter)) != v499_schema["top_key_set_sha256"]
            or v499_adapter.get("normalized_process_receipt") != normalized_v498_process
            or v499_adapter.get("normalized_process_receipt_sha256") != csha(normalized_v498_process)
            or v499_adapter.get("canonical_json_leaf_diff") != normalization_diff
            or any(v499_adapter.get("authorization", {}).values())
            or v499_process.get("status") != "passed_exact_once_readonly_adapter_no_authority"
            or v499_process.get("adapter_invocations") != 1
            or any(v499_process.get(name) != 0 for name in
                   ("authority_materializer_invocations", "outer_wrapper_invocations",
                    "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or v499_process.get("pre_post_snapshots_exactly_equal") is not True
            or v499_process.get("cleanup", {}).get("reaped") is not True
            or v499_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v499 adapter closure")

    v500_authority_record = record(contract["v500_authority_receipt"])
    v500_authority_tree = verify_tree(contract["v500_authority_registration_tree"])
    v500_materialization_tree = verify_tree(contract["v500_authority_materialization_evidence_tree"])
    v500_process_record = record(contract["v500_authority_materializer_process_receipt"])
    v500_process = json.loads(Path(v500_process_record["path"]).read_text())
    v500_deployment_tree = verify_tree(contract["v500_corrected_deployment_evidence_tree"])
    v500_deployment_process_record = record(contract["v500_corrected_deployment_process_receipt"])
    v500_deployment_process = json.loads(Path(v500_deployment_process_record["path"]).read_text())
    v501_forensic_record = record(contract["v501_preexecution_forensic_receipt"])
    v501_forensic_tree = verify_tree(contract["v501_preexecution_forensic_registration_tree"])
    v501_forensic = json.loads(Path(v501_forensic_record["path"]).read_text())
    v501_forensic_schema = contract["v501_preexecution_forensic_schema"]
    expected_invalid_ancestry = {
        "status": "invalid_unconsumed_preexecution_binding_mismatch",
        "outer_wrapper_invocations": 0, "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0, "retry_authorized": False,
        "invalid_outer_source": sources["v500_outer_wrapper_invalid_unconsumed"],
        "stale_inner_literal": {"path": sources["v500_inner_wrapper"]["path"],
                                "sha256": "c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f",
                                "logical_bytes": 101600},
        "actual_inner_source": sources["v500_inner_wrapper"],
        "root_cause": "v500 outer frozen INNER_WRAPPER_SHA and INNER_WRAPPER_BYTES do not match the authority-bound current inner source",
        "corrected_rule": "fresh authority must bind a fresh outer and fresh inner whose command and self-check records are identical",
    }
    if (v500_authority_record["sha256"], v500_authority_record["logical_bytes"]) != (
            "8c57944a115bcb26fd3709a0a7911a2ea58be2e82e238c0b6fffad04f5b22179", 222774):
        raise RuntimeError("v500 authority receipt")
    if (v500_authority_tree["root"] != str(V500_AUTHORITY_ROOT)
            or v500_authority_tree["file_count"] != 1
            or v500_authority_tree["sha256sum_lines_digest_sha256"] != "7633136acfd7aef756d39fecbfa4dc0343d775d9135b2644c58a52f9989d578e"
            or v500_authority_tree["canonical_json_triples_digest_sha256"] != "76fc17f7b90a98509b38069f597851bcac59544ace890020321ad3ba8bffcc79"):
        raise RuntimeError("v500 authority tree")
    if (v500_materialization_tree["root"] != str(V500_AUTHORITY_EVIDENCE_ROOT)
            or v500_materialization_tree["file_count"] != 6
            or v500_materialization_tree["logical_file_bytes"] != 167885
            or v500_materialization_tree["sha256sum_lines_digest_sha256"] != "37dffce5e5e4ed1d712ec44d7100c90bef3287559b96bd38c6137983134c6ca1"
            or v500_materialization_tree["canonical_json_triples_digest_sha256"] != "71de157900c60fe4abe088505036645617b1579264a9e28ff18a0b99d8cfcc6e"
            or (v500_process_record["sha256"], v500_process_record["logical_bytes"]) !=
               ("0fab24fca5e3b8a975f9cb4e32109297cb7b1094d0fb9a7c9289320633e46b11", 79870)
            or v500_process.get("status") != "passed_exact_once_authority_materialized_no_outer_or_inner_execution"
            or v500_process.get("authority_materializer_invocations") != 1
            or any(v500_process.get(k) != 0 for k in ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or v500_process.get("retry_authorized") is not False
            or v500_process.get("pre_post_snapshots_exactly_equal") is not True
            or v500_process.get("cleanup", {}).get("reaped") is not True
            or v500_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v500 materialization closure")
    if (v500_deployment_tree["root"] != str(V500_DEPLOYMENT_EVIDENCE_ROOT)
            or v500_deployment_tree["file_count"] != 6
            or v500_deployment_tree["logical_file_bytes"] != 59532
            or v500_deployment_tree["sha256sum_lines_digest_sha256"] != "792437bd87209c87174c59b7ee289ed442fe5d5a9346b78999c3ee05d7aaf8a4"
            or v500_deployment_tree["canonical_json_triples_digest_sha256"] != "beaa9c4744303a8818a067fbb0e44c2d2c722f4723f0d96c10ded57e1ebd0820"
            or (v500_deployment_process_record["sha256"], v500_deployment_process_record["logical_bytes"]) !=
               ("7e92b12a68bc1c150fcbfa7b6f0ff6cde7788000f4b92ddfc31693663f94d893", 31333)
            or v500_deployment_process.get("status") != "passed_exact_once_six_sources_deployed"
            or v500_deployment_process.get("passed") is not True
            or v500_deployment_process.get("deployer_invocations") != 1
            or v500_deployment_process.get("worker_returncode") != 0
            or v500_deployment_process.get("worker_summary", {}).get("deployment_writes") != 6
            or any(v500_deployment_process.get("worker_summary", {}).get(k) != 0 for k in
                   ("authority_materializer_invocations", "outer_wrapper_invocations",
                    "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or v500_deployment_process.get("pre_post_snapshots_exactly_equal") is not True
            or v500_deployment_process.get("cleanup", {}).get("reaped") is not True
            or v500_deployment_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v500 corrected deployment closure")
    if (v501_forensic_record["sha256"], v501_forensic_record["logical_bytes"]) != (
            "1c62d32c002ae2d3f3525ebc7a69c0c44f428d30a735e46f4edf53a1ad90aca9", 24950):
        raise RuntimeError("v501 forensic receipt")
    if (v501_forensic_tree["root"] != str(PREEXECUTION_FORENSIC_ROOT)
            or v501_forensic_tree["file_count"] != 1
            or v501_forensic_tree["sha256sum_lines_digest_sha256"] != "428939cb521865d1d6f904cd1a97abc2c609811cfba4ef2f58372e86083777fe"
            or v501_forensic_tree["canonical_json_triples_digest_sha256"] != "da31ad0603673ef1bcefd7feab55105dd59bf773d92e901c7ed7e29f31dfc077"):
        raise RuntimeError("v501 forensic tree")
    if (set(v501_forensic) != set(v501_forensic_schema.get("top_keys", []))
            or len(v501_forensic) != 34
            or v501_forensic.get("checks") != {k: True for k in v501_forensic.get("check_keys", [])}
            or len(v501_forensic.get("check_keys", [])) != 15
            or any(v501_forensic.get("authorization", {}).values())
            or v501_forensic.get("input_snapshots_exactly_equal") is not True
            or contract["v500_invalid_outer_unconsumed_ancestry"] != expected_invalid_ancestry):
        raise RuntimeError("v501 forensic schema and mismatch ancestry")

    v501_authority_record = record(contract["v501_authority_receipt"])
    v501_authority_tree = verify_tree(contract["v501_authority_registration_tree"])
    v501_materialization_tree = verify_tree(contract["v501_authority_materialization_evidence_tree"])
    v501_process_record = record(contract["v501_authority_materializer_process_receipt"])
    v501_authority = json.loads(Path(v501_authority_record["path"]).read_text())
    v501_process = json.loads(Path(v501_process_record["path"]).read_text())
    if ((v501_authority_record["sha256"], v501_authority_record["logical_bytes"])
            != ("8211d5c3d9b9a883eeda9249b803fe833079d58a0119486e72fd11c2b657245e", 263170)
            or v501_authority_tree["root"] != str(V501_AUTHORITY_ROOT)
            or v501_authority_tree["file_count"] != 1
            or v501_authority_tree["logical_file_bytes"] != 263170
            or v501_authority_tree["sha256sum_lines_digest_sha256"] != "9660235647c893aa8ba3c8515a42c3f98d2f03d965e3286aaf829f638bc6adf7"
            or v501_authority_tree["canonical_json_triples_digest_sha256"] != "53478de58923f8af35d484eaca613f156ef5466d052c2d240916b4a709ae34cd"
            or v501_materialization_tree["root"] != str(V501_AUTHORITY_EVIDENCE_ROOT)
            or v501_materialization_tree["file_count"] != 6
            or v501_materialization_tree["logical_file_bytes"] != 189453
            or v501_materialization_tree["sha256sum_lines_digest_sha256"] != "9e4242a3085ac9259a6ee41ec18ed00c7111d10a37157bf1fbe370ae2998c324"
            or v501_materialization_tree["canonical_json_triples_digest_sha256"] != "bc677282053341fdabf32a5ebb8f4a372a826048105b6db6b3ca4e2a7df60a89"
            or (v501_process_record["path"], v501_process_record["sha256"], v501_process_record["logical_bytes"])
               != (str(V501_PROCESS_PATH), "ad862d4cb750efe617c41bcb4d814255b07d8d5f652039d7a11e35e6712abe6a", 93795)
            or len(v501_authority) != 107 or len(v501_authority.get("checks", {})) != 96
            or v501_authority.get("passed") is not True
            or not all(value is True for value in v501_authority.get("checks", {}).values())
            or v501_authority.get("input_snapshots_exactly_equal") is not True
            or v501_process.get("passed") is not True
            or v501_process.get("authority_materializer_invocations") != 1
            or any(v501_process.get(key) != 0 for key in
                   ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or v501_process.get("retry_authorized") is not False
            or v501_process.get("pre_post_snapshots_exactly_equal") is not True
            or v501_process.get("cleanup", {}).get("reaped") is not True
            or v501_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v501 authority materialization closure")

    v502_forensic_record = sources["v502_preexecution_forensic_source"]
    if Path(v502_forensic_record["path"]) != V502_FORENSIC_PATH:
        raise RuntimeError("v502 forensic canonical path")
    v502_forensic = json.loads(V502_FORENSIC_PATH.read_text())
    v502_forensic_schema = {
        "format": "strict-track2-v502-v501-outer-preintent-materializer-path-mismatch-failure-forensic-v1",
        "status": "frozen_reconstructed_preintent_failure_consumed_outer_attempt_no_retry",
        "top_keys": sorted(v502_forensic),
        "top_key_set_sha256": csha(sorted(v502_forensic)),
        "check_keys": v502_forensic.get("check_keys"),
        "check_key_set_sha256": v502_forensic.get("check_key_set_sha256"),
        "checks_sha256": v502_forensic.get("checks_sha256"),
    }
    expected_v501_consumed = {
        "status": "failed_no_retry_preintent_path_binding_mismatch",
        "outer_wrapper_invocations": 1,
        "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "invalid_outer_source": sources["v501_outer_wrapper_invalid_consumed"],
        "actual_inner_source": sources["v501_inner_wrapper"],
        "authority_materializer_source": sources["v501_authority_materializer"],
        "stale_materializer_path": v502_forensic.get("path_mismatch", {}).get("outer_hardcoded_materializer_path"),
        "actual_materializer_path": sources["v501_authority_materializer"]["path"],
        "root_cause": "v501 outer hardcoded a stale materializer basename containing failure_tree_diagnostic while the authority-bound source used the canonical basename without that segment",
        "corrected_rule": "all source Path literals must match contract and authority records by canonical path, sha256, and logical bytes; reject same-sha wrong-path aliases",
    }
    mismatch = v502_forensic.get("path_mismatch", {})
    invocations = v502_forensic.get("invocation_partition", {})
    forensic_state = v502_forensic.get("state_after_failure", {})
    if (set(v502_forensic) != set(v502_forensic_schema["top_keys"])
            or len(v502_forensic) != 25 or len(v502_forensic.get("check_keys", [])) != 18
            or v502_forensic.get("format") != v502_forensic_schema["format"]
            or v502_forensic.get("status") != v502_forensic_schema["status"]
            or v502_forensic.get("checks") != {key: True for key in v502_forensic["check_keys"]}
            or any(v502_forensic.get("authorization", {}).values())
            or v502_forensic.get("native_exit_code") != 1
            or v502_forensic.get("native_persistent_capture_available") is not False
            or invocations.get("outer_wrapper_invocations") != 1
            or invocations.get("corrected_inner_wrapper_invocations") != 0
            or invocations.get("reconciler_r2_invocations") != 0
            or mismatch.get("authority_record_materializer_path") != sources["v501_authority_materializer"]["path"]
            or mismatch.get("same_sha256") != sources["v501_authority_materializer"]["sha256"]
            or mismatch.get("same_logical_bytes") != sources["v501_authority_materializer"]["logical_bytes"]
            or mismatch.get("paths_equal") is not False
            or v502_forensic.get("source_records") != {
                "authority_contract": sources["v501_authority_contract"],
                "authority_materializer": sources["v501_authority_materializer"],
                "inner_wrapper": sources["v501_inner_wrapper"],
                "outer_wrapper_invalid_consumed": sources["v501_outer_wrapper_invalid_consumed"],
            }
            or v502_forensic.get("v501_authority_receipt") != v501_authority_record
            or v502_forensic.get("v501_authority_registration_tree") != v501_authority_tree
            or any(not row.get("absent") or os.path.lexists(row.get("path", ""))
                   for row in forensic_state.get("required_absences", {}).values())
            or contract["v501_invalid_outer_consumed_ancestry"] != expected_v501_consumed
            or contract["v502_preexecution_forensic"] != v502_forensic
            or contract["v502_preexecution_forensic_schema"] != v502_forensic_schema):
        raise RuntimeError("v502 preexecution forensic and v501 consumed ancestry")
    expected_v496_ancestry = {
        "failed_status": "failed_no_retry", "materializer_invocations": 1,
        "outer_wrapper_invocations": 0, "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0, "retry_authorized": False,
        "root_cause": "v496 combined predicate failed; specific transient predicate unavailable because v496 did not persist per-predicate diagnostics; read-only postmortem all eight pass",
        "corrected_rule": "persist eight named expected/observed predicates and two independent snapshots before authority-root creation",
    }
    expected_v497_ancestry = {
        "failed_status": "failed_no_retry", "diagnostic_invocations": 1,
        "authority_materializer_invocations": 0, "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0, "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "root_cause": "v497 diagnostic referenced undefined V496_PROCESS_PATH before diagnostic registration",
        "corrected_rule": "derive V496_PROCESS_PATH only from the exact V496_PROCESS_RECORD path and execute the shared production prepublish builder in live self-test",
    }
    if (contract["v496_materializer_failure_ancestry"] != expected_v496_ancestry
            or contract["failed_v497_diagnostic_failure_ancestry"] != expected_v497_ancestry
            or record(contract["v496_authority_contract"]) != sources["v496_authority_contract"]):
        raise RuntimeError("v496-v497 ancestry aliases")
    v494_expected = {
        "v494_authority_contract": ("2f4cef2671dfd5c443d5a7f53dc0ea7b770b520da84cba3512a363884d56bd4a", 29803),
        "v494_authority_materializer": ("5b9396fd74e1d161fa41610108d823f356313992364cba708ed869519fa0a421", 39236),
        "v494_inner_wrapper": ("1f2c9de60703f12c22966b7aeb3526a6c32e0d5a0e0258094c9d208b9a56f4bf", 79760),
        "v494_outer_wrapper": ("bea4b6df050afb22c833908725a6af7417b292984d8ddb4fbb972b99128d3df5", 43024),
    }
    if any((sources[role]["sha256"], sources[role]["logical_bytes"]) != expected
           for role, expected in v494_expected.items()):
        raise RuntimeError("v494 frozen sources")
    if record(contract["v494_authority_contract"]) != sources["v494_authority_contract"]:
        raise RuntimeError("v494 authority contract alias")
    v494_contract = json.loads(Path(sources["v494_authority_contract"]["path"]).read_text())
    if (len(v494_contract) != 24
            or v494_contract.get("format") != "strict-track2-v494-v493-v490-interpreter-symlink-repair-execution-authority-design-contract-v1"
            or v494_contract.get("status") != CONTRACT_STATUS
            or v494_contract.get("seed") != 1636
            or v494_contract.get("source_closure", {}).get("corrected_inner_wrapper") != sources["v494_inner_wrapper"]
            or v494_contract.get("source_closure", {}).get("outer_execution_wrapper") != sources["v494_outer_wrapper"]):
        raise RuntimeError("v494 authority contract schema")
    forensic_record = record(contract["v494_materializer_failure_forensic"])
    transport_record = record(contract["v494_materializer_failure_transport_script"])
    if (forensic_record != sources["v494_failure_forensic"]
            or transport_record != sources["v494_failure_transport_script"]
            or forensic_record["sha256"] != "248c4a94a24c3b793bb29b6cc07a5955a93694f7e22b1a9363ef5bb7ee0ba0dc"
            or forensic_record["logical_bytes"] != 6409
            or transport_record["sha256"] != "0d4669e9a1fa52ee98fe2c459e9e42d880116e4428270edcefb0f50e5b95d49c"
            or transport_record["logical_bytes"] != 942):
        raise RuntimeError("v494 failure sources")
    forensic = json.loads(Path(forensic_record["path"]).read_text())
    failure_ancestry = contract["v494_materializer_failure_ancestry"]
    expected_failure_ancestry = {
        "failed_status": "frozen_reconstructed_v494_materializer_prestate_failure_no_retry",
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "native_exit_code": 1,
        "native_persistent_capture_available": False,
        "tool_capture_sha256": "4a65e33de2de1da29e8a36d39e5374e43b66435aa5e6bc3722ccfca574fb4a4c",
        "tool_capture_logical_bytes": 787,
        "root_cause": "v494 validator blanket-forbade function-local torch import required for exact runtime version evidence",
        "corrected_rule": "allow exactly one no-alias torch import and one Load torch.__version__ inside execution_interpreter_evidence; reject every other torch import, name, attribute, and call",
        "retry_authorized": False,
    }
    old_source_projection = {
        "authority_contract": sources["v494_authority_contract"],
        "authority_materializer": sources["v494_authority_materializer"],
        "corrected_inner_wrapper": sources["v494_inner_wrapper"],
        "outer_execution_wrapper": sources["v494_outer_wrapper"],
    }
    invocation = forensic.get("invocation", {})
    tool_capture = forensic.get("tool_capture", {})
    state = forensic.get("state_after_failure", {})
    if (failure_ancestry != expected_failure_ancestry
            or forensic.get("format") != "strict-track2-v495-v494-authority-materializer-failure-forensic-reconstructed-v1"
            or forensic.get("status") != expected_failure_ancestry["failed_status"]
            or forensic.get("occurred") is not True or forensic.get("passed") is not False
            or forensic.get("native_exit_code") != 1
            or forensic.get("native_persistent_capture_available") is not False
            or forensic.get("source_records") != old_source_projection
            or invocation.get("materializer_invocations") != 1
            or any(invocation.get(key) != 0 for key in ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "reconciler_r2_invocations"))
            or invocation.get("transport_script", {}).get("sha256") != transport_record["sha256"]
            or invocation.get("transport_script", {}).get("logical_bytes") != transport_record["logical_bytes"]
            or tool_capture.get("combined_output_sha256") != expected_failure_ancestry["tool_capture_sha256"]
            or tool_capture.get("combined_output_logical_bytes") != expected_failure_ancestry["tool_capture_logical_bytes"]
            or "corrected inner forbidden import" not in tool_capture.get("combined_output_text", "")
            or state.get("all_absent") is not True
            or state.get("live_reconciliation_processes_empty") is not True
            or state.get("gpu_compute_apps_empty") is not True):
        raise RuntimeError("v494 materializer failure forensic")
    v495_expected = {
        "v495_authority_contract": ("60d978a803bd148f6fe68906e71113cf33e61a3ae5bf77e22c8f2855c68c2823", 37084),
        "v495_authority_materializer": ("b0f4b59212ea67754a12394bf1f5a8e7d2c5389c93c44207e0099d9f2850fe18", 50978),
        "v495_inner_wrapper": ("b52e7eb29870141bda85eead81cbc1c3a47522ad86e6a7b6829f7052766a8f76", 90261),
        "v495_outer_wrapper": ("3aae6deb4059e143cc40a3e10bdf6f1cf51e20e884c0459b4aa950e0d480d526", 50749),
    }
    if any((sources[role]["sha256"], sources[role]["logical_bytes"]) != expected
           for role, expected in v495_expected.items()):
        raise RuntimeError("v495 frozen sources")
    if record(contract["v495_authority_contract"]) != sources["v495_authority_contract"]:
        raise RuntimeError("v495 authority contract alias")
    v495_contract = json.loads(Path(sources["v495_authority_contract"]["path"]).read_text())
    if (len(v495_contract) != 28
            or v495_contract.get("format") != "strict-track2-v495-v494-v490-interpreter-validator-repair-execution-authority-design-contract-v1"
            or v495_contract.get("status") != CONTRACT_STATUS
            or v495_contract.get("seed") != 1637
            or v495_contract.get("authority_materializer_source") != sources["v495_authority_materializer"]
            or v495_contract.get("source_closure", {}).get("corrected_inner_wrapper") != sources["v495_inner_wrapper"]
            or v495_contract.get("source_closure", {}).get("outer_execution_wrapper") != sources["v495_outer_wrapper"]):
        raise RuntimeError("v495 authority contract schema")
    # Freeze the expected closure here, but deliberately defer current-tree
    # equality and process-content gates until after the fsynced named8
    # checkpoint.  Thus an observed drift yields an exact first stdout line and
    # cannot be hidden by the old undifferentiated OR.
    v495_failure_tree = contract["v495_materializer_failure_tree"]
    v495_failure_process_record = contract["v495_materializer_failure_process_receipt"]
    if (not isinstance(v495_failure_tree, dict)
            or set(v495_failure_tree) != {"root", "inventory", "file_count", "logical_file_bytes",
                                          "sha256sum_lines_digest_sha256",
                                          "canonical_json_triples_digest_sha256"}
            or not isinstance(v495_failure_process_record, dict)
            or set(v495_failure_process_record) != {"path", "sha256", "logical_bytes"}):
        raise RuntimeError("v495 failure contract schema")
    v495_failure_ancestry = contract["v495_materializer_failure_ancestry"]
    expected_v495_failure_ancestry = {
        "failed_status": "failed_no_retry",
        "materializer_invocations": 1,
        "outer_wrapper_invocations": 0,
        "corrected_inner_wrapper_invocations": 0,
        "reconciler_r2_invocations": 0,
        "native_exit_code": 1,
        "retry_authorized": False,
        "root_cause": "v495 materializer required stale inner_wrapper_invocations absent from immutable v493 process receipt",
        "corrected_rule": "validate exact24 process receipt with corrected_inner_wrapper_invocations int zero and reject stale, missing, both, extra, nonzero, and bool variants",
    }
    if (v495_failure_tree["root"] != str(FAILED_V495_MATERIALIZATION_ROOT)
            or v495_failure_tree["file_count"] != 6
            or v495_failure_tree["logical_file_bytes"] != 32139
            or v495_failure_tree["sha256sum_lines_digest_sha256"] != "21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1b"
            or v495_failure_tree["canonical_json_triples_digest_sha256"] != "9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290"
            or not any(row == ["process_receipt.json", v495_failure_process_record["sha256"],
                               v495_failure_process_record["logical_bytes"]]
                       for row in v495_failure_tree["inventory"])
            or v495_failure_process_record["sha256"] != "72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4"
            or v495_failure_process_record["logical_bytes"] != 411):
        raise RuntimeError("v495 failure tree")
    if v495_failure_ancestry != expected_v495_failure_ancestry:
        raise RuntimeError("v495 materializer failure ancestry contract")
    phase_record = sources["phase_a_design_contract"]
    if (phase_record != contract["phase_a_design_contract_record"]
            or phase_record["sha256"] != "8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
            or phase_record["logical_bytes"] != 43960):
        raise RuntimeError("phase contract current record")
    validate_corrected_inner(Path(sources["corrected_inner_wrapper"]["path"]))
    path_literal_bijection = source_path_literal_bijection(
        Path(sources["corrected_inner_wrapper"]["path"]),
        Path(sources["outer_execution_wrapper"]["path"]),
        contract_record, materializer_record, sources)
    if contract["source_path_literal_bijection_contract"] != path_literal_bijection:
        raise RuntimeError("source path literal bijection contract")
    interpreter = execution_interpreter_evidence()
    if contract["execution_interpreter_contract"] != interpreter:
        raise RuntimeError("execution interpreter contract")

    repair_tree = verify_tree(contract["repair_formal_registration_tree"])
    static_tree = verify_tree(contract["postregistration_static_registration_tree"])
    f813_tree = verify_tree(contract["f813_registration_tree"])
    if repair_tree["file_count"] != 1 or static_tree["file_count"] != 1 or f813_tree["file_count"] != 2:
        raise RuntimeError("frozen tree cardinality")
    f813_path = Path(f813_tree["root"]) / "preregistration.json"
    f813 = json.loads(f813_path.read_text())
    phase_spec = f813.get("frozen_parent", {}).get("phase_a_design_contract")
    if (not isinstance(phase_spec, dict) or set(phase_spec) != {"path", "sha256"}
            or phase_spec != {"path": phase_record["path"], "sha256": phase_record["sha256"]}):
        raise RuntimeError("F813 phase contract exact2 projection")

    v493_contract = json.loads(Path(sources["v493_authority_contract"]["path"]).read_text())
    if record(contract["v493_authority_contract"]) != sources["v493_authority_contract"]:
        raise RuntimeError("v493 contract alias")
    v493_receipt_record = record(contract["v493_authority_receipt"])
    v493_receipt = json.loads(Path(v493_receipt_record["path"]).read_text())
    old_schema = v493_contract.get("authority_receipt_contract", {})
    if (set(v493_receipt) != set(old_schema.get("top_keys", []))
            or len(v493_receipt) != 39 or len(v493_receipt.get("checks", {})) != 26
            or v493_receipt.get("format") != old_schema.get("format")
            or v493_receipt.get("status") != old_schema.get("status")
            or v493_receipt.get("passed") is not True
            or v493_receipt.get("checks") != {key: True for key in old_schema.get("check_keys", [])}
            or v493_receipt.get("checks_sha256") != old_schema.get("checks_sha256")
            or v493_receipt.get("authorization") != old_schema.get("authorization_exact")
            or v493_receipt.get("runtime_observation") != old_schema.get("runtime_observation_exact")):
        raise RuntimeError("v493 authority receipt")
    v493_auth_tree = verify_tree(contract["v493_authority_registration_tree"])
    if (v493_auth_tree["file_count"] != 1
            or not any(row == ["authority_receipt.json", v493_receipt_record["sha256"],
                               v493_receipt_record["logical_bytes"]]
                       for row in v493_auth_tree["inventory"])):
        raise RuntimeError("v493 authority exact3")
    v493_evidence_tree = verify_tree(contract["v493_authority_materialization_evidence_tree"])
    v493_process_record = record(contract["v493_authority_materializer_process_receipt"])
    v493_process = json.loads(Path(v493_process_record["path"]).read_text())
    validate_v493_process_receipt(v493_process)
    process_schema = {
        "record": v493_process_record,
        "exact_keys": sorted(V493_PROCESS_KEYS),
        "key_set_sha256": csha(sorted(V493_PROCESS_KEYS)),
        "required_values": {
            "status": "passed_exact_once_no_outer_or_inner_execution",
            "passed": True,
            "materializer_invocations": 1,
            "outer_wrapper_invocations": 0,
            "corrected_inner_wrapper_invocations": 0,
            "r2_invocations": 0,
            "retry_authorized": False,
            "pre_post_snapshots_exactly_equal": True,
            "cleanup": {"reaped": True, "group_empty": True},
        },
        "forbidden_keys": ["inner_wrapper_invocations"],
    }
    if contract["v493_authority_materializer_process_schema"] != process_schema:
        raise RuntimeError("v493 process schema contract")
    if (v493_evidence_tree["file_count"] != 6
            or not any(row == ["process_receipt.json", v493_process_record["sha256"],
                               v493_process_record["logical_bytes"]]
                       for row in v493_evidence_tree["inventory"])
            or v493_process.get("passed") is not True):
        raise RuntimeError("v493 authority materialization evidence")

    failed_tree = verify_tree(contract["v493_outer_failure_tree"])
    failed_terminal_record = record(contract["v493_outer_terminal_receipt"])
    failed_terminal = json.loads(Path(failed_terminal_record["path"]).read_text())
    stderr_path = Path(failed_tree["root"]) / "corrected_inner_stderr.log"
    failure_spec = contract["v493_outer_failure_ancestry"]
    expected_failure_spec = {
        "failed_status": "failed_no_retry",
        "superseded_outer_wrapper_invocations": 1,
        "superseded_corrected_inner_wrapper_invocations": 1,
        "reconciler_r2_invocations": 0,
        "retry_authorized": False,
        "root_cause": "superseded corrected inner required lexical RLPY symlink to be regular nonsymlink",
        "corrected_rule": "bind exact lexical and intermediate symlinks, resolved regular target, hash, bytes, and runtime versions",
    }
    if (failed_tree["file_count"] != 4 or failure_spec != expected_failure_spec
            or not any(row == ["terminal_receipt.json", failed_terminal_record["sha256"],
                               failed_terminal_record["logical_bytes"]]
                       for row in failed_tree["inventory"])
            or failed_terminal.get("status") != "failed_no_retry"
            or failed_terminal.get("passed") is not False
            or failed_terminal.get("nested_corrected_inner_invocations") != 1
            or failed_terminal.get("retry_authorized") is not False
            or failed_terminal.get("cleanup", {}).get("reaped") is not True
            or failed_terminal.get("cleanup", {}).get("group_empty") is not True
            or failed_terminal.get("transparent_present") is not False
            or "RuntimeError" not in stderr_path.read_text()
            or "regular: /root/autodl-tmp/conda_envs/rlinf_track2/bin/python" not in stderr_path.read_text()):
        raise RuntimeError("v493 outer failed-no-retry ancestry")

    lineage = contract["lineage"]
    expected_lineage = {
        "execution_authority_root": str(AUTHORITY_ROOT),
        "authority_prep": str(AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + ".registration-prep")),
        "outer_evidence_root": str(OUTER_EVIDENCE_ROOT),
        "outer_evidence_prep": str(OUTER_EVIDENCE_ROOT.with_name(OUTER_EVIDENCE_ROOT.name + ".outer-prep")),
        "inner_attempt_root": str(INNER_ATTEMPT_ROOT),
        "inner_attempt_prep": str(INNER_ATTEMPT_ROOT.with_name(INNER_ATTEMPT_ROOT.name + ".attempt-prep")),
        "transparent_static_receipt_path": str(TRANSPARENT_PATH),
        "qualification_root": str(QUALIFICATION_ROOT),
    }
    if lineage != expected_lineage:
        raise RuntimeError("lineage")
    root = args.authority_root
    prep = root.with_name(root.name + ".registration-prep")
    if root != AUTHORITY_ROOT or root.resolve() != root or root.parent.is_symlink() or not root.parent.is_dir():
        raise RuntimeError("authority root")
    historical_keys = {
        "superseded_static_root", "superseded_static_prep", "v489_superseded_static_root",
        "v489_superseded_static_prep", "fresh_static_prep", "v490_authority_prep",
        "old_v490_inner_attempt_root", "old_v490_inner_attempt_prep", "transparent_output",
        "transparent_tmp", "qualification_root", "superseded_v491_authority_root",
        "superseded_v491_authority_prep", "superseded_v491_outer_evidence_root",
        "superseded_v491_outer_evidence_prep", "v492_authority_prep",
        "failed_v492_outer_evidence_prep", "v493_authority_prep",
        "failed_v493_outer_evidence_prep", "execution_authority_root",
        "failed_v493_inner_attempt_root", "failed_v493_inner_attempt_prep",
        "failed_v494_authority_root", "failed_v494_authority_prep",
        "failed_v494_outer_evidence_root", "failed_v494_outer_evidence_prep",
        "failed_v494_inner_attempt_root", "failed_v494_inner_attempt_prep",
        "failed_v495_authority_root", "failed_v495_authority_prep",
        "failed_v495_outer_evidence_root", "failed_v495_outer_evidence_prep",
        "failed_v495_inner_attempt_root", "failed_v495_inner_attempt_prep",
        "execution_authority_prep", "outer_evidence_root", "outer_evidence_prep",
        "corrected_inner_attempt_root", "corrected_inner_attempt_prep",
        "failed_v496_authority_root", "failed_v496_authority_prep",
        "failed_v496_outer_evidence_root", "failed_v496_outer_evidence_prep",
        "failed_v496_inner_attempt_root", "failed_v496_inner_attempt_prep",
        "failed_v497_diagnostic_root", "failed_v497_diagnostic_prep",
        "failed_v497_diagnostic_evidence_prep", "v498_diagnostic_prep",
        "v498_diagnostic_evidence_prep", "v499_adapter_prep", "v499_adapter_evidence_prep",
        "v500_authority_prep", "v500_outer_evidence_root", "v500_outer_evidence_prep",
        "v500_inner_attempt_root", "v500_inner_attempt_prep", "v501_preexecution_forensic_prep",
        "v501_authority_prep", "v501_outer_evidence_root", "v501_outer_evidence_prep",
        "v501_inner_attempt_root", "v501_inner_attempt_prep",
    }
    current_keys = historical_keys - {"execution_authority_root"}
    historical = decode_absences(contract["historical_absences"], historical_keys)
    current = decode_absences(contract["current_absences_after_authority"], current_keys)
    if (historical["execution_authority_root"] != root
            or historical["execution_authority_prep"] != prep
            or historical["failed_v493_outer_evidence_prep"] != FAILED_V493_OUTER_ROOT.with_name(FAILED_V493_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v493_inner_attempt_root"] != V493_FAILED_INNER_ROOT
            or historical["failed_v493_inner_attempt_prep"] != V493_FAILED_INNER_ROOT.with_name(V493_FAILED_INNER_ROOT.name + ".attempt-prep")
            or historical["failed_v494_authority_root"] != FAILED_V494_AUTHORITY_ROOT
            or historical["failed_v494_authority_prep"] != FAILED_V494_AUTHORITY_ROOT.with_name(FAILED_V494_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["failed_v494_outer_evidence_root"] != FAILED_V494_OUTER_ROOT
            or historical["failed_v494_outer_evidence_prep"] != FAILED_V494_OUTER_ROOT.with_name(FAILED_V494_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v494_inner_attempt_root"] != FAILED_V494_INNER_ROOT
            or historical["failed_v494_inner_attempt_prep"] != FAILED_V494_INNER_ROOT.with_name(FAILED_V494_INNER_ROOT.name + ".attempt-prep")
            or historical["failed_v495_authority_root"] != FAILED_V495_AUTHORITY_ROOT
            or historical["failed_v495_authority_prep"] != FAILED_V495_AUTHORITY_ROOT.with_name(FAILED_V495_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["failed_v495_outer_evidence_root"] != FAILED_V495_OUTER_ROOT
            or historical["failed_v495_outer_evidence_prep"] != FAILED_V495_OUTER_ROOT.with_name(FAILED_V495_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v495_inner_attempt_root"] != FAILED_V495_INNER_ROOT
            or historical["failed_v495_inner_attempt_prep"] != FAILED_V495_INNER_ROOT.with_name(FAILED_V495_INNER_ROOT.name + ".attempt-prep")
            or historical["old_v490_inner_attempt_root"] != OLD_V490_INNER_ROOT
            or historical["failed_v496_authority_root"] != FAILED_V496_AUTHORITY_ROOT
            or historical["failed_v496_authority_prep"] != FAILED_V496_AUTHORITY_ROOT.with_name(FAILED_V496_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["failed_v496_outer_evidence_root"] != FAILED_V496_OUTER_ROOT
            or historical["failed_v496_outer_evidence_prep"] != FAILED_V496_OUTER_ROOT.with_name(FAILED_V496_OUTER_ROOT.name + ".outer-prep")
            or historical["failed_v496_inner_attempt_root"] != FAILED_V496_INNER_ROOT
            or historical["failed_v496_inner_attempt_prep"] != FAILED_V496_INNER_ROOT.with_name(FAILED_V496_INNER_ROOT.name + ".attempt-prep")
            or historical["failed_v497_diagnostic_root"] != FAILED_V497_DIAGNOSTIC_ROOT
            or historical["failed_v497_diagnostic_prep"] != FAILED_V497_DIAGNOSTIC_ROOT.with_name(FAILED_V497_DIAGNOSTIC_ROOT.name + ".registration-prep")
            or historical["failed_v497_diagnostic_evidence_prep"] != FAILED_V497_DIAGNOSTIC_EVIDENCE_ROOT.with_name(FAILED_V497_DIAGNOSTIC_EVIDENCE_ROOT.name + ".execution-prep")
            or historical["v498_diagnostic_prep"] != V498_DIAGNOSTIC_ROOT.with_name(V498_DIAGNOSTIC_ROOT.name + ".registration-prep")
            or historical["v498_diagnostic_evidence_prep"] != V498_DIAGNOSTIC_EVIDENCE_ROOT.with_name(V498_DIAGNOSTIC_EVIDENCE_ROOT.name + ".execution-prep")
            or historical["v499_adapter_prep"] != V499_ADAPTER_ROOT.with_name(V499_ADAPTER_ROOT.name + ".registration-prep")
            or historical["v499_adapter_evidence_prep"] != V499_ADAPTER_EVIDENCE_ROOT.with_name(V499_ADAPTER_EVIDENCE_ROOT.name + ".execution-prep")
            or historical["v500_authority_prep"] != V500_AUTHORITY_ROOT.with_name(V500_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["v500_outer_evidence_root"] != V500_OUTER_EVIDENCE_ROOT
            or historical["v500_outer_evidence_prep"] != V500_OUTER_EVIDENCE_ROOT.with_name(V500_OUTER_EVIDENCE_ROOT.name + ".outer-prep")
            or historical["v500_inner_attempt_root"] != V500_INNER_ATTEMPT_ROOT
            or historical["v500_inner_attempt_prep"] != V500_INNER_ATTEMPT_ROOT.with_name(V500_INNER_ATTEMPT_ROOT.name + ".attempt-prep")
            or historical["v501_preexecution_forensic_prep"] != PREEXECUTION_FORENSIC_ROOT.with_name(PREEXECUTION_FORENSIC_ROOT.name + ".registration-prep")
            or historical["v501_authority_prep"] != V501_AUTHORITY_ROOT.with_name(V501_AUTHORITY_ROOT.name + ".registration-prep")
            or historical["v501_outer_evidence_root"] != V501_OUTER_EVIDENCE_ROOT
            or historical["v501_outer_evidence_prep"] != V501_OUTER_EVIDENCE_ROOT.with_name(V501_OUTER_EVIDENCE_ROOT.name + ".outer-prep")
            or historical["v501_inner_attempt_root"] != V501_INNER_ATTEMPT_ROOT
            or historical["v501_inner_attempt_prep"] != V501_INNER_ATTEMPT_ROOT.with_name(V501_INNER_ATTEMPT_ROOT.name + ".attempt-prep")
            or {key: historical[key] for key in current_keys} != current
            or any(os.path.lexists(path) for path in historical.values())):
        raise RuntimeError("absence prestate")
    live_fragments = {row["path"] for role, row in sources.items()
                      if role in {"corrected_inner_wrapper", "outer_execution_wrapper", "reconciler_r2"}}
    if live_processes(live_fragments) or gpu_processes():
        raise RuntimeError("process or GPU prestate")

    files = {"contract": str(args.contract), "materializer": str(args.materializer_source),
             "v493_authority_receipt": v493_receipt_record["path"],
             "v493_process_receipt": v493_process_record["path"],
             "failed_v493_terminal": failed_terminal_record["path"],
             "v495_failure_process": v495_failure_process_record["path"],
             "v496_failure_process": v496_failure_process_record["path"],
             "failed_v497_process": failed_v497_process_record["path"],
             "v498_diagnostic_receipt": v498_diagnostic_record["path"],
             "v498_diagnostic_process": v498_process_record["path"],
             "v499_adapter_receipt": v499_adapter_record["path"],
             "v499_adapter_process": v499_process_record["path"],
             "v500_authority_receipt": v500_authority_record["path"],
             "v500_authority_process": v500_process_record["path"],
             "v500_deployment_process": v500_deployment_process_record["path"],
             "v501_preexecution_forensic_receipt": v501_forensic_record["path"],
             "v501_authority_receipt": v501_authority_record["path"],
             "v501_authority_process": v501_process_record["path"],
             "v502_preexecution_forensic": v502_forensic_record["path"]}
    files.update({f"source::{name}": row["path"] for name, row in sources.items()})
    trees = {
        "v493_authority_REG": contract["v493_authority_registration_tree"]["root"],
        "v493_authority_materialization": contract["v493_authority_materialization_evidence_tree"]["root"],
        "failed_v493_outer": contract["v493_outer_failure_tree"]["root"],
        "v495_materializer_failure": contract["v495_materializer_failure_tree"]["root"],
        "v496_materializer_failure": contract["v496_materializer_failure_tree"]["root"],
        "failed_v497_diagnostic_evidence": contract["failed_v497_diagnostic_failure_tree"]["root"],
        "v498_diagnostic_REG": contract["v498_diagnostic_registration_tree"]["root"],
        "v498_diagnostic_evidence": contract["v498_diagnostic_execution_evidence_tree"]["root"],
        "v499_adapter_REG": contract["v499_adapter_registration_tree"]["root"],
        "v499_adapter_evidence": contract["v499_adapter_execution_evidence_tree"]["root"],
        "v500_authority_REG": contract["v500_authority_registration_tree"]["root"],
        "v500_authority_materialization": contract["v500_authority_materialization_evidence_tree"]["root"],
        "v500_corrected_deployment": contract["v500_corrected_deployment_evidence_tree"]["root"],
        "v501_preexecution_forensic_REG": contract["v501_preexecution_forensic_registration_tree"]["root"],
        "v501_authority_REG": contract["v501_authority_registration_tree"]["root"],
        "v501_authority_materialization": contract["v501_authority_materialization_evidence_tree"]["root"],
        "repair_formal_REG": contract["repair_formal_registration_tree"]["root"],
        "postregistration_static_REG": contract["postregistration_static_registration_tree"]["root"],
        "f813_REG": contract["f813_registration_tree"]["root"],
    }
    historical_pre = snapshot(files, trees, historical)
    pre = snapshot(files, trees, current)
    stable_absences = {name: path for name, path in current.items()
                       if name != "execution_authority_prep"}
    stable_pre = snapshot(files, trees, stable_absences)
    post = snapshot(files, trees, current)
    if (pre != post or historical_pre["files"] != pre["files"]
            or historical_pre["trees"] != pre["trees"]):
        raise RuntimeError("input drift")

    diagnostic_snapshot_1 = failure_snapshot()
    diagnostic_snapshot_2 = failure_snapshot()
    checkpoint_predicates = named_failure_predicates(diagnostic_snapshot_2)
    checkpoint_false_names = [row["name"] for row in checkpoint_predicates if not row["passed"]]
    checkpoint_all_pass = diagnostic_snapshot_1 == diagnostic_snapshot_2 and not checkpoint_false_names
    checkpoint_payload = {
        "format": "strict-track2-v502-materializer-pre-root-diagnostic-checkpoint-v1",
        "status": "passed_pre_root_named8_double_snapshot" if checkpoint_all_pass else "failed_pre_root_named8_no_authority",
        "passed": checkpoint_all_pass, "seed": 1644,
        "v498_diagnostic_receipt": v498_diagnostic_record,
        "v498_diagnostic_registration_tree": v498_registration,
        "v498_diagnostic_execution_evidence_tree": v498_evidence,
        "v498_diagnostic_process_receipt": v498_process_record,
        "v499_adapter_receipt": v499_adapter_record,
        "v499_adapter_registration_tree": v499_registration,
        "v499_adapter_execution_evidence_tree": v499_evidence,
        "v499_adapter_process_receipt": v499_process_record,
        "normalized_v498_diagnostic_process_receipt_sha256": csha(normalized_v498_process),
        "snapshot_1": diagnostic_snapshot_1, "snapshot_2": diagnostic_snapshot_2,
        "snapshots_exactly_equal": diagnostic_snapshot_1 == diagnostic_snapshot_2,
        "predicate_names": PREDICATE_NAMES, "predicates": checkpoint_predicates,
        "predicates_sha256": csha(checkpoint_predicates),
        "false_predicate_names": checkpoint_false_names,
        "root_cause_boundary": expected_v496_ancestry["root_cause"],
        "immutable_input_snapshot_sha256": pre["snapshot_sha256"],
        "runtime_observation": {"checkpoint_invocations": 1, "authority_materializer_invocations": 0,
                                "outer_wrapper_invocations": 0, "corrected_inner_wrapper_invocations": 0,
                                "reconciler_r2_invocations": 0},
        "authorization": {"authority_materialization_authorized": False,
                          "outer_execution_authorized": False, "corrected_inner_wrapper_authorized": False,
                          "reconciler_r2_authorized": False, "retry_authorized": False,
                          "training_authorized": False, "submission_authorized": False},
    }
    checkpoint_contract = contract["materializer_checkpoint_contract"]
    if (not isinstance(checkpoint_contract, dict)
            or checkpoint_contract.get("format") != checkpoint_payload["format"]
            or checkpoint_contract.get("top_keys") != sorted(checkpoint_payload)
            or checkpoint_contract.get("predicate_names") != PREDICATE_NAMES
            or checkpoint_contract.get("success_stdout_line_count") != 2
            or checkpoint_contract.get("failure_stdout_line_count") != 1):
        raise RuntimeError("materializer checkpoint contract")

    schema = contract["authority_receipt_contract"]
    checks = {key: True for key in AUTH_CHECK_KEYS}
    receipt = {
        "format": OUTPUT_FORMAT, "status": OUTPUT_STATUS, "passed": True,
        "authority_design_contract": contract_record,
        "authority_materializer_source": materializer_record,
        "corrected_inner_wrapper_source": sources["corrected_inner_wrapper"],
        "outer_execution_wrapper_source": sources["outer_execution_wrapper"],
        "v493_inner_wrapper_source": sources["v493_inner_wrapper"],
        "phase_a_design_contract_source": phase_record,
        "reconciler_r2_source": sources["reconciler_r2"],
        "v493_authority_contract": sources["v493_authority_contract"],
        "v493_authority_materializer_source": sources["v493_authority_materializer"],
        "v493_outer_wrapper_source": sources["v493_outer_wrapper"],
        "v494_authority_contract_source": sources["v494_authority_contract"],
        "v494_authority_materializer_source": sources["v494_authority_materializer"],
        "v494_outer_wrapper_source": sources["v494_outer_wrapper"],
        "v494_inner_wrapper_source": sources["v494_inner_wrapper"],
        "v494_materializer_failure_forensic": forensic_record,
        "v494_materializer_failure_transport_script": transport_record,
        "v494_materializer_failure_ancestry": failure_ancestry,
        "v495_authority_contract_source": sources["v495_authority_contract"],
        "v495_authority_materializer_source": sources["v495_authority_materializer"],
        "v495_outer_wrapper_source": sources["v495_outer_wrapper"],
        "v495_inner_wrapper_source": sources["v495_inner_wrapper"],
        "v495_materializer_failure_tree": v495_failure_tree,
        "v495_materializer_failure_process_receipt": v495_failure_process_record,
        "v495_materializer_failure_ancestry": v495_failure_ancestry,
        "v493_authority_materializer_process_schema": process_schema,
        "v493_authority_receipt": v493_receipt_record,
        "v493_authority_registration_tree": v493_auth_tree,
        "v493_authority_materialization_evidence_tree": v493_evidence_tree,
        "v493_authority_materializer_process_receipt": v493_process_record,
        "failed_v493_outer_execution_tree": failed_tree,
        "failed_v493_outer_terminal_receipt": failed_terminal_record,
        "repair_formal_registration_tree": repair_tree,
        "postregistration_static_registration_tree": static_tree,
        "f813_registration_tree": f813_tree,
        "source_closure": sources, "source_closure_sha256": csha(sources),
        "source_role_order": SOURCE_ROLE_ORDER, "source_aliases": SOURCE_ALIASES,
        "outer_evidence_root": str(OUTER_EVIDENCE_ROOT),
        "inner_attempt_root": str(INNER_ATTEMPT_ROOT),
        "transparent_static_receipt_path": str(TRANSPARENT_PATH),
        "historical_absences": {name: {"path": str(path), "absent": True}
                                for name, path in sorted(historical.items())},
        "required_absences": {name: {"path": str(path), "absent": True}
                              for name, path in sorted(current.items())},
        "checks": checks, "check_keys": AUTH_CHECK_KEYS,
        "check_key_set_sha256": csha(AUTH_CHECK_KEYS), "checks_sha256": csha(checks),
        "input_pre_snapshot": pre, "input_post_snapshot": post,
        "input_snapshots_exactly_equal": True,
        "authorization": AUTHORIZATION, "runtime_observation": RUNTIME,
        "execution_boundary": EXECUTION_BOUNDARY,
        "execution_interpreter_evidence": interpreter,
        "v496_authority_contract_source": sources["v496_authority_contract"],
        "v496_authority_materializer_source": sources["v496_authority_materializer"],
        "v496_inner_wrapper_source": sources["v496_inner_wrapper"],
        "v496_outer_wrapper_source": sources["v496_outer_wrapper"],
        "failed_v497_diagnostic_source": sources["failed_v497_diagnostic_source"],
        "failed_v497_diagnostic_helper_source": sources["failed_v497_diagnostic_helper"],
        "failed_v497_diagnostic_script_source": sources["failed_v497_diagnostic_script"],
        "v498_diagnostic_source": sources["v498_diagnostic_source"],
        "v498_diagnostic_helper_source": sources["v498_diagnostic_helper"],
        "v498_diagnostic_script_source": sources["v498_diagnostic_script"],
        "v499_adapter_source": sources["v499_adapter_source"],
        "v499_adapter_helper_source": sources["v499_adapter_helper"],
        "v499_adapter_script_source": sources["v499_adapter_script"],
        "v496_materializer_failure_tree": v496_failure_tree,
        "v496_materializer_failure_process_receipt": v496_failure_process_record,
        "v496_materializer_failure_ancestry": expected_v496_ancestry,
        "failed_v497_diagnostic_failure_tree": failed_v497_tree,
        "failed_v497_diagnostic_process_receipt": failed_v497_process_record,
        "failed_v497_diagnostic_failure_ancestry": expected_v497_ancestry,
        "v498_diagnostic_receipt": v498_diagnostic_record,
        "v498_diagnostic_registration_tree": v498_registration,
        "v498_diagnostic_execution_evidence_tree": v498_evidence,
        "v498_diagnostic_process_receipt": v498_process_record,
        "v498_diagnostic_schema": v498_schema,
        "v499_adapter_receipt": v499_adapter_record,
        "v499_adapter_registration_tree": v499_registration,
        "v499_adapter_execution_evidence_tree": v499_evidence,
        "v499_adapter_process_receipt": v499_process_record,
        "v499_adapter_schema": v499_schema,
        "normalized_v498_diagnostic_process_receipt": normalized_v498_process,
        "v500_authority_contract_source": sources["v500_authority_contract"],
        "v500_authority_materializer_source": sources["v500_authority_materializer"],
        "v500_inner_wrapper_source": sources["v500_inner_wrapper"],
        "v500_invalid_outer_wrapper_source": sources["v500_outer_wrapper_invalid_unconsumed"],
        "v500_authority_transport_helper_source": sources["v500_authority_transport_helper"],
        "v500_authority_transport_script_source": sources["v500_authority_transport_script"],
        "v500_corrected_deployer_source": sources["v500_corrected_deployer"],
        "v501_preexecution_forensic_source": sources["v501_preexecution_forensic_source"],
        "v500_authority_receipt": v500_authority_record,
        "v500_authority_registration_tree": v500_authority_tree,
        "v500_authority_materialization_evidence_tree": v500_materialization_tree,
        "v500_authority_materializer_process_receipt": v500_process_record,
        "v500_corrected_deployment_evidence_tree": v500_deployment_tree,
        "v500_corrected_deployment_process_receipt": v500_deployment_process_record,
        "v500_invalid_outer_unconsumed_ancestry": expected_invalid_ancestry,
        "v501_preexecution_forensic_receipt": v501_forensic_record,
        "v501_preexecution_forensic_registration_tree": v501_forensic_tree,
        "v501_preexecution_forensic_schema": v501_forensic_schema,
        "v501_authority_contract_source": sources["v501_authority_contract"],
        "v501_authority_materializer_source": sources["v501_authority_materializer"],
        "v501_inner_wrapper_source": sources["v501_inner_wrapper"],
        "v501_invalid_outer_wrapper_source": sources["v501_outer_wrapper_invalid_consumed"],
        "v501_authority_transport_helper_source": sources["v501_authority_transport_helper"],
        "v501_authority_transport_script_source": sources["v501_authority_transport_script"],
        "v502_preexecution_forensic_source": sources["v502_preexecution_forensic_source"],
        "v501_authority_receipt": v501_authority_record,
        "v501_authority_registration_tree": v501_authority_tree,
        "v501_authority_materialization_evidence_tree": v501_materialization_tree,
        "v501_authority_materializer_process_receipt": v501_process_record,
        "v501_invalid_outer_consumed_ancestry": expected_v501_consumed,
        "v502_preexecution_forensic": v502_forensic,
        "v502_preexecution_forensic_schema": v502_forensic_schema,
        "source_path_literal_bijection": path_literal_bijection,
        "materializer_pre_root_checkpoint": checkpoint_payload,
        "materializer_pre_root_checkpoint_sha256": csha(checkpoint_payload),
    }
    if (set(schema) != {"format", "status", "top_keys", "check_keys",
                        "check_key_set_sha256", "checks_sha256", "authorization_exact",
                        "runtime_observation_exact"}
            or set(receipt) != AUTH_TOP_KEYS or set(receipt) != set(schema["top_keys"])
            or schema["format"] != OUTPUT_FORMAT or schema["status"] != OUTPUT_STATUS
            or schema["check_keys"] != AUTH_CHECK_KEYS
            or schema["check_key_set_sha256"] != csha(AUTH_CHECK_KEYS)
            or schema["checks_sha256"] != csha(checks)
            or schema["authorization_exact"] != AUTHORIZATION
            or schema["runtime_observation_exact"] != RUNTIME
            or contract["execution_boundary"] != EXECUTION_BOUNDARY):
        raise RuntimeError("authority receipt schema")
    if args.read_only_preflight:
        print(json.dumps({"status": "passed_read_only_preflight_no_materialization",
                          "contract": contract_record,
                          "materializer": materializer_record,
                          "authority_root_absent": not os.path.lexists(root),
                          "authority_prep_absent": not os.path.lexists(prep),
                          "receipt_top_count": len(receipt),
                          "check_count": len(checks),
                          "source_count": len(sources),
                          "input_snapshots_exactly_equal": pre == post,
                          "execution_interpreter_evidence": interpreter}, sort_keys=True))
        return 0

    observed_checkpoint_sha = emit_checkpoint(checkpoint_payload)
    if observed_checkpoint_sha != csha(checkpoint_payload):
        raise RuntimeError("checkpoint stdout digest")
    if not checkpoint_all_pass:
        raise RuntimeError("checkpoint named predicate failure: " + ",".join(checkpoint_false_names))

    # Repeat closure and semantic checks independently after the persisted
    # checkpoint, still before authority mkdir.
    checkpoint_tree = {key: value for key, value in diagnostic_snapshot_2.items()
                       if key != "process_receipt_record"}
    if verify_tree(v495_failure_tree) != checkpoint_tree:
        raise RuntimeError("v495 post-checkpoint tree drift")
    v495_failure_process_record = record(v495_failure_process_record)
    v495_failure_process = json.loads(Path(v495_failure_process_record["path"]).read_text())
    if (v495_failure_process.get("format") != "strict-track2-v495-authority-materializer-process-receipt-v1"
            or v495_failure_process.get("status") != "failed_no_retry"
            or v495_failure_process.get("passed") is not False
            or type(v495_failure_process.get("materializer_invocations")) is not int
            or v495_failure_process.get("materializer_invocations") != 1
            or any(type(v495_failure_process.get(key)) is not int or v495_failure_process.get(key) != 0
                   for key in ("outer_wrapper_invocations", "corrected_inner_wrapper_invocations", "r2_invocations"))
            or v495_failure_process.get("retry_authorized") is not False
            or v495_failure_process.get("cleanup", {}).get("reaped") is not True
            or v495_failure_process.get("cleanup", {}).get("group_empty") is not True):
        raise RuntimeError("v495 post-checkpoint process ancestry")

    publish_authority_exact1(root, receipt, files, trees, stable_absences, stable_pre)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
