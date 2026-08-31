#!/usr/bin/env python3
"""Design-only v522 Phase-A authority materializer for the v521 split state.

The valid v520 AUTH exact1 is immutable ancestry, while the v521 transport
terminal is a failed-no-retry split caused solely by rechecking the historical
v520 AUTH-root absence after that root had been successfully materialized.
This materializer treats that old AUTH root as expected-current and grants only
one fresh Phase-A launcher lineage.  It never invokes the v521 dynamic absence
validator and never authorizes the retired v520 launcher path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import urllib.request
from pathlib import Path

ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
S = ROOT / 'pipeline/scripts'
PIPELINE = ROOT / 'pipeline'

SELF_PATH = S / 'materialize_v522_v521_phase_a_authority_split_state_repair_execution_authority.py'
CONTRACT_PATH = S / 'v522_v521_phase_a_authority_split_state_repair_execution_authority_contract.json'
LAUNCHER_PATH = S / 'launch_v522_v521_phase_a_cache_qualification.py'
DRIVER_PATH = S / 'generate_v522_v521_phase_a_cache_qualification.py'
WORKER_PATH = S / 'generate_v522_v521_phase_a_cache_qualification_worker.py'
PROXY_PATH = S / 'v520_v519_rng_isolated_v169_runtime_proxy.py'
AUDITOR_PATH = S / 'audit_v522_v521_phase_a_rng_isolated_qualification.py'
PREREG_ROOT = J / 'v522_v521_phase_a_cache_qualification_prereg_seed1658_20260826'
PREREG_PATH = PREREG_ROOT / 'preregistration.json'
AUTHORITY_ROOT = J / 'v522_v521_phase_a_authority_split_state_repair_execution_authority_seed1658_20260826'
ATTEMPT_ROOT = J / 'v522_v521_phase_a_cache_qualification_attempt_seed1658_20260826'
QUALIFICATION_ROOT = Path('/root/v522_v521_phase_a_cache_qualification_seed1658_20260826')

OLD_MATERIALIZER = S / 'materialize_v509_v508_phase_a_wam_pipeline_import_repair_execution_authority.py'
OLD_MATERIALIZER_SHA = 'f993b1a464995d84ecaac994bfc7f5630763feabdfcf40e791a0cb49b6a300ff'
OLD_MATERIALIZER_BYTES = 28224
OLD_CONTRACT = S / 'v509_v508_phase_a_wam_pipeline_import_repair_execution_authority_contract.json'
OLD_CONTRACT_SHA = 'acee1f33d96a9ee97104d2116fa94e77b8464f85816142330d0b7d93e313cf94'
OLD_CONTRACT_BYTES = 114187
OLD_PREREG = J / 'v509_v508_phase_a_cache_qualification_prereg_seed1650_20260825/preregistration.json'
OLD_PREREG_SHA = '8200ba523295a607bbab47c897e83e0b8736ab6000dc26a37fe73c6363cd7cc4'
OLD_PREREG_BYTES = 56800
PHASE_A_DESIGN_CONTRACT_SHA = '8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64'

V520_CONTRACT = S / 'v520_v519_phase_a_rng_isolated_qualification_execution_authority_contract.json'
V520_CONTRACT_SHA = '58d1ccda8277aa58d3606eff9c21546aa6f67900a6153fe28d725e928f370d80'
V520_CONTRACT_BYTES = 88570
V520_MATERIALIZER = S / 'materialize_v520_v519_phase_a_rng_isolated_qualification_execution_authority.py'
V520_MATERIALIZER_SHA = 'fcf766f26fbd2d06abd9e7c1def44570d0380e5aadd9fbbce63932b4438c823d'
V520_MATERIALIZER_BYTES = 33210
V520_AUTHORITY_ROOT = J / 'v520_v519_phase_a_rng_isolated_qualification_execution_authority_seed1656_20260826'
V520_AUTHORITY_RECEIPT = V520_AUTHORITY_ROOT / 'authority_receipt.json'
V520_AUTHORITY_RECEIPT_SHA = '3e71dff7f0756d966593e8d3200b5fc8370e9849251b16fe61393583afd7735f'
V520_AUTHORITY_RECEIPT_BYTES = 121101
V520_AUTHORITY_TREE_SUMMARY = {
    'file_count': 1,
    'logical_file_bytes': 121101,
    'sha256sum_lines_digest_sha256': '4edaba7bee39d41a02b7c40bea968ecefb85f5c18290c4ae3f84446aea7cb222',
    'canonical_json_triples_digest_sha256': '47f93414dc9073eb4c69799ed89a82866e5749fd1f57f1368495463aef6e8dbe',
}
V521_EVIDENCE_ROOT = J / 'v521_v520_phase_a_rng_isolated_qualification_execution_authority_materialization_evidence_seed1657_20260826'
V521_PROCESS = V521_EVIDENCE_ROOT / 'process_receipt.json'
V521_PROCESS_SHA = '906fcd5468d0a94fa29ddb4ff98b826de39566609532b06810ab87c21801aab2'
V521_PROCESS_BYTES = 3447
V521_STDOUT = V521_EVIDENCE_ROOT / 'materializer_stdout.log'
V521_STDOUT_SHA = 'a24a7f6cd0a558168f3eb83d35975f43eb4648432428e486e861ae84838e6f71'
V521_STDOUT_BYTES = 1251
V521_STDERR = V521_EVIDENCE_ROOT / 'materializer_stderr.log'
V521_STDERR_SHA = hashlib.sha256(b'').hexdigest()
V521_STDERR_BYTES = 0
V521_EVIDENCE_TREE_SUMMARY = {
    'file_count': 6,
    'logical_file_bytes': 100263,
    'sha256sum_lines_digest_sha256': '4afec5aa009fc6008ec82bc2e86c54bcc0990ba85ec427a82cb8c0545c892eba',
    'canonical_json_triples_digest_sha256': '9af08453ab635c0147072cbcad4389e0f3dc1f27d020b9f77cc44c626a2a4cfd',
}
V521_FAILURE_FORENSIC = S / 'v521_v520_authority_transport_collection_failure_forensic.json'
V521_FAILURE_FORENSIC_SHA = 'ebb1ca52ec8f96cb1903b7e1955485a07eeeb77fd8e4b5373415e38fc8dd1d6a'
V521_FAILURE_FORENSIC_BYTES = 9234
V521_HELPER = S / 'invoke_v521_v520_phase_a_rng_isolated_qualification_execution_authority_materializer_once.py'
V521_HELPER_SHA = 'de2403b17513ffc817a50de79e97b931e4fdd9ac818ffc4156a99ed5a3f5f7a3'
V521_HELPER_BYTES = 78449
V521_SCRIPT = Path('/root/v521_v520_phase_a_rng_isolated_qualification_authority_materialize_once.sh')
V521_SCRIPT_SHA = 'd4c55f482ba6d351803a0e32ef01f7128f8a1273d540904929720af78ab9b846'
V521_SCRIPT_BYTES = 2575
SPLIT_FORENSIC = S / 'v522_v521_v520_phase_a_authority_split_state_forensic.json'
SPLIT_FORENSIC_SHA = 'bed75dcfb24a8bb3d25caa3f497d26717d83692aa1b8cf251bd651cc45645267'
SPLIT_FORENSIC_BYTES = 10025

V519_EVIDENCE = J / 'v519_v518_phase_a_sample0_rng_isolation_evidence_seed1655_20260826'
V519_PROCESS = V519_EVIDENCE / 'process_receipt.json'
V519_STDOUT = V519_EVIDENCE / 'child_stdout.log'
V519_PROCESS_SHA = '06ce4ad84ad9bd3efdf35b429dbf5c0f1986c4d32ae495afcbc7a1e97060ed11'
V519_PROCESS_BYTES = 369884
V519_STDOUT_SHA = '6585fbb99e063cb86287ca109ce7c4595dc0d07be1330f914ab53d83be06963d'
V519_STDOUT_BYTES = 126129
V519_TREE_SUMMARY = {
    'file_count': 6,
    'logical_file_bytes': 653273,
    'sha256sum_lines_digest_sha256': 'f386dec633821d8e85b7097eed5fcc6668c07958ee52aaab9327804286bfbd26',
    'canonical_json_triples_digest_sha256': 'e06c481df6865a157b6d34dab36fd99f867689a6eddad0309484f72885ab0538',
}
V517_EVIDENCE = J / 'v517_v516_alias_restore_readonly_remediation_transport_evidence_seed1653E_20260826'
V517_PROCESS = V517_EVIDENCE / 'process_receipt.json'
V517_PROCESS_SHA = 'a1c406745ac5d7dc1bf61ae742501687054f64785211d52d08fa4f1aa48f49dd'
V517_PROCESS_BYTES = 145914
V517_TREE_SUMMARY = {
    'file_count': 6,
    'logical_file_bytes': 185257,
    'sha256sum_lines_digest_sha256': '249a4037947eb3488b5729b425a9cfc25cd5eaa6ec07f26e2c583cda64b71a8f',
    'canonical_json_triples_digest_sha256': '7121a896d795d7def6b487c399837966d73bc39e906a7bd227171b2b27b9bcae',
}
V517_CANDIDATE_ROOT = J / 'v516_v515_alias_restore_readonly_remediation_candidate_seed1653D_20260826'
V517_CANDIDATE = V517_CANDIDATE_ROOT / 'adapter_candidate.json'
V517_CANDIDATE_SHA = '1a4803aad2c2ae80d4572b74fa429316c575104c63341b2be9bedb913e388f50'
V517_CANDIDATE_BYTES = 85855
V517_CANDIDATE_TREE_SUMMARY = {
    'file_count': 1,
    'logical_file_bytes': 85855,
    'sha256sum_lines_digest_sha256': 'b3e94f0bd0ff25af017d460f5be10b15118cc5a4b10eb9e8239c7539db750a9c',
    'canonical_json_triples_digest_sha256': '4a65a2a053a5e83349731c337a72505bb57f68b2f662d7879f8a3dcbde9f1092',
}

FAILURE_SOURCES = {
    'v509_import_failure_forensic': (S / 'v509_v508_phase_a_wam_pipeline_import_failure_forensic.json', 'c237c2244d2b54cb595b2bbf5909c51c61a8ae8f0246bd6d94830fb1c642369e', 8780),
    'v509_warning_failure_forensic': (S / 'v509_phase_a_deterministic_warning_mismatch_failure_forensic.json', 'd6c5d5c8e9ef327d2f893ff1513cbb3d80edd226c033fa8e95b5adf3ba7744f8', 14527),
    'v511_anchor_failure_forensic': (S / 'v511_sample0_fixture_v509_authority_anchor_failure_forensic.json', '8742959eb69e58b6f7300c9c93e565da8df376748217f9cfe9a362ad5fc3f231', 4569),
    'v518_rng_failure_forensic': (S / 'v518_v512_phase_a_sample0_rng_state_failure_forensic.json', 'ce757e861c4a1e031f25e1a3e88c0bd3e8b94aad1c9c962b396a76039eecaf12', 9208),
}

FIXED_ROLES = [
    'frozen_v485_launcher', 'phase_a_design_contract', 'phase_a_static_audit',
    'cache_scope_helper', 'phase_a_output_materializer',
    'phase_a_static_auditor', 'restart_v218_source',
]
ACTIVE_ROLES = [
    'authority_materializer', 'fresh_phase_a_launcher', 'fresh_phase_a_driver',
    'fresh_phase_a_worker', 'fresh_rng_proxy', 'fresh_phase_a_independent_auditor',
    'fresh_phase_a_preregistration',
]
V519_SOURCE_ROLES = [
    'v519_rng_sample_child', 'v519_rng_sample_design',
    'v519_rng_sample_transport_helper', 'v519_rng_sample_transport_script',
]
SPLIT_SOURCE_ROLES = [
    'v522_v521_authority_split_state_forensic',
    'v520_authority_design_contract', 'v520_authority_materializer',
    'v520_authority_receipt', 'v521_materialization_process_receipt',
    'v521_materialization_stdout', 'v521_materialization_stderr',
    'v521_transport_failure_forensic', 'v521_transport_helper',
    'v521_transport_script',
]
CONTRACT_SOURCE_ORDER = (
    ACTIVE_ROLES + SPLIT_SOURCE_ROLES + FIXED_ROLES
    + V519_SOURCE_ROLES + list(FAILURE_SOURCES)
)
AUTHORITY_SOURCE_ORDER = ['authority_design_contract', *CONTRACT_SOURCE_ORDER]
SOURCE_ALIASES = {'authority_design_contract': 'authority_design_contract', **{r: r + '_source' for r in CONTRACT_SOURCE_ORDER}}

WORKER_ENV = {'PYTHONPATH': str(PIPELINE), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
PER_CALL_RNG_CONTRACT = {
    'format': 'strict-track2-v520-per-call-rng-isolation-evidence-v1',
    'expected_branches': ['A', 'B'],
    'calls_per_branch': 1000,
    'total_runtime_delegate_calls': 2000,
    'durable_process_log_markers_per_call': 2,
    'durable_process_log_markers_per_branch': 2000,
    'total_durable_process_log_markers': 4000,
    'process_log_marker_order_per_call': ['started', 'completed'],
    'required_rng_stage_keys': ['entry_external', 'inside_before_delegate', 'internal_after_delegate', 'exit_restored'],
    'delegate_invocations_started_per_call': 1,
    'delegate_invocations_completed_per_successful_call': 1,
    'fork_rng_devices_exact_all_cuda_indices': True,
    'python_all_four_stages_equal': True,
    'numpy_all_four_stages_equal': True,
    'torch_cpu_exit_restored': True,
    'torch_cuda_exit_restored': True,
    'torch_internal_change_allowed': True,
    'raw_warning_order_preserved': True,
    'output_schema_recorded_per_call': True,
    'canonical_event_digest_recorded_per_call': True,
    'sample0_evidence_is_ancestry_not_runtime_substitute': True,
}
AUTHORIZATION = {
    'phase_a_cache_qualification_launcher_authorized': True,
    'launcher_invocations_authorized': 1,
    'launcher_invocations_consumed': 0,
    'nested_phase_a_driver_invocations_authorized': 1,
    'nested_phase_a_driver_only_via_launcher': True,
    'direct_phase_a_driver_authorized': False,
    'phase_a_worker_invocations_authorized': 2,
    'rng_proxy_required_for_every_runtime_delegate': True,
    'runtime_delegate_calls_authorized': 2000,
    'runtime_delegate_calls_consumed': 0,
    'worker_environment_exact': WORKER_ENV,
    'service_environment_overrides_authorized': False,
    'retry_authorized': False,
    'cache_reuse_authorized': False,
    'training_authorized': False,
    'reward_read_authorized': False,
    'dev_hidden_final_outcome_read_authorized': False,
    'submission_authorized': False,
}
RUNTIME = {
    'execution_authority_materialized': True,
    'phase_a_launcher_executed': False,
    'phase_a_driver_executed': False,
    'phase_a_worker_invocations': 0,
    'rng_proxy_delegate_invocations': 0,
    'qualification_output_created': False,
    'cache_reused': False,
    'training_launched': False,
    'reward_read': False,
    'dev_hidden_final_outcome_read': False,
}
EXECUTION_BOUNDARY = {
    'authority_materialization_only': True,
    'phase_a_launcher_invocations': 0,
    'phase_a_driver_invocations': 0,
    'phase_a_worker_invocations': 0,
    'rng_proxy_delegate_invocations': 0,
    'training_invocations': 0,
    'reward_reads': 0,
    'dev_hidden_final_outcome_reads': 0,
}

CONTRACT_FORMAT = 'strict-track2-v522-v521-phase-a-authority-split-state-repair-execution-authority-design-contract-v1'
CONTRACT_STATUS = 'design_only_frozen_sources_pending_independent_review_no_authority'
OUTPUT_FORMAT = 'strict-track2-v522-v521-phase-a-authority-split-state-repair-execution-authority-v1'
OUTPUT_STATUS = 'authorized_exact_one_external_v522_phase_a_cache_qualification_attempt'

CHECK_KEYS = sorted({
    'contract_current', 'materializer_current', 'source_closure_current',
    'source_order_exact', 'source_aliases_exact', 'fresh_prereg_exact8',
    'v519_evidence_exact6', 'v519_process_receipt_exact', 'v519_stdout_exact1',
    'v519_sample_call_partition', 'v519_raw_warning_exact', 'v519_output_exact',
    'v519_rng_four_stage_full', 'v519_inputs_prepost_equal',
    'v517_candidate_exact1', 'v517_evidence_exact6', 'v517_process_passed',
    'v517_dual_bind_current', 'v517_normalization_diff_exact2',
    'old_phase_a_failures_no_retry', 'per_call_rng_contract_source_enforced',
    'worker_env_exact2', 'service_env_unchanged', 'fresh_roots_absent',
    'historical_absences', 'current_absences', 'input_snapshots_equal',
    'publish_stable_excludes_authority_prep', 'no_live_phase_a_process',
    'gpu_empty', 'authorization_boundary', 'execution_boundary', 'no_pending_values',
    'v520_authority_expected_current_exact1', 'v520_authority_top68_check33_source23',
    'v521_failed_transport_evidence_exact6', 'v521_transport_partition_exact',
    'v521_stdout_exact2_native_empty', 'split_forensic_exact',
    'split_transition_sole_authority_absence_leaf',
    'historical_absence_validator_not_reused_post_authority',
})

CONTRACT_TOP_KEYS = {
    'format', 'status', 'seed', 'lineage', 'source_closure', 'source_role_order',
    'source_aliases', 'source_closure_sha256', 'authority_materializer_source',
    'phase_a_launcher_source', 'phase_a_driver_source', 'phase_a_worker_source',
    'rng_proxy_source', 'phase_a_independent_auditor_source',
    'phase_a_preregistration_source', 'qualification_output_root',
    'worker_environment_exact', 'per_call_rng_evidence_contract',
    'v519_evidence_tree', 'v519_process_receipt', 'v519_child_stdout',
    'v519_output_record', 'v519_rng_isolation_evidence', 'v519_source_records',
    'v517_candidate_tree', 'v517_candidate_receipt', 'v517_evidence_tree',
    'v517_process_receipt', 'v517_normalized_current_view',
    'old_phase_a_failure_forensics', 'historical_absences',
    'v520_authority_registration_tree', 'v520_authority_receipt',
    'v520_authority_semantics', 'v521_materialization_evidence_tree',
    'v521_materialization_process_receipt', 'v521_materialization_stdout',
    'v521_materialization_stderr', 'v521_execution_partition',
    'authority_split_state_forensic', 'old_authority_expected_current',
    'current_absences_after_authority', 'authority_receipt_contract',
    'authorization', 'runtime_observation', 'execution_boundary',
}

def csha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()

def json_exact(left, right):
    """Canonical JSON equality that does not collapse bool and int values."""
    return json.dumps(left, sort_keys=True, separators=(',', ':'), ensure_ascii=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def regular(path, expected_sha=None, expected_bytes=None):
    path = Path(path)
    st = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(st.st_mode):
        raise RuntimeError('nonregular ' + str(path))
    record = {'path': str(path), 'sha256': sha(path), 'logical_bytes': st.st_size}
    if expected_sha is not None and record['sha256'] != expected_sha:
        raise RuntimeError('sha ' + str(path))
    if expected_bytes is not None and record['logical_bytes'] != expected_bytes:
        raise RuntimeError('bytes ' + str(path))
    return record


def exact_tree(root):
    root = Path(root)
    inventory = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        metadata = os.lstat(path)
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError('tree member ' + relative)
        inventory.append([relative, sha(path), metadata.st_size])
    lines = ''.join(f'{digest}  {relative}\n' for relative, digest, _size in inventory).encode()
    return {
        'root': str(root), 'inventory': inventory, 'file_count': len(inventory),
        'logical_file_bytes': sum(row[2] for row in inventory),
        'sha256sum_lines_digest_sha256': hashlib.sha256(lines).hexdigest(),
        'canonical_json_triples_digest_sha256': csha(inventory),
    }

def exact_summary(tree, expected):
    return all(tree.get(key) == value for key, value in expected.items())

def load_base():
    regular(OLD_MATERIALIZER, OLD_MATERIALIZER_SHA, OLD_MATERIALIZER_BYTES)
    spec = importlib.util.spec_from_file_location('v509_frozen', OLD_MATERIALIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.load_old()
    base = old.load_old().load_base()
    module.configure(base)
    base.CONTRACT_PATH = CONTRACT_PATH
    base.MATERIALIZER_PATH = SELF_PATH
    base.WRAPPER_PATH = LAUNCHER_PATH
    base.AUTHORITY_ROOT = AUTHORITY_ROOT
    base.WRAPPER_ATTEMPT_ROOT = ATTEMPT_ROOT
    base.CONTRACT_FORMAT = CONTRACT_FORMAT
    base.CONTRACT_STATUS = CONTRACT_STATUS
    base.OUTPUT_FORMAT = OUTPUT_FORMAT
    base.OUTPUT_STATUS = OUTPUT_STATUS
    base.AUTHORIZATION = AUTHORIZATION
    base.RUNTIME = RUNTIME
    base.EXECUTION_BOUNDARY = EXECUTION_BOUNDARY
    base.CONTRACT_SOURCE_ORDER = CONTRACT_SOURCE_ORDER
    base.AUTHORITY_SOURCE_ORDER = AUTHORITY_SOURCE_ORDER
    base.SOURCE_ALIASES = SOURCE_ALIASES
    base.CONTRACT_TOP_KEYS = CONTRACT_TOP_KEYS
    base.CHECK_KEYS = CHECK_KEYS
    return old, base

def absence_paths():
    historical = {
        'authority_root': AUTHORITY_ROOT,
        'authority_prep': AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + '.registration-prep'),
        'phase_a_attempt_root': ATTEMPT_ROOT,
        'phase_a_attempt_prep': ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + '.attempt-prep'),
        'qualification_output_root': QUALIFICATION_ROOT,
        'qualification_output_prep': QUALIFICATION_ROOT.with_name(QUALIFICATION_ROOT.name + '.attempt-prep'),
    }
    current = {key: value for key, value in historical.items() if key != 'authority_root'}
    return historical, current

def validate_v519(base):
    tree = base.tree(V519_EVIDENCE)
    if not exact_summary(tree, V519_TREE_SUMMARY):
        raise RuntimeError('v519 tree')
    process_record = regular(V519_PROCESS, V519_PROCESS_SHA, V519_PROCESS_BYTES)
    stdout_record = regular(V519_STDOUT, V519_STDOUT_SHA, V519_STDOUT_BYTES)
    process = json.loads(V519_PROCESS.read_text())
    child = json.loads(V519_STDOUT.read_text())
    rng = child.get('rng_isolation_evidence', {})
    if process.get('passed') is not True or process.get('status') != 'passed_exact_once_isolated_sample0_fixture':
        raise RuntimeError('v519 process')
    if process.get('sample0_calls_started') != 1 or process.get('sample0_calls_completed') != 1 or child.get('calls') != 1:
        raise RuntimeError('v519 calls')
    if process.get('v517_candidate_and_external_terminal_validated_current') is not True:
        raise RuntimeError('v519 v517 bind')
    if child.get('output_schema') != {'dtype': 'uint8', 'shape': [8, 256, 256, 3], 'sha256': '119d47797f50b24b2050c91f99b1733b9f2c0ed20a7987f61d44ca42ef3dfce3'}:
        raise RuntimeError('v519 output')
    if child.get('raw_warning_count') != 1 or child.get('canonical_warning_count') != 1 or child.get('filtered_warning_count') != 0 or child.get('deduplicated_warning_count') != 0 or child.get('other_warning_count') != 0 or len(child.get('raw_warnings_ordered', [])) != 1:
        raise RuntimeError('v519 warnings')
    required = {
        'format': 'strict-track2-v519-rng-isolation-four-stage-v1',
        'exact_seed': 1103331955,
        'cuda_device_count': 1,
        'cuda_device_indices': [0],
        'delegate_invocations_started': 1,
        'delegate_invocations_completed': 1,
        'fork_rng_devices_exact_all_cuda_indices': True,
        'python_all_four_stages_equal': True,
        'numpy_all_four_stages_equal': True,
        'torch_cpu_exit_restored': True,
        'torch_cuda_exit_restored': True,
        'torch_internal_change_allowed': True,
        'exception_finally_restoration_complete': True,
    }
    if not all(rng.get(key) == value for key, value in required.items()):
        raise RuntimeError('v519 rng flags')
    if not set(PER_CALL_RNG_CONTRACT['required_rng_stage_keys']) <= set(rng):
        raise RuntimeError('v519 rng stages')
    entry, inside, internal, exit_state = (rng[key] for key in PER_CALL_RNG_CONTRACT['required_rng_stage_keys'])
    if entry != exit_state or inside['python'] != internal['python'] or inside['numpy'] != internal['numpy']:
        raise RuntimeError('v519 rng values')
    if not all(process.get(key) is True for key in ('input_snapshot_prepost_exactly_equal', 'alias_held_prepost_exactly_equal', 'package_tree_prepost_exactly_equal', 'sample0_dataset_prepost_exactly_equal', 'health_prepost_exactly_equal', 'process_reaped', 'process_group_empty')):
        raise RuntimeError('v519 closure')
    return tree, process_record, stdout_record, process, child

def validate_v517(base):
    evidence_tree = base.tree(V517_EVIDENCE)
    candidate_tree = base.tree(V517_CANDIDATE_ROOT)
    if not exact_summary(evidence_tree, V517_TREE_SUMMARY) or not exact_summary(candidate_tree, V517_CANDIDATE_TREE_SUMMARY):
        raise RuntimeError('v517 trees')
    process_record = regular(V517_PROCESS, V517_PROCESS_SHA, V517_PROCESS_BYTES)
    candidate_record = regular(V517_CANDIDATE, V517_CANDIDATE_SHA, V517_CANDIDATE_BYTES)
    process = json.loads(V517_PROCESS.read_text())
    candidate = json.loads(V517_CANDIDATE.read_text())
    actual = process.get('candidate_actual_state', {})
    validation = process.get('candidate_validation', {})
    if process.get('passed') is not True or process.get('status') != 'passed_external_terminal' or process.get('candidate_consumable') is not True:
        raise RuntimeError('v517 process')
    if actual.get('receipt_record') != candidate_record or actual.get('tree') != candidate_tree:
        raise RuntimeError('v517 dual record')
    if validation.get('candidate_receipt_record') != candidate_record or validation.get('candidate_tree') != candidate_tree or validation.get('candidate_receipt') != candidate:
        raise RuntimeError('v517 dual validation')
    if candidate.get('authorization_leaf_diff_count') != 2 or candidate.get('authorization_leaf_diffs_canonical_sha256') != '1095bcaecfa85a1dc5dfe1e4e511490df6537640269ee9006a80fa5b307a044e' or candidate.get('normalized_process_receipt_canonical_sha256') != 'e707fc291c925ab9de87f3175f6e84c7e855cb2a986485cdf1b9edcaa429dc85':
        raise RuntimeError('v517 normalized diff')
    if candidate.get('standalone_consumable') is not False or candidate.get('external_terminal_required') is not True or candidate.get('publication_success_claimed') is not False:
        raise RuntimeError('v517 candidate truth')
    historical_snapshot = validation.get('independent_current_external_snapshot', {})
    source_row = historical_snapshot.get('source', {})
    target_row = historical_snapshot.get('target', {})
    source_current = regular(source_row.get('path', ''), source_row.get('sha256'), source_row.get('logical_bytes'))
    target_current = regular(target_row.get('path', ''), target_row.get('sha256'), target_row.get('logical_bytes'))
    f813_current = base.tree(Path(historical_snapshot.get('f813_tree', {}).get('root', '')))
    if f813_current != historical_snapshot.get('f813_tree'):
        raise RuntimeError('v517 f813 current')
    services_current = {}
    for port in ('8005', '18084'):
        expected = historical_snapshot.get('services', {}).get(port, {})
        with urllib.request.urlopen(expected.get('url'), timeout=10) as response:
            raw = response.read()
            code = response.status
        observed = {
            'url': expected.get('url'), 'http_code': code,
            'raw_body_sha256': hashlib.sha256(raw).hexdigest(),
            'raw_body_logical_bytes': len(raw), 'body': json.loads(raw),
        }
        if observed != expected:
            raise RuntimeError('v517 service current ' + port)
        services_current[port] = observed
    current_revalidation = {
        'source': source_current, 'target': target_current,
        'f813_tree': f813_current, 'services': services_current,
        'matching_pids': [], 'gpu_compute_pids': [],
    }
    view = {
        'candidate_tree': candidate_tree,
        'candidate_receipt': candidate_record,
        'evidence_tree': evidence_tree,
        'process_receipt': process_record,
        'authorization_leaf_diff_count': 2,
        'authorization_leaf_diffs_canonical_sha256': candidate['authorization_leaf_diffs_canonical_sha256'],
        'original_process_receipt_canonical_sha256': candidate['original_process_receipt_canonical_sha256'],
        'normalized_process_receipt_canonical_sha256': candidate['normalized_process_receipt_canonical_sha256'],
        'three_snapshots_and_current_equal': validation.get('three_snapshots_and_current_equal') is True,
        'external_terminal_passed': True,
        'normalized_current_view_revalidated': current_revalidation,
    }
    if view['three_snapshots_and_current_equal'] is not True:
        raise RuntimeError('v517 current view')
    return candidate_tree, candidate_record, evidence_tree, process_record, view

def validate_failures():
    records = {}
    values = {}
    for role, (path, digest, size) in FAILURE_SOURCES.items():
        records[role] = regular(path, digest, size)
        values[role] = json.loads(path.read_text())
    if values['v509_import_failure_forensic'].get('passed') is not True:
        raise RuntimeError('v509 import forensic')
    if values['v509_warning_failure_forensic'].get('passed') is not False or values['v509_warning_failure_forensic'].get('retry_authorized') is not False:
        raise RuntimeError('v509 warning forensic')
    if values['v511_anchor_failure_forensic'].get('status') != 'immutable_failed_no_retry_read_only_forensic':
        raise RuntimeError('v511 forensic')
    if values['v518_rng_failure_forensic'].get('status') != 'immutable_failed_no_retry_read_only_forensic' or values['v518_rng_failure_forensic'].get('authorization', {}).get('retry_authorized') is not False:
        raise RuntimeError('v518 forensic')
    raw = json.dumps(values, sort_keys=True)
    if 'no_retry' not in raw and 'retry_authorized' not in raw:
        raise RuntimeError('failure no-retry evidence')
    return records, values


def service_health():
    result = {}
    for key, url in {
        '8005_v1_health': 'http://127.0.0.1:8005/v1/health',
        '18084_health': 'http://127.0.0.1:18084/health',
    }.items():
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = response.read()
            result[key] = {
                'http_code': response.status,
                'body_sha256': hashlib.sha256(payload).hexdigest(),
                'body_bytes': len(payload),
                'json_model': json.loads(payload),
            }
    return result


def validate_split_state():
    records = {
        'v522_v521_authority_split_state_forensic': regular(
            SPLIT_FORENSIC, SPLIT_FORENSIC_SHA, SPLIT_FORENSIC_BYTES),
        'v520_authority_design_contract': regular(
            V520_CONTRACT, V520_CONTRACT_SHA, V520_CONTRACT_BYTES),
        'v520_authority_materializer': regular(
            V520_MATERIALIZER, V520_MATERIALIZER_SHA, V520_MATERIALIZER_BYTES),
        'v520_authority_receipt': regular(
            V520_AUTHORITY_RECEIPT, V520_AUTHORITY_RECEIPT_SHA, V520_AUTHORITY_RECEIPT_BYTES),
        'v521_materialization_process_receipt': regular(
            V521_PROCESS, V521_PROCESS_SHA, V521_PROCESS_BYTES),
        'v521_materialization_stdout': regular(
            V521_STDOUT, V521_STDOUT_SHA, V521_STDOUT_BYTES),
        'v521_materialization_stderr': regular(
            V521_STDERR, V521_STDERR_SHA, V521_STDERR_BYTES),
        'v521_transport_failure_forensic': regular(
            V521_FAILURE_FORENSIC, V521_FAILURE_FORENSIC_SHA, V521_FAILURE_FORENSIC_BYTES),
        'v521_transport_helper': regular(V521_HELPER, V521_HELPER_SHA, V521_HELPER_BYTES),
        'v521_transport_script': regular(V521_SCRIPT, V521_SCRIPT_SHA, V521_SCRIPT_BYTES),
    }
    authority_tree = exact_tree(V520_AUTHORITY_ROOT)
    evidence_tree = exact_tree(V521_EVIDENCE_ROOT)
    if not exact_summary(authority_tree, V520_AUTHORITY_TREE_SUMMARY):
        raise RuntimeError('v520 authority exact1 tree')
    if not exact_summary(evidence_tree, V521_EVIDENCE_TREE_SUMMARY):
        raise RuntimeError('v521 evidence exact6 tree')

    old_contract = json.loads(V520_CONTRACT.read_text(encoding='utf-8'))
    authority = json.loads(V520_AUTHORITY_RECEIPT.read_text(encoding='utf-8'))
    authority_schema = old_contract['authority_receipt_contract']
    if (sorted(authority) != authority_schema['top_keys'] or len(authority) != 68
            or authority.get('format') != 'strict-track2-v520-v519-phase-a-rng-isolated-qualification-execution-authority-v1'
            or authority.get('status') != 'authorized_exact_one_external_v520_phase_a_cache_qualification_attempt'
            or authority.get('passed') is not True
            or len(authority.get('checks', {})) != 33
            or authority.get('check_keys') != authority_schema['check_keys']
            or not all(type(value) is bool and value is True for value in authority['checks'].values())
            or authority.get('check_key_set_sha256') != csha(authority['check_keys'])
            or authority.get('checks_sha256') != csha(authority['checks'])
            or len(authority.get('source_closure', {})) != 23
            or authority.get('source_closure_sha256') != csha(authority['source_closure'])
            or authority.get('source_role_order') != ['authority_design_contract', *old_contract['source_role_order']]
            or set(authority['source_closure']) != set(authority['source_role_order'])
            or authority.get('source_aliases') != {
                'authority_design_contract': 'authority_design_contract',
                **old_contract['source_aliases'],
            }
            or authority.get('input_snapshots_exactly_equal') is not True
            or authority.get('input_pre_snapshot') != authority.get('input_post_snapshot')
            or not json_exact(authority.get('authorization'), old_contract['authorization'])
            or not json_exact(authority.get('runtime_observation'), old_contract['runtime_observation'])
            or not json_exact(authority.get('execution_boundary'), old_contract['execution_boundary'])
            or authority.get('authority_design_contract') != records['v520_authority_design_contract']
            or type(authority.get('authorization', {}).get('launcher_invocations_consumed')) is not int
            or authority.get('authorization', {}).get('launcher_invocations_consumed') != 0
            or type(authority.get('authorization', {}).get('runtime_delegate_calls_consumed')) is not int
            or authority.get('authorization', {}).get('runtime_delegate_calls_consumed') != 0):
        raise RuntimeError('v520 authority schema/semantics')
    for row in authority['source_closure'].values():
        if regular(row['path'], row['sha256'], row['logical_bytes']) != row:
            raise RuntimeError('v520 authority source current')
    for row in authority.get('required_absences', {}).values():
        if row.get('absent') is not True or os.path.lexists(row.get('path', '')):
            raise RuntimeError('v520 authority downstream absence')

    process = json.loads(V521_PROCESS.read_text(encoding='utf-8'))
    cleanup = process.get('cleanup', {})
    expected_zero = {
        'phase_a_launcher_invocations', 'phase_a_driver_invocations',
        'phase_a_worker_invocations', 'rng_proxy_delegate_invocations',
    }
    if (process.get('status') != 'failed_no_retry' or process.get('passed') is not False
            or process.get('error_type') != 'RuntimeError'
            or process.get('error') != 'v520 transport collection failure forensic'
            or type(process.get('transport_helper_invocations')) is not int
            or process.get('transport_helper_invocations') != 1
            or type(process.get('authority_materializer_invocations')) is not int
            or process.get('authority_materializer_invocations') != 1
            or any(type(process.get(key)) is not int or process.get(key) != 0 for key in expected_zero)
            or process.get('retry_authorized') is not False
            or process.get('stdout') != records['v521_materialization_stdout']
            or process.get('stderr') != records['v521_materialization_stderr']
            or process.get('prior_transport_failure_forensic') != records['v521_transport_failure_forensic']
            or process.get('transport_helper') != records['v521_transport_helper']
            or process.get('transport_script') != records['v521_transport_script']
            or cleanup != {'term_sent': False, 'kill_sent': False, 'reaped': True, 'group_empty': True}):
        raise RuntimeError('v521 failed process partition')
    stdout_bytes = V521_STDOUT.read_bytes()
    stdout_lines = stdout_bytes.splitlines()
    if len(stdout_lines) != 2:
        raise RuntimeError('v521 stdout exact2')
    stdout_values = [json.loads(line) for line in stdout_lines]
    if (stdout_values[0].get('line_origin') != 'transport_helper'
            or stdout_values[0].get('materializer_native_empty') is not True
            or stdout_values[0].get('authority_receipt') != records['v520_authority_receipt']
            or stdout_values[0].get('authority_registration_tree') != authority_tree
            or stdout_values[1] != {
                'authority_materializer_invocations': 1, 'committed_success': True,
                'line_origin': 'transport_helper', 'passed': True,
                'phase_a_driver_invocations': 0, 'phase_a_launcher_invocations': 0,
                'phase_a_worker_invocations': 0, 'rng_proxy_delegate_invocations': 0,
                'transport_helper_invocations': 1,
            } or V521_STDERR.read_bytes() != b''):
        raise RuntimeError('v521 stdout/stderr semantics')

    forensic = json.loads(SPLIT_FORENSIC.read_text(encoding='utf-8'))
    top_keys = {
        'format', 'status', 'passed', 'captured_at_date', 'lineage',
        'v520_materialized_authority_receipt', 'v520_materialized_authority_registration_tree',
        'v520_materialized_authority_schema', 'v521_materialization_evidence_tree',
        'v521_materialization_process_receipt', 'v521_materialization_stdout',
        'v521_materialization_stderr', 'sole_snapshot_transition',
        'other_seven_historical_absences_current', 'execution_partition',
        'frozen_downstream_sources', 'v521_transport_sources',
        'service_health_pre_and_current_equal', 'process_and_gpu_state', 'authorization',
    }
    transition = forensic.get('sole_snapshot_transition', {})
    other_seven = forensic.get('other_seven_historical_absences_current', [])
    partition = forensic.get('execution_partition', {})
    if (set(forensic) != top_keys
            or forensic.get('format') != 'strict-track2-v522-v521-phase-a-authority-split-state-forensic-v1'
            or forensic.get('passed') is not True
            or forensic.get('v520_materialized_authority_receipt') != records['v520_authority_receipt']
            or forensic.get('v520_materialized_authority_registration_tree') != authority_tree
            or forensic.get('v521_materialization_evidence_tree') != evidence_tree
            or forensic.get('v521_materialization_process_receipt', {}).get('path') != str(V521_PROCESS)
            or forensic.get('v521_materialization_process_receipt', {}).get('sha256') != V521_PROCESS_SHA
            or forensic.get('v521_materialization_process_receipt', {}).get('logical_bytes') != V521_PROCESS_BYTES
            or forensic.get('v521_materialization_stdout', {}).get('path') != str(V521_STDOUT)
            or forensic.get('v521_materialization_stdout', {}).get('sha256') != V521_STDOUT_SHA
            or forensic.get('v521_materialization_stderr', {}).get('sha256') != V521_STDERR_SHA
            or transition != {
                'logical_path': ['roots_after_failure', 'authority_root', 'absent'],
                'historical_value': True, 'current_value': False, 'current_state': 'present',
                'authority_root': str(V520_AUTHORITY_ROOT),
                'transition_cause': 'successful_fcf7_authority_materialization_exact1',
                'transition_is_valid': True, 'transport_post_snapshot_predicate_is_stale': True,
            }
            or not V520_AUTHORITY_ROOT.is_dir() or V520_AUTHORITY_ROOT.is_symlink()
            or len(other_seven) != 7 or not all(not os.path.lexists(path) for path in other_seven)
            or type(partition.get('v521_transport_helper_invocations')) is not int
            or partition.get('v521_transport_helper_invocations') != 1
            or type(partition.get('v520_authority_materializer_invocations')) is not int
            or partition.get('v520_authority_materializer_invocations') != 1
            or any(type(partition.get(key)) is not int or partition.get(key) != 0 for key in (
                'v520_phase_a_launcher_invocations', 'v520_phase_a_driver_invocations',
                'v520_phase_a_worker_invocations', 'v520_rng_proxy_delegate_invocations',
                'v520_independent_auditor_invocations', 'phase_a_qualification_outputs_created',
                'training_invocations', 'cache_reuse_invocations', 'reward_reads',
                'dev_hidden_final_outcome_reads'))
            or partition.get('retry_authorized') is not False
            or forensic.get('service_health_pre_and_current_equal') != service_health()
            or forensic.get('process_and_gpu_state', {}).get('cleanup_reaped') is not True
            or forensic.get('process_and_gpu_state', {}).get('cleanup_group_empty') is not True
            or any(forensic.get('authorization', {}).values())):
        raise RuntimeError('v521 split forensic')
    semantics = {
        'format': authority['format'], 'status': authority['status'], 'passed': True,
        'top_key_count': 68, 'check_count': 33, 'source_closure_count': 23,
        'checks_all_strict_true': True, 'source_closure_all_current': True,
        'input_snapshots_exactly_equal': True,
        'launcher_invocations_consumed': 0, 'runtime_delegate_calls_consumed': 0,
    }
    execution_partition = {
        'v521_transport_helper_invocations': 1,
        'v520_authority_materializer_invocations': 1,
        'fresh_v522_authority_materializer_invocations': 0,
        'phase_a_launcher_invocations': 0, 'phase_a_driver_invocations': 0,
        'phase_a_worker_invocations': 0, 'rng_proxy_delegate_invocations': 0,
        'training_invocations': 0, 'retry_authorized': False,
    }
    return {
        'records': records, 'forensic': forensic,
        'v520_authority_registration_tree': authority_tree,
        'v520_authority_receipt': records['v520_authority_receipt'],
        'v520_authority_semantics': semantics,
        'v521_materialization_evidence_tree': evidence_tree,
        'v521_materialization_process_receipt': records['v521_materialization_process_receipt'],
        'v521_materialization_stdout': records['v521_materialization_stdout'],
        'v521_materialization_stderr': records['v521_materialization_stderr'],
        'v521_execution_partition': execution_partition,
        'authority_split_state_forensic': forensic,
        'old_authority_expected_current': {
            'root': str(V520_AUTHORITY_ROOT), 'present': True,
            'receipt': records['v520_authority_receipt'],
            'registration_tree': authority_tree,
            'historical_absence_validator_reused_post_authority': False,
        },
    }

def validate_prereg(source_records):
    old = json.loads(OLD_PREREG.read_text())
    new = json.loads(PREREG_PATH.read_text())
    expected_roles = [
        'phase_a_preregistration_materializer', 'phase_a_driver', 'phase_a_process_worker',
        'phase_a_rng_isolation_proxy', 'cache_scope_helper', 'phase_a_independent_auditor',
        'phase_a_static_auditor', 'frozen_phase_a_launcher',
    ]
    if set(new) != set(old) or new.get('format') != old.get('format') or new.get('status') != old.get('status'):
        raise RuntimeError('prereg schema')
    if new.get('qualification_output_root') != str(QUALIFICATION_ROOT):
        raise RuntimeError('prereg root')
    rows = new.get('execution_source_records')
    if [row.get('role') for row in rows] != expected_roles or new.get('execution_sources_digest_sha256') != csha(rows):
        raise RuntimeError('prereg exact8')
    expected = {
        'phase_a_driver': source_records['fresh_phase_a_driver'],
        'phase_a_process_worker': source_records['fresh_phase_a_worker'],
        'phase_a_rng_isolation_proxy': source_records['fresh_rng_proxy'],
        'phase_a_independent_auditor': source_records['fresh_phase_a_independent_auditor'],
    }
    if any(new['execution_sources'].get(role) != record for role, record in expected.items()):
        raise RuntimeError('prereg active records')
    return new

def main():
    old_module, base = load_base()
    if os.sys.argv[1:] == ['--synthetic-self-test']:
        checks = {
            'source_roles': len(CONTRACT_SOURCE_ORDER) == 32,
            'worker_env_exact2': set(WORKER_ENV) == {'PYTHONPATH', 'CUBLAS_WORKSPACE_CONFIG'},
            'authorization_only_one_qualification': AUTHORIZATION['launcher_invocations_authorized'] == 1 and AUTHORIZATION['training_authorized'] is False,
            'per_call_not_sample_substitute': PER_CALL_RNG_CONTRACT['sample0_evidence_is_ancestry_not_runtime_substitute'] is True,
            'bool_int_canonical_json_distinct': not json_exact({'count': 1}, {'count': True}),
            'publish_selftest': base.publication_self_test(),
        }
        print(json.dumps({'passed': all(checks.values()), 'checks': checks, 'checks_sha256': csha(checks)}, sort_keys=True))
        return 0 if all(checks.values()) else 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha', required=True)
    parser.add_argument('--contract-bytes', type=int, required=True)
    parser.add_argument('--materializer-source', type=Path, required=True)
    parser.add_argument('--materializer-sha', required=True)
    parser.add_argument('--materializer-bytes', type=int, required=True)
    parser.add_argument('--authority-root', type=Path, required=True)
    args = parser.parse_args()
    if args.contract != CONTRACT_PATH or args.materializer_source != SELF_PATH or args.authority_root != AUTHORITY_ROOT:
        raise RuntimeError('canonical CLI')
    contract_record = regular(CONTRACT_PATH, args.contract_sha, args.contract_bytes)
    materializer_record = regular(SELF_PATH, args.materializer_sha, args.materializer_bytes)
    contract = json.loads(CONTRACT_PATH.read_text())
    if set(contract) != CONTRACT_TOP_KEYS or contract.get('format') != CONTRACT_FORMAT or contract.get('status') != CONTRACT_STATUS or contract.get('seed') != 1658:
        raise RuntimeError('contract schema')
    source_records = contract['source_closure']
    observed = {role: regular(row['path'], row['sha256'], row['logical_bytes']) for role, row in source_records.items()}
    if contract.get('source_role_order') != CONTRACT_SOURCE_ORDER or set(observed) != set(CONTRACT_SOURCE_ORDER) or observed != source_records or contract.get('source_closure_sha256') != csha(source_records):
        raise RuntimeError('source closure')
    if contract.get('source_aliases') != {role: SOURCE_ALIASES[role] for role in CONTRACT_SOURCE_ORDER}:
        raise RuntimeError('source aliases')
    expected_paths = {
        'authority_materializer': SELF_PATH, 'fresh_phase_a_launcher': LAUNCHER_PATH,
        'fresh_phase_a_driver': DRIVER_PATH, 'fresh_phase_a_worker': WORKER_PATH,
        'fresh_rng_proxy': PROXY_PATH, 'fresh_phase_a_independent_auditor': AUDITOR_PATH,
        'fresh_phase_a_preregistration': PREREG_PATH,
        'v522_v521_authority_split_state_forensic': SPLIT_FORENSIC,
        'v520_authority_design_contract': V520_CONTRACT,
        'v520_authority_materializer': V520_MATERIALIZER,
        'v520_authority_receipt': V520_AUTHORITY_RECEIPT,
        'v521_materialization_process_receipt': V521_PROCESS,
        'v521_materialization_stdout': V521_STDOUT,
        'v521_materialization_stderr': V521_STDERR,
        'v521_transport_failure_forensic': V521_FAILURE_FORENSIC,
        'v521_transport_helper': V521_HELPER,
        'v521_transport_script': V521_SCRIPT,
    }
    if any(observed[role]['path'] != str(path) for role, path in expected_paths.items()) or observed['authority_materializer'] != materializer_record:
        raise RuntimeError('active paths')
    old_contract = json.loads(OLD_CONTRACT.read_text())
    if any(observed[role] != old_contract['source_closure'][role] for role in FIXED_ROLES):
        raise RuntimeError('fixed sources')
    v519_tree, v519_process_record, v519_stdout_record, v519_process, v519_child = validate_v519(base)
    v517_candidate_tree, v517_candidate_record, v517_tree, v517_process_record, v517_view = validate_v517(base)
    failure_records, failure_values = validate_failures()
    split = validate_split_state()
    if any(observed[role] != failure_records[role] for role in failure_records):
        raise RuntimeError('failure source closure')
    if any(observed[role] != record for role, record in split['records'].items()):
        raise RuntimeError('split source closure')
    v519_map = v519_process['source_records']
    v519_expected = {
        'v519_rng_sample_child': v519_map['child_source'],
        'v519_rng_sample_design': v519_map['design'],
        'v519_rng_sample_transport_helper': v519_map['transport_helper'],
        'v519_rng_sample_transport_script': v519_map['transport_script'],
    }
    if any(observed[role] != record for role, record in v519_expected.items()):
        raise RuntimeError('v519 source closure')
    prereg = validate_prereg(observed)
    historical, current = absence_paths()
    if any(os.path.lexists(path) for path in historical.values()):
        raise RuntimeError('fresh roots')
    expected_lineage = {
        'name': 'v522_v521_phase_a_authority_split_state_repair',
        'fresh_authority_root': str(AUTHORITY_ROOT),
        'fresh_phase_a_attempt_root': str(ATTEMPT_ROOT),
        'fresh_qualification_output_root': str(QUALIFICATION_ROOT),
        'supersedes_v521_transport_failure_no_retry_without_reusing_old_launcher': True,
        'old_v520_authority_expected_current': True,
        'sole_semantic_change': 'fresh_phase_a_lineage_after_valid_authority_failed_transport_split',
    }
    if (not json_exact(contract.get('lineage'), expected_lineage)
            or not json_exact(contract.get('authorization'), AUTHORIZATION)
            or not json_exact(contract.get('runtime_observation'), RUNTIME)
            or not json_exact(contract.get('execution_boundary'), EXECUTION_BOUNDARY)):
        raise RuntimeError('contract boundary')
    exact_contract_values = {
        'authority_materializer_source': materializer_record,
        'phase_a_launcher_source': observed['fresh_phase_a_launcher'],
        'phase_a_driver_source': observed['fresh_phase_a_driver'],
        'phase_a_worker_source': observed['fresh_phase_a_worker'],
        'rng_proxy_source': observed['fresh_rng_proxy'],
        'phase_a_independent_auditor_source': observed['fresh_phase_a_independent_auditor'],
        'phase_a_preregistration_source': observed['fresh_phase_a_preregistration'],
        'qualification_output_root': str(QUALIFICATION_ROOT),
        'worker_environment_exact': WORKER_ENV,
        'per_call_rng_evidence_contract': PER_CALL_RNG_CONTRACT,
        'v519_evidence_tree': v519_tree,
        'v519_process_receipt': v519_process_record,
        'v519_child_stdout': v519_stdout_record,
        'v519_output_record': v519_child['output_schema'],
        'v519_rng_isolation_evidence': v519_child['rng_isolation_evidence'],
        'v519_source_records': v519_process['source_records'],
        'v517_candidate_tree': v517_candidate_tree,
        'v517_candidate_receipt': v517_candidate_record,
        'v517_evidence_tree': v517_tree,
        'v517_process_receipt': v517_process_record,
        'v517_normalized_current_view': v517_view,
        'old_phase_a_failure_forensics': failure_values,
        **{key: split[key] for key in (
            'v520_authority_registration_tree', 'v520_authority_receipt',
            'v520_authority_semantics', 'v521_materialization_evidence_tree',
            'v521_materialization_process_receipt', 'v521_materialization_stdout',
            'v521_materialization_stderr', 'v521_execution_partition',
            'authority_split_state_forensic', 'old_authority_expected_current',
        )},
    }
    if any(not json_exact(contract.get(key), value) for key, value in exact_contract_values.items()):
        raise RuntimeError('contract anchors')
    v517_current = v517_view['normalized_current_view_revalidated']
    files = {'contract': str(CONTRACT_PATH), 'materializer': str(SELF_PATH), **{'source_' + role: row['path'] for role, row in source_records.items()}, 'v519_process': str(V519_PROCESS), 'v519_stdout': str(V519_STDOUT), 'v517_process': str(V517_PROCESS), 'v517_candidate': str(V517_CANDIDATE), 'v517_alias_source': v517_current['source']['path'], 'v517_alias_target': v517_current['target']['path'], 'v520_authority_receipt_current': str(V520_AUTHORITY_RECEIPT), 'v521_process_current': str(V521_PROCESS), 'v521_stdout_current': str(V521_STDOUT), 'v521_stderr_current': str(V521_STDERR)}
    trees = {'v519_evidence': str(V519_EVIDENCE), 'v517_evidence': str(V517_EVIDENCE), 'v517_candidate': str(V517_CANDIDATE_ROOT), 'v517_f813_current': v517_current['f813_tree']['root'], 'v520_authority_current': str(V520_AUTHORITY_ROOT), 'v521_evidence_current': str(V521_EVIDENCE_ROOT)}
    pre = base.snapshot(files, trees, current)
    checks = {key: True for key in CHECK_KEYS}
    checks['no_live_phase_a_process'] = old_module.load_old().no_live_phase_a_process()
    checks['gpu_empty'] = base.gpu_empty()
    checks['no_pending_values'] = not base.pending(contract)
    authority_sources = {'authority_design_contract': contract_record, **source_records}
    receipt = {
        'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS, 'passed': True,
        'source_closure': authority_sources, 'source_role_order': AUTHORITY_SOURCE_ORDER,
        'source_aliases': SOURCE_ALIASES, 'source_closure_sha256': csha(authority_sources),
        **{SOURCE_ALIASES[role]: row for role, row in authority_sources.items()},
        **exact_contract_values,
        'phase_a_attempt_root': str(ATTEMPT_ROOT),
        'historical_absences': {key: {'path': str(path), 'absent': True} for key, path in historical.items()},
        'required_absences': {key: {'path': str(path), 'absent': True} for key, path in current.items()},
        'checks': checks, 'check_keys': CHECK_KEYS,
        'check_key_set_sha256': csha(CHECK_KEYS), 'checks_sha256': csha(checks),
        'input_pre_snapshot': pre, 'input_post_snapshot': base.snapshot(files, trees, current),
        'input_snapshots_exactly_equal': True, 'authorization': AUTHORIZATION,
        'runtime_observation': RUNTIME, 'execution_boundary': EXECUTION_BOUNDARY,
        'phase_a_cache_qualification_authorized': True, 'training_authorized': False,
        'preregistration_sha256': prereg and source_records['fresh_phase_a_preregistration']['sha256'],
        'contract_sha256': PHASE_A_DESIGN_CONTRACT_SHA,
    }
    receipt_schema = contract['authority_receipt_contract']
    if (receipt_schema.get('top_keys') != sorted(receipt)
            or receipt_schema.get('check_keys') != CHECK_KEYS
            or not json_exact(receipt_schema.get('authorization_exact'), AUTHORIZATION)
            or not json_exact(receipt_schema.get('per_call_rng_evidence_contract_exact'), PER_CALL_RNG_CONTRACT)):
        raise RuntimeError('receipt schema')
    if not all(checks.values()) or receipt['input_pre_snapshot'] != receipt['input_post_snapshot']:
        raise RuntimeError('receipt gates')
    publish_absences = {key: path for key, path in current.items() if key != 'authority_prep'}
    publish_pre = base.snapshot(files, trees, publish_absences)
    base.publish_exact1(AUTHORITY_ROOT, receipt, files, trees, publish_absences, publish_pre)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
