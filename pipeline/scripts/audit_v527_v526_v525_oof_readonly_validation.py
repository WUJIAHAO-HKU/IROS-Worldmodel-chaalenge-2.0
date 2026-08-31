#!/usr/bin/env python3
"""One-shot readonly validation of the normalized v524 qualification for a future OOF gate.

This source validates evidence only.  It never launches Phase A, an OOF model,
training, cache reuse, reward reads, or hidden/final outcome reads.
"""

import argparse
import ast
import copy
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
S = ROOT / 'pipeline/scripts'
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
SELF = S / 'audit_v527_v526_v525_oof_readonly_validation.py'
PREREG = S / 'v527_v526_v525_oof_readonly_gate_preregistration.json'
PREREG_SHA = '9fc4e53117309a77aac773a8eccc8550ad82446b8224872ca47afb8d763167ac'
PREREG_BYTES = 101898
CONTRACT = S / 'v527_v526_v525_oof_readonly_validation_execution_authority_contract.json'
AUTHORITY_ROOT = J / 'v527_v526_v525_oof_readonly_validation_execution_authority_seed1662_20260827'
AUTHORITY_RECEIPT = AUTHORITY_ROOT / 'authority_receipt.json'
AUTHORITY_PREP = AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name + '.authority-prep')
OUTPUT_ROOT = J / 'v527_v526_v525_oof_readonly_validation_attempt_seed1662_20260827'
OUTPUT_PREP = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + '.attempt-prep')
OUTPUT_MEMBER = 'oof_readonly_gate_validation_receipt.json'
EXTERNAL_EVIDENCE_ROOT = J / 'v527_v526_v525_oof_readonly_validation_external_evidence_seed1662_20260827'
EXTERNAL_EVIDENCE_PREP = EXTERNAL_EVIDENCE_ROOT.with_name(EXTERNAL_EVIDENCE_ROOT.name + '.evidence-prep')
EXTERNAL_PROCESS_MEMBER = 'process_receipt.json'
V526_HELPER_TEMP = S / '.invoke_v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_once.py.v526-owned-deploy-temp'
V526_SCRIPT_TEMP = Path('/root/.v525_phase_a_worker_receipt_format_readonly_reconciliation_once.sh.v526-owned-deploy-temp')

RECONCILER = S / 'reconcile_v525_v524_phase_a_worker_receipt_format_readonly.py'
RECONCILER_RECORD = ('81aa7da1eb525b5d03f8053da4dbce81e372b0660b6aa31e068dc3e2604b13f3', 48538)
V482_PREREG = J / 'v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json'
V482_SOURCES = {
    'v482_s0_execution_preregistration_ancestry': (V482_PREREG, '426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881', 294684),
    'v482_oof_design_contract': (S / 'v482_temporal_film_residual_model_design_contract.json', '19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb', 37030),
    'v482_static_auditor': (S / 'audit_v482_temporal8_residual_static.py', '088447774ad4ae0b67ec3470e4170ca5c8527e5fdfd0ee12fc000549bf785a2e', 16435),
    'v482_s0_validator': (S / 'audit_v482_temporal8_residual_s0.py', 'a24a20a8c0670e7bf53819bb48e78d204b482fbbdbe1060db61957488b3eed69', 20439),
    'v482_trainer_schema_ancestry': (S / 'train_v482_temporal8_residual_5fold.py', '674b68afed3b6c39663d629db38be11aa2c752e54692458b43d4254eb08b8c1d', 48428),
}
STAGE_A_AUTH = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_seed1661_20260827'
STAGE_A_EVIDENCE = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_execution_authority_materialization_evidence_seed1661_20260827'
CANDIDATE_ROOT = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_seed1661_20260827'
EXTERNAL_ROOT = J / 'v525_v524_phase_a_worker_receipt_format_readonly_reconciliation_attempt_seed1661_20260827'
QUALIFICATION_ROOT = Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
STAGE_A_TREE = (1, 95906, '1c3a3cce5bef518fb403cc632fd61236291fef99330badfdfbfbb1de0b0b1b98', '9c1f595ea80a08e021ecd14a7b20b5cb65294ba2a02066b114cb84e3e8656fb3')
STAGE_A_EVIDENCE_TREE = (6, 102271, 'd500f32ec9a5c60c086fb4eef9d98191d16e2e64311c3b731921ba165fe7aef6', '7c455785a8675f1214f793fd93355ee8145002d758e2eabe14bba29fb2da36b3')
CANDIDATE_TREE = (1, 43485, '37ce88ee4e1c57033bbbef9072141cf82f943787fbd650df14b54b271a7e1ffe', '1099268a0ae6419687a5203d182aa5489c86583eb2dddd386fb91c332a8032c3')
EXTERNAL_TREE = (6, 127746, '3cb39fa4b06cc230ccb1dda5dfbd9efc692af6ca5bd01de1a673ad200479fd6e', '16188ed7a3487dfd7827eaa3486b2422f775506ada301e385eb996c693a5f447')
QUALIFICATION_TREE = (20, 950124602, '6cdde75692d37b7eead3333d867b22dc6404f545bb5861b2f30b425fb4270dac', '223308d02ed5a4a082b4de639292030444a627bbbac7f361fac1f9c6a7fcbd1f')
STAGE_A_RECEIPT = ('1f468f2c5ed2f5928e9491e7113f1f34e1095dcd17c57c2a9c973107043a4057', 95906)
STAGE_A_PROCESS = ('fb30407693464e98feae18eb81ec52905a62a5a4d160889505e86ecaaa8a31f8', 31038)
CANDIDATE_RECEIPT = ('f577a285de806fdd9e27ed70708640da1b3686eebee66c5378cb9dcd0c5581f2', 43485)
EXTERNAL_PROCESS = ('e4674978630ebb85a824e31751ffba40b4cb06b9d737f7c5dc068635490df845', 38888)
ORIGINAL_CSHA = 'd00018deb7ea3ae660061e8879dc4a4f6d56bc65bd089c28c4a0580239c08101'
NORMALIZED_CSHA = 'c409c2b8f85641c23bc68fb06512cd664c84108d4b8b629480906193737f4b32'
DIFF_CSHA = 'fdc7999068d0e630272e6cb226b6b342692b5f8324cd506d593b5993b5f52779'
PREREG_TOP_KEYS = {
    'format', 'status', 'seed', 'classification', 'authority_contract_path',
    'fresh_authority_root', 'fresh_attempt_root', 'validator_source',
    'ancestry_sources', 'input_contract', 'normalized_qualification_view',
    'output_contract', 'authorization', 'current_state_contract',
}
AUTHORIZATION_KEYS = {
    'oof_readonly_gate_validator_invocations_authorized', 'oof_readonly_gate_validator_invocations_consumed',
    'oof_model_execution_invocations_authorized', 'phase_a_launcher_invocations_authorized',
    'phase_a_driver_invocations_authorized', 'phase_a_worker_invocations_authorized',
    'rng_proxy_delegate_invocations_authorized', 'phase_a_replay_invocations_authorized',
    'training_invocations_authorized', 'cache_reuse_invocations_authorized',
    'reward_read_invocations_authorized', 'dev_hidden_final_outcome_read_invocations_authorized',
    'submission_invocations_authorized', 'retry_authorized', 'oof_readonly_gate_validation_authorized',
    'oof_model_execution_authorized', 'phase_a_replay_authorized', 'training_authorized',
    'cache_reuse_authorized', 'reward_read_authorized', 'dev_hidden_final_outcome_read_authorized',
    'submission_authorized', 'qualification_readonly_validation_authorized',
    'qualification_mutation_authorized',
}
COUNT_AUTHORIZATION_KEYS = {
    key for key in AUTHORIZATION_KEYS if key.endswith('_invocations_authorized') or key.endswith('_invocations_consumed')
}
BOOL_AUTHORIZATION = {
    'retry_authorized': False,
    'oof_readonly_gate_validation_authorized': True,
    'oof_model_execution_authorized': False,
    'phase_a_replay_authorized': False,
    'training_authorized': False,
    'cache_reuse_authorized': False,
    'reward_read_authorized': False,
    'dev_hidden_final_outcome_read_authorized': False,
    'submission_authorized': False,
    'qualification_readonly_validation_authorized': True,
    'qualification_mutation_authorized': False,
}
EXPECTED_DIFF = [
    {'path': ['process_a', 'format'], 'original': 'strict-track2-v523-v522-rng-isolated-cache-worker-receipt-v1', 'normalized': 'strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1'},
    {'path': ['process_b', 'format'], 'original': 'strict-track2-v523-v522-rng-isolated-cache-worker-receipt-v1', 'normalized': 'strict-track2-v524-v523-rng-isolated-cache-worker-receipt-v1'},
]
NORMALIZED_VIEW_KEYS = {
    'format', 'original_worker_receipts', 'normalized_worker_receipts',
    'original_worker_receipts_canonical_sha256', 'normalized_worker_receipts_canonical_sha256',
    'recursive_diff', 'recursive_diff_count', 'recursive_diff_canonical_sha256',
    'recursive_diff_full_paths', 'candidate_exact1', 'external_terminal_exact6',
    'new_computation_claimed', 'historical_training_authority_not_inherited',
}
ANCESTRY_KEYS = {
    *V482_SOURCES,
    'v525_readonly_reconciler', 'v526_stage_b_predeploy_failure_fact',
    'v526_stage_b_deployer', 'v525_stage_b_helper', 'v525_stage_b_script',
}
INPUT_CONTRACT_KEYS = {
    'stage_a_authority', 'stage_a_external_evidence',
    'stage_b_candidate_and_external_terminal', 'qualification_tree',
    'qualification_terminal', 'qualification_report', 'qualification_independent_audit',
    'process_a', 'process_b', 'services',
}
OUTPUT_CONTRACT_KEYS = {'candidate', 'external_terminal'}
OUTPUT_FORMAT = 'strict-track2-v527-v526-v525-oof-readonly-gate-validation-candidate-v1'
OUTPUT_STATUS = 'readonly_oof_gate_inputs_verified_candidate_pending_external_terminal'
EXTERNAL_PROCESS_FORMAT = 'strict-track2-v527-v526-v525-oof-readonly-gate-validation-external-process-receipt-v1'
EXTERNAL_PROCESS_STATUS = 'passed_external_terminal'
CANDIDATE_OUTPUT_CONTRACT = {
    'root': str(OUTPUT_ROOT), 'prep_root': str(OUTPUT_PREP),
    'file_name': OUTPUT_MEMBER, 'exact_file_count': 1,
    'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS,
    'input_validation_passed': True, 'oof_gate_passed': None,
    'candidate_verified': True, 'passed': None,
    'standalone_consumable': False, 'external_terminal_required': True,
    'publication_success_claimed': False,
    'oof_execution_authorized': False, 'training_authorized': False,
    'qualification_mutated': False,
    'commit_semantics': 'current_exact1_visibility_after_noreplace_directory_rename',
    'crash_durability_claimed': False,
    'postcommit_parent_dir_fsync_best_effort': True,
    'postcommit_diagnostics_success_priority': True,
}
EXTERNAL_TERMINAL_EVIDENCE_CONTRACT = {
    'root': str(EXTERNAL_EVIDENCE_ROOT), 'prep_root': str(EXTERNAL_EVIDENCE_PREP),
    'process_receipt_member': EXTERNAL_PROCESS_MEMBER, 'exact_file_count': 6,
    'process_receipt_format': EXTERNAL_PROCESS_FORMAT,
    'process_receipt_status': EXTERNAL_PROCESS_STATUS,
    'passed': True, 'candidate_consumable': True,
    'standalone_candidate_consumable': False,
    'candidate_exact1_required': True,
    'candidate_nonterminal_schema_required': True,
    'candidate_and_external_terminal_dual_bind_required': True,
    'external_terminal_authoritative': True,
    'transport_helper_invocations_authorized': 1,
    'readonly_validator_invocations_authorized': 1,
    'oof_model_execution_invocations_authorized': 0,
    'phase_a_replay_invocations_authorized': 0,
    'training_invocations_authorized': 0,
    'cache_reuse_invocations_authorized': 0,
    'reward_read_invocations_authorized': 0,
    'dev_hidden_final_outcome_read_invocations_authorized': 0,
    'submission_invocations_authorized': 0,
    'commit_semantics': 'external_evidence_current_exact6_visibility_after_owned_temp_unlink',
    'crash_durability_claimed': False,
    'postcommit_diagnostics_success_priority': True,
    'retry_authorized': False,
}


def cbytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def csha(value: object) -> str:
    return hashlib.sha256(cbytes(value)).hexdigest()


def hash_fd(fd: int) -> tuple[str, int]:
    h = hashlib.sha256(); total = 0; offset = 0
    while True:
        block = os.pread(fd, 1024 * 1024, offset)
        if not block:
            break
        h.update(block); total += len(block); offset += len(block)
    return h.hexdigest(), total


def regular(path: Path, expected: tuple[str, int] | None = None) -> dict:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        stat_fd = os.fstat(fd); stat_path = os.lstat(path)
        if not os.path.isfile(path) or os.path.islink(path) or (stat_fd.st_dev, stat_fd.st_ino) != (stat_path.st_dev, stat_path.st_ino):
            raise RuntimeError(f'not held regular: {path}')
        actual = hash_fd(fd)
        if expected is not None and actual != expected:
            raise RuntimeError(f'record drift: {path}')
        if os.pread(fd, 1, actual[1]):
            raise RuntimeError(f'EOF drift: {path}')
        return {'path': str(path), 'sha256': actual[0], 'logical_bytes': actual[1]}
    finally:
        os.close(fd)


def json_exact(path: Path, expected: tuple[str, int]) -> tuple[dict, dict]:
    record = regular(path, expected)
    return json.loads(path.read_bytes()), record


def exact_tree(root: Path) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f'not exact tree: {root}')
    inventory = []
    for path in sorted(item for item in root.rglob('*') if item.is_file()):
        record = regular(path)
        inventory.append([path.relative_to(root).as_posix(), record['sha256'], record['logical_bytes']])
    lines = ''.join(f'{sha}  {name}\n' for name, sha, _ in inventory).encode()
    return {
        'root': str(root), 'file_count': len(inventory),
        'logical_file_bytes': sum(row[2] for row in inventory), 'inventory': inventory,
        'sha256sum_lines_digest_sha256': hashlib.sha256(lines).hexdigest(),
        'canonical_json_triples_digest_sha256': csha(inventory),
    }


def assert_tree(tree: dict, expected: tuple[int, int, str, str]) -> None:
    actual = (tree['file_count'], tree['logical_file_bytes'], tree['sha256sum_lines_digest_sha256'], tree['canonical_json_triples_digest_sha256'])
    if actual != expected:
        raise RuntimeError(f'tree drift: {tree["root"]}')


def records_current(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'logical_bytes'} and isinstance(value['path'], str):
            try:
                return regular(Path(value['path']), (value['sha256'], value['logical_bytes'])) == value
            except (OSError, RuntimeError):
                return False
        return all(records_current(item) for item in value.values())
    if isinstance(value, list):
        return all(records_current(item) for item in value)
    return True


def validate_authorization(value: dict, consumed: int) -> None:
    if set(value) != AUTHORIZATION_KEYS:
        raise RuntimeError('authorization keyset')
    for key in COUNT_AUTHORIZATION_KEYS:
        expected = 1 if key == 'oof_readonly_gate_validator_invocations_authorized' else consumed if key == 'oof_readonly_gate_validator_invocations_consumed' else 0
        if type(value.get(key)) is not int or value[key] != expected:
            raise RuntimeError(f'authorization count: {key}')
    for key, expected in BOOL_AUTHORIZATION.items():
        if value.get(key) is not expected:
            raise RuntimeError(f'authorization bool: {key}')


def validate_materialization_snapshot(value: dict, expected_absences: dict) -> None:
    snapshot_keys = {
        'files', 'trees', 'absences', 'services', 'gpu_compute_pids',
        'relevant_execution_pids', 'canonical_sha256',
    }
    if set(value) != snapshot_keys:
        raise RuntimeError('authority materialization snapshot keyset')
    if cbytes(value.get('absences')) != cbytes(expected_absences):
        raise RuntimeError('authority materialization snapshot absences')
    body = {key: item for key, item in value.items() if key != 'canonical_sha256'}
    if value.get('canonical_sha256') != csha(body):
        raise RuntimeError('authority materialization snapshot digest')
    if (not isinstance(value.get('files'), dict)
            or not isinstance(value.get('trees'), dict)
            or not isinstance(value.get('services'), dict)
            or type(value.get('gpu_compute_pids')) is not list
            or type(value.get('relevant_execution_pids')) is not list):
        raise RuntimeError('authority materialization snapshot schema')


def validate_prereg(value: dict) -> None:
    if set(value) != PREREG_TOP_KEYS:
        raise RuntimeError('prereg top keyset')
    if (value.get('format') != 'strict-track2-v527-v526-v525-oof-readonly-gate-preregistration-v1'
            or value.get('status') != 'preregistered_readonly_oof_gate_validation_pending_external_authority'
            or type(value.get('seed')) is not int or value['seed'] != 1662
            or value.get('classification') != 'readonly_validation_preregistration_not_oof_execution_or_training_authority'
            or value.get('authority_contract_path') != str(CONTRACT)
            or value.get('fresh_authority_root') != str(AUTHORITY_ROOT)
            or value.get('fresh_attempt_root') != str(OUTPUT_ROOT)):
        raise RuntimeError('prereg identity')
    path_literals = [value['authority_contract_path'], value['fresh_authority_root'], value['fresh_attempt_root'],
                     value.get('validator_source', {}).get('path', '')]
    path_literals.extend(row.get('path', '') for row in value.get('ancestry_sources', {}).values())
    if any(not path.startswith('/') or '\\' in path for path in path_literals):
        raise RuntimeError('prereg POSIX paths')
    if value.get('validator_source') != {'role': 'oof_readonly_gate_validator', 'path': str(SELF), 'sha256_and_bytes_binding': 'required_from_fresh_authority_source_closure'}:
        raise RuntimeError('prereg validator source')
    validate_authorization(value.get('authorization', {}), 0)
    state = value.get('current_state_contract', {})
    if set(state) != {'fresh_authority_root_initially_absent', 'fresh_authority_prep_initially_absent', 'fresh_attempt_root_initially_absent', 'fresh_attempt_prep_initially_absent', 'fresh_external_evidence_root_initially_absent', 'fresh_external_evidence_prep_initially_absent', 'qualification_tree_readonly_current_required', 'services_exact_required', 'phase_a_or_oof_execution_pids_required_empty', 'gpu_compute_pids_required_empty', 'historical_training_authority_not_inherited'} or not all(item is True for item in state.values()):
        raise RuntimeError('prereg current state')
    ancestry = value.get('ancestry_sources', {})
    if set(ancestry) != ANCESTRY_KEYS:
        raise RuntimeError('v482 ancestry roles')
    for role, (path, sha, size) in V482_SOURCES.items():
        if ancestry.get(role) != {'path': str(path), 'sha256': sha, 'logical_bytes': size}:
            raise RuntimeError(f'v482 ancestry: {role}')
    view = value.get('normalized_qualification_view', {})
    if (set(view) != NORMALIZED_VIEW_KEYS
            or view.get('format') != 'strict-track2-v527-v526-v525-normalized-qualification-view-v1'
            or view.get('original_worker_receipts_canonical_sha256') != ORIGINAL_CSHA
            or view.get('normalized_worker_receipts_canonical_sha256') != NORMALIZED_CSHA
            or view.get('recursive_diff_canonical_sha256') != DIFF_CSHA
            or csha(view.get('original_worker_receipts')) != ORIGINAL_CSHA
            or csha(view.get('normalized_worker_receipts')) != NORMALIZED_CSHA
            or view.get('recursive_diff') != EXPECTED_DIFF or view.get('recursive_diff_count') != 2
            or view.get('recursive_diff_full_paths') != [['process_a', 'format'], ['process_b', 'format']]
            or csha(view.get('recursive_diff')) != DIFF_CSHA
            or view.get('new_computation_claimed') is not False
            or view.get('historical_training_authority_not_inherited') is not True):
        raise RuntimeError('prereg normalized view')
    candidate = view.get('candidate_exact1', {})
    external = view.get('external_terminal_exact6', {})
    if (candidate != {
            'receipt': {'path': str(CANDIDATE_ROOT / 'reconciliation_receipt.json'), 'sha256': CANDIDATE_RECEIPT[0], 'logical_bytes': CANDIDATE_RECEIPT[1]},
            'tree': {'file_count': 1, 'logical_file_bytes': 43485, 'sha256sum_lines_digest_sha256': CANDIDATE_TREE[2], 'canonical_json_triples_digest_sha256': CANDIDATE_TREE[3]},
            'candidate_verified': True, 'passed': None, 'standalone_consumable': False,
            'external_terminal_required': True, 'publication_success_claimed': False,
        } or external != {
            'process_receipt': {'path': str(EXTERNAL_ROOT / 'process_receipt.json'), 'sha256': EXTERNAL_PROCESS[0], 'logical_bytes': EXTERNAL_PROCESS[1]},
            'tree': {'file_count': 6, 'logical_file_bytes': 127746, 'sha256sum_lines_digest_sha256': EXTERNAL_TREE[2], 'canonical_json_triples_digest_sha256': EXTERNAL_TREE[3]},
            'passed': True, 'status': 'passed_external_terminal', 'candidate_consumable': True,
            'standalone_candidate_consumable': False,
        }):
        raise RuntimeError('prereg candidate/external dual bind')
    inputs = value.get('input_contract', {})
    if (set(inputs) != INPUT_CONTRACT_KEYS
            or inputs.get('stage_b_candidate_and_external_terminal') != view
            or inputs.get('qualification_tree', {}).get('file_count') != QUALIFICATION_TREE[0]
            or inputs.get('qualification_tree', {}).get('logical_file_bytes') != QUALIFICATION_TREE[1]
            or inputs.get('qualification_tree', {}).get('sha256sum_lines_digest_sha256') != QUALIFICATION_TREE[2]
            or inputs.get('qualification_tree', {}).get('canonical_json_triples_digest_sha256') != QUALIFICATION_TREE[3]):
        raise RuntimeError('prereg input contract')
    output = value.get('output_contract', {})
    if (set(output) != OUTPUT_CONTRACT_KEYS
            or cbytes(output.get('candidate')) != cbytes(CANDIDATE_OUTPUT_CONTRACT)
            or cbytes(output.get('external_terminal')) != cbytes(EXTERNAL_TERMINAL_EVIDENCE_CONTRACT)):
        raise RuntimeError('prereg output contract')


def load_module(path: Path, expected: tuple[str, int], name: str):
    regular(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_authority(args, prereg: dict, self_record: dict) -> tuple[object, dict, dict, dict, dict]:
    contract = json.loads(args.contract.read_bytes()); authority = json.loads(args.authority_receipt.read_bytes())
    contract_record = regular(args.contract)
    authority_record = regular(args.authority_receipt)
    materializer_record = contract.get('authority_materializer_source', {})
    if materializer_record.get('path') != str(args.authority_materializer):
        raise RuntimeError('materializer path')
    module = load_module(args.authority_materializer, (materializer_record.get('sha256'), materializer_record.get('logical_bytes')), 'v527_authority_materializer')
    if (set(contract) != module.CONTRACT_TOP_KEYS or set(authority) != module.AUTH_TOP_KEYS
            or contract.get('format') != module.CONTRACT_FORMAT or contract.get('status') != module.CONTRACT_STATUS
            or authority.get('format') != module.OUTPUT_FORMAT or authority.get('status') != module.OUTPUT_STATUS
            or authority.get('passed') is not True):
        raise RuntimeError('authority schema')
    if (cbytes(contract.get('authorization')) != cbytes(module.AUTHORIZATION)
            or cbytes(authority.get('authorization')) != cbytes(module.AUTHORIZATION)
            or cbytes(contract.get('runtime_observation')) != cbytes(module.RUNTIME)
            or cbytes(authority.get('runtime_observation')) != cbytes(module.RUNTIME)
            or cbytes(contract.get('execution_boundary')) != cbytes(module.EXECUTION_BOUNDARY)
            or cbytes(authority.get('execution_boundary')) != cbytes(module.EXECUTION_BOUNDARY)):
        raise RuntimeError('authority boundary')
    validate_authorization(contract['authorization'], 0)
    validate_authorization(authority['authorization'], 0)
    checks = authority.get('checks', {})
    if set(checks) != set(module.CHECK_KEYS) or not all(value is True for value in checks.values()):
        raise RuntimeError('authority checks')
    if (authority.get('check_keys') != sorted(module.CHECK_KEYS)
            or authority.get('check_key_set_sha256') != csha(sorted(module.CHECK_KEYS))
            or authority.get('checks_sha256') != csha(checks)):
        raise RuntimeError('authority check digests')
    if authority.get('source_role_order') != module.AUTHORITY_SOURCE_ORDER or contract.get('source_role_order') != module.CONTRACT_SOURCE_ORDER:
        raise RuntimeError('authority source order')
    if cbytes(contract.get('source_aliases')) != cbytes(module.SOURCE_ALIASES) or cbytes(authority.get('source_aliases')) != cbytes(module.SOURCE_ALIASES):
        raise RuntimeError('authority aliases')
    if not records_current(contract) or not records_current(authority):
        raise RuntimeError('authority records current')
    contract_closure = contract.get('source_closure', {})
    authority_closure = authority.get('source_closure', {})
    if (set(contract_closure) != set(module.CONTRACT_SOURCE_ORDER)
            or set(authority_closure) != set(module.AUTHORITY_SOURCE_ORDER)
            or contract.get('source_closure_sha256') != csha(contract_closure)
            or authority.get('source_closure_sha256') != csha(authority_closure)):
        raise RuntimeError('authority source closure schema/digest')
    expected_authority_closure = {'authority_design_contract': contract_record, **contract_closure}
    if cbytes(authority_closure) != cbytes(expected_authority_closure):
        raise RuntimeError('authority design contract held path/record')
    for role, alias in module.SOURCE_ALIASES.items():
        if cbytes(authority.get(alias)) != cbytes(authority_closure[role]):
            raise RuntimeError(f'authority source alias value: {role}')
    for role, key in {
        'authority_materializer': 'authority_materializer_source',
        'oof_readonly_gate_validator': 'oof_readonly_gate_validator_source',
        'oof_readonly_gate_preregistration': 'oof_readonly_gate_preregistration_source',
    }.items():
        if cbytes(contract.get(key)) != cbytes(contract_closure[role]):
            raise RuntimeError(f'contract active source alias: {role}')
    prereg_record = regular(PREREG, (PREREG_SHA, PREREG_BYTES))
    if cbytes(self_record) != cbytes(contract_closure.get('oof_readonly_gate_validator')) or cbytes(prereg_record) != cbytes(contract_closure.get('oof_readonly_gate_preregistration')):
        raise RuntimeError('active source closure')
    if (Path(module.V526_HELPER_TEMP) != V526_HELPER_TEMP
            or Path(module.V526_SCRIPT_TEMP) != V526_SCRIPT_TEMP
            or Path(module.EXTERNAL_EVIDENCE_ROOT) != EXTERNAL_EVIDENCE_ROOT
            or Path(module.EXTERNAL_EVIDENCE_PREP) != EXTERNAL_EVIDENCE_PREP):
        raise RuntimeError('materializer absence source paths')
    expected_deployment = {
        'failure_fact': contract_closure['v526_stage_b_deploy_failure_fact'],
        'deployer': contract_closure['v526_stage_b_deployer'],
        'transport_helper': contract_closure['v525_stage_b_transport_helper'],
        'transport_script': contract_closure['v525_stage_b_transport_script'],
        'pair_current': True,
        'owned_deploy_temps': {
            'helper': {'path': str(V526_HELPER_TEMP), 'absent': True},
            'script': {'path': str(V526_SCRIPT_TEMP), 'absent': True},
        },
        'historical_self_match_not_retried': True,
    }
    if (cbytes(contract.get('v526_stage_b_transport_deployment')) != cbytes(expected_deployment)
            or cbytes(authority.get('v526_stage_b_transport_deployment')) != cbytes(expected_deployment)):
        raise RuntimeError('v526 stage-b deployment anchor')
    expected_absences = {
        'authority_root': {'path': str(AUTHORITY_ROOT), 'absent': True},
        'authority_prep': {'path': str(AUTHORITY_PREP), 'absent': True},
        'oof_validation_attempt_root': {'path': str(OUTPUT_ROOT), 'absent': True},
        'oof_validation_attempt_prep': {'path': str(OUTPUT_PREP), 'absent': True},
        'external_evidence_root': {'path': str(EXTERNAL_EVIDENCE_ROOT), 'absent': True},
        'external_evidence_prep': {'path': str(EXTERNAL_EVIDENCE_PREP), 'absent': True},
        'v526_helper_deploy_temp': {'path': str(V526_HELPER_TEMP), 'absent': True},
        'v526_script_deploy_temp': {'path': str(V526_SCRIPT_TEMP), 'absent': True},
    }
    materialization_pre = authority.get('input_pre_snapshot', {})
    materialization_post = authority.get('input_post_snapshot', {})
    validate_materialization_snapshot(materialization_pre, expected_absences)
    validate_materialization_snapshot(materialization_post, expected_absences)
    if (cbytes(materialization_pre) != cbytes(materialization_post)
            or authority.get('input_snapshots_exactly_equal') is not True):
        raise RuntimeError('authority materialization snapshots differ')
    historical_paths, current_paths = module.absences()
    contract_historical = {key: {'path': str(path)} for key, path in historical_paths.items()}
    contract_current = {key: {'path': str(path)} for key, path in current_paths.items()}
    authority_historical = {key: {'path': str(path), 'absent': True} for key, path in historical_paths.items()}
    authority_required = {key: {'path': str(path), 'absent': True} for key, path in current_paths.items()}
    if (cbytes(contract.get('historical_absences')) != cbytes(contract_historical)
            or cbytes(contract.get('current_absences_after_authority')) != cbytes(contract_current)
            or cbytes(authority.get('historical_absences')) != cbytes(authority_historical)
            or cbytes(authority.get('required_absences')) != cbytes(authority_required)):
        raise RuntimeError('authority absence contracts')
    exact = {
        'v525_normalized_qualification_view': prereg['normalized_qualification_view'],
        'oof_gate_input_contract': prereg['input_contract'],
        'oof_gate_output_contract': prereg['output_contract'],
        'oof_gate_candidate_contract': CANDIDATE_OUTPUT_CONTRACT,
        'oof_gate_external_terminal_evidence_contract': EXTERNAL_TERMINAL_EVIDENCE_CONTRACT,
    }
    for key, expected in exact.items():
        if cbytes(contract.get(key)) != cbytes(expected) or cbytes(authority.get(key)) != cbytes(expected):
            raise RuntimeError(f'authority embedded: {key}')
    return module, contract_record, authority_record, contract, authority


def service_health() -> dict:
    output = {}
    for key, url in [('8005', 'http://127.0.0.1:8005/v1/health'), ('18084', 'http://127.0.0.1:18084/health')]:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read()
        output[key] = {'http_code': response.status, 'body_sha256': hashlib.sha256(body).hexdigest(), 'body_bytes': len(body), 'json_model': json.loads(body)}
    expected = {
        '18084': {'http_code': 200, 'body_sha256': '31bf75f4c0a97cc1f7b60df824fa390b3be9ba014f29b63c87c698ba63d9a9fd', 'body_bytes': 18, 'json_model': {'status': 'ready'}},
        '8005': {'http_code': 200, 'body_sha256': '08dbc59225418d4d3064f9e122caeadbdf7c740d38fcbbaf708ad9baa44abab0', 'body_bytes': 106, 'json_model': {'api_version': '1.0', 'model_version': 'track2-v218-public-knn-blend-alpha070-route-aware', 'status': 'ready'}},
    }
    if output != expected:
        raise RuntimeError('service health')
    return output


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    return sorted(int(line) for line in result.stdout.splitlines() if line.strip())


def relevant_execution_pids() -> list[dict]:
    entrypoints = {
        str(S / 'launch_v524_v523_phase_a_cache_qualification.py'),
        str(S / 'generate_v524_v523_phase_a_cache_qualification.py'),
        str(S / 'generate_v524_v523_phase_a_cache_qualification_worker.py'),
        str(RECONCILER), str(S / 'train_v482_temporal8_residual_5fold.py'),
        str(S / 'audit_v482_temporal8_residual_s0.py'),
    }
    rows = []
    for child in Path('/proc').iterdir():
        if not child.name.isdigit() or int(child.name) == os.getpid():
            continue
        try:
            argv = [part.decode(errors='replace') for part in (child / 'cmdline').read_bytes().split(b'\0') if part]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matches = sorted(set(argv) & entrypoints)
        if matches:
            rows.append({'pid': int(child.name), 'argv': argv, 'exact_entrypoint_matches': matches})
    return sorted(rows, key=lambda row: row['pid'])


def validate_current(prereg: dict) -> dict:
    for role, row in prereg['ancestry_sources'].items():
        if set(row) != {'path', 'sha256', 'logical_bytes'}:
            raise RuntimeError(f'ancestry record schema: {role}')
        regular(Path(row['path']), (row['sha256'], row['logical_bytes']))
    v482, _ = json_exact(V482_PREREG, (V482_SOURCES['v482_s0_execution_preregistration_ancestry'][1], V482_SOURCES['v482_s0_execution_preregistration_ancestry'][2]))
    if (v482.get('format') != 'strict-track2-v482-temporal8-residual-preregistration-v1'
            or v482.get('status') != 'preregistered_public_train_temporal_s0_authorized'
            or v482.get('guards', {}).get('training_authorized') is not True):
        raise RuntimeError('v482 historical ancestry')
    stage_a_tree = exact_tree(STAGE_A_AUTH); assert_tree(stage_a_tree, STAGE_A_TREE)
    stage_a_evidence_tree = exact_tree(STAGE_A_EVIDENCE); assert_tree(stage_a_evidence_tree, STAGE_A_EVIDENCE_TREE)
    stage_a, stage_a_record = json_exact(STAGE_A_AUTH / 'authority_receipt.json', STAGE_A_RECEIPT)
    stage_a_process, stage_a_process_record = json_exact(STAGE_A_EVIDENCE / 'process_receipt.json', STAGE_A_PROCESS)
    if (stage_a.get('passed') is not True
            or stage_a.get('status') != 'authorized_exact_one_external_v525_readonly_reconciler_attempt'
            or stage_a_process.get('passed') is not True
            or stage_a_process.get('status') != 'passed_exact_once_authority_materialized_no_phase_a_execution'
            or stage_a_process.get('helper_returncode') != 0
            or stage_a_process.get('materializer_returncode') != 0):
        raise RuntimeError('stage A')
    candidate_tree = exact_tree(CANDIDATE_ROOT); assert_tree(candidate_tree, CANDIDATE_TREE)
    external_tree = exact_tree(EXTERNAL_ROOT); assert_tree(external_tree, EXTERNAL_TREE)
    candidate, candidate_record = json_exact(CANDIDATE_ROOT / 'reconciliation_receipt.json', CANDIDATE_RECEIPT)
    external, external_record = json_exact(EXTERNAL_ROOT / 'process_receipt.json', EXTERNAL_PROCESS)
    if (candidate.get('candidate_verified') is not True or candidate.get('passed') is not None
            or candidate.get('standalone_consumable') is not False or candidate.get('external_terminal_required') is not True
            or candidate.get('publication_success_claimed') is not False
            or external.get('passed') is not True or external.get('status') != 'passed_external_terminal'
            or external.get('candidate_consumable') is not True or external.get('standalone_candidate_consumable') is not False):
        raise RuntimeError('stage B dual bind')
    reconciler = load_module(RECONCILER, RECONCILER_RECORD, 'v525_readonly_reconciler')
    current = reconciler.validate_current_inputs()
    view = prereg['normalized_qualification_view']
    if (current['original_worker_receipts'] != view['original_worker_receipts']
            or current['normalized_worker_receipts'] != view['normalized_worker_receipts']
            or current['recursive_diff'] != EXPECTED_DIFF
            or csha(current['original_worker_receipts']) != ORIGINAL_CSHA
            or csha(current['normalized_worker_receipts']) != NORMALIZED_CSHA
            or csha(current['recursive_diff']) != DIFF_CSHA
            or candidate.get('normalized_worker_receipts') != current['normalized_worker_receipts']
            or external.get('per_call_validation') != current['per_call_validation']):
        raise RuntimeError('normalized current view')
    assert_tree(current['qualification_tree'], QUALIFICATION_TREE)
    services = service_health(); gpu = gpu_compute_pids(); pids = relevant_execution_pids()
    if gpu or pids:
        raise RuntimeError('live execution')
    return {
        'v482_historical_training_authority_not_inherited': True,
        'stage_a_authority_tree': stage_a_tree, 'stage_a_authority_receipt': stage_a_record,
        'stage_a_evidence_tree': stage_a_evidence_tree, 'stage_a_process_receipt': stage_a_process_record,
        'stage_b_candidate_tree': candidate_tree, 'stage_b_candidate_receipt': candidate_record,
        'stage_b_external_evidence_tree': external_tree, 'stage_b_external_process_receipt': external_record,
        'qualification_tree': current['qualification_tree'], 'qualification_terminal': current['qualification_terminal'],
        'qualification_report': current['qualification_report'], 'qualification_independent_audit': current['independent_final_audit'],
        'original_worker_receipts': current['original_worker_receipts'], 'normalized_worker_receipts': current['normalized_worker_receipts'],
        'recursive_diff': current['recursive_diff'], 'per_call_validation': current['per_call_validation'],
        'services': services, 'gpu_compute_pids': gpu, 'relevant_execution_pids': pids,
    }


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def read_fd_exact(fd: int, expected_bytes: int, pread=os.pread) -> bytes:
    output = bytearray(); offset = 0
    while offset < expected_bytes:
        block = pread(fd, min(1024 * 1024, expected_bytes - offset), offset)
        if not block:
            raise RuntimeError('candidate held member short read')
        output.extend(block); offset += len(block)
    if pread(fd, 1, expected_bytes):
        raise RuntimeError('candidate held member EOF drift')
    return bytes(output)


def assert_owned_candidate(path: Path, fd: int, identity: tuple[int, int],
                           expected_sha256: str, expected_bytes: int, pread=os.pread) -> dict:
    fd_stat = os.fstat(fd); path_stat = os.lstat(path)
    if (path.is_symlink() or not os.path.isfile(path)
            or (fd_stat.st_dev, fd_stat.st_ino) != identity
            or (path_stat.st_dev, path_stat.st_ino) != identity):
        raise RuntimeError('candidate held member identity')
    data = read_fd_exact(fd, expected_bytes, pread)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RuntimeError('candidate held member digest')
    return {'path': str(path), 'sha256': expected_sha256, 'logical_bytes': expected_bytes}


def cleanup_owned_candidate(prep: Path, prep_identity: tuple[int, int] | None,
                            member: Path, fd: int | None,
                            member_identity: tuple[int, int] | None,
                            directory_fsync=fsync_dir) -> None:
    if prep_identity is None:
        return
    try:
        prep_stat = os.lstat(prep)
    except FileNotFoundError:
        return
    if (prep_stat.st_dev, prep_stat.st_ino) != prep_identity:
        return
    entries = list(prep.iterdir())
    if entries:
        if (len(entries) != 1 or entries[0].name != OUTPUT_MEMBER
                or fd is None or member_identity is None):
            return
        try:
            fd_stat = os.fstat(fd); path_stat = os.lstat(member)
            if (member.is_symlink()
                    or (fd_stat.st_dev, fd_stat.st_ino) != member_identity
                    or (path_stat.st_dev, path_stat.st_ino) != member_identity):
                return
            read_fd_exact(fd, fd_stat.st_size)
        except (OSError, RuntimeError):
            return
        member.unlink()
    prep.rmdir(); directory_fsync(prep.parent)


def publish(receipt: dict, root: Path = OUTPUT_ROOT, prep: Path = OUTPUT_PREP,
            writer=os.write, pread=os.pread, phase_hook=None,
            rename=rename_noreplace, directory_fsync=fsync_dir,
            tree_reader=exact_tree) -> tuple[dict, dict | None]:
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    hook = phase_hook or (lambda _name: None)
    prep_identity = None; member_identity = None; fd = None; committed = False
    member = prep / OUTPUT_MEMBER; payload = b''; expected_sha256 = ''
    try:
        if os.path.lexists(root) or os.path.lexists(prep):
            raise FileExistsError(root)
        os.mkdir(prep, 0o700)
        prep_stat = os.lstat(prep); prep_identity = (prep_stat.st_dev, prep_stat.st_ino)
        payload = json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False).encode() + b'\n'
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        fd = os.open(member, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        fd_stat = os.fstat(fd); member_identity = (fd_stat.st_dev, fd_stat.st_ino)
        offset = 0
        while offset < len(payload):
            written = writer(fd, memoryview(payload)[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise OSError('candidate held member short write')
            offset += written
        os.fsync(fd)
        assert_owned_candidate(member, fd, member_identity, expected_sha256, len(payload), pread)
        hook('after_member_write')
        directory_fsync(prep)
        assert_owned_candidate(member, fd, member_identity, expected_sha256, len(payload), pread)
        hook('before_rename')
        assert_owned_candidate(member, fd, member_identity, expected_sha256, len(payload), pread)
        if {signal.SIGINT, signal.SIGTERM} & signal.sigpending():
            raise InterruptedError('pending termination signal before candidate commit')
        rename(prep, root); committed = True
        # Choice-A: the NOREPLACE directory rename is the visibility commit point.
        # Every later check is diagnostic-only and cannot roll back, relink, retry,
        # claim standalone consumption, or perform a second publication.
        diagnostics = []; tree = None; final_member = root / OUTPUT_MEMBER
        for name, operation in (
            ('phase_hook', lambda: hook('after_rename_before_postcommit')),
            ('parent_fsync', lambda: directory_fsync(root.parent)),
            ('held_readback', lambda: assert_owned_candidate(
                final_member, fd, member_identity, expected_sha256, len(payload), pread)),
            ('current_tree', lambda: tree_reader(root)),
        ):
            try:
                value = operation()
                if name == 'current_tree':
                    tree = value
            except BaseException as error:
                diagnostics.append({'operation': name, 'error_type': type(error).__name__, 'error': str(error)})
        expected_inventory = [[OUTPUT_MEMBER, expected_sha256, len(payload)]]
        current_exact = tree is not None and tree.get('file_count') == 1 and tree.get('inventory') == expected_inventory
        result = {
            'file_count': 1 if current_exact else None,
            'inventory': tree.get('inventory') if tree else None,
            'visibility_committed': True,
            'candidate_current_exact1': current_exact,
            'candidate_verified': current_exact,
            'passed': None,
            'standalone_consumable': False,
            'external_terminal_required': True,
            'publication_success_claimed': False,
            'postcommit_diagnostics_passed': not diagnostics and current_exact,
            'retry_authorized': False,
            'rollback_or_relink_performed': False,
            'second_publish_performed': False,
        }
        diagnostic = None if not diagnostics else {
            'postcommit_success_priority': True, 'diagnostics': diagnostics,
            'standalone_consumable': False, 'external_terminal_required': True,
        }
        return result, diagnostic
    finally:
        if not committed:
            cleanup_owned_candidate(prep, prep_identity, member, fd, member_identity, directory_fsync)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def synthetic_self_test() -> dict:
    prereg, _ = json_exact(PREREG, (PREREG_SHA, PREREG_BYTES)); validate_prereg(prereg)
    checks = {}
    source = Path(__file__).read_text(); tree = ast.parse(source)
    call_names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    checks['no_oof_train_predict_entry_calls'] = not ({'train', 'fit', 'predict', 'predict_one'} & call_names)
    checks['authorization_exact24'] = len(AUTHORIZATION_KEYS) == 24 and len(COUNT_AUTHORIZATION_KEYS) == 13 and len(BOOL_AUTHORIZATION) == 11
    for name, mutate in {
        'missing_top': lambda value: value.pop('classification'),
        'extra_top': lambda value: value.__setitem__('legacy', True),
        'bool_int_one': lambda value: value['authorization'].__setitem__('oof_readonly_gate_validator_invocations_authorized', True),
        'bool_int_zero': lambda value: value['authorization'].__setitem__('training_invocations_authorized', False),
        'legacy_training_inherited': lambda value: value['current_state_contract'].__setitem__('historical_training_authority_not_inherited', False),
        'diff_path': lambda value: value['normalized_qualification_view']['recursive_diff'][0].__setitem__('path', ['process_a', 'role']),
        'normalized_digest': lambda value: value['normalized_qualification_view'].__setitem__('normalized_worker_receipts_canonical_sha256', '0' * 64),
        'output_claim': lambda value: value['output_contract'].__setitem__('oof_gate_passed', True),
        'windows_path': lambda value: value.__setitem__('fresh_attempt_root', value['fresh_attempt_root'].replace('/', '\\')),
    }.items():
        tampered = copy.deepcopy(prereg); mutate(tampered)
        try:
            validate_prereg(tampered)
        except RuntimeError:
            checks['reject_' + name] = True
        else:
            checks['reject_' + name] = False
    with tempfile.TemporaryDirectory(prefix='v527-oof-validator-', dir='/dev/shm') as folder:
        base = Path(folder); root = base / 'output'; prep = base / 'output.prep'
        fixture = {'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS, 'input_validation_passed': True, 'oof_gate_passed': None}
        result, diagnostic = publish(fixture, root, prep)
        checks['noreplace_exact1'] = result['file_count'] == 1 and diagnostic is None and not prep.exists()
        try:
            publish(fixture, root, prep)
        except FileExistsError:
            checks['no_second_publish'] = True
        else:
            checks['no_second_publish'] = False
    if not all(checks.values()):
        raise RuntimeError(checks)
    return {'passed': True, 'checks': checks, 'checks_sha256': csha(checks),
            'validator_invocations': 0, 'oof_model_execution_invocations': 0,
            'phase_a_replay_invocations': 0, 'training_invocations': 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-self-test', action='store_true')
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--authority-receipt', type=Path)
    parser.add_argument('--authority-materializer', type=Path)
    parser.add_argument('--preregistration', type=Path)
    parser.add_argument('--validator-source', type=Path)
    parser.add_argument('--output-root', type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic_self_test:
        print(json.dumps(synthetic_self_test(), sort_keys=True)); return 0
    expected = {
        'contract': CONTRACT, 'authority_receipt': AUTHORITY_RECEIPT,
        'preregistration': PREREG, 'validator_source': SELF, 'output_root': OUTPUT_ROOT,
    }
    if any(getattr(args, key) != value for key, value in expected.items()) or args.authority_materializer is None:
        raise RuntimeError('exact CLI')
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    if (os.path.lexists(OUTPUT_ROOT) or os.path.lexists(OUTPUT_PREP)
            or os.path.lexists(EXTERNAL_EVIDENCE_ROOT) or os.path.lexists(EXTERNAL_EVIDENCE_PREP)
            or os.path.lexists(AUTHORITY_PREP)):
        raise RuntimeError('fresh roots')
    prereg, prereg_record = json_exact(PREREG, (PREREG_SHA, PREREG_BYTES)); validate_prereg(prereg)
    self_record = regular(SELF)
    module, contract_record, authority_record, contract, authority = validate_authority(args, prereg, self_record)
    before = validate_current(prereg)
    middle = validate_current(prereg)
    if csha(before) != csha(middle):
        raise RuntimeError('double input snapshot')
    consumed = copy.deepcopy(module.AUTHORIZATION)
    consumed['oof_readonly_gate_validator_invocations_consumed'] = 1
    validate_authorization(consumed, 1)
    receipt = {
        'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS,
        'passed': None, 'candidate_verified': True,
        'input_validation_passed': True, 'oof_gate_passed': None,
        'standalone_consumable': False, 'external_terminal_required': True,
        'publication_success_claimed': False,
        'standalone_oof_execution_authority': False,
        'candidate_publication_contract': CANDIDATE_OUTPUT_CONTRACT,
        'external_terminal_evidence_contract': EXTERNAL_TERMINAL_EVIDENCE_CONTRACT,
        'authority_contract': contract_record, 'authority_receipt': authority_record,
        'preregistration': prereg_record, 'validator_source': self_record,
        'authorization': consumed,
        'runtime_partition': {
            'oof_readonly_gate_validator_invocations': 1,
            'oof_model_execution_invocations': 0,
            'phase_a_launcher_invocations': 0, 'phase_a_driver_invocations': 0,
            'phase_a_worker_invocations': 0, 'rng_proxy_delegate_invocations': 0,
            'phase_a_replay_invocations': 0, 'training_invocations': 0,
            'cache_reuse_invocations': 0, 'reward_read_invocations': 0,
            'dev_hidden_final_outcome_read_invocations': 0, 'submission_invocations': 0,
        },
        'historical_v482_training_authority_not_inherited': True,
        'normalized_qualification_view': prereg['normalized_qualification_view'],
        'input_pre_snapshot': before, 'input_post_snapshot': middle,
        'input_snapshots_exactly_equal': True,
        'qualification_mutated': False,
        'retry_authorized': False,
        'commit_semantics': 'current_exact1_visibility_after_noreplace_directory_rename',
        'crash_durability_claimed': False,
        'postcommit_parent_dir_fsync_best_effort': True,
        'postcommit_diagnostics_success_priority': True,
    }
    # Third full gate occurs after receipt construction and immediately before publication.
    third = validate_current(prereg)
    if csha(middle) != csha(third):
        raise RuntimeError('third prepublish input gate')
    tree, diagnostic = publish(receipt)
    print(json.dumps({'passed': None, 'candidate_verified': tree['candidate_verified'],
                      'standalone_consumable': False, 'external_terminal_required': True,
                      'publication_success_claimed': False,
                      'input_validation_passed': True, 'oof_gate_passed': None,
                      'output_tree': tree, 'postcommit_diagnostic': diagnostic,
                      'oof_model_execution_invocations': 0, 'training_invocations': 0,
                      'phase_a_replay_invocations': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
