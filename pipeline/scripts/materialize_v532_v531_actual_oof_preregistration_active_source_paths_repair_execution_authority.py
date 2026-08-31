#!/usr/bin/env python3
"""Materialize exactly one fresh v532 authority after the v531 registry failure."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import signal
import stat
from pathlib import Path

ROOT = Path('/root/autodl-tmp/IROS_WAM_2.0 challenge')
S = ROOT / 'pipeline/scripts'
J = ROOT / 'artifacts/strict_track2_joint_augmentation_20260810'
SELF = S / 'materialize_v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority.py'
CONTRACT = S / 'v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_contract.json'
PREREG = S / 'v532_v531_actual_oof_execution_preregistration.json'
MANIFEST = S / 'v532_v531_actual_oof_execution_manifest.json'
EXECUTOR = S / 'execute_v532_v531_actual_oof.py'
AUDITOR = S / 'audit_v532_v531_actual_oof.py'
LAUNCHER = S / 'launch_v532_v531_actual_oof.py'
FORENSIC = S / 'v532_v531_actual_oof_preregistration_active_source_paths_failure_forensic.json'

PREREG_EXPECTED = ('7ef48984ac29d86edd22f75d4fab2d82e12237c1bd72615172da59dd8fdf70a5', 14210)
MANIFEST_EXPECTED = ('3a3e83eb03415c9c0f573a10112b4cb18a0593bda10fe6eaff1feed843e7996e', 1296036)
EXECUTOR_EXPECTED = ('7f5d154a6bd2d4ae94d372091d36fc6bf45d770889e692b6d9e499745f9e9932', 62773)
AUDITOR_EXPECTED = ('b8f98db595fc0ddbff1cab1e9fce20a0c4bd7d6fbe8372ae57d4a4c092fc2893', 11763)
LAUNCHER_EXPECTED = ('ca1ecc8bec51776935b10df70121ed4c0c12ad94c3689604bec63d214a3dea4d', 15315)
FORENSIC_EXPECTED = ('9f8edc098b04f548fdc0638f5796dd1f4b7d5635b2b3b22a6baadab756563506', 10911)

AUTH_ROOT = J / 'v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_seed1666_20260827'
AUTH_PREP = AUTH_ROOT.with_name(AUTH_ROOT.name + '.authority-prep')
ATTEMPT_ROOT = J / 'v532_v531_actual_oof_attempt_seed1666_20260827'
ATTEMPT_PREP = ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name + '.attempt-prep')
OOF_ROOT = Path('/root/v532_v531_actual_oof_seed1666_20260827')
OOF_PREP = OOF_ROOT.with_name(OOF_ROOT.name + '.oof-prep')
CANDIDATE_ROOT = J / 'v527_v526_v525_oof_readonly_validation_attempt_seed1662_20260827'
V530_ROOT = J / 'v530_v529_stage_b_candidate_external_terminal_evidence_seed1664_20260827'
QUAL_ROOT = Path('/root/v524_v523_phase_a_cache_qualification_seed1660_20260826')
DATASET_ROOT = Path('/root/v478_temporal8_dataset_seed1622_20260824')

CONTRACT_FORMAT = 'strict-track2-v532-v531-actual-oof-execution-authority-design-contract-v1'
CONTRACT_STATUS = 'design_only_frozen_actual_oof_sources_pending_independent_authority_materialization'
OUTPUT_FORMAT = 'strict-track2-v532-v531-actual-oof-execution-authority-v1'
OUTPUT_STATUS = 'authorized_exact_one_actual_oof_execution_boundary_pending_external_execution'

AUTHORIZATION = {
    'actual_oof_execution_boundary_invocations_authorized': 1,
    'actual_oof_execution_boundary_invocations_consumed': 0,
    'direct_executor_invocations_authorized': 0,
    'direct_auditor_invocations_authorized': 0,
    'oof_readonly_validator_invocations_authorized': 0,
    'model_runtime_delegate_invocations_authorized': 0,
    'phase_a_launcher_invocations_authorized': 0,
    'phase_a_driver_invocations_authorized': 0,
    'phase_a_worker_invocations_authorized': 0,
    'phase_a_replay_invocations_authorized': 0,
    'training_invocations_authorized': 0,
    'cache_reuse_invocations_authorized': 0,
    'reward_read_invocations_authorized': 0,
    'dev_hidden_final_outcome_read_invocations_authorized': 0,
    'submission_invocations_authorized': 0,
    'retry_authorized': False,
    'actual_oof_execution_authorized': True,
    'qualification_readonly': True,
    'qualification_mutation_authorized': False,
    'hidden_or_final_inputs_authorized': False,
    'reward_read_authorized': False,
    'training_authorized': False,
    'cache_reuse_authorized': False,
    'submission_authorized': False,
}
RUNTIME = {
    'authority_materializer_invocations': 1,
    'actual_oof_execution_boundary_invocations': 0,
    'launcher_invocations': 0,
    'executor_invocations': 0,
    'auditor_import_invocations': 0,
    'direct_executor_invocations': 0,
    'direct_auditor_invocations': 0,
    'oof_readonly_validator_invocations': 0,
    'model_runtime_delegate_invocations': 0,
    'phase_a_invocations': 0,
    'training_invocations': 0,
    'cache_reuse_invocations': 0,
    'reward_read_invocations': 0,
    'dev_hidden_final_outcome_read_invocations': 0,
    'submission_invocations': 0,
    'qualification_mutated': False,
}
EXECUTION_BOUNDARY = {
    'authority_materialization_only': True,
    'actual_oof_execution_boundary_invocations': 0,
    'actual_oof_launcher_executor_auditor_invocations': 0,
    'direct_execution_authorized': False,
    'qualification_readonly': True,
    'qualification_mutation_authorized': False,
    'dataset_mutation_authorized': False,
    'hidden_reward_training_cache_submission_authorized': False,
    'authority_publication_noreplace': True,
    'authority_publication_commit_semantics': 'current_exact1_visibility_after_noreplace_directory_rename',
    'crash_durability_claimed': False,
    'postcommit_diagnostics_best_effort_success_priority': True,
    'rollback_relink_retry_or_second_publish_after_commit': False,
}

ACTIVE_SOURCE_ROLE_ORDER = [
    'authority_design_contract', 'authority_materializer', 'actual_oof_execution_preregistration',
    'actual_oof_execution_manifest', 'actual_oof_executor', 'actual_oof_auditor', 'actual_oof_launcher',
]
ACTIVE_SOURCE_PATHS = {
    'authority_design_contract': str(CONTRACT), 'authority_materializer': str(SELF),
    'actual_oof_execution_preregistration': str(PREREG), 'actual_oof_execution_manifest': str(MANIFEST),
    'actual_oof_executor': str(EXECUTOR), 'actual_oof_auditor': str(AUDITOR),
    'actual_oof_launcher': str(LAUNCHER),
}
SOURCE_ROLES = [
    'authority_materializer', 'actual_oof_execution_preregistration', 'actual_oof_execution_manifest',
    'actual_oof_executor', 'actual_oof_auditor', 'actual_oof_launcher', 'v531_failure_forensic',
    'v531_invalid_authority_design_contract', 'v531_invalid_authority_materializer',
    'v531_invalid_actual_oof_execution_preregistration', 'v531_invalid_actual_oof_execution_manifest',
    'v531_invalid_actual_oof_executor', 'v531_invalid_actual_oof_auditor', 'v531_invalid_actual_oof_launcher',
    'v531_unconsumed_authority_receipt', 'v530_external_terminal_process', 'v527_candidate_receipt',
    'v527_authority_receipt', 'v528_deployment_receipt', 'v524_qualification_terminal',
    'v524_qualification_report', 'v524_qualification_audit', 'v524_cache_manifest', 'v482_preregistration',
    'v482_runtime_source', 'v482_trainer_source', 'v482_model_design_contract', 'v482_s0_auditor',
    'v478_selection',
]
AUTHORITY_SOURCE_ROLES = ['authority_design_contract', *SOURCE_ROLES]
SOURCE_ALIASES = {'authority_design_contract': 'authority_design_contract', **{r: r + '_source' for r in SOURCE_ROLES}}
LINEAGE = {
    'canonical_v532_execution_not_performed': True, 'fresh_actual_oof_authority_only': True,
    'v531_failed_attempt_or_output_not_published': True,
    'v531_authority_durable_consumption_receipt_available': False,
    'v531_authority_consumed_field_zero': True,
    'v531_physical_launcher_executor_entry_recorded_no_retry': True,
    'v531_failed_lineage_retry_authorized': False, 'v531_failure_forensic_current': True,
    'v524_qualification_exact20_readonly': True, 'v482_dataset_model_source_current_revalidated': True,
}
CHECK_KEYS = sorted({
    'contract_schema', 'preregistration_schema', 'manifest_schema', 'authorization_exact',
    'runtime_boundary_exact', 'execution_boundary_exact', 'source_role_order_exact',
    'source_aliases_exact', 'source_closure_current', 'active_sources_current',
    'v530_candidate_external_process_tri_bind', 'v530_candidate_exact1_unchanged',
    'v530_external_evidence_exact6_passed', 'qualification_exact20_readonly',
    'qualification_input_records_current', 'dataset_exact411_current', 'dataset_selection_current',
    'v482_model_and_sources_current', 'v169_release_library_closure_recomputed',
    'ordered_manifest_rows_exact1000', 'fold_branch_order_exact', 'fresh_roots_absent',
    'input_prepost_equal', 'services_current', 'gpu_empty', 'execution_pids_empty',
    'historical_training_authority_not_inherited', 'hidden_reward_training_cache_disabled',
    'publish_noreplace_exact1', 'active_source_role_order_exact', 'active_source_paths_exact',
    'contract_active_source_records_exact6', 'authority_active_source_records_exact7',
    'v531_failure_forensic_exact', 'v531_old_failure_roots_absent_unconsumed_authority',
})

CONTRACT_TOP_KEYS = {
    'format', 'status', 'seed', 'lineage', 'active_source_role_order', 'active_source_paths',
    'active_source_records', 'source_closure', 'source_role_order', 'source_aliases',
    'source_closure_sha256', 'authority_materializer_source', 'actual_oof_execution_preregistration_source',
    'actual_oof_execution_manifest_source', 'actual_oof_executor_source', 'actual_oof_auditor_source',
    'actual_oof_launcher_source', 'actual_oof_input_contract', 'actual_oof_output_contract',
    'v530_tri_bind', 'qualification_contract', 'dataset_contract', 'model_and_source_contract',
    'ordered_execution_manifest_contract', 'historical_absences', 'current_absences_after_authority',
    'authority_receipt_contract', 'authorization', 'runtime_observation', 'execution_boundary',
}
ANCHOR_KEYS = {
    'actual_oof_input_contract', 'actual_oof_output_contract', 'v530_tri_bind', 'qualification_contract',
    'dataset_contract', 'model_and_source_contract', 'ordered_execution_manifest_contract',
}
AUTH_TOP_KEYS = {
    'format', 'status', 'passed', 'lineage', 'active_source_role_order', 'active_source_paths',
    'active_source_records', 'source_closure', 'source_role_order', 'source_aliases',
    'source_closure_sha256', *SOURCE_ALIASES.values(), *ANCHOR_KEYS,
    'actual_oof_attempt_root', 'actual_oof_output_root', 'historical_absences', 'required_absences',
    'checks', 'check_keys', 'check_key_set_sha256', 'checks_sha256', 'input_pre_snapshot',
    'input_post_snapshot', 'input_snapshots_exactly_equal', 'authorization', 'runtime_observation',
    'execution_boundary',
}


def cbytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def csha(value):
    return hashlib.sha256(cbytes(value)).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path, expected=None):
    path = Path(path)
    metadata = os.lstat(path)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError('nonregular:' + str(path))
    record = {'path': str(path), 'sha256': sha(path), 'logical_bytes': metadata.st_size}
    if expected is not None and (record['sha256'], record['logical_bytes']) != expected:
        raise RuntimeError('record:' + str(path))
    return record


def exact_keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise RuntimeError(label + ':keys')


def json_exact(left, right):
    return cbytes(left) == cbytes(right)


def load_executor():
    regular(EXECUTOR, EXECUTOR_EXPECTED)
    spec = importlib.util.spec_from_file_location('v532_authority_executor_schema', EXECUTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_specs():
    return {
        'actual_oof_execution_preregistration': (PREREG, PREREG_EXPECTED),
        'actual_oof_execution_manifest': (MANIFEST, MANIFEST_EXPECTED),
        'actual_oof_executor': (EXECUTOR, EXECUTOR_EXPECTED),
        'actual_oof_auditor': (AUDITOR, AUDITOR_EXPECTED),
        'actual_oof_launcher': (LAUNCHER, LAUNCHER_EXPECTED),
        'v531_failure_forensic': (FORENSIC, FORENSIC_EXPECTED),
        'v531_invalid_authority_design_contract': (S / 'v531_v530_actual_oof_execution_authority_contract.json', ('065152e018ccec41688bd8d0e3d8716a58b1e4feecb763a123f1175ba696f109', 99457)),
        'v531_invalid_authority_materializer': (S / 'materialize_v531_v530_actual_oof_execution_authority.py', ('845494276b0f2fc946dd5c5a48c46edd97ec36592e98aba5ddd69265cf6a8525', 33005)),
        'v531_invalid_actual_oof_execution_preregistration': (S / 'v531_v530_actual_oof_execution_preregistration.json', ('6fc85dd57bd6693a276c6f3ddfbbda1ff6cf42af7f5a1417dcdfbab460045329', 15071)),
        'v531_invalid_actual_oof_execution_manifest': (S / 'v531_v530_actual_oof_execution_manifest.json', ('ec0a5c396972475ead9eb70db93320799a239257e88938490453b6236ecd7fc9', 1485362)),
        'v531_invalid_actual_oof_executor': (S / 'execute_v531_v530_actual_oof.py', ('14537cfe9f54ba4f1d47609102c12768a9c215317cd9c8f66346f0a7bb33587b', 47719)),
        'v531_invalid_actual_oof_auditor': (S / 'audit_v531_v530_actual_oof.py', ('8251aedf27fcbd9966a46e9c5ef7ef3ce55b0e986251a7b9dcc31a820f1d3792', 10155)),
        'v531_invalid_actual_oof_launcher': (S / 'launch_v531_v530_actual_oof.py', ('653de64004c2582ef776bd4965d52693f738e2e12f99377c448ac79f00287895', 5515)),
        'v531_unconsumed_authority_receipt': (J / 'v531_v530_actual_oof_execution_authority_seed1665_20260827/authority_receipt.json', ('e8b6d34606dbfcb09ba25b7ecef13b99c63eeb50062cc84234bcdb5e82396072', 299373)),
        'v530_external_terminal_process': (V530_ROOT / 'process_receipt.json', ('2de06ea2b50577beda7904137338f9d7423bcd3d3ea57ab8c571a210b7a6c045', 28903)),
        'v527_candidate_receipt': (CANDIDATE_ROOT / 'oof_readonly_gate_validation_receipt.json', ('5b443f2275c9caced1974a0fa0f10e8da414faa15f672eae83fd4a8630861439', 152048)),
        'v527_authority_receipt': (J / 'v527_v526_v525_oof_readonly_validation_execution_authority_seed1662_20260827/authority_receipt.json', ('e658fc7e8ac7ae0afc431bdc0a1b756fd202d933cfae0bec61e693e3d3e71b4a', 181901)),
        'v528_deployment_receipt': (J / 'v528_v527_stage_a_authority_source_deployment_evidence_seed1663_20260827/deployment_receipt.json', ('681b30606e5029049e6dd7ecadcde6ed1da82b799fa6c77ba7bd0c8d4917ea22', 33739)),
        'v524_qualification_terminal': (QUAL_ROOT / 'terminal_receipt.json', ('93d38e010b37275a740bf3ccae2b935b08e40b6953657c12838780e5e2b542a1', 83762)),
        'v524_qualification_report': (QUAL_ROOT / 'qualification_report.json', ('6514b497f994bc502fc46dc6c5a90865dbb126402935cfa359a1af2fe0fbc173', 87681)),
        'v524_qualification_audit': (QUAL_ROOT / 'independent_final_audit.json', ('3d702b36a3589d737c9f21e0a1182c59f636753faa1143a2adc85d90d3a674bc', 9935)),
        'v524_cache_manifest': (QUAL_ROOT / 'process_a/cache/manifest.json', ('716d62575ea149242f0e0918ed508e22ae803ff12d3b66fb26bb4b8f5a3e6f94', 18453)),
        'v482_preregistration': (J / 'v482_temporal8_residual_s0_r3_seed1624_20260824/preregistration.json', ('426d4f7a774520458f6af41d20660cf31202791f99603dacec114f771da6a881', 294684)),
        'v482_runtime_source': (ROOT / 'pipeline/wam_pipeline/v482_temporal8_residual_runtime.py', ('6939e3f6f52c1eb83bfd4324372d7e75ed4ee6d493777a39f38c26e5f4db7471', 21745)),
        'v482_trainer_source': (S / 'train_v482_temporal8_residual_5fold.py', ('674b68afed3b6c39663d629db38be11aa2c752e54692458b43d4254eb08b8c1d', 48428)),
        'v482_model_design_contract': (S / 'v482_temporal_film_residual_model_design_contract.json', ('19eea774f4b8871528054a352bf6c7ed6cefd39afcfc1956c9cf7a969335fddb', 37030)),
        'v482_s0_auditor': (S / 'audit_v482_temporal8_residual_s0.py', ('a24a20a8c0670e7bf53819bb48e78d204b482fbbdbe1060db61957488b3eed69', 20439)),
        'v478_selection': (J / 'v478_temporal200_seed1622_20260824/selection.json', ('f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398', None)),
    }


def source_records(materializer_record):
    records = {'authority_materializer': materializer_record}
    for role, (path, expected) in source_specs().items():
        if expected[1] is None:
            row = regular(path)
            if row['sha256'] != expected[0]:
                raise RuntimeError('selection record')
            records[role] = row
        else:
            records[role] = regular(path, expected)
    if list(records) != SOURCE_ROLES:
        raise RuntimeError('source role order construction')
    return records


def tree_summary(module, path):
    return module.tree(Path(path))


def absences():
    return {
        'authority_root': AUTH_ROOT, 'authority_prep': AUTH_PREP,
        'actual_oof_attempt_root': ATTEMPT_ROOT, 'actual_oof_attempt_prep': ATTEMPT_PREP,
        'actual_oof_output_root': OOF_ROOT, 'actual_oof_output_prep': OOF_PREP,
    }


def relevant_pids():
    names = {EXECUTOR.name, AUDITOR.name, LAUNCHER.name, 'train_v482_temporal8_residual_5fold.py', 'v482_temporal8_residual_runtime.py'}
    excluded = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            pid = int((Path('/proc') / str(pid) / 'stat').read_text().split()[3])
        except Exception:
            break
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            argv = [item.decode(errors='replace') for item in (proc / 'cmdline').read_bytes().split(b'\0') if item]
        except Exception:
            continue
        matches = sorted({Path(item).name for item in argv if Path(item).name in names})
        if matches:
            rows.append({'pid': int(proc.name), 'argv': argv, 'matched_entrypoints': matches})
    return sorted(rows, key=lambda row: row['pid'])


def snapshot(module, records, absence_map):
    values = {
        'files': {role: regular(Path(row['path'])) for role, row in records.items()},
        'trees': {
            'qualification_exact20': tree_summary(module, QUAL_ROOT),
            'dataset_exact411': tree_summary(module, DATASET_ROOT),
            'v527_candidate_exact1': tree_summary(module, CANDIDATE_ROOT),
            'v530_external_evidence_exact6': tree_summary(module, V530_ROOT),
        },
        'absences': {name: {'path': str(path), 'absent': not os.path.lexists(path)} for name, path in sorted(absence_map.items())},
        'services': module.services_snapshot(),
        'gpu_compute_pids': module.gpu_processes(),
        'relevant_execution_pids': relevant_pids(),
    }
    values['canonical_sha256'] = csha(values)
    return values


def validate_live_inputs(module, prereg, manifest):
    module.validate_static_documents(prereg, manifest)
    module.validate_input_records(prereg)
    module.validate_v482_data_model(manifest)
    for name, document in [('preregistration', prereg), ('manifest', manifest)]:
        if document.get('active_source_role_order') != ACTIVE_SOURCE_ROLE_ORDER or not json_exact(document.get('active_source_paths'), ACTIVE_SOURCE_PATHS) or 'active_source_records' in document:
            raise RuntimeError(name + ' active source registry')
    if not json_exact(prereg['authorization'], AUTHORIZATION):
        raise RuntimeError('prereg authorization')
    if prereg['execution_manifest'] != regular(MANIFEST, MANIFEST_EXPECTED):
        raise RuntimeError('manifest binding')
    rows = manifest['ordered_rows']
    if len(rows) != 1000 or manifest['ordered_rows_canonical_sha256'] != '14aa8b1d8ac70c70c375df6cb4737850e2894d901ba0eb408c8097b1158676c2':
        raise RuntimeError('ordered rows')
    candidate = json.loads((CANDIDATE_ROOT / 'oof_readonly_gate_validation_receipt.json').read_bytes())
    external = json.loads((V530_ROOT / 'process_receipt.json').read_bytes())
    if external.get('status') != 'passed_external_terminal' or external.get('passed') is not True or external.get('candidate_consumable') is not True:
        raise RuntimeError('v530 external terminal')
    if external.get('candidate_receipt') != regular(CANDIDATE_ROOT / 'oof_readonly_gate_validation_receipt.json', ('5b443f2275c9caced1974a0fa0f10e8da414faa15f672eae83fd4a8630861439', 152048)):
        raise RuntimeError('v530 candidate bind')
    if candidate.get('candidate_verified') is not True or candidate.get('passed') is not None or candidate.get('standalone_consumable') is not False:
        raise RuntimeError('candidate truth')
    forensic = json.loads(FORENSIC.read_bytes())
    if forensic.get('status') != 'frozen_failed_no_retry_pending_fresh_lineage_repair' or forensic.get('repair_boundary', {}).get('required_role_order') != ACTIVE_SOURCE_ROLE_ORDER:
        raise RuntimeError('v531 forensic')
    old_authority = json.loads((J / 'v531_v530_actual_oof_execution_authority_seed1665_20260827/authority_receipt.json').read_bytes())
    if old_authority.get('authorization', {}).get('actual_oof_execution_boundary_invocations_consumed') != 0:
        raise RuntimeError('v531 authority consumed field')
    old_failed = forensic['actual_state']['v531_failed_roots']
    if not all(row.get('absent') is True and not os.path.lexists(row['path']) for row in old_failed.values()):
        raise RuntimeError('v531 failed roots current absence')
    qualification = module.tree(QUAL_ROOT)
    expected_qualification = prereg['input_contract']['qualification']['qualification_tree']
    normalized = {
        'root': str(QUAL_ROOT), 'file_count': qualification['file_count'],
        'logical_file_bytes': qualification['logical_file_bytes'],
        'sha256sum_lines_digest_sha256': qualification['sha256sum_lines_digest_sha256'],
        'canonical_json_triples_digest_sha256': qualification['canonical_json_triples_digest_sha256'],
    }
    if not json_exact(normalized, {'root': str(QUAL_ROOT), **expected_qualification}):
        raise RuntimeError('qualification exact20')
    dataset = module.tree(DATASET_ROOT)
    expected_dataset = prereg['input_contract']['dataset']
    dataset_pair_digest = csha([[relative, digest] for relative, digest, _ in dataset['inventory']])
    dataset['canonical_json_pairs_digest_sha256'] = dataset_pair_digest
    if dataset['file_count'] != expected_dataset['file_count'] or dataset['logical_file_bytes'] != expected_dataset['logical_file_bytes'] or dataset_pair_digest != expected_dataset['canonical_tree_sha256']:
        raise RuntimeError('dataset exact411')
    return candidate, external, qualification, dataset


def contract_anchors(prereg, manifest, module):
    candidate, external, qualification, dataset = validate_live_inputs(module, prereg, manifest)
    candidate_tree = module.tree(CANDIDATE_ROOT)
    external_tree = module.tree(V530_ROOT)
    tri_bind = {
        'candidate_receipt': regular(CANDIDATE_ROOT / 'oof_readonly_gate_validation_receipt.json', ('5b443f2275c9caced1974a0fa0f10e8da414faa15f672eae83fd4a8630861439', 152048)),
        'candidate_tree': candidate_tree,
        'candidate_nonterminal_verified': True,
        'external_process_receipt': regular(V530_ROOT / 'process_receipt.json', ('2de06ea2b50577beda7904137338f9d7423bcd3d3ea57ab8c571a210b7a6c045', 28903)),
        'external_evidence_tree': external_tree,
        'external_status': external['status'],
        'external_candidate_consumable': external['candidate_consumable'],
        'external_candidate_receipt_bind': external['candidate_receipt'],
    }
    return {
        'actual_oof_input_contract': prereg['input_contract'],
        'actual_oof_output_contract': prereg['output_contract'],
        'v530_tri_bind': tri_bind,
        'qualification_contract': {'root': str(QUAL_ROOT), 'tree': qualification, 'readonly': True, 'mutation_authorized': False},
        'dataset_contract': {'root': str(DATASET_ROOT), 'tree': dataset, 'selection': prereg['input_contract']['dataset']['selection'], 'mutation_authorized': False},
        'model_and_source_contract': prereg['input_contract']['model_and_sources'],
        'ordered_execution_manifest_contract': {
            'manifest': regular(MANIFEST, MANIFEST_EXPECTED),
            'ordered_rows_count': 1000,
            'ordered_rows_canonical_sha256': manifest['ordered_rows_canonical_sha256'],
            'fold_order': manifest['fold_order'], 'branches': manifest['branches'],
            'fold_row_counts': manifest['fold_row_counts'],
        },
    }


def rename_noreplace(source, target):
    function = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if function is None:
        raise RuntimeError('renameat2 unavailable')
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), str(target))


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def held_read(fd, size):
    data = os.pread(fd, size + 1, 0)
    if len(data) != size or os.pread(fd, 1, size) != b'':
        raise RuntimeError('held EOF')
    return data


def gate_owned_staged(prep, prep_identity, member, member_identity, descriptor, payload):
    prep_metadata = os.lstat(prep)
    if prep.is_symlink() or not stat.S_ISDIR(prep_metadata.st_mode) or (prep_metadata.st_dev, prep_metadata.st_ino) != prep_identity:
        raise RuntimeError('prep identity')
    if sorted(entry.name for entry in os.scandir(prep)) != ['authority_receipt.json']:
        raise RuntimeError('prep exact1')
    member_metadata = os.lstat(member)
    fd_metadata = os.fstat(descriptor)
    if member.is_symlink() or not stat.S_ISREG(member_metadata.st_mode) or (member_metadata.st_dev, member_metadata.st_ino) != member_identity or (fd_metadata.st_dev, fd_metadata.st_ino) != member_identity:
        raise RuntimeError('held identity')
    if fd_metadata.st_size != len(payload) or held_read(descriptor, len(payload)) != payload:
        raise RuntimeError('held payload')


def publish_exact1(receipt, stable_snapshot, module, records):
    if os.path.lexists(AUTH_ROOT) or os.path.lexists(AUTH_PREP):
        raise RuntimeError('authority prestate')
    oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    prep_identity = None
    member_identity = None
    descriptor = None
    committed = False
    payload = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        os.mkdir(AUTH_PREP, 0o700)
        metadata = os.lstat(AUTH_PREP)
        prep_identity = (metadata.st_dev, metadata.st_ino)
        member = AUTH_PREP / 'authority_receipt.json'
        descriptor = os.open(member, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        metadata = os.fstat(descriptor)
        member_identity = (metadata.st_dev, metadata.st_ino)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if type(written) is not int or written <= 0 or written > len(payload) - offset:
                raise RuntimeError('held write')
            offset += written
        os.fsync(descriptor)
        gate_owned_staged(AUTH_PREP, prep_identity, member, member_identity, descriptor, payload)
        fsync_dir(AUTH_PREP)
        stable_absences = {k: v for k, v in absences().items() if k not in {'authority_root', 'authority_prep'}}
        third = snapshot(module, records, stable_absences)
        if third != stable_snapshot:
            raise RuntimeError('third snapshot drift')
        gate_owned_staged(AUTH_PREP, prep_identity, member, member_identity, descriptor, payload)
        rename_noreplace(AUTH_PREP, AUTH_ROOT)
        committed = True
        diagnostics = []
        tree = None
        for name, operation in [
            ('parent_fsync', lambda: fsync_dir(AUTH_ROOT.parent)),
            ('held_readback', lambda: held_read(descriptor, len(payload))),
            ('current_tree', lambda: module.tree(AUTH_ROOT)),
        ]:
            try:
                result = operation()
                if name == 'current_tree':
                    tree = result
            except BaseException as error:
                diagnostics.append({'operation': name, 'error_type': type(error).__name__, 'error': str(error)})
        exact = tree is not None and tree.get('inventory') == [['authority_receipt.json', digest, len(payload)]]
        return {'visibility_committed': True, 'current_exact1_consumable': exact, 'postcommit_diagnostics_passed': not diagnostics and exact, 'postcommit_diagnostics': diagnostics, 'retry_authorized': False}
    finally:
        if not committed and prep_identity is not None and AUTH_PREP.is_dir() and not AUTH_PREP.is_symlink():
            try:
                metadata = os.lstat(AUTH_PREP)
                member = AUTH_PREP / 'authority_receipt.json'
                if (metadata.st_dev, metadata.st_ino) == prep_identity and descriptor is not None and os.path.lexists(member):
                    current = os.lstat(member)
                    fd_metadata = os.fstat(descriptor)
                    if not member.is_symlink() and (current.st_dev, current.st_ino) == member_identity and (fd_metadata.st_dev, fd_metadata.st_ino) == member_identity and fd_metadata.st_size == len(payload) and held_read(descriptor, len(payload)) == payload:
                        member.unlink()
                if not any(AUTH_PREP.iterdir()):
                    AUTH_PREP.rmdir(); fsync_dir(AUTH_PREP.parent)
            except BaseException:
                pass
        if descriptor is not None:
            try: os.close(descriptor)
            except OSError: pass
        signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)


def expected_sources(materializer_record):
    return source_records(materializer_record)


def validate_context(args):
    if args.contract != CONTRACT or args.materializer_source != SELF or args.authority_root != AUTH_ROOT:
        raise RuntimeError('canonical CLI')
    contract_record = regular(CONTRACT, (args.contract_sha, args.contract_bytes))
    materializer_record = regular(SELF, (args.materializer_sha, args.materializer_bytes))
    prereg_record = regular(PREREG, PREREG_EXPECTED)
    manifest_record = regular(MANIFEST, MANIFEST_EXPECTED)
    regular(EXECUTOR, EXECUTOR_EXPECTED); regular(AUDITOR, AUDITOR_EXPECTED); regular(LAUNCHER, LAUNCHER_EXPECTED)
    prereg = json.loads(PREREG.read_bytes()); manifest = json.loads(MANIFEST.read_bytes()); contract = json.loads(CONTRACT.read_bytes())
    module = load_executor()
    validate_live_inputs(module, prereg, manifest)
    exact_keys(contract, CONTRACT_TOP_KEYS, 'contract')
    if contract['format'] != CONTRACT_FORMAT or contract['status'] != CONTRACT_STATUS or type(contract['seed']) is not int or contract['seed'] != 1666:
        raise RuntimeError('contract header')
    if not json_exact(contract['lineage'], LINEAGE) or not json_exact(contract['authorization'], AUTHORIZATION) or not json_exact(contract['runtime_observation'], RUNTIME) or not json_exact(contract['execution_boundary'], EXECUTION_BOUNDARY):
        raise RuntimeError('contract boundary')
    sources = expected_sources(materializer_record)
    if contract['source_role_order'] != SOURCE_ROLES or not json_exact(contract['source_closure'], sources) or contract['source_closure_sha256'] != csha(sources) or contract['source_aliases'] != SOURCE_ALIASES:
        raise RuntimeError('contract source closure')
    shared_active = ACTIVE_SOURCE_ROLE_ORDER[1:]
    active_records = {role: sources[role] for role in shared_active}
    if contract['active_source_role_order'] != ACTIVE_SOURCE_ROLE_ORDER or not json_exact(contract['active_source_paths'], ACTIVE_SOURCE_PATHS) or not json_exact(contract['active_source_records'], active_records):
        raise RuntimeError('contract active source registry')
    for role in shared_active:
        if contract[SOURCE_ALIASES[role]] != sources[role]:
            raise RuntimeError('contract source alias:' + role)
    anchors = contract_anchors(prereg, manifest, module)
    if any(not json_exact(contract[key], value) for key, value in anchors.items()):
        raise RuntimeError('contract anchor')
    historical = absences()
    current = {k: v for k, v in historical.items() if k != 'authority_root'}
    if contract['historical_absences'] != {k: {'path': str(v)} for k, v in historical.items()} or contract['current_absences_after_authority'] != {k: {'path': str(v)} for k, v in current.items()}:
        raise RuntimeError('contract absences')
    if any(os.path.lexists(path) for path in historical.values()):
        raise RuntimeError('fresh roots')
    schema = contract['authority_receipt_contract']
    expected_schema = {
        'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS, 'top_keys': sorted(AUTH_TOP_KEYS),
        'check_keys': CHECK_KEYS, 'check_key_set_sha256': csha(CHECK_KEYS),
        'checks_sha256': csha({k: True for k in CHECK_KEYS}), 'authorization_exact': AUTHORIZATION,
        'runtime_observation_exact': RUNTIME, 'execution_boundary_exact': EXECUTION_BOUNDARY,
        'actual_oof_input_contract_exact': prereg['input_contract'], 'actual_oof_output_contract_exact': prereg['output_contract'],
        'v530_tri_bind_exact': anchors['v530_tri_bind'], 'ordered_execution_manifest_contract_exact': anchors['ordered_execution_manifest_contract'],
        'active_source_role_order_exact': ACTIVE_SOURCE_ROLE_ORDER,
        'active_source_paths_exact': ACTIVE_SOURCE_PATHS,
        'contract_active_source_records_exact': active_records,
        'contract_active_source_records_count': 6,
        'contract_design_record_excluded_to_avoid_self_hash_cycle': True,
        'authority_active_source_records_count': 7,
        'authority_design_record_included': True,
    }
    if not json_exact(schema, expected_schema):
        raise RuntimeError('receipt schema')
    records = {'authority_design_contract': contract_record, **sources}
    pre = snapshot(module, records, historical)
    return contract, contract_record, materializer_record, prereg, manifest, sources, anchors, historical, current, records, pre, module


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha', required=True)
    parser.add_argument('--contract-bytes', type=int, required=True)
    parser.add_argument('--materializer-source', type=Path, required=True)
    parser.add_argument('--materializer-sha', required=True)
    parser.add_argument('--materializer-bytes', type=int, required=True)
    parser.add_argument('--authority-root', type=Path, required=True)
    parser.add_argument('--read-only-preflight', action='store_true')
    return parser.parse_args()


def main():
    if os.sys.argv[1:] == ['--synthetic-self-test']:
        checks = {
            'contract_top_count': len(CONTRACT_TOP_KEYS) == 30,
            'source_count': len(SOURCE_ROLES) == 29,
            'authority_source_count': len(AUTHORITY_SOURCE_ROLES) == 30,
            'active_role_count': len(ACTIVE_SOURCE_ROLE_ORDER) == 7,
            'active_path_bijection': list(ACTIVE_SOURCE_PATHS) == ACTIVE_SOURCE_ROLE_ORDER and len(set(ACTIVE_SOURCE_PATHS.values())) == 7,
            'authorization_count': len(AUTHORIZATION) == 24,
            'authorization_strict_types': all(type(value) is int for key, value in AUTHORIZATION.items() if key.endswith('_invocations_authorized') or key.endswith('_invocations_consumed')),
            'only_actual_oof_one': AUTHORIZATION['actual_oof_execution_boundary_invocations_authorized'] == 1 and AUTHORIZATION['actual_oof_execution_boundary_invocations_consumed'] == 0,
            'all_unsafe_disabled': all(AUTHORIZATION[key] is False for key in ('retry_authorized', 'qualification_mutation_authorized', 'hidden_or_final_inputs_authorized', 'reward_read_authorized', 'training_authorized', 'cache_reuse_authorized', 'submission_authorized')),
            'fresh_roots_distinct': len(set(map(str, absences().values()))) == 6,
        }
        print(json.dumps({'passed': all(checks.values()), 'checks': checks, 'checks_sha256': csha(checks)}, sort_keys=True))
        return 0 if all(checks.values()) else 1
    args = parse()
    contract, contract_record, materializer_record, prereg, manifest, sources, anchors, historical, current, records, pre, module = validate_context(args)
    post = snapshot(module, records, historical)
    if pre != post:
        raise RuntimeError('input prepost drift')
    checks = {key: True for key in CHECK_KEYS}
    checks['gpu_empty'] = post['gpu_compute_pids'] == []
    checks['execution_pids_empty'] = post['relevant_execution_pids'] == []
    authority_sources = {'authority_design_contract': contract_record, **sources}
    authority_active_records = {role: authority_sources[role] for role in ACTIVE_SOURCE_ROLE_ORDER}
    receipt = {
        'format': OUTPUT_FORMAT, 'status': OUTPUT_STATUS, 'passed': True,
        'lineage': LINEAGE, 'active_source_role_order': ACTIVE_SOURCE_ROLE_ORDER,
        'active_source_paths': ACTIVE_SOURCE_PATHS, 'active_source_records': authority_active_records,
        'source_closure': authority_sources, 'source_role_order': AUTHORITY_SOURCE_ROLES,
        'source_aliases': SOURCE_ALIASES, 'source_closure_sha256': csha(authority_sources),
        **{SOURCE_ALIASES[role]: record for role, record in authority_sources.items()}, **anchors,
        'actual_oof_attempt_root': str(ATTEMPT_ROOT), 'actual_oof_output_root': str(OOF_ROOT),
        'historical_absences': {k: {'path': str(v), 'absent': True} for k, v in historical.items()},
        'required_absences': {k: {'path': str(v), 'absent': True} for k, v in current.items()},
        'checks': checks, 'check_keys': CHECK_KEYS, 'check_key_set_sha256': csha(CHECK_KEYS),
        'checks_sha256': csha(checks), 'input_pre_snapshot': pre, 'input_post_snapshot': post,
        'input_snapshots_exactly_equal': True, 'authorization': AUTHORIZATION,
        'runtime_observation': RUNTIME, 'execution_boundary': EXECUTION_BOUNDARY,
    }
    exact_keys(receipt, AUTH_TOP_KEYS, 'authority receipt')
    if not all(type(value) is bool and value is True for value in checks.values()):
        raise RuntimeError('checks')
    if args.read_only_preflight:
        print(json.dumps({'passed': True, 'status': 'passed_read_only_preflight_no_materialization', 'contract_top_count': len(contract), 'authority_top_count': len(receipt), 'check_count': len(checks), 'source_count': len(sources), 'authority_source_count': len(authority_sources), 'input_snapshot_sha256': pre['canonical_sha256']}, sort_keys=True))
        return 0
    stable = {k: v for k, v in current.items() if k != 'authority_prep'}
    stable_snapshot = snapshot(module, records, stable)
    publish_exact1(receipt, stable_snapshot, module, records)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
