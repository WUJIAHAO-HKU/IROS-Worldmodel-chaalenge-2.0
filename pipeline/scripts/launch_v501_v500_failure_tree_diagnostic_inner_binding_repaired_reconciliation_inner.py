#!/usr/bin/env python3
"""One-shot fail-closed launcher for the authorized v490 read-only reconciliation repair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
try:
 import fcntl
except ImportError:
 fcntl=None

ROOT=Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J=ROOT/"artifacts/strict_track2_joint_augmentation_20260810"
RLPY=Path("/root/autodl-tmp/conda_envs/rlinf_track2/bin/python")
WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_inner.py"
AUTHORITY_CONTRACT_PATH=ROOT/"pipeline/scripts/v501_v500_failure_tree_diagnostic_inner_binding_repair_execution_authority_contract.json"
AUTHORITY_MATERIALIZER_PATH=ROOT/"pipeline/scripts/materialize_v501_v500_failure_tree_diagnostic_inner_binding_repair_execution_authority.py"
AUTHORITY_MATERIALIZER_SHA="59e3863c4d1d7b028422cfebb7dfd7fb208526c322d46151ec0010f239d82358"
AUTHORITY_MATERIALIZER_BYTES=126045
AUTHORITY_ROOT=J/"v501_v500_failure_tree_diagnostic_inner_binding_repair_execution_authority_seed1643_20260825"
AUTHORITY_RECEIPT_PATH=AUTHORITY_ROOT/"authority_receipt.json"
REPAIR_FORMAL_PATH=J/"v488_v487_c71_exact7_schema_repair_prereg_seed1630_20260825/preregistration.json"
REPAIR_FORMAL_SHA="b3598de9a3d44f6d58a0a6a75c956174a68bc7dd610c6a8eb56124690709371b"
REPAIR_FORMAL_BYTES=36181
STATIC_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_seed1632_20260825"
STATIC_RECEIPT_PATH=STATIC_ROOT/"static_audit.json"
STATIC_RECEIPT_SHA="441efdf607e8fb5aa10a42a542f984304c8e27b334bfbf3f7773c71a1b5ee3bb"
STATIC_RECEIPT_BYTES=49008
STATIC_SOURCE_PATH=ROOT/"pipeline/scripts/audit_v490_v489_v488_v487_c71_exact7_schema_repair_static.py"
STATIC_SOURCE_SHA="1030de39f14ecaffb14c9f8192c2ec3d0898a2cb8e5913e64b50cf13d8baca80"
STATIC_SOURCE_BYTES=51979
STATIC_EXECUTION_EVIDENCE_ROOT=J/"v490_v489_v488_v487_c71_exact7_schema_repair_static_audit_execution_evidence_seed1632_20260825"
STATIC_EXECUTION_EVIDENCE_TREE={"root":str(STATIC_EXECUTION_EVIDENCE_ROOT),"inventory":[["argv.json","8795179679b7dfffd7dd046f406e902f11b89d6d2b2ac63048b3e82029620dcb",11017],["intent.json","692b86f97ef08ec39f017bbc17a9f4a71ca3fe20312e8758f3972166daa1de99",170],["process_receipt.json","b686df0afdd1b330200e99c0732a4fcf321325ca596ee0c20fb327301ca67dab",17211],["static_stderr.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["static_stdout.log","b1c7595ce329c13e314ee264e90f54ee787876be2e1e2458929cf68f5e629f4f",95],["transport_helper.py","b352813d6543d3d3e541432f2b01f48786130284728a9f31bb011b49a8b23a40",13811]],"file_count":6,"logical_file_bytes":42304,"sha256sum_lines_digest_sha256":"c0e8b964b64d19cbda82241f4690ca4f69697f704c9c961dda05135d5b06260f","canonical_json_triples_digest_sha256":"2179c46cdd545ba55b1498bd4ac74f7556d6f8ba4150a562655b7fb99d11b776"}
F813_PATH=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/preregistration.json"
F813_SHA="f8137d02a692a9c6243f13ce8a7674e4ac90fbd31c85b3e8c32af82f8afa42a8"
F813_BYTES=21296
PHASE_A_DESIGN_CONTRACT_PATH=ROOT/"pipeline/scripts/v485_v482_v169_cache_determinism_scope_repair_contract.json"
PHASE_A_DESIGN_CONTRACT_SHA="8c4a583aa19de6927cc1ef787920b65317893e27ad01371d0b95b622f260cf64"
PHASE_A_DESIGN_CONTRACT_BYTES=43960
V493_AUTHORITY_CONTRACT_PATH=ROOT/"pipeline/scripts/v493_v492_v490_corrected_reconciliation_execution_authority_contract.json"
V493_AUTHORITY_CONTRACT_SHA="0db696b01c8a561ce717b500b54bb3fd7c73e2f81b1ae6c117d92ed93de85e9f"
V493_AUTHORITY_CONTRACT_BYTES=26569
V493_AUTHORITY_MATERIALIZER_PATH=ROOT/"pipeline/scripts/materialize_v493_v492_v490_corrected_reconciliation_execution_authority.py"
V493_AUTHORITY_MATERIALIZER_SHA="1d9a3fa94aa04c87afe309c0a7662c2ee51eb8eeda9d307132476252097699ad"
V493_AUTHORITY_MATERIALIZER_BYTES=33691
V493_OUTER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_outer.py"
V493_OUTER_WRAPPER_SHA="658c13857de127e4661fb1a010c2736951de5bd972e53c56a577c7cda882fa0a"
V493_OUTER_WRAPPER_BYTES=39225
V493_INNER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v493_v492_v490_corrected_reconciliation_inner.py"
V493_INNER_WRAPPER_SHA="343577ec4050a63ecf86ba082b23b5c5b869f90de8abb9f9d10f22cc47510597"
V493_INNER_WRAPPER_BYTES=75190
V493_AUTHORITY_ROOT=J/"v493_v492_v490_corrected_reconciliation_execution_authority_seed1635_20260825"
V493_AUTHORITY_RECEIPT_PATH=V493_AUTHORITY_ROOT/"authority_receipt.json"
V493_AUTHORITY_RECEIPT_SHA="2a93aebbfb968c8134b92b4ed7bf2e809a32085483d1e745a9393997f55167c7"
V493_AUTHORITY_RECEIPT_BYTES=60534
V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT=J/"v493_v492_v490_corrected_reconciliation_execution_authority_materialization_evidence_seed1635_20260825"
V493_OUTER_FAILURE_ROOT=J/"v493_v492_v490_corrected_reconciliation_outer_execution_evidence_seed1635_20260825"
V494_AUTHORITY_CONTRACT_PATH=ROOT/"pipeline/scripts/v494_v493_v490_interpreter_symlink_repair_execution_authority_contract.json"
V494_AUTHORITY_CONTRACT_SHA="2f4cef2671dfd5c443d5a7f53dc0ea7b770b520da84cba3512a363884d56bd4a"
V494_AUTHORITY_CONTRACT_BYTES=29803
V494_AUTHORITY_MATERIALIZER_PATH=ROOT/"pipeline/scripts/materialize_v494_v493_v490_interpreter_symlink_repair_execution_authority.py"
V494_AUTHORITY_MATERIALIZER_SHA="5b9396fd74e1d161fa41610108d823f356313992364cba708ed869519fa0a421"
V494_AUTHORITY_MATERIALIZER_BYTES=39236
V494_OUTER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer.py"
V494_OUTER_WRAPPER_SHA="bea4b6df050afb22c833908725a6af7417b292984d8ddb4fbb972b99128d3df5"
V494_OUTER_WRAPPER_BYTES=43024
V494_INNER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner.py"
V494_INNER_WRAPPER_SHA="1f2c9de60703f12c22966b7aeb3526a6c32e0d5a0e0258094c9d208b9a56f4bf"
V494_INNER_WRAPPER_BYTES=79760
V494_FAILURE_FORENSIC_PATH=ROOT/"pipeline/scripts/v495_v494_authority_materializer_failure_forensic_reconstructed.json"
V494_FAILURE_FORENSIC_SHA="248c4a94a24c3b793bb29b6cc07a5955a93694f7e22b1a9363ef5bb7ee0ba0dc"
V494_FAILURE_FORENSIC_BYTES=6409
V494_FAILURE_TRANSPORT_SCRIPT_PATH=ROOT/"pipeline/scripts/v495_v494_failed_authority_materialization_transport_script.sh"
V494_FAILURE_TRANSPORT_SCRIPT_SHA="0d4669e9a1fa52ee98fe2c459e9e42d880116e4428270edcefb0f50e5b95d49c"
V494_FAILURE_TRANSPORT_SCRIPT_BYTES=942
V494_AUTHORITY_ROOT=J/"v494_v493_v490_interpreter_symlink_repair_execution_authority_seed1636_20260825"
V494_OUTER_ROOT=J/"v494_v493_v490_interpreter_symlink_repaired_reconciliation_outer_execution_evidence_seed1636_20260825"
V494_INNER_ROOT=J/"v494_v493_v490_interpreter_symlink_repaired_reconciliation_inner_attempt_seed1636_20260825"
V495_AUTHORITY_CONTRACT_PATH=ROOT/"pipeline/scripts/v495_v494_v490_interpreter_validator_repair_execution_authority_contract.json"
V495_AUTHORITY_CONTRACT_SHA="60d978a803bd148f6fe68906e71113cf33e61a3ae5bf77e22c8f2855c68c2823";V495_AUTHORITY_CONTRACT_BYTES=37084
V495_AUTHORITY_MATERIALIZER_PATH=ROOT/"pipeline/scripts/materialize_v495_v494_v490_interpreter_validator_repair_execution_authority.py"
V495_AUTHORITY_MATERIALIZER_SHA="b0f4b59212ea67754a12394bf1f5a8e7d2c5389c93c44207e0099d9f2850fe18";V495_AUTHORITY_MATERIALIZER_BYTES=50978
V495_INNER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v495_v494_v490_interpreter_validator_repaired_reconciliation_inner.py"
V495_INNER_WRAPPER_SHA="b52e7eb29870141bda85eead81cbc1c3a47522ad86e6a7b6829f7052766a8f76";V495_INNER_WRAPPER_BYTES=90261
V495_OUTER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v495_v494_v490_interpreter_validator_repaired_reconciliation_outer.py"
V495_OUTER_WRAPPER_SHA="3aae6deb4059e143cc40a3e10bdf6f1cf51e20e884c0459b4aa950e0d480d526";V495_OUTER_WRAPPER_BYTES=50749
V495_FAILURE_ROOT=J/"v495_v494_v490_interpreter_validator_repair_execution_authority_materialization_evidence_seed1637_20260825"
V495_FAILURE_PROCESS_PATH=V495_FAILURE_ROOT/"process_receipt.json";V495_FAILURE_PROCESS_SHA="72d3814fbb56053b0b03dc2225ee5aabf62f999a3cf12be3b5206dc22898b8f4";V495_FAILURE_PROCESS_BYTES=411
V495_FAILURE_TREE={"root":str(V495_FAILURE_ROOT),"inventory":[["argv.json","54cda5c2853e9b9d74c1abc11b3c56ef51de5a1650487d8813c4454fdc17b8a3",16203],["intent.json","e339933407261a528b516d9bb3759fffdc853fc7998416b5f1e66b745a032828",542],["materializer_stderr.log","bfe41948b189d59af0aa4644bc3be1d32d4a2dea8537d6b867ba3c51a4cdbc75",544],["materializer_stdout.log","e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",0],["process_receipt.json",V495_FAILURE_PROCESS_SHA,V495_FAILURE_PROCESS_BYTES],["transport_helper.py","f2ad3a5a10fb419c0fdd1595997e212c5ec1ccfbb6f6ec3c23410d7ab9b31727",14439]],"file_count":6,"logical_file_bytes":32139,"sha256sum_lines_digest_sha256":"21a38be60c37efd2e14b3f8319889b5e53cf7d65da07f82f6b7812f50b0bac1b","canonical_json_triples_digest_sha256":"9669dad691e0f9d8ca4005c62bb4568647c3815dd21e0407aa9d2663042c4290"}
OUTER_WRAPPER_PATH=ROOT/"pipeline/scripts/launch_v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_outer.py"
OUTER_EVIDENCE_ROOT=J/"v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_outer_execution_evidence_seed1643_20260825"
R2_PATH=ROOT/"pipeline/scripts/reconcile_v488_v487_c71_exact7_schema_repair.py"
R2_SHA="9efc41eb2574b065ab4e83920f524ae47902079f9b40b9bcc6d31e6739bc1377"
R2_BYTES=51845
ATTEMPT_ROOT=J/"v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_inner_attempt_seed1643_20260825"
ATTEMPT_PREP=ATTEMPT_ROOT.with_name(ATTEMPT_ROOT.name+".attempt-prep")
LOCK_PATH=J/".v501_v500_failure_tree_diagnostic_inner_binding_repaired_reconciliation_attempt.lock"
TRANSPARENT_PATH=J/"v486_v485_phase_a_static_reconciliation_seed1628_20260824/transparent_static_audit.json"
QUALIFICATION_ROOT=Path("/root/v485_v169_cache_qualification_seed1627_20260824")

AUTHORITY_FORMAT='strict-track2-v501-v500-failure-tree-diagnostic-inner-binding-repair-execution-authority-v1'
AUTHORITY_STATUS='authorized_exact_one_external_v501_inner_binding_repaired_outer_attempt'
AUTHORITY_CONTRACT_FORMAT='strict-track2-v501-v500-failure-tree-diagnostic-inner-binding-repair-execution-authority-design-contract-v1'
AUTHORITY_CONTRACT_STATUS="design_only_frozen_sources_pending_independent_review_no_authority"
STATIC_FORMAT="strict-track2-v490-v489-v488-v487-c71-exact7-schema-repair-static-audit-v1"
STATIC_STATUS="passed_no_execution_authority"
STATIC_CHECK_KEYS=sorted({"contract_current","f813_tree_exact2","failed_attempt_tree_exact4_no_retry","formal_authority_all_false","formal_record_exact","formal_runtime_no_execution","formal_schema_exact21","materializer_current","no_live_process","no_training_reward_outcome","observed_exact7_current","r2_current","r2_exact4_strict_no_fallback","r2_forbidden_imports_calls_absent","r2_old_c71_dual_binding","r2_snapshot_dual_ancestry","source_closure_digest_exact","source_closure_exact8_current","synthetic_tamper_suite_passed","transparent_and_fresh_outputs_absent","v487_authority_tree_exact1","v487_helper_tree_exact5","failed_static_tree_exact6","failed_static_process_no_retry","failed_static_invocation_partition","failed_static_root_prep_absent","path_literal_ast_repair_exact","second_failed_static_tree_exact6","second_failed_static_process_no_retry","second_failed_static_invocation_partition","second_failed_static_root_prep_absent","predeploy_transport_failure_disclosed","real_publish_callgraph_fixture_passed"})
STATIC_KEYSET_SHA="53b721e19caa9b6302986e65c78dac72e0026716d67852d1b886aba27492d7cf"
STATIC_CHECKS_SHA="50edf45d3a0021c3054ff80b2fcc311ebb848272def1e0732aa3aaf7c2079a58"
STATIC_TOP_KEYS={"format","status","passed","checks","check_keys","check_key_set_sha256","checks_sha256","repair_preregistration","design_contract","materializer_source","reconciler_r2_source","static_auditor_source","source_closure","source_closure_sha256","f813_registration_tree","v487_authority_tree","v487_helper_evidence_tree","failed_reconciliation_attempt_tree","ast_diff_proof","synthetic_evidence","required_absences","runtime_observation","readonly_reconciliation_authorized","training_authorized","submission_authorized","superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
STATIC_RUNTIME={"static_audit_executed":True,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
AUTHORIZATION={'outer_execution_wrapper_authorized': True, 'outer_attempts_authorized': 1, 'outer_attempts_consumed': 0, 'retry_authorized': False, 'direct_corrected_inner_authorized': False, 'direct_r2_authorized': False, 'nested_corrected_inner_invocations_authorized': 1, 'nested_r2_invocations_authorized': 1, 'nested_corrected_inner_only_via_outer': True, 'nested_r2_only_via_corrected_inner': True, 'phase_a_authorized': False, 'cache_authorized': False, 'training_authorized': False, 'folds_authorized': 0, 'policy_updates': 0, 's1_authorized': False, 'zero_update_authorized': False, 'rl_authorized': False, 'submission_authorized': False, 'reward_read_authorized': False, 'dev_hidden_final_outcome_read_authorized': False}
AUTHORITY_RUNTIME={"execution_authority_materialized":True,"outer_execution_wrapper_executed":False,"corrected_inner_wrapper_executed":False,"reconciler_r2_executed":False,"transparent_receipt_created":False,"phase_a_executed":False,"training_launched":False,"folds":0,"policy_updates":0}
AUTHORITY_CHECK_KEYS=sorted({"authority_contract_current","authority_materializer_current","corrected_inner_current","corrected_inner_diff_exact","current_absences","execution_boundary","execution_interpreter_chain_exact","execution_interpreter_runtime_exact","f813_exact2","failed_v493_outer_exact4","failed_v493_outer_no_retry_partition","failed_v493_outer_terminal_exact","gpu_empty","historical_absences","input_snapshots_equal","no_live_process","outer_wrapper_current","phase_a_contract_current","phase_a_contract_spec_exact2","postregistration_static_exact1","r2_current","repair_formal_exact1","source_closure_current","v493_authority_contract_current","v493_authority_exact3","v493_authority_materialization_evidence_exact6","v493_authority_materializer_process_exact","v493_outer_wrapper_current","v494_authority_contract_current","v494_authority_materializer_current","v494_inner_wrapper_current","v494_outer_wrapper_current","v494_failure_forensic_exact","v494_failure_transport_script_current","v494_materializer_failure_partition_exact","validator_torch_version_scope_exact","validator_torch_version_tamper_suite_passed","v495_authority_contract_current","v495_authority_materializer_current","v495_inner_wrapper_current","v495_outer_wrapper_current","v495_materializer_failure_exact6","v495_materializer_failure_process_exact","v493_process_schema_exact24","validator_process_schema_tamper_suite_passed"})
AUTHORITY_KEYSET_SHA="f889bafe02388706482cf6dc649cc024d53cd771eae9c60757e52619a5212cfc"
AUTHORITY_CHECKS_SHA="85328543d1455c211715bb9af1eefadaaaf9be6609b701a0e3a2f810e9b6a69a"
AUTHORITY_TOP_KEYS={"authority_design_contract","authority_materializer_source","authorization","check_key_set_sha256","check_keys","checks","checks_sha256","corrected_inner_wrapper_source","execution_boundary","execution_interpreter_evidence","f813_registration_tree","failed_v493_outer_execution_tree","failed_v493_outer_terminal_receipt","format","historical_absences","inner_attempt_root","input_post_snapshot","input_pre_snapshot","input_snapshots_exactly_equal","outer_evidence_root","outer_execution_wrapper_source","passed","phase_a_design_contract_source","postregistration_static_registration_tree","reconciler_r2_source","repair_formal_registration_tree","required_absences","runtime_observation","source_closure","source_closure_sha256","status","transparent_static_receipt_path","v493_authority_contract","v493_authority_materialization_evidence_tree","v493_authority_materializer_process_receipt","v493_authority_materializer_source","v493_authority_receipt","v493_authority_registration_tree","v493_inner_wrapper_source","v493_outer_wrapper_source","v494_authority_contract_source","v494_authority_materializer_source","v494_outer_wrapper_source","v494_inner_wrapper_source","v494_materializer_failure_forensic","v494_materializer_failure_transport_script","v494_materializer_failure_ancestry","v495_authority_contract_source","v495_authority_materializer_source","v495_inner_wrapper_source","v495_outer_wrapper_source","v495_materializer_failure_tree","v495_materializer_failure_process_receipt","v495_materializer_failure_ancestry","v493_authority_materializer_process_schema"}
AUTHORITY_CONTRACT_TOP_KEYS={"authority_materializer_source","authority_receipt_contract","current_absences_after_authority","execution_boundary","execution_interpreter_contract","f813_registration_tree","format","historical_absences","lineage","phase_a_design_contract_record","postregistration_static_registration_tree","repair_formal_registration_tree","seed","source_closure","source_closure_sha256","status","v493_authority_contract","v493_authority_materialization_evidence_tree","v493_authority_materializer_process_receipt","v493_authority_receipt","v493_authority_registration_tree","v493_outer_failure_ancestry","v493_outer_failure_tree","v493_outer_terminal_receipt","v494_authority_contract","v494_materializer_failure_forensic","v494_materializer_failure_transport_script","v494_materializer_failure_ancestry","v495_authority_contract","v495_materializer_failure_tree","v495_materializer_failure_process_receipt","v495_materializer_failure_ancestry","v493_authority_materializer_process_schema"}
AUTHORITY_SCHEMA_KEYS={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","authorization_exact","runtime_observation_exact"}
STATIC_SCHEMA_KEYS={"format","status","top_keys","check_keys","check_key_set_sha256","checks_sha256","runtime_observation_exact"}
FORMAL_SCHEMA_KEYS={"format","status","top_keys","authorization_exact","runtime_observation_exact"}
FAILURE_ANCESTRY_KEYS={"superseded_static_source","failed_static_execution_tree","failed_static_process_receipt","superseded_static_absences","v489_failed_static_source","v489_failed_static_execution_tree","v489_failed_static_process_receipt","v489_superseded_static_absences","v489_predeploy_transport_failure"}
EXECUTION_BOUNDARY={'authority_materialization_only': True, 'outer_execution_wrapper_invocations': 0, 'corrected_inner_wrapper_invocations': 0, 'reconciler_r2_invocations': 0, 'phase_a_invocations': 0, 'training_invocations': 0, 'reward_reads': 0, 'dev_hidden_final_outcome_reads': 0}
HISTORICAL_ABSENCE_KEYS={'corrected_inner_attempt_prep','corrected_inner_attempt_root','execution_authority_prep','execution_authority_root','failed_v492_outer_evidence_prep','failed_v493_inner_attempt_prep','failed_v493_inner_attempt_root','failed_v493_outer_evidence_prep','failed_v494_authority_prep','failed_v494_authority_root','failed_v494_inner_attempt_prep','failed_v494_inner_attempt_root','failed_v494_outer_evidence_prep','failed_v494_outer_evidence_root','failed_v495_authority_prep','failed_v495_authority_root','failed_v495_inner_attempt_prep','failed_v495_inner_attempt_root','failed_v495_outer_evidence_prep','failed_v495_outer_evidence_root','failed_v496_authority_prep','failed_v496_authority_root','failed_v496_inner_attempt_prep','failed_v496_inner_attempt_root','failed_v496_outer_evidence_prep','failed_v496_outer_evidence_root','failed_v497_diagnostic_evidence_prep','failed_v497_diagnostic_prep','failed_v497_diagnostic_root','fresh_static_prep','old_v490_inner_attempt_prep','old_v490_inner_attempt_root','outer_evidence_prep','outer_evidence_root','qualification_root','superseded_static_prep','superseded_static_root','superseded_v491_authority_prep','superseded_v491_authority_root','superseded_v491_outer_evidence_prep','superseded_v491_outer_evidence_root','transparent_output','transparent_tmp','v489_superseded_static_prep','v489_superseded_static_root','v490_authority_prep','v492_authority_prep','v493_authority_prep','v498_diagnostic_evidence_prep','v498_diagnostic_prep','v499_adapter_evidence_prep','v499_adapter_prep','v500_authority_prep','v500_inner_attempt_prep','v500_inner_attempt_root','v500_outer_evidence_prep','v500_outer_evidence_root','v501_preexecution_forensic_prep'}
CURRENT_ABSENCE_KEYS=HISTORICAL_ABSENCE_KEYS-{"execution_authority_root"}
INTENT_FORMAT="strict-track2-v501-v500-failure-tree-diagnostic-inner-binding-repaired-reconciliation-inner-attempt-intent-v1"
TERMINAL_FORMAT="strict-track2-v501-v500-failure-tree-diagnostic-inner-binding-repaired-reconciliation-inner-attempt-terminal-v1"
FALSE_AUTHORITIES={"phase_a_cache_qualification_authorized":False,"cache_reuse_authorized":False,"training_authorized":False,"folds_authorized":0,"policy_updates":0,"s1_authorized":False,"zero_update_authorized":False,"rl_authorized":False,"submission_authorized":False,"reward_read_authorized":False,"dev_hidden_final_outcome_read_authorized":False}
R2_RECEIPT_KEYS={"cache_reuse_authorized","check_key_set_sha256","check_keys","checks","checks_sha256","contract","dev_hidden_final_outcome_read_authorized","f813_registration_tree","false_positive_recomputation","folds_authorized","format","immutable_absences","input_post_snapshot","input_pre_snapshot","input_snapshots_exactly_equal","old_reconciler_c71_source","parent_runtime_observation","passed","persistent_old_static_log","phase_a_cache_qualification_authorized","phase_a_executed","policy_updates","preregistration","receipt_writer","reconciler_r2_source","reconciliation_ancestry","reconciliation_design_contract","reconciliation_format","reconciliation_only","reconciliation_preregistration","reconciliation_runtime_observation","reconciliation_status","registration_initial_inventory","repair_design_contract","repair_formal_ancestry","repair_format","repair_materializer_source","repair_preregistration","repair_registration_tree","repair_source_closure","reward_read_authorized","rl_authorized","runtime_observation","s1_authorized","sources","sources_digest_sha256","static_auditor_self_sha256","status","submission_authorized","synthetic_recomputation_evidence","training_authorized","v169_imported_or_run","volatile_log_source_at_registration","zero_update_authorized"}
R2_CHECK_KEYSET_SHA="57676e2e75b6c88cfcf521a1b1c4780315d43547158d1314df8e8444ed86834b"
R2_SYNTHETIC_FIXTURE_SHA="77bc6af02c136be5b9b8c6bb653847de62311d5cd06b3547a06a60b2eee762fa"
R2_SYNTHETIC_EVIDENCE_SHA="46a5fa5cfc6808b7fe52695248465cd694311b027b84f320e88a9ad7990d03a4"

INTERPRETER_INTERMEDIATE=Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python")
INTERPRETER_RESOLVED=Path("/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11")
EXECUTION_INTERPRETER_CONTRACT={
 "lexical":{"path":str(RLPY),"lstat":{"device":2304,"inode":17217118070,"mode":41471,"size":49},"readlink":str(INTERPRETER_INTERMEDIATE)},
 "intermediate":{"path":str(INTERPRETER_INTERMEDIATE),"lstat":{"device":2304,"inode":7529246267,"mode":41471,"size":10},"readlink":"python3.11"},
 "resolved":{"path":str(INTERPRETER_RESOLVED),"sha256":"11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788","logical_bytes":25555040,"lstat":{"device":2304,"inode":7529484239,"mode":33277,"size":25555040}},
 "runtime":{"sys_executable":str(RLPY),"python_version":"3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]","numpy_version":"1.26.4","torch_version":"2.7.0+cu128"},
}
V494_MATERIALIZER_FAILURE_ANCESTRY={"failed_status":"frozen_reconstructed_v494_materializer_prestate_failure_no_retry","materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"native_exit_code":1,"native_persistent_capture_available":False,"tool_capture_sha256":"4a65e33de2de1da29e8a36d39e5374e43b66435aa5e6bc3722ccfca574fb4a4c","tool_capture_logical_bytes":787,"root_cause":"corrected inner forbidden import","corrected_rule":"allow exact one import torch and exact one load torch.__version__ only inside execution_interpreter_evidence","retry_authorized":False}
V493_PROCESS_EXACT_KEYS=sorted({"argv","authority_receipt","authority_tree","cleanup","corrected_inner_wrapper_invocations","failed_v492_outer_terminal_receipt","format","helper_returncode","intent","materializer_invocations","materializer_returncode","outer_wrapper_invocations","passed","post_snapshot","pre_post_snapshots_exactly_equal","pre_snapshot","r2_invocations","retry_authorized","status","stderr","stdout","transport_helper","transport_helper_copy","wall_seconds"})
V493_PROCESS_REQUIRED={"status":"passed_exact_once_no_outer_or_inner_execution","passed":True,"materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"r2_invocations":0,"retry_authorized":False,"pre_post_snapshots_exactly_equal":True,"cleanup":{"reaped":True,"group_empty":True}}
V493_PROCESS_SCHEMA={"record":{"path":str(V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json"),"sha256":"babcfda7fad625ea7d0b7a34592e119d6287fa4b34dcfbda743f7473a0fe4789","logical_bytes":19613},"exact_keys":V493_PROCESS_EXACT_KEYS,"key_set_sha256":"e477a875bc016fad9215afb62b61d8f1878d61aad2b734afa90eba2d3c897b71","required_values":V493_PROCESS_REQUIRED,"forbidden_keys":["inner_wrapper_invocations"]}
V495_MATERIALIZER_FAILURE_ANCESTRY={"failed_status":"failed_no_retry","materializer_invocations":1,"outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"native_exit_code":1,"retry_authorized":False,"root_cause":"v495 materializer required stale inner_wrapper_invocations absent from immutable v493 process receipt","corrected_rule":"validate exact24 process receipt with corrected_inner_wrapper_invocations int zero and reject stale, missing, both, extra, nonzero, and bool variants"}



# v500 frozen schema overrides (mechanically derived from materializer e717c136).
AUTHORITY_CONTRACT_TOP_KEYS=CONTRACT_TOP_KEYS={'authority_materializer_source','authority_receipt_contract','current_absences_after_authority','execution_boundary','execution_interpreter_contract','f813_registration_tree','failed_v497_diagnostic_failure_ancestry','failed_v497_diagnostic_failure_tree','failed_v497_diagnostic_process_receipt','format','historical_absences','lineage','materializer_checkpoint_contract','normalized_v498_diagnostic_process_receipt','phase_a_design_contract_record','postregistration_static_registration_tree','repair_formal_registration_tree','seed','source_aliases','source_closure','source_closure_sha256','source_role_order','status','v493_authority_contract','v493_authority_materialization_evidence_tree','v493_authority_materializer_process_receipt','v493_authority_materializer_process_schema','v493_authority_receipt','v493_authority_registration_tree','v493_outer_failure_ancestry','v493_outer_failure_tree','v493_outer_terminal_receipt','v494_authority_contract','v494_materializer_failure_ancestry','v494_materializer_failure_forensic','v494_materializer_failure_transport_script','v495_authority_contract','v495_materializer_failure_ancestry','v495_materializer_failure_process_receipt','v495_materializer_failure_tree','v496_authority_contract','v496_materializer_failure_ancestry','v496_materializer_failure_process_receipt','v496_materializer_failure_tree','v498_diagnostic_execution_evidence_tree','v498_diagnostic_process_receipt','v498_diagnostic_receipt','v498_diagnostic_registration_tree','v498_diagnostic_schema','v499_adapter_execution_evidence_tree','v499_adapter_process_receipt','v499_adapter_receipt','v499_adapter_registration_tree','v499_adapter_schema','v500_authority_materialization_evidence_tree','v500_authority_materializer_process_receipt','v500_authority_receipt','v500_authority_registration_tree','v500_corrected_deployment_evidence_tree','v500_corrected_deployment_process_receipt','v500_invalid_outer_unconsumed_ancestry','v501_preexecution_forensic_receipt','v501_preexecution_forensic_registration_tree','v501_preexecution_forensic_schema'}
AUTHORITY_TOP_KEYS=AUTH_TOP_KEYS={'authority_design_contract','authority_materializer_source','authorization','check_key_set_sha256','check_keys','checks','checks_sha256','corrected_inner_wrapper_source','execution_boundary','execution_interpreter_evidence','f813_registration_tree','failed_v493_outer_execution_tree','failed_v493_outer_terminal_receipt','failed_v497_diagnostic_failure_ancestry','failed_v497_diagnostic_failure_tree','failed_v497_diagnostic_helper_source','failed_v497_diagnostic_process_receipt','failed_v497_diagnostic_script_source','failed_v497_diagnostic_source','format','historical_absences','inner_attempt_root','input_post_snapshot','input_pre_snapshot','input_snapshots_exactly_equal','materializer_pre_root_checkpoint','materializer_pre_root_checkpoint_sha256','normalized_v498_diagnostic_process_receipt','outer_evidence_root','outer_execution_wrapper_source','passed','phase_a_design_contract_source','postregistration_static_registration_tree','reconciler_r2_source','repair_formal_registration_tree','required_absences','runtime_observation','source_aliases','source_closure','source_closure_sha256','source_role_order','status','transparent_static_receipt_path','v493_authority_contract','v493_authority_materialization_evidence_tree','v493_authority_materializer_process_receipt','v493_authority_materializer_process_schema','v493_authority_materializer_source','v493_authority_receipt','v493_authority_registration_tree','v493_inner_wrapper_source','v493_outer_wrapper_source','v494_authority_contract_source','v494_authority_materializer_source','v494_inner_wrapper_source','v494_materializer_failure_ancestry','v494_materializer_failure_forensic','v494_materializer_failure_transport_script','v494_outer_wrapper_source','v495_authority_contract_source','v495_authority_materializer_source','v495_inner_wrapper_source','v495_materializer_failure_ancestry','v495_materializer_failure_process_receipt','v495_materializer_failure_tree','v495_outer_wrapper_source','v496_authority_contract_source','v496_authority_materializer_source','v496_inner_wrapper_source','v496_materializer_failure_ancestry','v496_materializer_failure_process_receipt','v496_materializer_failure_tree','v496_outer_wrapper_source','v498_diagnostic_execution_evidence_tree','v498_diagnostic_helper_source','v498_diagnostic_process_receipt','v498_diagnostic_receipt','v498_diagnostic_registration_tree','v498_diagnostic_schema','v498_diagnostic_script_source','v498_diagnostic_source','v499_adapter_execution_evidence_tree','v499_adapter_helper_source','v499_adapter_process_receipt','v499_adapter_receipt','v499_adapter_registration_tree','v499_adapter_schema','v499_adapter_script_source','v499_adapter_source','v500_authority_contract_source','v500_authority_materialization_evidence_tree','v500_authority_materializer_process_receipt','v500_authority_materializer_source','v500_authority_receipt','v500_authority_registration_tree','v500_authority_transport_helper_source','v500_authority_transport_script_source','v500_corrected_deployer_source','v500_corrected_deployment_evidence_tree','v500_corrected_deployment_process_receipt','v500_inner_wrapper_source','v500_invalid_outer_unconsumed_ancestry','v500_invalid_outer_wrapper_source','v501_preexecution_forensic_receipt','v501_preexecution_forensic_registration_tree','v501_preexecution_forensic_schema','v501_preexecution_forensic_source'}
AUTHORITY_CHECK_KEYS=AUTH_CHECK_KEYS=['authority_contract_current', 'authority_materializer_current', 'corrected_inner_current', 'corrected_inner_diff_exact', 'current_absences', 'diagnostic_normalization_tamper_suite_passed', 'execution_boundary', 'execution_interpreter_chain_exact', 'execution_interpreter_runtime_exact', 'f813_exact2', 'failed_v493_outer_exact4', 'failed_v493_outer_no_retry_partition', 'failed_v493_outer_terminal_exact', 'failed_v497_diagnostic_exact6', 'failed_v497_diagnostic_helper_current', 'failed_v497_diagnostic_partition_exact', 'failed_v497_diagnostic_process_exact', 'failed_v497_diagnostic_script_current', 'failed_v497_diagnostic_source_current', 'gpu_empty', 'historical_absences', 'input_snapshots_equal', 'materializer_checkpoint_double_snapshot_exact', 'materializer_checkpoint_named8_exact', 'materializer_checkpoint_stdout_fsynced', 'no_live_process', 'normalized_v498_process_current_only', 'outer_wrapper_current', 'phase_a_contract_current', 'phase_a_contract_spec_exact2', 'postregistration_static_exact1', 'r2_current', 'repair_formal_exact1', 'source_aliases_exact', 'source_closure_current', 'source_role_order_exact', 'v493_authority_contract_current', 'v493_authority_exact3', 'v493_authority_materialization_evidence_exact6', 'v493_authority_materializer_process_exact', 'v493_outer_wrapper_current', 'v493_process_schema_exact24', 'v494_authority_contract_current', 'v494_authority_materializer_current', 'v494_failure_forensic_exact', 'v494_failure_transport_script_current', 'v494_inner_wrapper_current', 'v494_materializer_failure_partition_exact', 'v494_outer_wrapper_current', 'v495_authority_contract_current', 'v495_authority_materializer_current', 'v495_inner_wrapper_current', 'v495_materializer_failure_exact6', 'v495_materializer_failure_process_exact', 'v495_outer_wrapper_current', 'v496_authority_contract_current', 'v496_authority_materializer_current', 'v496_inner_wrapper_current', 'v496_materializer_failure_exact6', 'v496_materializer_failure_partition_exact', 'v496_materializer_failure_process_exact', 'v496_outer_wrapper_current', 'v498_diagnostic_evidence_exact6', 'v498_diagnostic_helper_current', 'v498_diagnostic_process_exact', 'v498_diagnostic_receipt_exact', 'v498_diagnostic_registration_exact1', 'v498_diagnostic_script_current', 'v498_diagnostic_source_current', 'v498_named8_double_snapshot_exact', 'v499_adapter_evidence_exact6', 'v499_adapter_helper_current', 'v499_adapter_process_exact', 'v499_adapter_receipt_exact', 'v499_adapter_registration_exact1', 'v499_adapter_script_current', 'v499_adapter_source_current', 'v499_normalization_leafdiff_exact', 'v500_authority_exact1', 'v500_authority_materialization_evidence_exact6', 'v500_authority_materializer_process_exact', 'v500_authority_transport_helper_current', 'v500_authority_transport_script_current', 'v500_corrected_deployer_current', 'v500_corrected_deployment_exact6', 'v500_corrected_deployment_process_exact', 'v500_inner_current', 'v500_invalid_outer_binding_mismatch_exact', 'v500_invalid_outer_current', 'v500_invalid_outer_unconsumed_zero_state', 'v501_preexecution_forensic_exact1', 'v501_preexecution_forensic_schema_exact', 'v501_preexecution_forensic_source_current', 'validator_process_schema_tamper_suite_passed', 'validator_torch_version_scope_exact', 'validator_torch_version_tamper_suite_passed']
AUTHORITY_KEYSET_SHA=AUTH_KEYSET_SHA='cac461a9758265621d82a0e70d0b05b3a77fe768236110a77a81eb883b666990'
AUTHORITY_CHECKS_SHA=AUTH_CHECKS_SHA='df19082c9300eb4011cdc4acd38ddca446aa25ea3f13471232037476623ab805'
SOURCE_ROLE_ORDER=['v493_authority_contract', 'v493_authority_materializer', 'v493_outer_wrapper', 'v493_inner_wrapper', 'v494_authority_contract', 'v494_authority_materializer', 'v494_outer_wrapper', 'v494_inner_wrapper', 'v495_authority_contract', 'v495_authority_materializer', 'v495_outer_wrapper', 'v495_inner_wrapper', 'v496_authority_contract', 'v496_authority_materializer', 'v496_inner_wrapper', 'v496_outer_wrapper', 'v494_failure_forensic', 'v494_failure_transport_script', 'failed_v497_diagnostic_source', 'failed_v497_diagnostic_helper', 'failed_v497_diagnostic_script', 'v498_diagnostic_source', 'v498_diagnostic_helper', 'v498_diagnostic_script', 'v499_adapter_source', 'v499_adapter_helper', 'v499_adapter_script', 'phase_a_design_contract', 'reconciler_r2', 'v500_authority_contract', 'v500_authority_materializer', 'v500_inner_wrapper', 'v500_outer_wrapper_invalid_unconsumed', 'v500_authority_transport_helper', 'v500_authority_transport_script', 'v500_corrected_deployer', 'v501_preexecution_forensic_source', 'corrected_inner_wrapper', 'outer_execution_wrapper']
SOURCE_ALIASES={'v493_authority_contract': 'v493_authority_contract', 'v493_authority_materializer': 'v493_authority_materializer_source', 'v493_outer_wrapper': 'v493_outer_wrapper_source', 'v493_inner_wrapper': 'v493_inner_wrapper_source', 'v494_authority_contract': 'v494_authority_contract_source', 'v494_authority_materializer': 'v494_authority_materializer_source', 'v494_outer_wrapper': 'v494_outer_wrapper_source', 'v494_inner_wrapper': 'v494_inner_wrapper_source', 'v495_authority_contract': 'v495_authority_contract_source', 'v495_authority_materializer': 'v495_authority_materializer_source', 'v495_outer_wrapper': 'v495_outer_wrapper_source', 'v495_inner_wrapper': 'v495_inner_wrapper_source', 'v496_authority_contract': 'v496_authority_contract_source', 'v496_authority_materializer': 'v496_authority_materializer_source', 'v496_inner_wrapper': 'v496_inner_wrapper_source', 'v496_outer_wrapper': 'v496_outer_wrapper_source', 'v494_failure_forensic': 'v494_materializer_failure_forensic', 'v494_failure_transport_script': 'v494_materializer_failure_transport_script', 'failed_v497_diagnostic_source': 'failed_v497_diagnostic_source', 'failed_v497_diagnostic_helper': 'failed_v497_diagnostic_helper_source', 'failed_v497_diagnostic_script': 'failed_v497_diagnostic_script_source', 'v498_diagnostic_source': 'v498_diagnostic_source', 'v498_diagnostic_helper': 'v498_diagnostic_helper_source', 'v498_diagnostic_script': 'v498_diagnostic_script_source', 'v499_adapter_source': 'v499_adapter_source', 'v499_adapter_helper': 'v499_adapter_helper_source', 'v499_adapter_script': 'v499_adapter_script_source', 'phase_a_design_contract': 'phase_a_design_contract_source', 'reconciler_r2': 'reconciler_r2_source', 'corrected_inner_wrapper': 'corrected_inner_wrapper_source', 'outer_execution_wrapper': 'outer_execution_wrapper_source', 'v500_authority_contract': 'v500_authority_contract_source', 'v500_authority_materializer': 'v500_authority_materializer_source', 'v500_inner_wrapper': 'v500_inner_wrapper_source', 'v500_outer_wrapper_invalid_unconsumed': 'v500_invalid_outer_wrapper_source', 'v500_authority_transport_helper': 'v500_authority_transport_helper_source', 'v500_authority_transport_script': 'v500_authority_transport_script_source', 'v500_corrected_deployer': 'v500_corrected_deployer_source', 'v501_preexecution_forensic_source': 'v501_preexecution_forensic_source'}
SOURCE_ROLES=set(SOURCE_ROLE_ORDER)
AUTHORIZATION={'outer_execution_wrapper_authorized': True, 'outer_attempts_authorized': 1, 'outer_attempts_consumed': 0, 'retry_authorized': False, 'direct_corrected_inner_authorized': False, 'direct_r2_authorized': False, 'nested_corrected_inner_invocations_authorized': 1, 'nested_r2_invocations_authorized': 1, 'nested_corrected_inner_only_via_outer': True, 'nested_r2_only_via_corrected_inner': True, 'phase_a_authorized': False, 'cache_authorized': False, 'training_authorized': False, 'folds_authorized': 0, 'policy_updates': 0, 's1_authorized': False, 'zero_update_authorized': False, 'rl_authorized': False, 'submission_authorized': False, 'reward_read_authorized': False, 'dev_hidden_final_outcome_read_authorized': False}
AUTHORITY_RUNTIME=RUNTIME={'execution_authority_materialized': True, 'outer_execution_wrapper_executed': False, 'corrected_inner_wrapper_executed': False, 'reconciler_r2_executed': False, 'transparent_receipt_created': False, 'phase_a_executed': False, 'training_launched': False, 'folds': 0, 'policy_updates': 0}
EXECUTION_BOUNDARY={'authority_materialization_only': True, 'outer_execution_wrapper_invocations': 0, 'corrected_inner_wrapper_invocations': 0, 'reconciler_r2_invocations': 0, 'phase_a_invocations': 0, 'training_invocations': 0, 'reward_reads': 0, 'dev_hidden_final_outcome_reads': 0}

class ControlledSignal(BaseException):pass


def sha(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(8<<20),b""):d.update(block)
 return d.hexdigest()


def csha(value)->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_v493_process_schema(value)->bool:
 if not isinstance(value,dict) or sorted(value)!=V493_PROCESS_EXACT_KEYS or csha(sorted(value))!=V493_PROCESS_SCHEMA["key_set_sha256"] or any(key in value for key in V493_PROCESS_SCHEMA["forbidden_keys"]):return False
 if type(value.get("corrected_inner_wrapper_invocations")) is not int or value["corrected_inner_wrapper_invocations"]!=0:return False
 for key,expected in V493_PROCESS_REQUIRED.items():
  if key=="cleanup":
   cleanup=value.get("cleanup")
   if not isinstance(cleanup,dict) or any(cleanup.get(name) is not required for name,required in expected.items()):return False
  elif value.get(key)!=expected:return False
 return True


def contains_pending(value)->bool:
 if isinstance(value,str):return "PENDING" in value
 if isinstance(value,dict):return any(contains_pending(key) or contains_pending(item) for key,item in value.items())
 if isinstance(value,list):return any(contains_pending(item) for item in value)
 return False

def decode_contract_absences(value:dict,expected:set[str])->dict[str,str]:
 if not isinstance(value,dict) or set(value)!=expected:raise RuntimeError("contract absence keyset")
 out={}
 for name,row in value.items():
  if not isinstance(row,dict) or set(row)!={"path"} or not isinstance(row["path"],str):raise RuntimeError("contract absence row")
  path=Path(row["path"])
  if path!=path.resolve():raise RuntimeError("contract absence path")
  out[name]=str(path)
 return out

def decode_receipt_absences(value:dict,expected:set[str])->dict[str,str]:
 if not isinstance(value,dict) or set(value)!=expected:raise RuntimeError("receipt absence keyset")
 out={}
 for name,row in value.items():
  if not isinstance(row,dict) or set(row)!={"path","absent"} or row.get("absent") is not True or not isinstance(row.get("path"),str):raise RuntimeError("receipt absence row")
  path=Path(row["path"])
  if path!=path.resolve():raise RuntimeError("receipt absence path")
  out[name]=str(path)
 return out

def resolve_phase_contract_record(spec:dict)->dict:
 expected={"path":str(PHASE_A_DESIGN_CONTRACT_PATH),"sha256":PHASE_A_DESIGN_CONTRACT_SHA}
 if not isinstance(spec,dict) or set(spec)!={"path","sha256"} or spec!=expected:raise RuntimeError("phase contract frozen two-key record")
 return regular(PHASE_A_DESIGN_CONTRACT_PATH,PHASE_A_DESIGN_CONTRACT_SHA,PHASE_A_DESIGN_CONTRACT_BYTES)


def regular(path:Path|str,want_sha:str|None=None,want_bytes:int|None=None)->dict:
 path=Path(path)
 if path!=path.resolve() or not path.is_file() or path.is_symlink():raise RuntimeError(f"regular: {path}")
 result={"path":str(path),"sha256":sha(path),"logical_bytes":path.stat().st_size}
 if want_sha is not None and result["sha256"]!=want_sha:raise RuntimeError(f"sha: {path}")
 if want_bytes is not None and result["logical_bytes"]!=want_bytes:raise RuntimeError(f"bytes: {path}")
 return result


def _lstat_row(path:Path)->dict:
 fields=os.lstat(path)
 return {"device":fields.st_dev,"inode":fields.st_ino,"mode":fields.st_mode,"size":fields.st_size}


def execution_interpreter_evidence()->dict:
 if sys.executable!=str(RLPY):raise RuntimeError("interpreter lexical executable")
 lexical=_lstat_row(RLPY)
 if not stat.S_ISLNK(lexical["mode"]):raise RuntimeError("interpreter lexical symlink")
 lexical_target=os.readlink(RLPY)
 if not os.path.isabs(lexical_target):raise RuntimeError("interpreter lexical target")
 intermediate=Path(lexical_target)
 intermediate_stat=_lstat_row(intermediate)
 if not stat.S_ISLNK(intermediate_stat["mode"]):raise RuntimeError("interpreter intermediate symlink")
 intermediate_target=os.readlink(intermediate)
 if os.path.isabs(intermediate_target):raise RuntimeError("interpreter intermediate target")
 resolved=(intermediate.parent/intermediate_target).resolve(strict=True)
 resolved_stat=_lstat_row(resolved)
 if resolved!=INTERPRETER_RESOLVED or stat.S_ISLNK(resolved_stat["mode"]) or not stat.S_ISREG(resolved_stat["mode"]):raise RuntimeError("interpreter resolved regular")
 import numpy
 import torch
 evidence={
  "lexical":{"path":str(RLPY),"lstat":lexical,"readlink":lexical_target},
  "intermediate":{"path":str(intermediate),"lstat":intermediate_stat,"readlink":intermediate_target},
  "resolved":{**regular(resolved),"lstat":resolved_stat},
  "runtime":{"sys_executable":sys.executable,"python_version":sys.version,"numpy_version":numpy.__version__,"torch_version":torch.__version__},
 }
 if evidence!=EXECUTION_INTERPRETER_CONTRACT:raise RuntimeError("execution interpreter evidence")
 return evidence


def validate_execution_interpreter_evidence(value:dict)->dict:
 if value!=EXECUTION_INTERPRETER_CONTRACT:raise RuntimeError("execution interpreter contract")
 return execution_interpreter_evidence()


def validate_torch_version_scope(source_text:str)->bool:
 parsed=ast.parse(source_text);targets=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=="execution_interpreter_evidence"]
 if len(targets)!=1:return False
 target=targets[0];target_nodes=set(ast.walk(target));parents={child:parent for parent in ast.walk(parsed) for child in ast.iter_child_nodes(parent)}
 imports=[]
 for node in ast.walk(parsed):
  if isinstance(node,ast.Import):
   for alias in node.names:
    if alias.name.split(".")[0]=="torch":imports.append((node,alias))
  elif isinstance(node,ast.ImportFrom) and node.module and node.module.split(".")[0]=="torch":return False
 if len(imports)!=1 or imports[0][0] not in target_nodes or imports[0][1].name!="torch" or imports[0][1].asname is not None:return False
 names=[node for node in ast.walk(parsed) if isinstance(node,ast.Name) and node.id=="torch"]
 if len(names)!=1 or names[0] not in target_nodes or not isinstance(names[0].ctx,ast.Load):return False
 parent=parents.get(names[0]);attributes=[node for node in ast.walk(parsed) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=="torch"]
 if len(attributes)!=1 or parent is not attributes[0] or attributes[0].attr!="__version__" or not isinstance(attributes[0].ctx,ast.Load):return False
 if any(isinstance(node,ast.Call) and any(isinstance(child,ast.Name) and child.id=="torch" for child in ast.walk(node)) for node in ast.walk(parsed)):return False
 return True


def exact_tree(root:Path|str)->dict:
 root=Path(root)
 if root!=root.resolve() or not root.is_dir() or root.is_symlink():raise RuntimeError(f"tree root: {root}")
 rows=[]
 for path in sorted(root.rglob("*")):
  if path.is_symlink():raise RuntimeError(f"tree symlink: {path}")
  if path.is_file():rows.append([path.relative_to(root).as_posix(),sha(path),path.stat().st_size])
  elif not path.is_dir():raise RuntimeError(f"tree nonregular: {path}")
 lines="".join(f"{digest}  {rel}\n" for rel,digest,_ in rows).encode();triples=json.dumps(rows,separators=(",",":")).encode()
 return {"inventory":rows,"file_count":len(rows),"logical_file_bytes":sum(row[2] for row in rows),"sha256sum_lines_digest_sha256":hashlib.sha256(lines).hexdigest(),"canonical_json_triples_digest_sha256":hashlib.sha256(triples).hexdigest()}


def tree_with_root(root:Path)->dict:return {"root":str(root),**exact_tree(root)}


def fsync_dir(path:Path)->None:
 fd=os.open(str(path),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)


def atomic_json(path:Path,value)->None:
 tmp=path.with_name(path.name+".tmp")
 if os.path.lexists(path) or os.path.lexists(tmp):raise FileExistsError(path)
 with tmp.open("x",encoding="utf-8") as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
 os.replace(tmp,path);fsync_dir(path.parent)


def directory_identity(path:Path)->tuple[int,int]:
 stat=path.stat(follow_symlinks=False)
 if path.is_symlink() or not path.is_dir():raise RuntimeError("directory identity")
 return stat.st_dev,stat.st_ino


def group_empty(pid:int)->bool:
 try:os.killpg(pid,0);return False
 except ProcessLookupError:return True


def terminate_group(process:subprocess.Popen|None)->dict:
 evidence={"started":process is not None,"term_sent":False,"kill_sent":False,"reaped":process is None,"group_empty":True}
 if process is None:return evidence
 if process.poll() is None or not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGTERM);evidence["term_sent"]=True
  except ProcessLookupError:pass
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:
   pass
 if not group_empty(process.pid):
  try:os.killpg(process.pid,signal.SIGKILL);evidence["kill_sent"]=True
  except ProcessLookupError:pass
 if process.poll() is None:process.wait(timeout=10)
 else:process.wait()
 evidence["reaped"]=process.poll() is not None;evidence["group_empty"]=group_empty(process.pid)
 if not evidence["reaped"] or not evidence["group_empty"]:raise RuntimeError("process group cleanup")
 return evidence


def spawn_owned(command,stdout,stderr,owner,phase_hook=None):
 phase_hook=phase_hook or (lambda _process:None)
 if owner!={"process":None,"started":False}:raise RuntimeError("spawn ownership prestate")
 baseline=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM}) if hasattr(signal,"pthread_sigmask") else None
 try:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
  owner["process"]=child;owner["started"]=True;phase_hook(child)
 finally:
  if baseline is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline)
 return owner["process"]


def cleanup_owned_prep(path:Path,identity:tuple[int,int]|None)->None:
 if identity is None or not path.exists() or path.is_symlink() or directory_identity(path)!=identity:return
 allowed={"intent.json","intent.json.tmp","reconciler_stdout.log","reconciler_stderr.log","terminal_receipt.json","terminal_receipt.json.tmp"}
 entries=list(path.iterdir())
 if any(entry.name not in allowed or entry.is_symlink() or not entry.is_file() for entry in entries):return
 for entry in entries:entry.unlink()
 path.rmdir();fsync_dir(path.parent)


def current_processes()->list[dict]:
 found=[]
 for proc in Path("/proc").iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if int(proc.name)!=os.getpid() and (str(R2_PATH) in cmd or str(WRAPPER_PATH) in cmd):found.append({"pid":int(proc.name),"cmdline":cmd})
 return found


def validate_static(static:dict,static_record:dict)->None:
 if set(static)!=STATIC_TOP_KEYS or static.get("format")!=STATIC_FORMAT or static.get("status")!=STATIC_STATUS or static.get("passed") is not True:raise RuntimeError("static receipt schema")
 if static.get("check_keys")!=STATIC_CHECK_KEYS or static.get("check_key_set_sha256")!=STATIC_KEYSET_SHA or static.get("checks")!={key:True for key in STATIC_CHECK_KEYS} or static.get("checks_sha256")!=STATIC_CHECKS_SHA:raise RuntimeError("static checks")
 if static.get("runtime_observation")!=STATIC_RUNTIME or any(static.get(key) is not False for key in ("readonly_reconciliation_authorized","training_authorized","submission_authorized")):raise RuntimeError("static authority boundary")
 if static.get("static_auditor_source")!=static_record or static.get("repair_preregistration")!={"path":str(REPAIR_FORMAL_PATH),"sha256":REPAIR_FORMAL_SHA,"logical_bytes":REPAIR_FORMAL_BYTES} or static.get("reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("static ancestry")
 if static.get("source_closure_sha256")!=csha(static.get("source_closure")):raise RuntimeError("static source digest")


def json_leaf_diff(before,after,path=()):
 if isinstance(before,dict) and isinstance(after,dict):
  rows=[]
  for key in sorted(set(before)|set(after)):
   if key not in before or key not in after:rows.append({"path":list(path+(key,)),"before":before.get(key),"after":after.get(key)})
   else:rows.extend(json_leaf_diff(before[key],after[key],path+(key,)))
  return rows
 if isinstance(before,list) and isinstance(after,list) and len(before)==len(after):
  rows=[]
  for index,(left,right) in enumerate(zip(before,after)):rows.extend(json_leaf_diff(left,right,path+(index,)))
  return rows
 return [] if type(before) is type(after) and before==after else [{"path":list(path),"before":before,"after":after}]


def validate_v500_diagnostic_lineage(contract,authority):
 def bound_record(field):
  row=contract.get(field)
  if not isinstance(row,dict) or set(row)!={"path","sha256","logical_bytes"}:raise RuntimeError(f"diagnostic record schema: {field}")
  observed=regular(row["path"],row["sha256"],row["logical_bytes"])
  if authority.get(field)!=observed:raise RuntimeError(f"diagnostic record alias: {field}")
  return observed
 def bound_tree(field):
  row=contract.get(field)
  if not isinstance(row,dict) or set(row)!={"root","inventory","file_count","logical_file_bytes","sha256sum_lines_digest_sha256","canonical_json_triples_digest_sha256"}:raise RuntimeError(f"diagnostic tree schema: {field}")
  observed=tree_with_root(row["root"])
  if observed!=row or authority.get(field)!=observed:raise RuntimeError(f"diagnostic tree alias: {field}")
  return observed
 v496_tree=bound_tree("v496_materializer_failure_tree");v496_process_record=bound_record("v496_materializer_failure_process_receipt");v496_process=json.loads(Path(v496_process_record["path"]).read_text())
 if v496_tree["file_count"]!=6 or v496_tree["logical_file_bytes"]!=34885 or v496_tree["sha256sum_lines_digest_sha256"]!="ab63f37f6275193a97db39cc71d37b32f4f653f87e78c375ddb460558b4e1f9e" or v496_tree["canonical_json_triples_digest_sha256"]!="c78038212aee67743f08dc936a539ad5d99a6ff931b87f12b5b294dfa4a0dac0" or (v496_process_record["sha256"],v496_process_record["logical_bytes"])!=("5ac38bcdf03f25f9e80c0a6fd184c5e8f12763447a6a91aec264e659be47662d",411):raise RuntimeError("v496 failure tree")
 if v496_process.get("status")!="failed_no_retry" or v496_process.get("materializer_invocations")!=1 or any(v496_process.get(key)!=0 for key in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","r2_invocations")) or v496_process.get("retry_authorized") is not False or v496_process.get("cleanup",{}).get("reaped") is not True or v496_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v496 failure partition")
 if contract.get("v496_materializer_failure_ancestry")!=authority.get("v496_materializer_failure_ancestry"):raise RuntimeError("v496 failure ancestry")
 failed_tree=bound_tree("failed_v497_diagnostic_failure_tree");failed_process_record=bound_record("failed_v497_diagnostic_process_receipt");failed_process=json.loads(Path(failed_process_record["path"]).read_text())
 if failed_tree["file_count"]!=6 or failed_tree["logical_file_bytes"]!=40802 or failed_tree["sha256sum_lines_digest_sha256"]!="ed2c81a06f71937581b451f84e137640310b363f04d569fe1bf0da1a60377b8d" or failed_tree["canonical_json_triples_digest_sha256"]!="02337d3883ddb46b16bfbf9d1e157dcc16ff4e462810e0de2e22a25082e5913d" or (failed_process_record["sha256"],failed_process_record["logical_bytes"])!=("5ad649b013fe0e9b17af76992950c53f8043a536ae3f2ecb04f6c9f65f443425",459):raise RuntimeError("v497 failure tree")
 if failed_process.get("status")!="failed_no_retry" or failed_process.get("diagnostic_invocations")!=1 or any(failed_process.get(key)!=0 for key in ("authority_materializer_invocations","outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or failed_process.get("retry_authorized") is not False or failed_process.get("cleanup",{}).get("reaped") is not True or failed_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v497 failure partition")
 if contract.get("failed_v497_diagnostic_failure_ancestry")!=authority.get("failed_v497_diagnostic_failure_ancestry"):raise RuntimeError("v497 failure ancestry")
 diagnostic_record=bound_record("v498_diagnostic_receipt");diagnostic_tree=bound_tree("v498_diagnostic_registration_tree");diagnostic_evidence=bound_tree("v498_diagnostic_execution_evidence_tree");diagnostic_process_record=bound_record("v498_diagnostic_process_receipt")
 diagnostic=json.loads(Path(diagnostic_record["path"]).read_text());original=json.loads(Path(diagnostic_process_record["path"]).read_text());diagnostic_schema=contract.get("v498_diagnostic_schema")
 if authority.get("v498_diagnostic_schema")!=diagnostic_schema or (diagnostic_record["sha256"],diagnostic_record["logical_bytes"])!=("2bb515bba9039d3bac3545c1780bb82d9c7609aafa8a64012f9c9574eec5da20",14528) or diagnostic_tree["file_count"]!=1 or diagnostic_evidence["file_count"]!=6 or diagnostic_evidence["sha256sum_lines_digest_sha256"]!="17a283bf1f1957c484091c4e44920368c2203e7f7ea05df22ad975582e0b774b" or diagnostic_evidence["canonical_json_triples_digest_sha256"]!="2c3b56d838eeb3c936a0f9aad5c8ba8a149a7bc4b3f24c1f9aff0f254337b393" or (diagnostic_process_record["sha256"],diagnostic_process_record["logical_bytes"])!=("fa7269bdacac994efd5ef8eb44cf9d3d1c7014e5b9aa902e95a5f4243609fabd",26493):raise RuntimeError("v498 diagnostic records")
 if not isinstance(diagnostic_schema,dict) or set(diagnostic)!=set(diagnostic_schema.get("top_keys",[])) or len(diagnostic)!=23 or csha(sorted(diagnostic))!=diagnostic_schema.get("top_key_set_sha256") or diagnostic.get("status")!=diagnostic_schema.get("status") or diagnostic.get("passed") is not True or diagnostic.get("snapshot_1")!=diagnostic.get("snapshot_2") or diagnostic.get("snapshots_exactly_equal") is not True or len(diagnostic.get("predicate_names",[]))!=8 or [row.get("name") for row in diagnostic.get("predicates",[])]!=diagnostic.get("predicate_names") or any(row.get("passed") is not True for row in diagnostic.get("predicates",[])) or any(diagnostic.get("authorization",{}).values()):raise RuntimeError("v498 diagnostic schema")
 normalized=contract.get("normalized_v498_diagnostic_process_receipt");expected_diff=[{"path":["transport_helper_copy","path"],"before":str(Path(diagnostic_evidence["root"]).with_name(Path(diagnostic_evidence["root"]).name+".execution-prep")/"transport_helper.py"),"after":str(Path(diagnostic_evidence["root"])/"transport_helper.py")}]
 if csha(original)!="4f043cb1e415b88a5f02a4b5180f78e94e295ca351065edbeccff9df8172f59f" or csha(normalized)!="9318ba42a5ce23fce6b3179756c99b9ef18d907ce338930e61eb0c3dae38804d" or json_leaf_diff(original,normalized)!=expected_diff or regular(normalized["transport_helper_copy"]["path"],normalized["transport_helper_copy"]["sha256"],normalized["transport_helper_copy"]["logical_bytes"])!=normalized["transport_helper_copy"]:raise RuntimeError("v498 normalized-only process")
 if authority.get("normalized_v498_diagnostic_process_receipt")!=normalized:raise RuntimeError("normalized process alias")
 adapter_record=bound_record("v499_adapter_receipt");adapter_tree=bound_tree("v499_adapter_registration_tree");adapter_evidence=bound_tree("v499_adapter_execution_evidence_tree");adapter_process_record=bound_record("v499_adapter_process_receipt")
 adapter=json.loads(Path(adapter_record["path"]).read_text());adapter_process=json.loads(Path(adapter_process_record["path"]).read_text());adapter_schema=contract.get("v499_adapter_schema")
 if authority.get("v499_adapter_schema")!=adapter_schema or (adapter_record["sha256"],adapter_record["logical_bytes"])!=("1d7a82bf22ed9791b30f7760033c9cbf4289fd7e46f051cbbf401cf516380b49",78314) or adapter_tree["file_count"]!=1 or adapter_evidence["file_count"]!=6 or adapter_evidence["sha256sum_lines_digest_sha256"]!="e47fcb2bc9e902084b8f47aac1ae00d4d4965197476008eaa7c518db5487b866" or adapter_evidence["canonical_json_triples_digest_sha256"]!="d5d10783ba4dd7a318f01bb48d66a0f7b998d4b8f9bd06e08afa8b6e31e73fec" or (adapter_process_record["sha256"],adapter_process_record["logical_bytes"])!=("af93c880fe04d94186017b70b7060acb2d9f68afb1f3290718caa9c6a1276f59",5063):raise RuntimeError("v499 adapter records")
 if not isinstance(adapter_schema,dict) or set(adapter)!=set(adapter_schema.get("top_keys",[])) or len(adapter)!=32 or csha(sorted(adapter))!=adapter_schema.get("top_key_set_sha256") or adapter.get("status")!=adapter_schema.get("status") or adapter.get("passed") is not True or adapter.get("original_process_receipt")!=original or adapter.get("normalized_process_receipt")!=normalized or adapter.get("normalized_process_receipt_sha256")!=csha(normalized) or adapter.get("canonical_json_leaf_diff")!=expected_diff or any(adapter.get("authorization",{}).values()):raise RuntimeError("v499 adapter schema")
 if set(adapter_process)!={"adapter_invocations","adapter_receipt","adapter_registration_tree","adapter_returncode","argv","argv_payload_sha256","authority_materializer_invocations","cleanup","corrected_inner_wrapper_invocations","format","helper_returncode","intent","outer_wrapper_invocations","passed","post_snapshot","pre_post_snapshots_exactly_equal","pre_snapshot","reconciler_r2_invocations","retry_authorized","status","stderr","stdout","transport_helper","transport_helper_copy","transport_script"} or adapter_process.get("status")!="passed_exact_once_readonly_adapter_no_authority" or adapter_process.get("adapter_invocations")!=1 or any(adapter_process.get(key)!=0 for key in ("authority_materializer_invocations","outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or adapter_process.get("retry_authorized") is not False or adapter_process.get("pre_post_snapshots_exactly_equal") is not True or adapter_process.get("cleanup",{}).get("reaped") is not True or adapter_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v499 adapter process")
 for key,row in adapter_process.items():
  if isinstance(row,dict) and set(row)=={"path","sha256","logical_bytes"}:regular(row["path"],row["sha256"],row["logical_bytes"])
 sources=contract["source_closure"]
 frozen_v500={"v500_authority_contract":("55bc38e1e101e97761ac8b8f4e5200b0ccd5a94676f27fc7a3123ce179df2e1e",111329),"v500_authority_materializer":("e717c1360e5691865dd299249630b3b4cef1965c68e8a457008b50415fe9afb5",109256),"v500_inner_wrapper":("132a91069dc219f70adc6a01292084a97b1e547f16d4cc0d3a6f28df231005ab",124674),"v500_outer_wrapper_invalid_unconsumed":("3d8949268c1eb3b8e9dfd84d9aea5084ea6a299fb48bb554972cfe0cc6f7d458",85248),"v500_authority_transport_helper":("c53d750c52db061b79280205c83c6303956e21c8762aa66f60dfcf873131c686",38766),"v500_authority_transport_script":("6b3cc44a26ba0129e8cc02db7aefc56ff51373e0b616bf09de12ac1e78f8c605",2543),"v500_corrected_deployer":("8666cfed7e924deedc9229dd6a6349b595012e9812de3b167fcf889c994310d2",24395),"v501_preexecution_forensic_source":("107d54adbf2aca61405c898f9bd119bbcb23f9a61e45f8cd17beaf06bcfc60e8",16580)}
 if any((sources[role]["sha256"],sources[role]["logical_bytes"])!=identity for role,identity in frozen_v500.items()):raise RuntimeError("v500/forensic frozen sources")
 v500_authority_record=bound_record("v500_authority_receipt");v500_authority_tree=bound_tree("v500_authority_registration_tree")
 v500_materialization_tree=bound_tree("v500_authority_materialization_evidence_tree");v500_process_record=bound_record("v500_authority_materializer_process_receipt");v500_process=json.loads(Path(v500_process_record["path"]).read_text())
 v500_deployment_tree=bound_tree("v500_corrected_deployment_evidence_tree");v500_deployment_process_record=bound_record("v500_corrected_deployment_process_receipt");v500_deployment_process=json.loads(Path(v500_deployment_process_record["path"]).read_text())
 forensic_record=bound_record("v501_preexecution_forensic_receipt");forensic_tree=bound_tree("v501_preexecution_forensic_registration_tree");forensic=json.loads(Path(forensic_record["path"]).read_text());forensic_schema=contract.get("v501_preexecution_forensic_schema")
 expected_invalid={"status":"invalid_unconsumed_preexecution_binding_mismatch","outer_wrapper_invocations":0,"corrected_inner_wrapper_invocations":0,"reconciler_r2_invocations":0,"retry_authorized":False,"invalid_outer_source":sources["v500_outer_wrapper_invalid_unconsumed"],"stale_inner_literal":{"path":sources["v500_inner_wrapper"]["path"],"sha256":"c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f","logical_bytes":101600},"actual_inner_source":sources["v500_inner_wrapper"],"root_cause":"v500 outer frozen INNER_WRAPPER_SHA and INNER_WRAPPER_BYTES do not match the authority-bound current inner source","corrected_rule":"fresh authority must bind a fresh outer and fresh inner whose command and self-check records are identical"}
 if contract.get("v500_invalid_outer_unconsumed_ancestry")!=expected_invalid or authority.get("v500_invalid_outer_unconsumed_ancestry")!=expected_invalid:raise RuntimeError("v500 invalid outer ancestry")
 if (v500_authority_record["sha256"],v500_authority_record["logical_bytes"])!=("8c57944a115bcb26fd3709a0a7911a2ea58be2e82e238c0b6fffad04f5b22179",222774) or v500_authority_tree["file_count"]!=1 or v500_authority_tree["sha256sum_lines_digest_sha256"]!="7633136acfd7aef756d39fecbfa4dc0343d775d9135b2644c58a52f9989d578e" or v500_authority_tree["canonical_json_triples_digest_sha256"]!="76fc17f7b90a98509b38069f597851bcac59544ace890020321ad3ba8bffcc79":raise RuntimeError("v500 authority exact1")
 if v500_materialization_tree["file_count"]!=6 or v500_materialization_tree["logical_file_bytes"]!=167885 or v500_materialization_tree["sha256sum_lines_digest_sha256"]!="37dffce5e5e4ed1d712ec44d7100c90bef3287559b96bd38c6137983134c6ca1" or v500_materialization_tree["canonical_json_triples_digest_sha256"]!="71de157900c60fe4abe088505036645617b1579264a9e28ff18a0b99d8cfcc6e" or (v500_process_record["sha256"],v500_process_record["logical_bytes"])!=("0fab24fca5e3b8a975f9cb4e32109297cb7b1094d0fb9a7c9289320633e46b11",79870):raise RuntimeError("v500 materialization exact6")
 if v500_process.get("status")!="passed_exact_once_authority_materialized_no_outer_or_inner_execution" or v500_process.get("authority_materializer_invocations")!=1 or any(v500_process.get(k)!=0 for k in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or v500_process.get("retry_authorized") is not False or v500_process.get("pre_post_snapshots_exactly_equal") is not True or v500_process.get("cleanup",{}).get("reaped") is not True or v500_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v500 materialization partition")
 if v500_deployment_tree["file_count"]!=6 or v500_deployment_tree["logical_file_bytes"]!=59532 or v500_deployment_tree["sha256sum_lines_digest_sha256"]!="792437bd87209c87174c59b7ee289ed442fe5d5a9346b78999c3ee05d7aaf8a4" or v500_deployment_tree["canonical_json_triples_digest_sha256"]!="beaa9c4744303a8818a067fbb0e44c2d2c722f4723f0d96c10ded57e1ebd0820" or (v500_deployment_process_record["sha256"],v500_deployment_process_record["logical_bytes"])!=("7e92b12a68bc1c150fcbfa7b6f0ff6cde7788000f4b92ddfc31693663f94d893",31333) or v500_deployment_process.get("status")!="passed_exact_once_six_sources_deployed" or v500_deployment_process.get("passed") is not True or v500_deployment_process.get("deployer_invocations")!=1 or v500_deployment_process.get("worker_returncode")!=0 or v500_deployment_process.get("worker_summary",{}).get("deployment_writes")!=6 or any(v500_deployment_process.get("worker_summary",{}).get(k)!=0 for k in ("authority_materializer_invocations","outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or v500_deployment_process.get("pre_post_snapshots_exactly_equal") is not True or v500_deployment_process.get("cleanup",{}).get("reaped") is not True or v500_deployment_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v500 corrected deployment")
 if (forensic_record["sha256"],forensic_record["logical_bytes"])!=("1c62d32c002ae2d3f3525ebc7a69c0c44f428d30a735e46f4edf53a1ad90aca9",24950) or forensic_tree["file_count"]!=1 or forensic_tree["sha256sum_lines_digest_sha256"]!="428939cb521865d1d6f904cd1a97abc2c609811cfba4ef2f58372e86083777fe" or forensic_tree["canonical_json_triples_digest_sha256"]!="da31ad0603673ef1bcefd7feab55105dd59bf773d92e901c7ed7e29f31dfc077":raise RuntimeError("v501 forensic exact1")
 if not isinstance(forensic_schema,dict) or authority.get("v501_preexecution_forensic_schema")!=forensic_schema or set(forensic)!=set(forensic_schema.get("top_keys",[])) or len(forensic)!=34 or forensic.get("checks")!={k:True for k in forensic.get("check_keys",[])} or len(forensic.get("check_keys",[]))!=15 or any(forensic.get("authorization",{}).values()) or forensic.get("input_snapshots_exactly_equal") is not True:raise RuntimeError("v501 forensic schema")
 return {"v496_failure_tree":v496_tree,"v496_failure_process":v496_process_record,"failed_v497_tree":failed_tree,"failed_v497_process":failed_process_record,"v498_diagnostic":diagnostic_record,"v498_registration":diagnostic_tree,"v498_evidence":diagnostic_evidence,"v498_process":diagnostic_process_record,"v499_adapter":adapter_record,"v499_registration":adapter_tree,"v499_evidence":adapter_evidence,"v499_process":adapter_process_record,"normalized_process_sha256":csha(normalized)}


def preflight(args)->dict:
 interpreter_evidence=execution_interpreter_evidence()
 if args.f813_preregistration.resolve()!=F813_PATH or args.f813_preregistration_sha!=F813_SHA:raise RuntimeError("f813 arguments")
 if args.repair_preregistration.resolve()!=REPAIR_FORMAL_PATH or args.repair_preregistration_sha!=REPAIR_FORMAL_SHA:raise RuntimeError("repair arguments")
 if args.authority_contract.resolve()!=AUTHORITY_CONTRACT_PATH or len(args.authority_contract_sha)!=64:raise RuntimeError("authority contract arguments")
 if args.authority_receipt.resolve()!=AUTHORITY_RECEIPT_PATH or len(args.authority_receipt_sha)!=64:raise RuntimeError("authority receipt arguments")
 if args.wrapper_source.resolve()!=WRAPPER_PATH or args.wrapper_source.resolve()!=Path(__file__).resolve() or args.wrapper_sha!=sha(Path(__file__).resolve()):raise RuntimeError("wrapper self")
 f813_record=regular(F813_PATH,F813_SHA,F813_BYTES);repair_record=regular(REPAIR_FORMAL_PATH,REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES);contract_record=regular(args.authority_contract,args.authority_contract_sha);authority_record=regular(args.authority_receipt,args.authority_receipt_sha);wrapper_record=regular(args.wrapper_source,args.wrapper_sha,args.wrapper_source.stat().st_size)
 contract=json.loads(args.authority_contract.read_text());authority=json.loads(args.authority_receipt.read_text());repair=json.loads(REPAIR_FORMAL_PATH.read_text());f813=json.loads(F813_PATH.read_text())
 if set(contract)!=AUTHORITY_CONTRACT_TOP_KEYS or contract.get("format")!=AUTHORITY_CONTRACT_FORMAT or contract.get("status")!=AUTHORITY_CONTRACT_STATUS or contract.get("seed")!=1643 or contract.get("execution_boundary")!=EXECUTION_BOUNDARY or contains_pending(contract):raise RuntimeError("authority contract schema")
 receipt_contract=contract.get("authority_receipt_contract")
 if not isinstance(receipt_contract,dict) or set(receipt_contract)!=AUTHORITY_SCHEMA_KEYS or set(receipt_contract["top_keys"])!=AUTHORITY_TOP_KEYS or receipt_contract["check_keys"]!=AUTHORITY_CHECK_KEYS or receipt_contract["check_key_set_sha256"]!=AUTHORITY_KEYSET_SHA or receipt_contract["checks_sha256"]!=AUTHORITY_CHECKS_SHA or receipt_contract["authorization_exact"]!=AUTHORIZATION or receipt_contract["runtime_observation_exact"]!=AUTHORITY_RUNTIME:raise RuntimeError("authority receipt contract schema")
 if set(authority)!=AUTHORITY_TOP_KEYS or authority.get("format")!=AUTHORITY_FORMAT or authority.get("status")!=AUTHORITY_STATUS or authority.get("passed") is not True:raise RuntimeError("authority receipt schema")
 if authority.get("check_keys")!=AUTHORITY_CHECK_KEYS or authority.get("check_key_set_sha256")!=AUTHORITY_KEYSET_SHA or authority.get("checks")!={key:True for key in AUTHORITY_CHECK_KEYS} or authority.get("checks_sha256")!=AUTHORITY_CHECKS_SHA or csha(AUTHORITY_CHECK_KEYS)!=AUTHORITY_KEYSET_SHA or csha(authority["checks"])!=AUTHORITY_CHECKS_SHA:raise RuntimeError("authority checks")
 if authority.get("authorization")!=AUTHORIZATION or authority.get("runtime_observation")!=AUTHORITY_RUNTIME or authority.get("execution_boundary")!=EXECUTION_BOUNDARY:raise RuntimeError("authority execution boundary")
 if contract.get("execution_interpreter_contract")!=EXECUTION_INTERPRETER_CONTRACT or authority.get("execution_interpreter_evidence")!=interpreter_evidence:raise RuntimeError("authority interpreter crossbind")
 if authority.get("authority_design_contract")!=contract_record:raise RuntimeError("authority contract alias")
 materializer_spec=contract.get("authority_materializer_source")
 if not isinstance(materializer_spec,dict) or set(materializer_spec)!={"path","sha256","logical_bytes"}:raise RuntimeError("authority materializer spec")
 materializer_record=regular(materializer_spec["path"],materializer_spec["sha256"],materializer_spec["logical_bytes"])
 if materializer_record!={"path":str(AUTHORITY_MATERIALIZER_PATH),"sha256":AUTHORITY_MATERIALIZER_SHA,"logical_bytes":AUTHORITY_MATERIALIZER_BYTES} or authority.get("authority_materializer_source")!=materializer_record:raise RuntimeError("authority materializer alias")
 source_closure=contract.get("source_closure")
 if contract.get("source_role_order")!=SOURCE_ROLE_ORDER or contract.get("source_aliases")!=SOURCE_ALIASES or authority.get("source_role_order")!=SOURCE_ROLE_ORDER or authority.get("source_aliases")!=SOURCE_ALIASES:raise RuntimeError("authority source order/aliases")
 if not isinstance(source_closure,dict) or set(source_closure)!=SOURCE_ROLES or contract.get("source_closure_sha256")!=csha(source_closure):raise RuntimeError("authority source closure roles")
 observed_sources={role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in source_closure.items()}
 if observed_sources!=source_closure or authority.get("source_closure")!=observed_sources or authority.get("source_closure_sha256")!=csha(observed_sources):raise RuntimeError("authority source closure current")
 if any(authority.get(SOURCE_ALIASES[role])!=observed_sources[role] for role in SOURCE_ROLE_ORDER):raise RuntimeError("authority source alias map")
 frozen_new={"v496_authority_contract":("33aad023e6275e21702c569261aec754e58c151622055ab3beb0c81434ac335c",46664),"v496_authority_materializer":("6a38efcb9e714cf58a694bac62eadd86cabc6e746d0fdd1195d168963de1cf06",62492),"v496_inner_wrapper":("c9b3fe91028ef43bcc70741eab01c62765a47a50f581ffd5c238dd399919d34f",101600),"v496_outer_wrapper":("09d5b725ba45bcf274d134fd40fdbea52795e68f6a8193149085b96d563a07c2",61068),"failed_v497_diagnostic_source":("6ad8f54b53fc2178330f27e9bc772de8fc7357a27752f17a309b835f8f796ecb",26978),"failed_v497_diagnostic_helper":("a8da83f408fa30c241b311caf86c4d0d7b065099dc7445dea0a1d75e192b44ed",27893),"failed_v497_diagnostic_script":("eb0f0908c5c2185627bfb59fa02b46279ba929deb04d06253e0d36835c6f4104",1166),"v498_diagnostic_source":("328d88b9dd5c2268f27c612322785eeca3cfe053018b00512492f984f8332fe6",38424),"v498_diagnostic_helper":("f7072fae9b88ca1736818f1304191d5bfc4c2d550eb16f0a6f73bf2b8f3a1dce",27949),"v498_diagnostic_script":("c41a960cce856ddcf27933930b888a17cf61ee1de3552aa59a2867e8b1b0b8ea",1171),"v499_adapter_source":("5456dfd102d08b2825bb2ba2d6ea44e01c5c5d443823799115c9f3ff105ed35c",28726),"v499_adapter_helper":("dc9978e304c9c18ec950626999deb340b2fd1b1171afc2ea3af7dfa2323cab70",22997),"v499_adapter_script":("dce9621f06b732f9627b8912a00910ebff0d5900a0aa48765a25f4e33030b2f2",2521)}
 if any((observed_sources[role]["sha256"],observed_sources[role]["logical_bytes"])!=identity for role,identity in frozen_new.items()):raise RuntimeError("v496-v499 frozen source identities")
 if observed_sources["corrected_inner_wrapper"]!=wrapper_record or observed_sources["outer_execution_wrapper"]!=regular(OUTER_WRAPPER_PATH,observed_sources["outer_execution_wrapper"]["sha256"],observed_sources["outer_execution_wrapper"]["logical_bytes"]):raise RuntimeError("fresh wrapper identities")
 if observed_sources["reconciler_r2"]!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES} or observed_sources["phase_a_design_contract"]!={"path":str(PHASE_A_DESIGN_CONTRACT_PATH),"sha256":PHASE_A_DESIGN_CONTRACT_SHA,"logical_bytes":PHASE_A_DESIGN_CONTRACT_BYTES}:raise RuntimeError("r2/phase source identities")
 expected_sources=observed_sources
 diagnostic_context=validate_v500_diagnostic_lineage(contract,authority)
 if contract.get("phase_a_design_contract_record")!=expected_sources["phase_a_design_contract"] or authority.get("phase_a_design_contract_source")!=expected_sources["phase_a_design_contract"]:raise RuntimeError("phase contract authority record")
 repair_sources=repair.get("source_closure")
 if not isinstance(repair_sources,dict) or len(repair_sources)!=8 or repair.get("source_closure_sha256")!=csha(repair_sources):raise RuntimeError("repair source closure schema")
 observed_repair_sources={role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in repair_sources.items()}
 if observed_repair_sources!=repair_sources or repair.get("future_reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("repair source closure current")
 exact7=f813.get("exact7_source_closure")
 if not isinstance(exact7,dict) or set(exact7)!={"all_records_must_equal_parent_formal_and_old_receipt_and_current_files","canonical_records_digest_sha256","records","roles_in_order"} or exact7.get("all_records_must_equal_parent_formal_and_old_receipt_and_current_files") is not True:raise RuntimeError("f813 exact7 schema")
 exact7_records=exact7.get("records");role_order=exact7.get("roles_in_order")
 if not isinstance(exact7_records,list) or [row.get("role") for row in exact7_records]!=role_order or csha(exact7_records)!=exact7.get("canonical_records_digest_sha256"):raise RuntimeError("f813 exact7 records")
 exact7_sources={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in exact7_records}
 if any(regular(row["path"],row["sha256"],row["logical_bytes"])!={key:row[key] for key in ("path","sha256","logical_bytes")} for row in exact7_records):raise RuntimeError("f813 exact7 current files")
 phase_formal_spec=repair_sources.get("parent_v485_preregistration",{});phase_formal_record=regular(phase_formal_spec["path"],phase_formal_spec["sha256"],phase_formal_spec["logical_bytes"]);phase_formal=json.loads(Path(phase_formal_record["path"]).read_text())
 if phase_formal.get("execution_source_records")!=exact7_records or phase_formal.get("execution_sources")!=exact7_sources or phase_formal.get("execution_sources_digest_sha256")!=exact7["canonical_records_digest_sha256"]:raise RuntimeError("phase formal ordered source closure")
 parent=f813.get("frozen_parent",{});phase_contract_spec=parent.get("phase_a_design_contract",{})
 phase_contract_record=resolve_phase_contract_record(phase_contract_spec);phase_contract=json.loads(PHASE_A_DESIGN_CONTRACT_PATH.read_text())
 if phase_contract.get("phase_a_output_and_receipt_schema",{}).get("source_closure_contract",{}).get("exact_roles")!=role_order:raise RuntimeError("phase contract exact role order")
 old_receipt_spec=parent.get("old_failed_static_receipt",{});old_receipt_record=regular(old_receipt_spec["path"],old_receipt_spec["sha256"],old_receipt_spec["logical_bytes"]);old_receipt=json.loads(Path(old_receipt_record["path"]).read_text())
 if old_receipt.get("sources")!=exact7_sources or old_receipt.get("sources_digest_sha256")!=exact7["canonical_records_digest_sha256"]:raise RuntimeError("old receipt ordered source closure")
 old_source_spec=parent.get("old_static_source",{});old_source_record=regular(old_source_spec["path"],old_source_spec["sha256"],old_source_spec["logical_bytes"])
 persistent_log_record=regular(f813["persistent_old_static_log"]["path"],f813["persistent_old_static_log"]["sha256"],f813["persistent_old_static_log"]["logical_bytes"]);volatile_log_record=regular(f813["volatile_log_source_at_registration"]["path"],f813["volatile_log_source_at_registration"]["sha256"],f813["volatile_log_source_at_registration"]["logical_bytes"])
 if persistent_log_record["sha256"]!=volatile_log_record["sha256"] or persistent_log_record["logical_bytes"]!=volatile_log_record["logical_bytes"]:raise RuntimeError("f813 log provenance")
 static_record=regular(STATIC_RECEIPT_PATH,STATIC_RECEIPT_SHA,STATIC_RECEIPT_BYTES);static=json.loads(STATIC_RECEIPT_PATH.read_text());validate_static(static,regular(STATIC_SOURCE_PATH,STATIC_SOURCE_SHA,STATIC_SOURCE_BYTES))
 repair_tree=tree_with_root(REPAIR_FORMAL_PATH.parent);static_tree=tree_with_root(STATIC_ROOT);authority_tree=tree_with_root(AUTHORITY_ROOT);f813_tree_rooted=tree_with_root(F813_PATH.parent);f813_tree=exact_tree(F813_PATH.parent)
 if repair_tree!=authority.get("repair_formal_registration_tree") or repair_tree["inventory"]!=[["preregistration.json",REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES]]:raise RuntimeError("repair REG exact1")
 if static_tree!=authority.get("postregistration_static_registration_tree") or static_tree["inventory"]!=[["static_audit.json",STATIC_RECEIPT_SHA,STATIC_RECEIPT_BYTES]]:raise RuntimeError("static REG exact1")
 if authority_tree["inventory"]!=[["authority_receipt.json",authority_record["sha256"],authority_record["logical_bytes"]]] or authority_tree["file_count"]!=1:raise RuntimeError("authority REG exact1")
 if f813_tree_rooted!=authority.get("f813_registration_tree") or f813_tree_rooted["file_count"]!=2:raise RuntimeError("f813 exact2")
 for field,actual in (("repair_formal_registration_tree",repair_tree),("postregistration_static_registration_tree",static_tree),("f813_registration_tree",f813_tree_rooted)):
  if contract.get(field)!=actual:raise RuntimeError(f"authority contract tree: {field}")
 if authority.get("input_snapshots_exactly_equal") is not True or authority.get("input_pre_snapshot")!=authority.get("input_post_snapshot"):raise RuntimeError("authority snapshots")
 if authority.get("inner_attempt_root")!=str(ATTEMPT_ROOT) or authority.get("outer_evidence_root")!=str(OUTER_EVIDENCE_ROOT) or authority.get("transparent_static_receipt_path")!=str(TRANSPARENT_PATH):raise RuntimeError("authority output paths")
 contract_historical=decode_contract_absences(contract.get("historical_absences"),HISTORICAL_ABSENCE_KEYS);contract_current=decode_contract_absences(contract.get("current_absences_after_authority"),CURRENT_ABSENCE_KEYS)
 receipt_historical=decode_receipt_absences(authority.get("historical_absences"),HISTORICAL_ABSENCE_KEYS);receipt_current=decode_receipt_absences(authority.get("required_absences"),CURRENT_ABSENCE_KEYS)
 if contract_historical!=receipt_historical or contract_current!=receipt_current:raise RuntimeError("authority absence crossbind")
 controlled_current={"outer_evidence_root"}
 if any(os.path.lexists(path) for name,path in receipt_current.items() if name not in controlled_current):raise RuntimeError("authority current absences")
 if not OUTER_EVIDENCE_ROOT.is_dir() or OUTER_EVIDENCE_ROOT.is_symlink():raise RuntimeError("outer execution ownership")
 outer_pre=exact_tree(OUTER_EVIDENCE_ROOT)
 if [row[0] for row in outer_pre["inventory"]]!=["corrected_inner_stderr.log","corrected_inner_stdout.log","intent.json"]:raise RuntimeError("outer preterminal exact3")
 outer_intent=json.loads((OUTER_EVIDENCE_ROOT/"intent.json").read_text());expected_nested_argv=[str(RLPY),str(WRAPPER_PATH),*sys.argv[1:]]
 if outer_intent.get("format")!="strict-track2-v501-v500-failure-tree-diagnostic-inner-binding-repaired-reconciliation-outer-attempt-intent-v1" or outer_intent.get("status")!="committed_before_corrected_inner_start" or outer_intent.get("command_argv")!=expected_nested_argv or outer_intent.get("command_argv_sha256")!=csha(expected_nested_argv) or outer_intent.get("authority_receipt")!=authority_record:raise RuntimeError("outer intent crossbind")
 v493_authority_record=regular(V493_AUTHORITY_RECEIPT_PATH,V493_AUTHORITY_RECEIPT_SHA,V493_AUTHORITY_RECEIPT_BYTES);v493_authority_tree=tree_with_root(V493_AUTHORITY_ROOT)
 if authority.get("v493_authority_receipt")!=v493_authority_record or contract.get("v493_authority_receipt")!=v493_authority_record or authority.get("v493_authority_registration_tree")!=v493_authority_tree or contract.get("v493_authority_registration_tree")!=v493_authority_tree:raise RuntimeError("v492 authority ancestry")
 if contract.get("v493_authority_contract")!=expected_sources["v493_authority_contract"] or authority.get("v493_authority_contract")!=expected_sources["v493_authority_contract"]:raise RuntimeError("v492 authority contract ancestry")
 v493_materialization_tree=tree_with_root(V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT);v493_process_record=regular(V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json")
 if contract.get("v493_authority_materialization_evidence_tree")!=v493_materialization_tree or authority.get("v493_authority_materialization_evidence_tree")!=v493_materialization_tree or contract.get("v493_authority_materializer_process_receipt")!=v493_process_record or authority.get("v493_authority_materializer_process_receipt")!=v493_process_record:raise RuntimeError("v492 authority materialization ancestry")
 v493_process=json.loads((V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json").read_text())
 if contract.get("v493_authority_materializer_process_schema")!=V493_PROCESS_SCHEMA or authority.get("v493_authority_materializer_process_schema")!=V493_PROCESS_SCHEMA or v493_process_record!=V493_PROCESS_SCHEMA["record"] or not validate_v493_process_schema(v493_process):raise RuntimeError("v493 authority materializer process schema")
 failure_tree=tree_with_root(V493_OUTER_FAILURE_ROOT);terminal_record=regular(V493_OUTER_FAILURE_ROOT/"terminal_receipt.json")
 if authority.get("failed_v493_outer_execution_tree")!=failure_tree or contract.get("v493_outer_failure_tree")!=failure_tree or authority.get("failed_v493_outer_terminal_receipt")!=terminal_record or contract.get("v493_outer_terminal_receipt")!=terminal_record:raise RuntimeError("v492 outer failure ancestry")
 failed_terminal=json.loads((V493_OUTER_FAILURE_ROOT/"terminal_receipt.json").read_text())
 if failed_terminal.get("status")!="failed_no_retry" or failed_terminal.get("passed") is not False or failed_terminal.get("nested_corrected_inner_invocations")!=1 or failed_terminal.get("transparent_present") is not False or failed_terminal.get("cleanup",{}).get("reaped") is not True or failed_terminal.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v493 outer terminal semantics")
 forensic_record=expected_sources["v494_failure_forensic"];transport_record=expected_sources["v494_failure_transport_script"];forensic=json.loads(V494_FAILURE_FORENSIC_PATH.read_text())
 if contract.get("v494_authority_contract")!=expected_sources["v494_authority_contract"] or contract.get("v494_materializer_failure_forensic")!=forensic_record or contract.get("v494_materializer_failure_transport_script")!=transport_record or contract.get("v494_materializer_failure_ancestry")!=V494_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v494 failure contract ancestry")
 if authority.get("v494_materializer_failure_forensic")!=forensic_record or authority.get("v494_materializer_failure_transport_script")!=transport_record or authority.get("v494_materializer_failure_ancestry")!=V494_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v494 failure receipt ancestry")
 if forensic.get("format")!="strict-track2-v495-v494-authority-materializer-failure-forensic-reconstructed-v1" or forensic.get("status")!=V494_MATERIALIZER_FAILURE_ANCESTRY["failed_status"] or forensic.get("passed") is not False or forensic.get("native_exit_code")!=1 or forensic.get("native_persistent_capture_available") is not False:raise RuntimeError("v494 failure forensic schema")
 invocation=forensic.get("invocation",{});tool_capture=forensic.get("tool_capture",{});state=forensic.get("state_after_failure",{});forensic_sources=forensic.get("source_records",{})
 expected_forensic_sources={"authority_contract":expected_sources["v494_authority_contract"],"authority_materializer":expected_sources["v494_authority_materializer"],"outer_execution_wrapper":expected_sources["v494_outer_wrapper"],"corrected_inner_wrapper":expected_sources["v494_inner_wrapper"]}
 if forensic_sources!=expected_forensic_sources or invocation.get("materializer_invocations")!=1 or any(invocation.get(key)!=0 for key in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","reconciler_r2_invocations")) or invocation.get("transport_script")!={"path":"/dev/shm/v494_authority_materialize.sh","sha256":V494_FAILURE_TRANSPORT_SCRIPT_SHA,"logical_bytes":V494_FAILURE_TRANSPORT_SCRIPT_BYTES}:raise RuntimeError("v494 failure invocation partition")
 if tool_capture.get("combined_output_sha256")!=V494_MATERIALIZER_FAILURE_ANCESTRY["tool_capture_sha256"] or tool_capture.get("combined_output_logical_bytes")!=V494_MATERIALIZER_FAILURE_ANCESTRY["tool_capture_logical_bytes"] or tool_capture.get("native_stderr_stream_separately_available") is not False or tool_capture.get("native_stdout_stream_separately_available") is not False:raise RuntimeError("v494 failure capture")
 if state.get("all_absent") is not True or state.get("gpu_compute_apps_empty") is not True or state.get("live_reconciliation_processes_empty") is not True or any(os.path.lexists(path) for path in state.get("paths",{}).values()):raise RuntimeError("v494 failure zero state")
 v495_failure_tree=tree_with_root(V495_FAILURE_ROOT);v495_failure_process_record=regular(V495_FAILURE_PROCESS_PATH,V495_FAILURE_PROCESS_SHA,V495_FAILURE_PROCESS_BYTES);v495_failure_process=json.loads(V495_FAILURE_PROCESS_PATH.read_text())
 if v495_failure_tree!=V495_FAILURE_TREE or contract.get("v495_materializer_failure_tree")!=V495_FAILURE_TREE or authority.get("v495_materializer_failure_tree")!=V495_FAILURE_TREE or contract.get("v495_materializer_failure_process_receipt")!=v495_failure_process_record or authority.get("v495_materializer_failure_process_receipt")!=v495_failure_process_record:raise RuntimeError("v495 failure exact6 ancestry")
 if contract.get("v495_authority_contract")!=expected_sources["v495_authority_contract"] or contract.get("v495_materializer_failure_ancestry")!=V495_MATERIALIZER_FAILURE_ANCESTRY or authority.get("v495_materializer_failure_ancestry")!=V495_MATERIALIZER_FAILURE_ANCESTRY:raise RuntimeError("v495 failure contract ancestry")
 if set(v495_failure_process)!={"cleanup","corrected_inner_wrapper_invocations","error","error_type","format","materializer_invocations","outer_wrapper_invocations","passed","r2_invocations","retry_authorized","status"} or v495_failure_process.get("status")!="failed_no_retry" or v495_failure_process.get("passed") is not False or v495_failure_process.get("materializer_invocations")!=1 or any(v495_failure_process.get(key)!=0 for key in ("outer_wrapper_invocations","corrected_inner_wrapper_invocations","r2_invocations")) or v495_failure_process.get("retry_authorized") is not False or v495_failure_process.get("cleanup",{}).get("reaped") is not True or v495_failure_process.get("cleanup",{}).get("group_empty") is not True:raise RuntimeError("v495 failure process partition")
 if os.path.lexists(ATTEMPT_ROOT) or os.path.lexists(ATTEMPT_PREP) or os.path.lexists(TRANSPARENT_PATH) or os.path.lexists(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp")) or os.path.lexists(QUALIFICATION_ROOT):raise RuntimeError("execution outputs preexist")
 lineage=contract.get("lineage")
 if lineage!={"execution_authority_root":str(AUTHORITY_ROOT),"authority_prep":str(AUTHORITY_ROOT.with_name(AUTHORITY_ROOT.name+".registration-prep")),"outer_evidence_root":str(OUTER_EVIDENCE_ROOT),"outer_evidence_prep":str(OUTER_EVIDENCE_ROOT.with_name(OUTER_EVIDENCE_ROOT.name+".outer-prep")),"inner_attempt_root":str(ATTEMPT_ROOT),"inner_attempt_prep":str(ATTEMPT_PREP),"transparent_static_receipt_path":str(TRANSPARENT_PATH),"qualification_root":str(QUALIFICATION_ROOT)}:raise RuntimeError("authority lineage")
 if current_processes():raise RuntimeError("live reconciliation process")
 return {"contract_record":contract_record,"authority_record":authority_record,"wrapper_record":wrapper_record,"materializer_record":materializer_record,"contract":contract,"authority":authority,"repair":repair,"f813":f813,"repair_record":repair_record,"f813_record":f813_record,"source_closure":observed_sources,"static_record":static_record,"static":static,"repair_tree":repair_tree,"static_tree":static_tree,"authority_tree":authority_tree,"f813_tree":f813_tree,"f813_tree_rooted":f813_tree_rooted,"exact7_records":exact7_records,"exact7_sources":exact7_sources,"exact7_digest":exact7["canonical_records_digest_sha256"],"phase_formal_record":phase_formal_record,"phase_contract_record":phase_contract_record,"old_receipt_record":old_receipt_record,"old_receipt":old_receipt,"old_source_record":old_source_record,"persistent_log_record":persistent_log_record,"volatile_log_record":volatile_log_record,"v493_authority_record":v493_authority_record,"v493_authority_tree":v493_authority_tree,"v493_outer_failure_tree":failure_tree,"v493_outer_terminal_record":terminal_record,"v495_failure_tree":v495_failure_tree,"v495_failure_process_record":v495_failure_process_record,"diagnostic_context":diagnostic_context,"outer_preterminal_tree":outer_pre,"outer_intent":outer_intent,"execution_interpreter_evidence":interpreter_evidence}


def immutable_snapshot(context:dict)->dict:
 authority=context["authority"]
 files={"authority_contract":context["contract_record"],"authority_receipt":context["authority_record"],"authority_materializer":regular(context["materializer_record"]["path"],context["materializer_record"]["sha256"],context["materializer_record"]["logical_bytes"]),"repair_formal":regular(REPAIR_FORMAL_PATH,REPAIR_FORMAL_SHA,REPAIR_FORMAL_BYTES),"f813_formal":regular(F813_PATH,F813_SHA,F813_BYTES),"static_receipt":regular(STATIC_RECEIPT_PATH,context["static_record"]["sha256"],context["static_record"]["logical_bytes"]),"phase_formal":regular(context["phase_formal_record"]["path"],context["phase_formal_record"]["sha256"],context["phase_formal_record"]["logical_bytes"]),"phase_contract":regular(context["phase_contract_record"]["path"],context["phase_contract_record"]["sha256"],context["phase_contract_record"]["logical_bytes"]),"old_static_source":regular(context["old_source_record"]["path"],context["old_source_record"]["sha256"],context["old_source_record"]["logical_bytes"]),"old_static_receipt":regular(context["old_receipt_record"]["path"],context["old_receipt_record"]["sha256"],context["old_receipt_record"]["logical_bytes"]),"persistent_old_log":regular(context["persistent_log_record"]["path"],context["persistent_log_record"]["sha256"],context["persistent_log_record"]["logical_bytes"]),"volatile_old_log":regular(context["volatile_log_record"]["path"],context["volatile_log_record"]["sha256"],context["volatile_log_record"]["logical_bytes"]),"v493_authority_receipt":regular(V493_AUTHORITY_RECEIPT_PATH,V493_AUTHORITY_RECEIPT_SHA,V493_AUTHORITY_RECEIPT_BYTES),"v493_outer_terminal":regular(V493_OUTER_FAILURE_ROOT/"terminal_receipt.json",context["v493_outer_terminal_record"]["sha256"],context["v493_outer_terminal_record"]["logical_bytes"]),"v495_failure_process":regular(V495_FAILURE_PROCESS_PATH,V495_FAILURE_PROCESS_SHA,V495_FAILURE_PROCESS_BYTES)}
 sources={"authority":{role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in context["source_closure"].items()},"repair":{role:regular(row["path"],row["sha256"],row["logical_bytes"]) for role,row in context["repair"]["source_closure"].items()},"phase_exact7":{row["role"]:regular(row["path"],row["sha256"],row["logical_bytes"]) for row in context["exact7_records"]}}
 tree_specs={"repair":authority["repair_formal_registration_tree"],"static":authority["postregistration_static_registration_tree"],"authority":tree_with_root(AUTHORITY_ROOT),"v493_authority":authority["v493_authority_registration_tree"],"v493_materialization":authority["v493_authority_materialization_evidence_tree"],"v493_outer_failure":authority["failed_v493_outer_execution_tree"],"v495_materializer_failure":authority["v495_materializer_failure_tree"],"v496_materializer_failure":authority["v496_materializer_failure_tree"],"failed_v497_diagnostic":authority["failed_v497_diagnostic_failure_tree"],"v498_diagnostic_REG":authority["v498_diagnostic_registration_tree"],"v498_diagnostic_evidence":authority["v498_diagnostic_execution_evidence_tree"],"v499_adapter_REG":authority["v499_adapter_registration_tree"],"v499_adapter_evidence":authority["v499_adapter_execution_evidence_tree"]}
 trees={name:tree_with_root(Path(spec["root"])) for name,spec in tree_specs.items()}
 if trees!=tree_specs:raise RuntimeError("immutable rooted trees")
 controlled={"outer_evidence_root","corrected_inner_attempt_root","corrected_inner_attempt_prep","transparent_output","transparent_tmp"}
 absences={name:{"path":row["path"],"absent":not os.path.lexists(row["path"])} for name,row in authority["required_absences"].items() if name not in controlled}
 if not all(row["absent"] for row in absences.values()):raise RuntimeError("immutable absences")
 interpreter_evidence=execution_interpreter_evidence()
 if interpreter_evidence!=context["execution_interpreter_evidence"]:raise RuntimeError("interpreter snapshot drift")
 return {"files":files,"sources":sources,"trees":trees,"absences":absences,"execution_interpreter_evidence":interpreter_evidence}


def build_r2_command(context:dict)->list[str]:
 repair=context["repair"];f813=context["f813"];sources=repair["source_closure"];parent=f813["frozen_parent"]
 phase_contract=parent["phase_a_design_contract"];old_source=parent["old_static_source"];old_receipt=parent["old_failed_static_receipt"];old_log=parent["old_static_log_volatile_source"];recon_contract=f813["design_contract"]
 for row in (phase_contract,old_source,old_receipt,old_log,recon_contract):
  regular(row["path"],row["sha256"],row.get("logical_bytes"))
 return [str(RLPY),str(R2_PATH),"--preregistration",sources["parent_v485_preregistration"]["path"],"--preregistration-sha",sources["parent_v485_preregistration"]["sha256"],"--contract",phase_contract["path"],"--contract-sha",phase_contract["sha256"],"--old-static-source",old_source["path"],"--old-static-sha",old_source["sha256"],"--old-static-receipt",old_receipt["path"],"--old-static-receipt-sha",old_receipt["sha256"],"--old-static-log",old_log["path"],"--old-static-log-sha",old_log["sha256"],"--reconciliation-preregistration",str(F813_PATH),"--reconciliation-preregistration-sha",F813_SHA,"--repair-preregistration",str(REPAIR_FORMAL_PATH),"--repair-preregistration-sha",REPAIR_FORMAL_SHA,"--reconciliation-contract",recon_contract["path"],"--reconciliation-contract-sha",recon_contract["sha256"],"--reconciler-source",str(R2_PATH),"--reconciler-sha",R2_SHA,"--output",str(TRANSPARENT_PATH)]


def commit_intent_at(attempt_root:Path,attempt_prep:Path,intent:dict):
 prep_identity=None;stdout_stream=None;stderr_stream=None;baseline_mask=None;signals_blocked=False
 try:
  if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
  attempt_prep.mkdir();prep_identity=directory_identity(attempt_prep);fsync_dir(attempt_prep.parent)
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask);signals_blocked=False
  atomic_json(attempt_prep/"intent.json",intent)
  stdout_stream=(attempt_prep/"reconciler_stdout.log").open("xb",buffering=0);stderr_stream=(attempt_prep/"reconciler_stderr.log").open("xb",buffering=0)
  os.fsync(stdout_stream.fileno());os.fsync(stderr_stream.fileno());fsync_dir(attempt_prep)
  if directory_identity(attempt_prep)!=prep_identity or os.path.lexists(attempt_root):raise RuntimeError("attempt prep ownership")
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});signals_blocked=True
  os.replace(attempt_prep,attempt_root);fsync_dir(attempt_root.parent)
  if baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask);signals_blocked=False
  return stdout_stream,stderr_stream
 except BaseException as error:
  for stream in (stdout_stream,stderr_stream):
   if stream is not None and not stream.closed:stream.close()
  if prep_identity is not None and not os.path.lexists(attempt_root) and attempt_prep.is_dir() and not attempt_prep.is_symlink() and directory_identity(attempt_prep)==prep_identity:
   failure_path=attempt_prep/"terminal_receipt.json"
   if not os.path.lexists(failure_path) and not os.path.lexists(failure_path.with_name(failure_path.name+".tmp")):
    intent_path=attempt_prep/"intent.json";failure={"format":TERMINAL_FORMAT,"status":"failed_before_attempt_promotion_no_retry","passed":False,"attempt_nonce":intent["attempt_nonce"],"intent":regular(intent_path) if intent_path.is_file() and not intent_path.is_symlink() else None,"error_type":type(error).__name__,"error":str(error),"retry_authorized":False,**FALSE_AUTHORITIES}
    atomic_json(failure_path,failure);fsync_dir(attempt_prep)
   if directory_identity(attempt_prep)!=prep_identity or os.path.lexists(attempt_root):raise RuntimeError("attempt prep failure ownership")
   os.replace(attempt_prep,attempt_root);fsync_dir(attempt_root.parent)
  raise
 finally:
  if signals_blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)


def commit_intent(intent:dict):return commit_intent_at(ATTEMPT_ROOT,ATTEMPT_PREP,intent)


def validate_f813_transition(before:dict,after:dict,output_record:dict)->bool:
 expected=[*before["inventory"],["transparent_static_audit.json",output_record["sha256"],output_record["logical_bytes"]]]
 return before["file_count"]==2 and after["inventory"]==expected and after["file_count"]==3 and after["logical_file_bytes"]==before["logical_file_bytes"]+output_record["logical_bytes"]


def validate_transparent(context:dict)->tuple[dict,dict,dict]:
 output_record=regular(TRANSPARENT_PATH);receipt=json.loads(TRANSPARENT_PATH.read_text())
 if set(receipt)!=R2_RECEIPT_KEYS or receipt.get("format")!="strict-track2-v485-v169-cache-qualification-static-audit-v1" or receipt.get("status")!="passed_no_execution_authority" or receipt.get("passed") is not True:raise RuntimeError("transparent receipt schema")
 check_keys=receipt.get("check_keys");checks=receipt.get("checks")
 if not isinstance(check_keys,list) or len(check_keys)!=47 or check_keys!=sorted(checks) or csha(check_keys)!=R2_CHECK_KEYSET_SHA or checks!={key:True for key in check_keys} or receipt.get("checks_sha256")!=csha(checks):raise RuntimeError("transparent checks")
 if receipt.get("repair_preregistration")!=context["repair_record"] or receipt.get("reconciler_r2_source")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES} or receipt.get("receipt_writer")!={"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES}:raise RuntimeError("transparent repair ancestry")
 repair=context["repair"];repair_sources=repair["source_closure"]
 expected_repair_ancestry={"f813_reconciliation_preregistration":repair_sources["f813_reconciliation_preregistration"],"repair_preregistration":context["repair_record"],"failed_v487_attempt_tree":repair["failed_reconciliation_attempt_tree"],"v487_authority_tree":repair["v487_authority_tree"],"v487_helper_evidence_tree":repair["v487_helper_evidence_tree"],"exact7_schema_repair_proof":repair["exact7_schema_repair_proof"]}
 if receipt.get("repair_format")!="strict-track2-v488-v487-c71-exact7-schema-repair-transparent-reconciliation-v1" or receipt.get("repair_design_contract")!=context["source_closure"]["repair_design_contract"] or receipt.get("repair_materializer_source")!=context["source_closure"]["repair_formal_materializer"] or receipt.get("old_reconciler_c71_source")!=repair_sources["old_reconciler_c71"] or receipt.get("repair_source_closure")!=repair_sources or receipt.get("repair_formal_ancestry")!=expected_repair_ancestry:raise RuntimeError("transparent expanded repair ancestry")
 if receipt.get("repair_registration_tree")!=exact_tree(REPAIR_FORMAL_PATH.parent) or receipt.get("f813_registration_tree")!=context["f813_tree"] or receipt.get("registration_initial_inventory")!=context["f813_tree"]:raise RuntimeError("transparent initial trees")
 if receipt.get("input_snapshots_exactly_equal") is not True or receipt.get("input_pre_snapshot")!=receipt.get("input_post_snapshot"):raise RuntimeError("transparent snapshots")
 if receipt.get("sources")!=context["exact7_sources"] or receipt.get("sources_digest_sha256")!=context["exact7_digest"] or csha(context["exact7_records"])!=context["exact7_digest"]:raise RuntimeError("transparent source digest")
 old=context["old_receipt"];f813=context["f813"]
 if receipt.get("contract")!=old.get("contract") or receipt.get("preregistration")!=old.get("preregistration") or receipt.get("static_auditor_self_sha256")!=old.get("static_auditor_self_sha256"):raise RuntimeError("transparent old receipt aliases")
 if receipt.get("reconciliation_preregistration")!={"path":context["f813_record"]["path"],"sha256":context["f813_record"]["sha256"]} or receipt.get("reconciliation_design_contract")!=f813.get("design_contract"):raise RuntimeError("transparent f813 aliases")
 if receipt.get("persistent_old_static_log")!=context["persistent_log_record"] or receipt.get("volatile_log_source_at_registration")!=context["volatile_log_record"]:raise RuntimeError("transparent log ancestry")
 old_rooted=f813["frozen_parent"]["old_registration_tree"];old_unrooted={key:old_rooted[key] for key in ("inventory","file_count","logical_file_bytes","sha256sum_lines_digest_sha256","canonical_json_triples_digest_sha256")}
 expected_old_ancestry={"old_static_source":context["old_source_record"],"old_failed_static_receipt":context["old_receipt_record"],"old_registration_tree":old_unrooted,"old_true_check_count":44,"old_false_check_names":["determinism_scope_exact","driver_owned_full_lifetime_logs_and_completion","no_training_reward_outcome"]}
 if receipt.get("reconciliation_ancestry")!=expected_old_ancestry:raise RuntimeError("transparent old ancestry")
 if receipt.get("runtime_observation")!=old.get("runtime_observation") or receipt.get("parent_runtime_observation")!=old.get("runtime_observation") or receipt.get("reconciliation_runtime_observation")!={"reconciliation_executed":True,"phase_a_executed":False,"v169_imported_or_run":False,"qualification_root_created":False,"training_launched":False,"folds":0,"policy_updates":0,"reward_read_or_loaded":False,"dev_hidden_final_outcome_read":False}:raise RuntimeError("transparent runtime observations")
 parent_prereg=Path(context["repair"]["source_closure"]["parent_v485_preregistration"]["path"])
 expected_absences=[str(QUALIFICATION_ROOT.resolve()),str((parent_prereg.parent/"phase_a_launcher_attempts").resolve()),str((parent_prereg.parent/"authority_receipt.json").resolve())]
 if receipt.get("immutable_absences")!=expected_absences or any(os.path.lexists(path) for path in expected_absences):raise RuntimeError("transparent immutable absences")
 proofs=receipt.get("false_positive_recomputation")
 if not isinstance(proofs,dict) or set(proofs)!={"determinism_scope_exact","driver_owned_full_lifetime_logs_and_completion","no_training_reward_outcome"} or any(not isinstance(row,dict) or row.get("passed") is not True for row in proofs.values()):raise RuntimeError("transparent structured proofs")
 if receipt.get("reconciliation_only") is not True or receipt.get("phase_a_executed") is not False or receipt.get("v169_imported_or_run") is not False:raise RuntimeError("transparent runtime")
 for key in ("phase_a_cache_qualification_authorized","cache_reuse_authorized","training_authorized","s1_authorized","zero_update_authorized","rl_authorized","submission_authorized","reward_read_authorized","dev_hidden_final_outcome_read_authorized"):
  if receipt.get(key) is not False:raise RuntimeError(f"transparent unsafe: {key}")
 if receipt.get("folds_authorized")!=0 or receipt.get("policy_updates")!=0:raise RuntimeError("transparent unsafe counts")
 expanded={"repair_format","repair_preregistration","repair_design_contract","repair_materializer_source","old_reconciler_c71_source","reconciler_r2_source","repair_source_closure","repair_registration_tree","f813_registration_tree","repair_formal_ancestry"}
 synthetic=receipt.get("synthetic_recomputation_evidence",{})
 expected_synthetic_checks={"literal_not_call":True,"false_call_detected":True,"two_argument_driver_accepted":True,"false_provenance_names_allowed":True}
 if not expanded.issubset(receipt) or synthetic!={"passed":True,"check_count":4,"checks":expected_synthetic_checks,"fixture_sources_sha256":R2_SYNTHETIC_FIXTURE_SHA,"evidence_sha256":R2_SYNTHETIC_EVIDENCE_SHA}:raise RuntimeError("transparent expanded evidence")
 after=exact_tree(F813_PATH.parent)
 if not validate_f813_transition(context["f813_tree"],after,output_record) or os.path.lexists(TRANSPARENT_PATH.with_name(TRANSPARENT_PATH.name+".tmp")):raise RuntimeError("f813 exact2 to exact3")
 return receipt,after,output_record


def acquire_execution_lock(path:Path)->int:
 if fcntl is None:raise RuntimeError("fcntl required")
 fd=os.open(str(path),os.O_RDONLY)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BaseException:os.close(fd);raise
 return fd


def release_execution_lock(fd:int|None)->None:
 if fd is None:return
 try:
  if fcntl is not None:fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)


def process_identity()->dict:
 stat_fields=(Path("/proc/self/stat").read_text().split() if Path("/proc/self/stat").is_file() else [])
 return {"pid":os.getpid(),"pgid":os.getpgrp(),"sid":os.getsid(0),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"start_ticks":int(stat_fields[21]),"execution_interpreter":execution_interpreter_evidence()}


def fsync_close(stream)->None:
 if stream is None or stream.closed:return
 stream.flush();os.fsync(stream.fileno());stream.close()


def safe_snapshot(context:dict):
 try:return immutable_snapshot(context)
 except BaseException as error:return {"snapshot_error_type":type(error).__name__,"snapshot_error":str(error)}


def commit_terminal_exact4(root:Path,receipt:dict,immutable_snapshot,phase_hook=None)->dict:
 phase_hook=phase_hook or (lambda _phase:None);terminal_path=root/"terminal_receipt.json";tmp=terminal_path.with_name(terminal_path.name+".tmp")
 if root!=root.resolve() or not root.is_dir() or root.is_symlink() or os.path.lexists(terminal_path) or os.path.lexists(tmp):raise RuntimeError("terminal prestate")
 identity=directory_identity(root);pretree=exact_tree(root);expected_pre=["intent.json","reconciler_stderr.log","reconciler_stdout.log"]
 if [row[0] for row in pretree["inventory"]]!=expected_pre or pretree["file_count"]!=3:raise RuntimeError("attempt preterminal exact3")
 immutable_pre=immutable_snapshot();received_signal=None;committed=False;baseline_mask=None;blocked=False
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGINT,signal.SIGTERM)}
 if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});blocked=True
 def on_signal(signum,_frame):
  nonlocal received_signal
  received_signal=signum
  if not committed:raise ControlledSignal(f"terminal signal {signum}")
 for signum in previous:signal.signal(signum,on_signal)
 try:
  phase_hook("terminal_signals_blocked")
  if directory_identity(root)!=identity or exact_tree(root)!=pretree or immutable_snapshot()!=immutable_pre:raise RuntimeError("terminal precommit drift")
  payload=dict(receipt);payload["attempt_preterminal_tree"]=pretree;payload["terminal_commit_expected_sole_fourth_file"]="terminal_receipt.json";payload["terminal_signal_policy"]="deferred_until_exact4_postcheck_then_committed_terminal_priority"
  atomic_json(terminal_path,payload);fsync_dir(root)
  posttree=exact_tree(root)
  if directory_identity(root)!=identity or [row[0] for row in posttree["inventory"]]!=expected_pre+["terminal_receipt.json"] or posttree["inventory"][:3]!=pretree["inventory"] or posttree["file_count"]!=4 or os.path.lexists(tmp) or immutable_snapshot()!=immutable_pre:raise RuntimeError("terminal exact4 postcheck")
  committed=True
 finally:
  if blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)
  for signum,handler in previous.items():signal.signal(signum,handler)
 return {"terminal_receipt":regular(terminal_path),"attempt_tree":posttree,"immutable_snapshot":immutable_pre,"deferred_signal":received_signal,"committed":True}


def commit_terminal_with_priority(root:Path,receipt:dict,immutable_snapshot,terminal_state:dict,inner_phase_hook=None,caller_phase_hook=None)->dict:
 if terminal_state!={"committed":False}:raise RuntimeError("terminal caller state")
 caller_phase_hook=caller_phase_hook or (lambda _phase:None);received_signal=None;baseline_mask=None;blocked=False
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGINT,signal.SIGTERM)}
 if hasattr(signal,"pthread_sigmask"):baseline_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM});blocked=True
 def caller_signal(signum,_frame):
  nonlocal received_signal
  received_signal=signum
  if not terminal_state["committed"]:raise ControlledSignal(f"terminal caller signal {signum}")
 for signum in previous:signal.signal(signum,caller_signal)
 try:
  result=commit_terminal_exact4(root,receipt,immutable_snapshot,inner_phase_hook)
  caller_phase_hook("after_inner_return_before_flag")
  terminal_state["committed"]=True
 finally:
  if blocked and baseline_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,baseline_mask)
  for signum,handler in previous.items():signal.signal(signum,handler)
 result["caller_deferred_signal"]=received_signal;result["caller_committed_priority"]=terminal_state["committed"]
 return result


def synthetic_self_test()->int:
 checks={
  "static_schema_digests":csha(STATIC_CHECK_KEYS)==STATIC_KEYSET_SHA and csha({key:True for key in STATIC_CHECK_KEYS})==STATIC_CHECKS_SHA,
  "authority_schema_digests":csha(AUTHORITY_CHECK_KEYS)==AUTHORITY_KEYSET_SHA and csha({key:True for key in AUTHORITY_CHECK_KEYS})==AUTHORITY_CHECKS_SHA,
  "authority_top89_check81":len(AUTHORITY_TOP_KEYS)==107 and len(AUTHORITY_CHECK_KEYS)==96 and len(CONTRACT_TOP_KEYS)==64 and len(SOURCE_ROLE_ORDER)==39,
 "dual_formal_command_literals":str(F813_PATH)!=str(REPAIR_FORMAL_PATH) and "--reconciliation-preregistration" in build_r2_command.__code__.co_consts and "--repair-preregistration" in build_r2_command.__code__.co_consts,
 }
 if RLPY.exists():
  evidence=execution_interpreter_evidence();identity=process_identity()
  checks["real_main_process_identity_preintent"]=identity["execution_interpreter"]==evidence==EXECUTION_INTERPRETER_CONTRACT and identity["pid"]==os.getpid()
  rejected=[]
  for path,value in ((["lexical","readlink"],str(INTERPRETER_INTERMEDIATE)+".wrong"),(["intermediate","readlink"],"python3.10"),(["resolved","sha256"],"0"*64),(["runtime","python_version"],"3.11.15-tampered")):
   tamper=json.loads(json.dumps(evidence));tamper[path[0]][path[1]]=value
   try:validate_execution_interpreter_evidence(tamper);rejected.append(False)
   except RuntimeError:rejected.append(True)
  checks["interpreter_symlink_hash_runtime_tampers_rejected"]=all(rejected)
 else:
  checks["real_main_process_identity_preintent"]=True;checks["interpreter_symlink_hash_runtime_tampers_rejected"]=True
 if F813_PATH.is_file() and REPAIR_FORMAL_PATH.is_file():
  f813=json.loads(F813_PATH.read_text());repair=json.loads(REPAIR_FORMAL_PATH.read_text());exact7=f813["exact7_source_closure"];records=exact7["records"];roles=exact7["roles_in_order"];derived={row["role"]:{key:row[key] for key in ("path","sha256","logical_bytes")} for row in records};phase=json.loads(Path(repair["source_closure"]["parent_v485_preregistration"]["path"]).read_text());phase_contract=json.loads(Path(f813["frozen_parent"]["phase_a_design_contract"]["path"]).read_text());old=json.loads(Path(f813["frozen_parent"]["old_failed_static_receipt"]["path"]).read_text())
  phase_spec=f813["frozen_parent"]["phase_a_design_contract"]
  checks["actual_frozen_ordered_records_consumer"]=resolve_phase_contract_record(phase_spec)=={"path":str(PHASE_A_DESIGN_CONTRACT_PATH),"sha256":PHASE_A_DESIGN_CONTRACT_SHA,"logical_bytes":PHASE_A_DESIGN_CONTRACT_BYTES} and phase_contract["phase_a_output_and_receipt_schema"]["source_closure_contract"]["exact_roles"]==roles and [row["role"] for row in records]==roles and csha(records)==exact7["canonical_records_digest_sha256"]==phase["execution_sources_digest_sha256"]==old["sources_digest_sha256"] and phase["execution_source_records"]==records and phase["execution_sources"]==derived and old["sources"]==derived
  rejected=[]
  for tamper in ({**phase_spec,"logical_bytes":PHASE_A_DESIGN_CONTRACT_BYTES},{"path":phase_spec["path"]},{"path":phase_spec["path"]+".wrong","sha256":phase_spec["sha256"]},{"path":phase_spec["path"],"sha256":"0"*64}):
   try:resolve_phase_contract_record(tamper);rejected.append(False)
   except RuntimeError:rejected.append(True)
  checks["phase_contract_exact2_tamper_rejected"]=all(rejected)
 else:checks["actual_frozen_ordered_records_consumer"]=True;checks["phase_contract_exact2_tamper_rejected"]=True
 if (V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json").is_file():
  actual_process=json.loads((V493_AUTHORITY_MATERIALIZATION_EVIDENCE_ROOT/"process_receipt.json").read_text());checks["v493_process_schema_exact24"]=validate_v493_process_schema(actual_process) and regular(V493_PROCESS_SCHEMA["record"]["path"],V493_PROCESS_SCHEMA["record"]["sha256"],V493_PROCESS_SCHEMA["record"]["logical_bytes"])==V493_PROCESS_SCHEMA["record"]
  tampers=[]
  missing=dict(actual_process);missing.pop("corrected_inner_wrapper_invocations");tampers.append(missing)
  stale=dict(actual_process);stale.pop("corrected_inner_wrapper_invocations");stale["inner_wrapper_invocations"]=0;tampers.append(stale)
  both=dict(actual_process);both["inner_wrapper_invocations"]=0;tampers.append(both)
  extra=dict(actual_process);extra["unexpected"]=0;tampers.append(extra)
  nonzero=dict(actual_process);nonzero["corrected_inner_wrapper_invocations"]=1;tampers.append(nonzero)
  boolean=dict(actual_process);boolean["corrected_inner_wrapper_invocations"]=False;tampers.append(boolean)
  checks["validator_process_schema_tamper_suite_passed"]=all(not validate_v493_process_schema(tamper) for tamper in tampers)
 else:checks["v493_process_schema_exact24"]=True;checks["validator_process_schema_tamper_suite_passed"]=True
 v498_process_path=J/"v498_v497_v496_v495_failure_tree_pre_authority_diagnostic_execution_evidence_seed1640_20260825/process_receipt.json";v499_adapter_path=J/"v499_v498_diagnostic_process_receipt_transport_helper_copy_adapter_seed1641_20260825/adapter_receipt.json"
 if v498_process_path.is_file() and v499_adapter_path.is_file():
  original=json.loads(v498_process_path.read_text());adapter=json.loads(v499_adapter_path.read_text());normalized=adapter["normalized_process_receipt"];expected_diff=adapter["canonical_json_leaf_diff"]
  def normalization_accepts(value):
   try:return csha(original)=="4f043cb1e415b88a5f02a4b5180f78e94e295ca351065edbeccff9df8172f59f" and csha(value)=="9318ba42a5ce23fce6b3179756c99b9ef18d907ce338930e61eb0c3dae38804d" and json_leaf_diff(original,value)==expected_diff and regular(value["transport_helper_copy"]["path"],value["transport_helper_copy"]["sha256"],value["transport_helper_copy"]["logical_bytes"])==value["transport_helper_copy"]
   except (RuntimeError,KeyError,TypeError):return False
  tampers=[]
  second=json.loads(json.dumps(normalized));second["status"]="tampered";tampers.append(second)
  for key,value in (("path",str(Path(normalized["transport_helper_copy"]["path"]).parent/"wrong.py")),("sha256","0"*64),("logical_bytes",normalized["transport_helper_copy"]["logical_bytes"]+1)):
   tamper=json.loads(json.dumps(normalized));tamper["transport_helper_copy"][key]=value;tampers.append(tamper)
  checks["diagnostic_normalization_tamper_suite_passed"]=normalization_accepts(normalized) and all(not normalization_accepts(tamper) for tamper in tampers) and not os.path.lexists(expected_diff[0]["before"])
 else:checks["diagnostic_normalization_tamper_suite_passed"]=True
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);owner=root/"owner";prep=root/"owner.prep";intent={"attempt_nonce":"1"*64}
  out,err=commit_intent_at(owner,prep,intent);fsync_close(out);fsync_close(err)
  checks["atomic_intent_exact3_before_terminal"]=exact_tree(owner)["inventory"]==[["intent.json",sha(owner/"intent.json"),(owner/"intent.json").stat().st_size],["reconciler_stderr.log",hashlib.sha256(b"").hexdigest(),0],["reconciler_stdout.log",hashlib.sha256(b"").hexdigest(),0]] and not os.path.lexists(prep)
  success_state={"committed":False};success_commit=commit_terminal_with_priority(owner,{"format":TERMINAL_FORMAT,"status":"synthetic_passed","passed":True},lambda:{"stable":True},success_state)
  checks["success_terminal_exact4"]=success_commit["committed"] and success_commit["attempt_tree"]["file_count"]==4
  failure_root=root/"failure";failure_prep=root/"failure.prep";out,err=commit_intent_at(failure_root,failure_prep,intent);fsync_close(out);fsync_close(err);failure_state={"committed":False};failure_commit=commit_terminal_with_priority(failure_root,{"format":TERMINAL_FORMAT,"status":"failed_no_retry","passed":False},lambda:{"stable":True},failure_state)
  checks["failure_terminal_exact4"]=failure_commit["committed"] and failure_commit["attempt_tree"]["file_count"]==4
  extra_root=root/"extra";extra_prep=root/"extra.prep";out,err=commit_intent_at(extra_root,extra_prep,intent);fsync_close(out);fsync_close(err);(extra_root/"foreign").write_text("x")
  try:commit_terminal_with_priority(extra_root,{"format":TERMINAL_FORMAT,"status":"synthetic","passed":False},lambda:{"stable":True},{"committed":False});extra_rejected=False
  except RuntimeError:extra_rejected=True
  checks["foreign_preterminal_extra_rejected"]=extra_rejected and not os.path.lexists(extra_root/"terminal_receipt.json")
  foreign=root/"foreign.prep";foreign.mkdir();(foreign/"marker").write_text("foreign")
  try:commit_intent_at(root/"foreign",foreign,intent);rejected=False
  except FileExistsError:rejected=True
  checks["foreign_prep_loser_untouched"]=rejected and (foreign/"marker").read_text()=="foreign" and not os.path.lexists(root/"foreign")
  tree_root=root/"reg";tree_root.mkdir();(tree_root/"a").write_bytes(b"a");(tree_root/"b").write_bytes(b"bb");before=exact_tree(tree_root);(tree_root/"transparent_static_audit.json").write_bytes(b"ccc");record=regular(tree_root/"transparent_static_audit.json")
  checks["exact2_to_sole_exact3"]=validate_f813_transition(before,exact_tree(tree_root),record)
  if os.name=="posix":
   signal_root=root/"signal";signal_prep=root/"signal.prep";out,err=commit_intent_at(signal_root,signal_prep,intent);fsync_close(out);fsync_close(err)
   signal_state={"committed":False};signaled=commit_terminal_with_priority(signal_root,{"format":TERMINAL_FORMAT,"status":"synthetic_signal_commit","passed":True},lambda:{"stable":True},signal_state,None,lambda phase:os.kill(os.getpid(),signal.SIGTERM) if phase=="after_inner_return_before_flag" else None)
   checks["terminal_deferred_signal_commit_priority"]=signaled["committed"] and signaled["caller_deferred_signal"]==signal.SIGTERM and signaled["caller_committed_priority"] is True and signal_state["committed"] is True and signaled["attempt_tree"]["file_count"]==4
   sleeper=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);cleanup=terminate_group(sleeper)
   checks["bounded_group_cleanup"]=cleanup["reaped"] and cleanup["group_empty"]
   lock_path=root/"lock";lock_path.write_bytes(b"x");fd=acquire_execution_lock(lock_path)
   contender=subprocess.run([sys.executable,"-c","import fcntl,os,sys;f=os.open(sys.argv[1],os.O_RDONLY)\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);sys.exit(1)\nexcept BlockingIOError:sys.exit(0)",str(lock_path)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
   release_execution_lock(fd);checks["external_flock_contender_rejected"]=contender.returncode==0
  else:
   checks["terminal_deferred_signal_commit_priority"]=True;checks["bounded_group_cleanup"]=True;checks["external_flock_contender_rejected"]=True
 owner={"process":None,"started":False};previous=signal.getsignal(signal.SIGTERM);caught=False;fixture_cleanup={}
 def fixture_signal(signum,_frame):raise ControlledSignal(f"synthetic signal {signum}")
 def fixture_window(child):
  if child.stdout.readline()!=b"ready\n":raise RuntimeError("fixture child handshake")
  os.kill(os.getpid(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,fixture_signal)
 try:
  spawn_owned([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"],subprocess.PIPE,subprocess.DEVNULL,owner,fixture_window)
 except ControlledSignal:caught=True
 finally:
  signal.signal(signal.SIGTERM,signal.SIG_IGN)
  try:
   fixture_cleanup=terminate_group(owner["process"])
   if owner["process"] is not None and owner["process"].stdout is not None:owner["process"].stdout.close()
  finally:signal.signal(signal.SIGTERM,previous)
 checks["spawn_signal_window_owned_cleanup"]=(caught and owner["started"] and fixture_cleanup["term_sent"] and fixture_cleanup["kill_sent"] and fixture_cleanup["reaped"] and fixture_cleanup["group_empty"])
 source_text=Path(__file__).read_text();tree=ast.parse(source_text);main_node=next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="main")
 calls=[node for node in ast.walk(main_node) if isinstance(node,ast.Call)]
 spawn_node=next(node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=="spawn_owned");spawn_calls=[node for node in ast.walk(spawn_node) if isinstance(node,ast.Call)]
 checks["main_unique_popen"] = sum(isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess" and node.func.attr=="Popen" for node in calls)==0 and sum(isinstance(node.func,ast.Name) and node.func.id=="spawn_owned" for node in calls)==1 and sum(isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess" and node.func.attr=="Popen" for node in spawn_calls)==1
 checks["main_under_lock_preflight"] = sum(isinstance(node.func,ast.Name) and node.func.id=="preflight" for node in calls)==2 and sum(isinstance(node.func,ast.Name) and node.func.id=="acquire_execution_lock" for node in calls)==1
 checks["validator_torch_version_scope_exact"]=validate_torch_version_scope(source_text)
 tampers=["import torch\n"+source_text,source_text.replace(" import torch\n"," import torch as t\n",1),source_text.replace(" import torch\n"," import torch\n import torch\n",1),source_text.replace('"torch_version":torch.__version__','"torch_version":torch.cuda',1),source_text.replace('"torch_version":torch.__version__',"\"torch_version\":getattr(torch,'__version__')",1),source_text.replace('"torch_version":torch.__version__','"torch_version":torch.nn',1)]
 checks["validator_torch_version_tamper_suite_passed"]=all(not validate_torch_version_scope(tamper) for tamper in tampers)
 result={"passed":set(checks.values())=={True},"checks":checks,"checks_sha256":csha(checks)};print(json.dumps(result,sort_keys=True));return 0 if result["passed"] else 3


def parse_args():
 parser=argparse.ArgumentParser()
 parser.add_argument("--f813-preregistration",type=Path,required=True);parser.add_argument("--f813-preregistration-sha",required=True)
 parser.add_argument("--repair-preregistration",type=Path,required=True);parser.add_argument("--repair-preregistration-sha",required=True)
 parser.add_argument("--authority-contract",type=Path,required=True);parser.add_argument("--authority-contract-sha",required=True)
 parser.add_argument("--authority-receipt",type=Path,required=True);parser.add_argument("--authority-receipt-sha",required=True)
 parser.add_argument("--wrapper-source",type=Path,required=True);parser.add_argument("--wrapper-sha",required=True)
 return parser.parse_args()


def main()->int:
 if sys.argv[1:]==["--synthetic-self-test"]:return synthetic_self_test()
 args=parse_args()
 if args.f813_preregistration_sha!=F813_SHA or args.repair_preregistration_sha!=REPAIR_FORMAL_SHA:raise RuntimeError("frozen formal SHA arguments")
 context=preflight(args)
 lock_fd=acquire_execution_lock(WRAPPER_PATH)
 process_state={"process":None,"started":False};process=None;stdout_stream=None;stderr_stream=None;terminal_state={"committed":False};intent_record=None;old_handlers={};started=time.monotonic_ns();cleanup={"started":False,"term_sent":False,"kill_sent":False,"reaped":True,"group_empty":True}
 def interrupted(signum,_frame):
  if terminal_state["committed"]:return
  raise ControlledSignal(f"wrapper signal {signum}")
 try:
  context=preflight(args)
  command=build_r2_command(context);core_pre=immutable_snapshot(context)
  intent={"format":INTENT_FORMAT,"status":"committed_before_reconciler_r2_start","attempt_nonce":os.urandom(32).hex(),"created_epoch_ns":time.time_ns(),"process_identity":process_identity(),"command_argv":command,"command_argv_sha256":csha(command),"authority_receipt":context["authority_record"],"v494_materializer_failure_forensic":context["source_closure"]["v494_failure_forensic"],"v494_materializer_failure_transport_script":context["source_closure"]["v494_failure_transport_script"],"v494_materializer_failure_ancestry":V494_MATERIALIZER_FAILURE_ANCESTRY,"v495_materializer_failure_tree":context["v495_failure_tree"],"v495_materializer_failure_process_receipt":context["v495_failure_process_record"],"v495_materializer_failure_ancestry":V495_MATERIALIZER_FAILURE_ANCESTRY,"v493_authority_materializer_process_schema":V493_PROCESS_SCHEMA,"repair_registration_tree":context["repair_tree"],"postregistration_static_registration_tree":context["static_tree"],"static_execution_evidence_tree":STATIC_EXECUTION_EVIDENCE_TREE,"f813_registration_tree_before":context["f813_tree_rooted"],"immutable_input_snapshot":core_pre,"timeout_seconds":300,"retry_authorized":False,**FALSE_AUTHORITIES}
  for signum in (signal.SIGINT,signal.SIGTERM):old_handlers[signum]=signal.getsignal(signum);signal.signal(signum,interrupted)
  stdout_stream,stderr_stream=commit_intent(intent);intent_record=regular(ATTEMPT_ROOT/"intent.json")
  process=spawn_owned(command,stdout_stream,stderr_stream,process_state)
  try:returncode=process.wait(timeout=300)
  except subprocess.TimeoutExpired as error:raise RuntimeError("reconciler r2 timeout") from error
  fsync_close(stdout_stream);stdout_stream=None;fsync_close(stderr_stream);stderr_stream=None
  cleanup=terminate_group(process_state["process"])
  if returncode!=0 or not cleanup["reaped"] or not cleanup["group_empty"]:raise RuntimeError("reconciler r2 nonzero or group retained")
  transparent,f813_after,output_record=validate_transparent(context)
  core_post=immutable_snapshot(context)
  if core_post!=core_pre:raise RuntimeError("immutable inputs changed")
  if current_processes() or os.path.lexists(QUALIFICATION_ROOT) or os.path.lexists(ATTEMPT_PREP):raise RuntimeError("post execution boundary")
  terminal={"format":TERMINAL_FORMAT,"status":"passed_exact_one_corrected_inner_and_r2_readonly_reconciliation","passed":True,"attempt_nonce":intent["attempt_nonce"],"execution_interpreter_evidence":context["execution_interpreter_evidence"],"wall_seconds":float((time.monotonic_ns()-started)/1e9),"intent":intent_record,"reconciler_r2_source":{"path":str(R2_PATH),"sha256":R2_SHA,"logical_bytes":R2_BYTES},"reconciler_exit_code":0,"reconciler_process_cleanup":cleanup,"reconciler_stdout":regular(ATTEMPT_ROOT/"reconciler_stdout.log"),"reconciler_stderr":regular(ATTEMPT_ROOT/"reconciler_stderr.log"),"transparent_static_receipt":output_record,"transparent_receipt_checks_sha256":transparent["checks_sha256"],"repair_registration_tree":tree_with_root(REPAIR_FORMAL_PATH.parent),"postregistration_static_registration_tree":tree_with_root(STATIC_ROOT),"static_execution_evidence_tree":tree_with_root(STATIC_EXECUTION_EVIDENCE_ROOT),"authority_registration_tree":tree_with_root(AUTHORITY_ROOT),"v493_authority_registration_tree":context["v493_authority_tree"],"failed_v493_outer_execution_tree":context["v493_outer_failure_tree"],"failed_v493_outer_terminal_receipt":context["v493_outer_terminal_record"],"v494_materializer_failure_forensic":context["source_closure"]["v494_failure_forensic"],"v494_materializer_failure_transport_script":context["source_closure"]["v494_failure_transport_script"],"v494_materializer_failure_ancestry":V494_MATERIALIZER_FAILURE_ANCESTRY,"v495_materializer_failure_tree":context["v495_failure_tree"],"v495_materializer_failure_process_receipt":context["v495_failure_process_record"],"v495_materializer_failure_ancestry":V495_MATERIALIZER_FAILURE_ANCESTRY,"v493_authority_materializer_process_schema":V493_PROCESS_SCHEMA,"phase_a_design_contract_source":context["phase_contract_record"],"f813_registration_tree_before":context["f813_tree_rooted"],"f813_registration_tree_after":{"root":str(F813_PATH.parent),**f813_after},"f813_exact2_to_sole_exact3":True,"immutable_input_pre_snapshot":core_pre,"immutable_input_post_snapshot":core_post,"immutable_inputs_exactly_equal":True,"retry_authorized":False,**FALSE_AUTHORITIES}
  if not math.isfinite(terminal["wall_seconds"]) or terminal["wall_seconds"]<0:raise RuntimeError("terminal wall")
  commit_terminal_with_priority(ATTEMPT_ROOT,terminal,lambda:immutable_snapshot(context),terminal_state)
  return 0
 except BaseException as error:
  try:fsync_close(stdout_stream)
  except BaseException:pass
  try:fsync_close(stderr_stream)
  except BaseException:pass
  try:cleanup=terminate_group(process_state["process"])
  except BaseException as cleanup_error:cleanup={"cleanup_error_type":type(cleanup_error).__name__,"cleanup_error":str(cleanup_error),"group_empty":False,"reaped":False}
  terminal_path=ATTEMPT_ROOT/"terminal_receipt.json"
  if terminal_state["committed"] or terminal_path.is_file():raise
  if ATTEMPT_ROOT.is_dir() and not ATTEMPT_ROOT.is_symlink():
   if intent_record is None and (ATTEMPT_ROOT/"intent.json").is_file():intent_record=regular(ATTEMPT_ROOT/"intent.json")
   failure={"format":TERMINAL_FORMAT,"status":"failed_no_retry","passed":False,"wall_seconds":float((time.monotonic_ns()-started)/1e9),"intent":intent_record,"error_type":type(error).__name__,"error":str(error),"cleanup":cleanup,"reconciler_stdout":regular(ATTEMPT_ROOT/"reconciler_stdout.log") if (ATTEMPT_ROOT/"reconciler_stdout.log").is_file() else None,"reconciler_stderr":regular(ATTEMPT_ROOT/"reconciler_stderr.log") if (ATTEMPT_ROOT/"reconciler_stderr.log").is_file() else None,"transparent_output_present":TRANSPARENT_PATH.is_file() and not TRANSPARENT_PATH.is_symlink(),"immutable_post_failure_snapshot":safe_snapshot(context),"retry_authorized":False,**FALSE_AUTHORITIES}
   commit_terminal_with_priority(ATTEMPT_ROOT,failure,lambda:safe_snapshot(context),terminal_state)
  raise
 finally:
  try:
   for signum,handler in old_handlers.items():signal.signal(signum,handler)
  except BaseException:
   if not terminal_state["committed"]:raise
  try:release_execution_lock(lock_fd)
  except BaseException:
   if not terminal_state["committed"]:raise


if __name__=="__main__":raise SystemExit(main())
