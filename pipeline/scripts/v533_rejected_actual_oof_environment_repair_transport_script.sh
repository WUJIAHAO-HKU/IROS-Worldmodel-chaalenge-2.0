#!/bin/bash
set -euo pipefail

RLPY='/root/autodl-tmp/conda_envs/rlinf_track2/bin/python'
RLPY_INTERMEDIATE='/root/autodl-tmp/conda_envs/isaacsim51/bin/python'
RLPY_RESOLVED='/root/autodl-tmp/conda_envs/isaacsim51/bin/python3.11'
EXPECTED_RLPY_SHA='11e245a5a0d85eef88b5b851e421935e06a77bdf57bb7329eeecd10fb4f76788'
EXPECTED_RLPY_BYTES='25555040'

ROOT='/root/autodl-tmp/IROS_WAM_2.0 challenge'
J="$ROOT/artifacts/strict_track2_joint_augmentation_20260810"
SCRIPTS="$ROOT/pipeline/scripts"
SCRIPT='/root/v533_v532_actual_oof_execution_environment_repair_once.sh'
DEPLOYMENT_RECORD="$SCRIPTS/v533_v532_actual_oof_execution_environment_repair_transport_deployment_record.json"
FORENSIC="$SCRIPTS/v533_v532_actual_oof_execution_environment_failure_forensic.json"
FORENSIC_SHA='0592fcc92340e1bb58a8c87b9e868a26055686ae8e100048a9a1b8b7c684228d'
FORENSIC_BYTES='6124'
CONTRACT="$SCRIPTS/v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_contract.json"
CONTRACT_SHA='92766e99940a0ba3d27ff886487808155810c312791e9d77b341f32711bfa9d2'
CONTRACT_BYTES='110854'
AUTH_ROOT="$J/v532_v531_actual_oof_preregistration_active_source_paths_repair_execution_authority_seed1666_20260827"
AUTH_PREP="$AUTH_ROOT.authority-prep"
AUTH_RECEIPT="$AUTH_ROOT/authority_receipt.json"
AUTH_RECEIPT_SHA='420c5f09895f49f0d80e7d149dbf1754cc3b792f8011f79bdf3fec8cf0d03b34'
AUTH_RECEIPT_BYTES='317060'
PREREGISTRATION="$SCRIPTS/v532_v531_actual_oof_execution_preregistration.json"
PREREGISTRATION_SHA='7ef48984ac29d86edd22f75d4fab2d82e12237c1bd72615172da59dd8fdf70a5'
PREREGISTRATION_BYTES='14210'
MANIFEST="$SCRIPTS/v532_v531_actual_oof_execution_manifest.json"
MANIFEST_SHA='3a3e83eb03415c9c0f573a10112b4cb18a0593bda10fe6eaff1feed843e7996e'
MANIFEST_BYTES='1296036'
LAUNCHER="$SCRIPTS/launch_v532_v531_actual_oof.py"
LAUNCHER_SHA='ca1ecc8bec51776935b10df70121ed4c0c12ad94c3689604bec63d214a3dea4d'
LAUNCHER_BYTES='15315'
EXECUTOR_SOURCE="$SCRIPTS/execute_v532_v531_actual_oof.py"
EXECUTOR_SHA='7f5d154a6bd2d4ae94d372091d36fc6bf45d770889e692b6d9e499745f9e9932'
EXECUTOR_BYTES='62773'
AUDITOR_SOURCE="$SCRIPTS/audit_v532_v531_actual_oof.py"
AUDITOR_SHA='b8f98db595fc0ddbff1cab1e9fce20a0c4bd7d6fbe8372ae57d4a4c092fc2893'
AUDITOR_BYTES='11763'
SEED='1666'
ATTEMPT_ROOT="$J/v532_v531_actual_oof_attempt_seed1666_20260827"
ATTEMPT_PREP="$ATTEMPT_ROOT.attempt-prep"
OUTPUT_ROOT='/root/v532_v531_actual_oof_seed1666_20260827'
OUTPUT_PREP="$OUTPUT_ROOT.oof-prep"
[[ "$SEED" =~ ^[0-9]+$ && "$SEED" == '1666' ]]

SELF_TEST=false
if [[ "${1-}" == '--transport-self-test' ]]; then
  [[ "$#" == '1' ]]
  SELF_TEST=true
elif [[ "$#" != '0' ]]; then
  exit 2
fi

# Parent state must match the frozen failure fact: both required variables absent.
[[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" && -z "${PYTHONHASHSEED+x}" ]]
ENV_WITHOUT_REQUIRED_BEFORE="$(env | LC_ALL=C sort | sha256sum | awk '{print $1}')"
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export PYTHONHASHSEED='0'
[[ "$CUBLAS_WORKSPACE_CONFIG" == ':4096:8' && "$PYTHONHASHSEED" == '0' ]]
ENV_WITHOUT_REQUIRED_AFTER="$(env -u CUBLAS_WORKSPACE_CONFIG -u PYTHONHASHSEED | LC_ALL=C sort | sha256sum | awk '{print $1}')"
[[ "$ENV_WITHOUT_REQUIRED_AFTER" == "$ENV_WITHOUT_REQUIRED_BEFORE" ]]

[[ -L "$RLPY" ]]
[[ "$(readlink -- "$RLPY")" == "$RLPY_INTERMEDIATE" ]]
[[ -L "$RLPY_INTERMEDIATE" ]]
[[ "$(readlink -- "$RLPY_INTERMEDIATE")" == 'python3.11' ]]
[[ "$(readlink -f -- "$RLPY")" == "$RLPY_RESOLVED" ]]
[[ -f "$RLPY_RESOLVED" && ! -L "$RLPY_RESOLVED" ]]
[[ "$(sha256sum -- "$RLPY_RESOLVED" | awk '{print $1}')" == "$EXPECTED_RLPY_SHA" ]]
[[ "$(stat -c %s -- "$RLPY_RESOLVED")" == "$EXPECTED_RLPY_BYTES" ]]

check_regular() {
  local path="$1" expected_sha="$2" expected_bytes="$3"
  [[ -f "$path" && ! -L "$path" ]]
  [[ "$(sha256sum -- "$path" | awk '{print $1}')" == "$expected_sha" ]]
  [[ "$(stat -c %s -- "$path")" == "$expected_bytes" ]]
}
check_regular "$CONTRACT" "$CONTRACT_SHA" "$CONTRACT_BYTES"
check_regular "$AUTH_RECEIPT" "$AUTH_RECEIPT_SHA" "$AUTH_RECEIPT_BYTES"
check_regular "$PREREGISTRATION" "$PREREGISTRATION_SHA" "$PREREGISTRATION_BYTES"
check_regular "$MANIFEST" "$MANIFEST_SHA" "$MANIFEST_BYTES"
check_regular "$LAUNCHER" "$LAUNCHER_SHA" "$LAUNCHER_BYTES"
check_regular "$EXECUTOR_SOURCE" "$EXECUTOR_SHA" "$EXECUTOR_BYTES"
check_regular "$AUDITOR_SOURCE" "$AUDITOR_SHA" "$AUDITOR_BYTES"

CHECK_FORENSIC="$FORENSIC"
CHECK_SCRIPT="$SCRIPT"
CHECK_DEPLOYMENT_RECORD="$DEPLOYMENT_RECORD"
if [[ "$SELF_TEST" == true ]]; then
  CHECK_FORENSIC="${V533_ACTUAL_OOF_TRANSPORT_SELFTEST_FORENSIC:-$FORENSIC}"
  CHECK_SCRIPT="${V533_ACTUAL_OOF_TRANSPORT_SELFTEST_SCRIPT:-${BASH_SOURCE[0]}}"
  CHECK_DEPLOYMENT_RECORD="${V533_ACTUAL_OOF_TRANSPORT_SELFTEST_DEPLOYMENT_RECORD:-$DEPLOYMENT_RECORD}"
fi
check_regular "$CHECK_FORENSIC" "$FORENSIC_SHA" "$FORENSIC_BYTES"

[[ -d "$AUTH_ROOT" && ! -L "$AUTH_ROOT" ]]
shopt -s nullglob dotglob
auth_members=("$AUTH_ROOT"/*)
[[ "${#auth_members[@]}" == '1' && "${auth_members[0]}" == "$AUTH_RECEIPT" ]]
for path in "$AUTH_PREP" "$ATTEMPT_ROOT" "$ATTEMPT_PREP" "$OUTPUT_ROOT" "$OUTPUT_PREP"; do
  [[ ! -e "$path" && ! -L "$path" ]]
done

[[ -f "$CHECK_SCRIPT" && ! -L "$CHECK_SCRIPT" ]]
SELF_SHA="$(sha256sum -- "$CHECK_SCRIPT" | awk '{print $1}')"
SELF_BYTES="$(stat -c %s -- "$CHECK_SCRIPT")"
[[ -f "$CHECK_DEPLOYMENT_RECORD" && ! -L "$CHECK_DEPLOYMENT_RECORD" ]]
IFS= read -r deployment_line < "$CHECK_DEPLOYMENT_RECORD"
[[ -n "$deployment_line" && "$(wc -l < "$CHECK_DEPLOYMENT_RECORD")" == '1' ]]
expected_deployment_line='{"canonical_readback_required":true,"classification":"design_only_non_authority_non_execution_transport_manifest","deployment_authority":"external_atomic_noreplace_deployer_required","deployment_executed":false,"failure_forensic":{"logical_bytes":6124,"path":"/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v533_v532_actual_oof_execution_environment_failure_forensic.json","sha256":"0592fcc92340e1bb58a8c87b9e868a26055686ae8e100048a9a1b8b7c684228d"},"format":"strict-track2-v533-v532-actual-oof-execution-environment-repair-transport-deployment-record-v1","overwrite_authorized":false,"status":"preregistered_atomic_noreplace_transport_pending_external_deployment","target":{"logical_bytes":'"$SELF_BYTES"',"path":"/root/v533_v532_actual_oof_execution_environment_repair_once.sh","role":"actual_oof_execution_environment_repair_transport_script","sha256":"'"$SELF_SHA"'"}}'
[[ "$deployment_line" == "$expected_deployment_line" ]]

if [[ "$SELF_TEST" == true ]]; then
  ENV_PROBE="$("$RLPY" -c 'import json,os,sys; assert "torch" not in sys.modules and "torch.cuda" not in sys.modules; assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8" and os.environ.get("PYTHONHASHSEED")=="0"; print(json.dumps({"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"],"PYTHONHASHSEED":os.environ["PYTHONHASHSEED"],"torch_cuda_imported":False,"torch_imported":False},sort_keys=True,separators=(",",":")))')"
  [[ "$ENV_PROBE" == '{"CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONHASHSEED":"0","torch_cuda_imported":false,"torch_imported":false}' ]]
  AUTH_SUMMARY="$("$RLPY" -c 'import json,sys; a=json.load(open(sys.argv[1],"rb")); p=json.load(open(sys.argv[2],"rb")); m=json.load(open(sys.argv[3],"rb")); f=json.load(open(sys.argv[4],"rb")); roles=["authority_design_contract","authority_materializer","actual_oof_execution_preregistration","actual_oof_execution_manifest","actual_oof_executor","actual_oof_auditor","actual_oof_launcher"]; assert len(a)==62 and len(a["check_keys"])==35 and len(a["source_closure"])==30 and a["active_source_role_order"]==roles and p["active_source_role_order"]==m["active_source_role_order"]==roles; z=a["authorization"]; assert type(z["actual_oof_execution_boundary_invocations_authorized"]) is int and z["actual_oof_execution_boundary_invocations_authorized"]==1 and type(z["actual_oof_execution_boundary_invocations_consumed"]) is int and z["actual_oof_execution_boundary_invocations_consumed"]==0; assert f["root_cause"]["required_environment"]=={"CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONHASHSEED":"0"} and f["root_cause"]["observed_environment"]=={"CUBLAS_WORKSPACE_CONFIG":None,"PYTHONHASHSEED":None}; print("62:35:30:7:1:0")' "$AUTH_RECEIPT" "$PREREGISTRATION" "$MANIFEST" "$CHECK_FORENSIC")"
  [[ "$AUTH_SUMMARY" == '62:35:30:7:1:0' ]]
  printf '{"active_source_roles":7,"actual_oof_auditor_invocations":0,"actual_oof_execution_boundary_invocations":0,"actual_oof_executor_invocations":0,"actual_oof_launcher_invocations":0,"authority_checks":35,"authority_receipt_sha256":"%s","authority_source_roles":30,"authority_top_keys":62,"authorized":1,"consumed":0,"environment_probe":%s,"failure_forensic_sha256":"%s","fresh_roots_absent":true,"passed":true,"script_bytes":%s,"script_sha256":"%s","seed":%s}\n' \
    "$AUTH_RECEIPT_SHA" "$ENV_PROBE" "$FORENSIC_SHA" "$SELF_BYTES" "$SELF_SHA" "$SEED"
  exit 0
fi

[[ "$(readlink -f -- "${BASH_SOURCE[0]}")" == "$SCRIPT" ]]
argv=(
  "$RLPY"
  "$LAUNCHER"
  --preregistration "$PREREGISTRATION"
  --preregistration-sha "$PREREGISTRATION_SHA"
  --manifest "$MANIFEST"
  --manifest-sha "$MANIFEST_SHA"
  --contract "$CONTRACT"
  --contract-sha "$CONTRACT_SHA"
  --authority-receipt "$AUTH_RECEIPT"
  --authority-receipt-sha "$AUTH_RECEIPT_SHA"
  --executor-source "$EXECUTOR_SOURCE"
  --executor-sha "$EXECUTOR_SHA"
  --auditor-source "$AUDITOR_SOURCE"
  --auditor-sha "$AUDITOR_SHA"
  --launcher-sha "$LAUNCHER_SHA"
  --seed "$SEED"
  --output-root "$OUTPUT_ROOT"
)
exec "${argv[@]}"
