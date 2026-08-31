#!/bin/bash
set -euo pipefail
cmd=(
  "/root/autodl-tmp/conda_envs/rlinf_track2/bin/python"
  "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v494_v493_v490_interpreter_symlink_repair_execution_authority.py"
  "--contract"
  "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/v494_v493_v490_interpreter_symlink_repair_execution_authority_contract.json"
  "--contract-sha"
  "2f4cef2671dfd5c443d5a7f53dc0ea7b770b520da84cba3512a363884d56bd4a"
  "--materializer-source"
  "/root/autodl-tmp/IROS_WAM_2.0 challenge/pipeline/scripts/materialize_v494_v493_v490_interpreter_symlink_repair_execution_authority.py"
  "--materializer-sha"
  "5b9396fd74e1d161fa41610108d823f356313992364cba708ed869519fa0a421"
  "--authority-root"
  "/root/autodl-tmp/IROS_WAM_2.0 challenge/artifacts/strict_track2_joint_augmentation_20260810/v494_v493_v490_interpreter_symlink_repair_execution_authority_seed1636_20260825"
)
exec "${cmd[@]}"
