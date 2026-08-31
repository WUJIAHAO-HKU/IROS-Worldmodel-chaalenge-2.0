#!/usr/bin/env python3
"""Freeze the immutable evidence closure for v479 reconciliation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/root/autodl-tmp/IROS_WAM_2.0 challenge")
J = ROOT / "artifacts/strict_track2_joint_augmentation_20260810"
OLD_REG = J / "v478_endpoint_oracle12_seed1622_20260824"
NEW_REG = J / "v479_v478_endpoint_oracle12_reconciliation_20260824"
ORACLE = Path("/root/v478_endpoint_oracle12_seed1622_20260824")
EXPECTED = {
    "phase_a_preregistration": (OLD_REG / "preregistration_r2.json", "167841e7c0d6a3380a4f524d186bcb9db37c7cf2a73550f711b35cffa547e6b9"),
    "selection": (J / "v478_temporal200_seed1622_20260824/selection.json", "f9d62a9b6a8db9d90225e1821dc5016bd56874b47baa48b5486ca5501d850398"),
    "collector": (ROOT / "pipeline/scripts/collect_v460_endpoint_only.py", "6c64e4a42f273f43b9e64523bf12e4dc5f24618d5673848db6c1582b5547e846"),
    "legacy_auditor": (ROOT / "pipeline/scripts/audit_v478_endpoint_oracle12.py", "6cb920fbccfa039ac3a12839ee2626614a6498397adb563c3b91cd5c573cb4a2"),
    "phase_a_launcher": (ROOT / "pipeline/scripts/launch_v478_endpoint_oracle12.sh", "bae7c423678a58602768c48be725bd4aa4f950da0c13ea3fe5f846d5c060f33c"),
    "generation_report": (ORACLE / "generation_report.json", "7f9972f2eb795402777641b2f12c2b10ffbba2033f4c70524664b9ce4c3492d8"),
    "legacy_launcher_failure": (OLD_REG / "launcher_failure_receipt.json", "678cf8433109fd8ade32c44f709fde41a4708a7fbe8a26b3f3e7429e81c8acdf"),
    "legacy_audit_tmp": (OLD_REG / "audit_receipt.json.tmp", "27e545c00c1733218483d03a846d7b283f09ca65f66cfef48b1040835a0c58d6"),
    "legacy_auditor_log": (OLD_REG / "auditor.log", "e7fd1ed72dd2c6adce9dc916b7deb12267d0dece8974a8f347e6b0e1802bcbac"),
    "legacy_generator_log": (OLD_REG / "generator.log", "7638e2a9b51184817180d53e77f5d3df988b5165562d3e4004e1c60bc28a4f5e"),
    "legacy_restore_v218_log": (OLD_REG / "restore_v218.log", "bf36c8bfe0d2f0af1690b20840a80e622e67e3029daab4b4985a88fc2989acb1"),
    "task_config": (ROOT / "artifacts/strict_track2_official_20260810/official_deps/RoboTwin_RLinf_support/task_config/demo_clean.yml", "865af205a90c3e7ffe4b4dd3db06955609f1b58c76d997e18894d8195a361566"),
    "resize_source": (ROOT / "pipeline/wam_pipeline/data.py", "d060b8b3503e582ccd5d9fcccbb4efd708c812ae25324bf804c20c7dc7026f97"),
}
ROWS = [
    (12,102,"afa376a777cef27dfe56154975ccb0c7ad11b183ba986d981ae91d87644d3ad5","64eaae6ee63248d9a214e2fd72c75ed7c35bc8a500a99e719a825560e3390dea"),
    (30,82,"bb79dad71242af6c21f6b80e5f3722af31147fe8e0508171f2060733302d91cd","2e5d3fcaa2a47cf1108e24eb60f34abc3d8a6f5deea2c2fd9a82cd35f384f372"),
    (30,125,"0c1f8b7b446ddb46124286e3929d3ec1468204f5105f723ab4a9bf7ecb01138b","433d564a05eb6f4d4a4898b20ad0b8401f93724d7bcca4ee790b774b772d61c7"),
    (32,93,"354f6d7485707ff0b9e4a98630507c6c272e60f7f9d0227c303ff101adc98c22","32fbfc6a5f6e95998ab4bb1a367dd52d38b656093ba7e3431a2b6898f58588e0"),
    (33,97,"93f254961447803ff6473403ff2d5396d2d740ef1a4f8b8e6a3a0e390c04af24","4f30737b4574293bf91660d23c9640d5a03c0bef475d2802c159c1ebf3004efc"),
    (37,91,"45e5bc33db1b4755514d5886a066240e3fca126ce6586e5ee88b76e48d9bd4c1","4a51b9904f94d2783f8b3dc0323fd5267b2b6b0d7b5d3df1998c6e7ab1dcee73"),
    (40,102,"dd159e3c000d6d6fab1660be26f79c5a5dab8886b07b1912507edb7ecc0c28be","820cf8af68e134ff8bf034aed48f1eceefa21bbdbe3882139eb323a013cffd82"),
    (40,108,"d49259070eb34d1317ee7834ac4eb199086cb8c38d5328ad49ef48a1eb0a7296","833a47d5f1d722c989cb233cc0ed43d69648d33d6cb799d0550cc0849ee80994"),
    (46,121,"0c322573fd6731fb51ca5612ed717f47fb8e84ec7c047f97664926a55f98cbcd","56bfae0f60c6c782d9b4dbf2886af1f8d432dd7800c6aec10b4db062fde1cc40"),
    (47,72,"08d8b9d37726e87f4513f2e8143ac08df50b2ca18df08ebd1177bcb4a2d015aa","fa1f7df3e68797c4de4e81052cecc4558d663422b29dbf4d7c5fd60884843023"),
    (49,76,"f0520d329f8aae3c24e5bcbd2070803d0877c907a71a6613fa506b19d86f54e2","0992c834867daed5f14ab6aed1d29b1b7f8b82e0883fd637823b417da72d6955"),
    (49,119,"9b76ddf87039833946478ac810325c0522a5fe92bbd7c82c5c6c79f2b32cf9f2","fbcdd4640c6ee6b5119e464343cc3e693fe9189b97cf716828688d144ba4b83d"),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, expected: str) -> dict:
    path = path.resolve()
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"evidence drift: {path}")
    return {"path": str(path), "sha256": expected}


def tree_inventory(root: Path):
    root = root.resolve()
    paths = list(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise RuntimeError("oracle tree symlink forbidden")
    items = [[p.relative_to(root).as_posix(),p.stat().st_size,sha(p)] for p in sorted(paths) if p.is_file()]
    digest = hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return items,digest


def atomic(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if path.exists() or tmp.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--reconciler",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.resolve()!=NEW_REG/"preregistration_r3.json":
        raise RuntimeError("wrong v479 preregistration path")
    immutable={key:record(path,expected) for key,(path,expected) in EXPECTED.items()}
    phase=json.loads(Path(immutable["phase_a_preregistration"]["path"]).read_text())
    failure=json.loads(Path(immutable["legacy_launcher_failure"]["path"]).read_text())
    if phase.get("format")!="strict-track2-v478-endpoint-oracle12-preregistration-v1" or phase.get("status")!="preregistered_public_train_endpoint_oracle_collection_authorized":
        raise RuntimeError("wrong phase-A parent")
    if failure.get("format")!="strict-track2-v478-endpoint-oracle12-launcher-failure-v1" or failure.get("stage")!="endpoint_oracle12_audit" or failure.get("exit_code")!=1 or failure.get("phase_b_temporal_collection_authorized") is not False or failure.get("retry_under_same_lineage_authorized") is not False:
        raise RuntimeError("wrong immutable failure")
    if "Object of type bool_ is not JSON serializable" not in Path(immutable["legacy_auditor_log"]["path"]).read_text(errors="replace"):
        raise RuntimeError("wrong serialization failure")
    if (OLD_REG/"audit_receipt.json").exists() or (OLD_REG/"launcher_receipt.json").exists():
        raise RuntimeError("legacy lineage unexpectedly has a formal success receipt")
    row_files=[]
    for episode,start,npz_sha,receipt_sha in ROWS:
        row=ORACLE/"rows"/f"episode{episode}_start{start:05d}"
        row_files.append({"episode":episode,"start":start,"npz":record(row/"endpoint.npz",npz_sha),"receipt":record(row/"receipt.json",receipt_sha)})
    items,digest=tree_inventory(ORACLE)
    if digest!="240eb073f89de75100cabfc8fbf278fea2e5cf7fe1f08e016ec306367cede7e0" or len(items)!=25 or sum(x[1] for x in items)!=14034631:
        raise RuntimeError("wrong immutable oracle tree")
    reconciler=record(args.reconciler,sha(args.reconciler))
    prepare=record(Path(__file__),sha(Path(__file__)))
    value={
      "format":"strict-track2-v479-v478-endpoint-oracle12-reconciliation-preregistration-v1",
      "status":"preregistered_immutable_reconciliation_authorized",
      "registry_root":str(NEW_REG),"oracle_root":str(ORACLE),
      "immutable_evidence":immutable,"row_files":row_files,
      "oracle_tree":{"digest_sha256":digest,"file_count":25,"total_bytes":14034631},
      "execution_closure":{"reconciler":reconciler,"prepare":prepare},
      "legacy_recompute_arguments":{"support_root":phase["execution_closure"]["support_root"]["path"]},
      "outputs":{"recomputed_v478_audit":str(NEW_REG/"recomputed_v478_audit_receipt.json"),"reconciliation_receipt":str(NEW_REG/"reconciliation_receipt.json")},
      "authorized_change":{"only":"canonical JSON serialization of numpy.generic via item() and numpy.ndarray via tolist()","legacy_audit_computation":"unchanged imported main()"},
      "authorization_before_reconciliation":{"phase_b_temporal_collection_authorized":False,"training_authorized":False,"s1_authorized":False,"zero_update_authorized":False,"policy_updates":0,"rl_authorized":False},
      "required_absence":[str(OLD_REG/"audit_receipt.json"),str(OLD_REG/"launcher_receipt.json")],
      "guards":{"simulator_execution_authorized":False,"endpoint_oracle_retry_authorized":False,"old_tree_mutation_authorized":False,"old_tmp_overwrite_authorized":False,"phase_b_requires_reconciliation_pass":True},
    }
    atomic(args.output,value)
    return 0


if __name__=="__main__":raise SystemExit(main())
