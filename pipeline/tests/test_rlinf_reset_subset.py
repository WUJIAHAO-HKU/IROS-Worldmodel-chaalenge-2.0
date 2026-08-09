import json
import os

import numpy as np

from scripts.make_rlinf_reset_subset import build_subset


def test_build_subset_uses_only_requested_hardlinks(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "subset"
    source.mkdir()
    records = []
    for episode_id in range(4):
        filename = f"episode{episode_id}.npy"
        np.save(source / filename, np.asarray([episode_id]))
        records.append({"episode": f"episode{episode_id}.hdf5", "reset_file": filename})
    (source / "manifest.json").write_text(json.dumps({"episodes": records}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_episodes": [3, 1]}))

    manifest = build_subset(source, split, "train_episodes", output)

    assert manifest["episode_ids"] == [1, 3]
    assert sorted(path.name for path in output.glob("*.npy")) == ["episode1.npy", "episode3.npy"]
    assert os.stat(source / "episode1.npy").st_ino == os.stat(output / "episode1.npy").st_ino
    assert json.loads((output / "manifest.json").read_text())["split_key"] == "train_episodes"
