from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from wam_pipeline.external_robotwin_data_v260 import ExternalRandomizedWindowDataset


def test_external_alignment_and_shapes(tmp_path: Path):
    episode = tmp_path / "episode_7" / "episode_7.hdf5"
    episode.parent.mkdir()
    frames = []
    for index in range(15):
        image = Image.fromarray(np.full((32, 48, 3), index, dtype=np.uint8), mode="RGB")
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        frames.append(np.bytes_(buffer.getvalue()))
    actions = np.arange(15 * 14, dtype=np.float32).reshape(15, 14)
    with h5py.File(episode, "w") as handle:
        handle.create_dataset("observations/images/cam_high", data=frames)
        handle.create_dataset("action", data=actions)

    dataset = ExternalRandomizedWindowDataset(tmp_path, stride=99)
    context, history, future, target = dataset[0]
    assert context.shape == (5, 256, 256, 3)
    assert target.shape == (8, 256, 256, 3)
    assert history.shape == (4, 14)
    assert future.shape == (8, 14)
    np.testing.assert_array_equal(history.numpy(), actions[:4])
    np.testing.assert_array_equal(future.numpy(), actions[4:12])
    assert context[:, 0, 0, 0].tolist() == list(range(5))
    assert target[:, 0, 0, 0].tolist() == list(range(5, 13))
