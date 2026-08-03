#!/usr/bin/env python3
"""Download the public iVideoGPT-64 action-conditioned checkpoint from HF mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve


REPOSITORY = "thuml/ivideogpt-bair-64-act-cond"
REVISION = "2863382adad16f49a8a6bc8e61c08a03e98bc5e5"
FILES = (
    "README.md",
    "tokenizer/config.json",
    "tokenizer/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/model.safetensors",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/upstream/ivideogpt-bair-64-act-cond")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    output = Path(args.output)
    for filename in FILES:
        destination = output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{args.endpoint}/{REPOSITORY}/resolve/{REVISION}/{filename}"
        print(f"downloading {filename}")
        urlretrieve(url, destination)
    (output / "pipeline_source.json").write_text(
        json.dumps({"repository": REPOSITORY, "revision": REVISION, "license": "MIT"}, indent=2) + "\n"
    )
    print(f"downloaded checkpoint to {output}")


if __name__ == "__main__":
    main()
