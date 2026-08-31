"""Read-only numerical integrity check for a PyTorch policy checkpoint."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tensors = [(name, value) for name, value in state.items() if isinstance(value, torch.Tensor)]
    nonfinite = [name for name, value in tensors if not torch.isfinite(value).all()]
    max_abs = max(float(value.detach().abs().max()) for _, value in tensors if value.numel())
    print({"tensor_keys": len(tensors), "nonfinite_keys": len(nonfinite), "max_abs": max_abs, "first_nonfinite": nonfinite[:3]})


if __name__ == "__main__":
    main()
