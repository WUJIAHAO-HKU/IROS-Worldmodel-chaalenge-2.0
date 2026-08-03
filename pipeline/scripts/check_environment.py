#!/usr/bin/env python3
"""Report the isolated runtime prerequisites without installing or changing them."""

from __future__ import annotations

import importlib
import sys


def version(package: str) -> str:
    try:
        module = importlib.import_module(package)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return f"MISSING ({exc})"


print("python", sys.version.split()[0])
for package in ("torch", "numpy", "PIL", "h5py", "fastapi", "uvicorn", "httpx", "transformers", "diffusers", "accelerate", "safetensors"):
    print(package, version(package))
try:
    import torch

    print("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception:
    pass
