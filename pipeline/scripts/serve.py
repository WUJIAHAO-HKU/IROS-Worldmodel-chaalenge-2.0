#!/usr/bin/env python3
"""Run the local implementation of the official Track 2 API."""

import os

import uvicorn

from wam_pipeline.service import create_app, settings_from_env


if __name__ == "__main__":
    uvicorn.run(create_app(settings_from_env()), host="0.0.0.0", port=int(os.environ.get("WAM_PORT", "8000")))
