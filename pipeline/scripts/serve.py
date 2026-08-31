#!/usr/bin/env python3
"""Run the local implementation of the official Track 2 API."""

import os

import uvicorn

from wam_pipeline.service import create_app, settings_from_env


if __name__ == "__main__":
    # TLS is opt-in so existing local HTTP development remains compatible.  In
    # production/acceptance runs both paths are explicitly configured and the
    # same bearer-authenticated ASGI app is served over HTTPS.
    certfile = os.environ.get("WAM_SSL_CERTFILE") or None
    keyfile = os.environ.get("WAM_SSL_KEYFILE") or None
    if bool(certfile) != bool(keyfile):
        raise SystemExit("WAM_SSL_CERTFILE and WAM_SSL_KEYFILE must be provided together")
    uvicorn.run(
        create_app(settings_from_env()),
        host="0.0.0.0",
        port=int(os.environ.get("WAM_PORT", "8000")),
        ssl_certfile=certfile,
        ssl_keyfile=keyfile,
    )
