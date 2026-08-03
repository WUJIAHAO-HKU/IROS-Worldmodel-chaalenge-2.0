"""The published WorldArena Track 2 profile, centralized for all pipeline stages."""

from __future__ import annotations

API_VERSION = "1.0"
OFFICIAL_PROFILE_ID = "wa2-track2-robotwin-adjustbottle-v1"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
CONTEXT_FRAMES = 5
CONTEXT_ACTIONS = CONTEXT_FRAMES - 1
PREDICTION_FRAMES = 8
ACTION_DIM = 14
ACTION_REPRESENTATION = "robotwin_aloha_agilex_abs14"
MAX_BATCH_SIZE = 8
MAX_REQUEST_BYTES = 33_554_432
MAX_CONCURRENCY = 1
RECOMMENDED_TIMEOUT_MS = 600_000


def capabilities(model_version: str) -> dict:
    """Return the immutable capability response for the published profile."""
    return {
        "api_version": API_VERSION,
        "model_version": model_version,
        "profiles": [
            {
                "profile_id": OFFICIAL_PROFILE_ID,
                "image": {
                    "encoding": "png_base64",
                    "color_space": "RGB",
                    "height": IMAGE_HEIGHT,
                    "width": IMAGE_WIDTH,
                    "channels": IMAGE_CHANNELS,
                },
                "context_frames": {"min": CONTEXT_FRAMES, "max": CONTEXT_FRAMES},
                "prediction_frames": {"min": PREDICTION_FRAMES, "max": PREDICTION_FRAMES},
                "action": {
                    "dtype": "float32",
                    "dimension": ACTION_DIM,
                    "representation": ACTION_REPRESENTATION,
                },
                "state": {"supported": False, "dimension": None},
                "instruction": {"supported": True},
            }
        ],
        "limits": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_concurrency": MAX_CONCURRENCY,
            "recommended_timeout_ms": RECOMMENDED_TIMEOUT_MS,
        },
        "determinism": {"seed_supported": True, "same_seed_same_pixels": True},
    }
