"""Movie Editor sidecar configuration.

All values are overridable via environment variables so the app can point at a
remote ComfyUI on a rental box without code changes.
"""
import os
from pathlib import Path

# ComfyUI inference backend (the GPU box). No trailing slash.
COMFY_URL = os.environ.get("FUNPACK_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

# Where the editor stores projects (JSON) and, later, cached assets/latents.
DATA_DIR = Path(os.environ.get("FUNPACK_MOVIE_DATA", Path.home() / ".funpack_movie"))
PROJECTS_DIR = DATA_DIR / "projects"

# API-format workflow template the app fills and queues (user-exported; step 0).
BACKEND_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = Path(
    os.environ.get("FUNPACK_MOVIE_TEMPLATE", BACKEND_DIR / "templates" / "ltxav_chain.api.json")
)
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

# Sidecar bind.
HOST = os.environ.get("FUNPACK_MOVIE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FUNPACK_MOVIE_PORT", "8200"))

# Comma-separated CORS origins; "*" by default for local dev.
CORS_ORIGINS = [o.strip() for o in os.environ.get("FUNPACK_MOVIE_CORS", "*").split(",") if o.strip()]


def ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
