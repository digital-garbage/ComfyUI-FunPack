"""Movie Editor configuration.

The editor now runs INSIDE ComfyUI (routes registered on ComfyUI's aiohttp
PromptServer), so there is no separate server to bind. We only need storage paths
and the template location; self-HTTP calls target ComfyUI's own host:port, resolved
at runtime from comfy.cli_args.
"""
import os
from pathlib import Path

# Where the editor stores projects (JSON) and, later, cached assets/latents.
DATA_DIR = Path(os.environ.get("FUNPACK_MOVIE_DATA", Path.home() / ".funpack_movie"))
PROJECTS_DIR = DATA_DIR / "projects"

# API-format workflow template the app fills and queues (user-exported; step 0).
BACKEND_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = Path(
    os.environ.get("FUNPACK_MOVIE_TEMPLATE", BACKEND_DIR / "templates" / "ltxav_chain.api.json")
)
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
# Saved copy of the user's real workflow — read only, to seed fixed-core widget
# values (the editor builds the graph in code; this is not a runtime template).
REFERENCE_DIR = BACKEND_DIR / "reference"


def comfy_base_url() -> str:
    """Base URL of the ComfyUI server we run inside, for loopback /prompt calls.

    The browser never needs this — the frontend is served by ComfyUI, so its API
    calls are same-origin (relative) and hit the correct port automatically. This is
    only for the backend talking to ComfyUI's own endpoints. We resolve the REAL port
    (not the 8188 default) from the running server / CLI args; env override wins.
    """
    override = os.environ.get("FUNPACK_COMFY_URL")
    if override:
        return override.rstrip("/")
    port = None
    # 1) The live server instance knows the port it actually bound.
    try:
        from server import PromptServer
        port = getattr(PromptServer.instance, "port", None)
    except Exception:
        pass
    # 2) CLI args (--port).
    if not port:
        try:
            from comfy.cli_args import args
            port = getattr(args, "port", None)
        except Exception:
            pass
    port = int(port or 8188)
    return f"http://127.0.0.1:{port}"  # always loopback for self-calls


def ensure_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
