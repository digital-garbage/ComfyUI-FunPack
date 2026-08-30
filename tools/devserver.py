"""Dev server: serves the app exactly as ComfyUI will, without ComfyUI.

It mounts the REAL routes -- `core.routes.register` takes a route table rather
than reaching for ComfyUI's server -- so what the browser gets here is what
ComfyUI would give it, refusals and all. The previous version reimplemented the
handlers over http.server, which meant looking at the app exercised a copy: it
answered GET /api/pipeline with a payload assembled here, and answered POST with
501, so the whole edit-and-queue path could not be seen at all.

    python tools/devserver.py [port]

There is no queue behind it. Generating reaches /prompt, which ComfyUI serves
and this does not, and the app says so -- which is the honest answer and worth
seeing, since that is exactly what a person hitting Generate with ComfyUI down
would get.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _add_comfyui() -> str:
    """Put ComfyUI on the path, because most modules import it.

    Without this the dev server loads only the modules that need nothing from
    ComfyUI -- one out of nine -- and the app looks almost empty while reporting
    nothing wrong. Looking at the app is supposed to be how the app gets tested,
    so the preview has to see what ComfyUI would see.
    """
    import os
    candidates = [os.environ.get("COMFYUI_ROOT"),
                  str(Path.home() / "Documents" / "ComfyUI")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "comfy" / "supported_models.py").is_file():
            sys.path.append(candidate)
            return candidate
    return ""


def _reexec_with_comfyui_python(comfyui: str) -> None:
    """Restart under ComfyUI's own interpreter when this one cannot import torch.

    Finding ComfyUI's SOURCE is not enough: the modules import torch, and the
    interpreter that launches this may be a bare system python. Getting that
    wrong is not a crash, it is nine modules quietly absent and an app that
    looks almost empty -- so the server corrects it rather than serving a
    misleading page.
    """
    import os
    if os.environ.get("FUNPACK_DEVSERVER_REEXEC"):
        return                                   # already tried; do not loop
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    candidate = Path(comfyui) / "venv" / "bin" / "python"
    if not candidate.is_file():
        print(f"  WARNING: {sys.executable} cannot import torch and there is no "
              f"venv at {candidate}. Modules needing it will be absent.")
        return

    print(f"  re-running under {candidate} (this one has no torch)")
    os.environ["FUNPACK_DEVSERVER_REEXEC"] = "1"
    os.execv(str(candidate), [str(candidate), *sys.argv])


COMFYUI = _add_comfyui()
if COMFYUI:
    _reexec_with_comfyui_python(COMFYUI)

from aiohttp import web                          # noqa: E402
from core import config, routes as funpack_routes  # noqa: E402

P = config.UI_PREFIX


def build_app() -> web.Application:
    app = web.Application()
    routes = web.RouteTableDef()
    funpack_routes.register(routes)

    @routes.get("/")
    async def _root(_req):
        raise web.HTTPFound(P + "/")

    app.add_routes(routes)
    return app


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    print(f"FunPack v5 dev server: http://127.0.0.1:{port}{P}/")
    if COMFYUI:
        print(f"  with ComfyUI from {COMFYUI}")
    else:
        # Said out loud rather than left to be discovered as a half-empty page.
        print("  WITHOUT ComfyUI: every module that imports it will be absent. "
              "Set COMFYUI_ROOT to fix.")
    print("  no queue behind it: Generate will report that /prompt is not here")
    web.run_app(build_app(), host="127.0.0.1", port=port, print=None)
