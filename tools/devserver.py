"""Dev server: serves the app exactly as ComfyUI will, without ComfyUI.

Routes through core.serve, so the extension allowlist and the traversal guard are
exercised every time you look at the catalogue.

    python tools/devserver.py [port]
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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

from core import config, routes, serve as static  # noqa: E402

P = config.UI_PREFIX


def _pipeline_payload():
    """The same answer the aiohttp route gives, so the preview shows what
    ComfyUI would."""
    from core import graph as graph_mod
    slots = []
    for _spec, make in routes.modules().providers("default_pipeline"):
        slots = make()
        break
    prompt, incomplete = graph_mod.build(slots)
    return {"slots": slots, "refused": [], "incomplete": incomplete,
            "queueable": not incomplete}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # The same manifest the aiohttp route serves, so what you see in the
        # browser here is what ComfyUI would send.
        if path == P + "/api/pipeline":
            body = json.dumps(_pipeline_payload()).encode()
            served = static.Served(200, body, "application/json")
        elif path == P + "/api/log":
            query = parse_qs(parsed.query)
            level = (query.get("level") or [None])[0]
            body = json.dumps({"levels": list(routes.log.LEVELS),
                               "records": routes.log.history(level)}).encode()
            served = static.Served(200, body, "application/json")
        elif path == P + "/api/modules":
            raw = parse_qs(parsed.query).get("traits")
            traits = [t for t in raw[0].split(",") if t] if raw else None
            body = json.dumps(routes.manifest(traits)).encode()
            served = static.Served(200, body, "application/json")
        elif path in (P, P + "/"):
            served = static.serve(config.APP_DIR, "index.html", config.APP_EXTS)
        elif path.startswith(P + "/app/"):
            served = static.serve(config.APP_DIR, path[len(P) + 5:], config.APP_EXTS)
        elif path.startswith(P + "/modules/"):
            served = static.serve(config.MODULES_DIR, path[len(P) + 9:], config.MODULE_EXTS)
        elif path == "/":
            self.send_response(302)
            self.send_header("Location", P + "/")
            self.end_headers()
            return
        else:
            served = static.Served(404)

        self.send_response(served.status)
        self.send_header("Content-Type", served.content_type)
        self.send_header("Content-Length", str(len(served.body)))
        for k, v in served.headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(served.body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    print(f"FunPack v5 dev server: http://127.0.0.1:{port}{P}/")
    if COMFYUI:
        print(f"  with ComfyUI from {COMFYUI}")
    else:
        # Said out loud rather than left to be discovered as a half-empty page.
        print("  WITHOUT ComfyUI: every module that imports it will be absent. "
              "Set COMFYUI_ROOT to fix.")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
