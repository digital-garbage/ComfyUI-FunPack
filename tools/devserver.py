"""Dev server: serves the app exactly as ComfyUI will, without ComfyUI.

Routes through core.serve, so the extension allowlist and the traversal guard are
exercised every time you look at the catalogue.

    python tools/devserver.py [port]
"""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config, serve as static  # noqa: E402

P = config.UI_PREFIX


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path in (P, P + "/"):
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
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
