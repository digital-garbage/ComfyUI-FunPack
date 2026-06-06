"""FunPack Movie Editor sidecar — FastAPI app.

Run:  uvicorn movie_editor.backend.app:app --host 127.0.0.1 --port 8200
(or:  python -m movie_editor.backend.app)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .routes import generate as generate_routes
from .routes import library as library_routes
from .routes import projects as project_routes
from .routes import timeline as timeline_routes

config.ensure_dirs()

app = FastAPI(title="FunPack Movie Editor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_routes.router)
app.include_router(timeline_routes.router)
app.include_router(generate_routes.router)
app.include_router(library_routes.router)


@app.get("/api/health")
def health():
    return {"ok": True, "comfy_url": config.COMFY_URL, "template": str(config.TEMPLATE_PATH)}


# Serve the frontend at root (mounted last so /api/* wins).
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
