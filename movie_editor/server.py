"""Movie Editor routes, served by ComfyUI's own aiohttp server.

UI:   GET  /funpack/movie/            (and static assets under /funpack/movie/<file>)
API:  /funpack/movie/api/*

Registered on PromptServer like the other /funpack/* routes (see templates.py,
batch_training.py). Pure logic lives in backend/ (timeline, projects, workflow);
bridge.py talks to ComfyUI (parse/library in-process, queue/history/view over loopback).
"""
from __future__ import annotations

import mimetypes
from typing import Optional

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI
    web = None
    PromptServer = None

from .backend import bridge, config, nodes, projects, workflow
from .backend.timeline import Project, build_combined_prompt

UI_PREFIX = "/funpack/movie"


# ── helpers ──────────────────────────────────────────────────────────────────

def _media_from_history(hist_entry: dict) -> list[dict]:
    out: list[dict] = []
    for node_id, node_out in (hist_entry.get("outputs") or {}).items():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename")
                if not fn:
                    continue
                out.append({
                    "node_id": node_id, "kind": key, "filename": fn,
                    "subfolder": item.get("subfolder", ""), "type": item.get("type", "output"),
                    "format": item.get("format"),
                })
    return out


def _project_or_404(pid: str) -> Project:
    p = projects.get(pid)
    if p is None:
        raise web.HTTPNotFound(reason=f"Project {pid} not found")
    return p


def _solo(p: Project, only_scene: Optional[str]) -> Project:
    if not only_scene:
        return p
    scene = next((s for s in p.scenes if s.id == only_scene), None)
    if scene is None:
        raise web.HTTPNotFound(reason=f"Scene {only_scene} not found")
    clone = Project.from_dict(p.to_dict())
    clone.scenes = [scene]
    return clone


def _serve_static(tail: str) -> "web.Response":
    root = config.FRONTEND_DIR.resolve()
    rel = (tail or "index.html").lstrip("/")
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise web.HTTPForbidden()
    if target.is_dir():
        target = target / "index.html"
    if not target.is_file():
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    if target.suffix == ".js":
        ctype = "text/javascript"  # be explicit; some platforms guess text/plain
    return web.Response(
        body=target.read_bytes(), content_type=ctype,
        headers={"Cache-Control": "no-store, max-age=0"},  # editor iterates fast; never cache
    )


# ── registration ─────────────────────────────────────────────────────────────

if web is not None and PromptServer is not None:
    routes = PromptServer.instance.routes
    config.ensure_dirs()

    # --- API: projects ---
    @routes.get(UI_PREFIX + "/api/health")
    async def _health(_req):
        return web.json_response({
            "ok": True,
            "comfy_url": config.comfy_base_url(),
            "template": str(config.TEMPLATE_PATH),
            "template_exists": config.TEMPLATE_PATH.exists(),
        })

    @routes.get(UI_PREFIX + "/api/projects")
    async def _list(_req):
        return web.json_response({"projects": projects.list_projects()})

    @routes.post(UI_PREFIX + "/api/projects")
    async def _create(req):
        body = await req.json()
        return web.json_response(projects.create(str(body.get("name") or "Untitled")).to_dict())

    @routes.get(UI_PREFIX + "/api/projects/{pid}")
    async def _get(req):
        return web.json_response(_project_or_404(req.match_info["pid"]).to_dict())

    @routes.put(UI_PREFIX + "/api/projects/{pid}")
    async def _update(req):
        pid = req.match_info["pid"]
        existing = _project_or_404(pid)
        body = await req.json()
        body["id"] = pid
        body.setdefault("created_at", existing.created_at)
        return web.json_response(projects.save(Project.from_dict(body)).to_dict())

    @routes.delete(UI_PREFIX + "/api/projects/{pid}")
    async def _delete(req):
        pid = req.match_info["pid"]
        if not projects.delete(pid):
            raise web.HTTPNotFound(reason=f"Project {pid} not found")
        return web.json_response({"deleted": pid})

    # --- API: timeline preview (round-trip integrity check) ---
    @routes.get(UI_PREFIX + "/api/projects/{pid}/preview")
    async def _preview(req):
        p = _project_or_404(req.match_info["pid"])
        include_excluded = req.query.get("include_excluded") == "true"
        prompt = build_combined_prompt(p, include_excluded=include_excluded)
        result = {"combined_prompt": prompt}
        try:
            parsed = bridge.parse_timeline(prompt, seed=p.seed)
            result["parsed"] = parsed
            expected = len([s for s in p.scenes if include_excluded or not s.excluded])
            got = len(parsed.get("scenes", []))
            result["expected_scenes"] = expected
            result["parsed_scenes"] = got
            if expected and got != expected:
                result["warning"] = (
                    f"Studio split produced {got} scene(s) but the timeline has {expected}. "
                    f"Check transition markers (each scene needs a recognized trigger before it)."
                )
        except Exception as e:  # noqa: BLE001
            result["parse_error"] = str(e)
        return web.json_response(result)

    # --- API: library ---
    @routes.get(UI_PREFIX + "/api/library/transitions")
    async def _transitions(_req):
        try:
            return web.json_response(bridge.transitions())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"transitions": [], "error": str(e)})

    # --- API: generate / status / result ---
    @routes.post(UI_PREFIX + "/api/projects/{pid}/generate")
    async def _generate(req):
        p = _project_or_404(req.match_info["pid"])
        body = await req.json() if req.can_read_body else {}
        target = _solo(p, body.get("only_scene"))
        prompt = build_combined_prompt(target)
        if not prompt.strip():
            raise web.HTTPBadRequest(reason="Nothing to generate — no active scene text.")
        try:
            graph = workflow.load_template()
            graph, applied = workflow.inject(graph, {
                "prompt": prompt, "seed": target.seed,
                "num_frames_per_scene": target.num_frames_per_scene,
                "frame_rate": target.frame_rate, "max_scenes": target.max_scenes,
            })
        except workflow.WorkflowError as e:
            raise web.HTTPBadRequest(reason=str(e))
        try:
            result = await bridge.queue_prompt(graph)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"Failed to queue: {e}")
        return web.json_response({"prompt_id": result.get("prompt_id"), "injected": applied})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/status/{prompt_id}")
    async def _status(req):
        prompt_id = req.match_info["prompt_id"]
        try:
            hist = await bridge.history(prompt_id)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"history unavailable: {e}")
        if prompt_id in hist:
            entry = hist[prompt_id]
            return web.json_response({
                "state": "completed",
                "media": _media_from_history(entry),
                "status": entry.get("status", {}),
            })
        try:
            running = await bridge.is_running(prompt_id)
        except Exception:  # noqa: BLE001
            running = True
        return web.json_response({"state": "running" if running else "pending", "media": []})

    @routes.get(UI_PREFIX + "/api/projects/{pid}/result")
    async def _result(req):
        try:
            data, ctype = await bridge.fetch_view(
                req.query.get("filename", ""),
                req.query.get("subfolder", ""),
                req.query.get("type", "output"),
            )
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"could not fetch result: {e}")
        return web.Response(body=data, content_type=ctype.split(";")[0])

    # --- API: models / pluggable node slots ---
    @routes.get(UI_PREFIX + "/api/node-roles")
    async def _node_roles(_req):
        return web.json_response({"roles": nodes.roles_payload()})

    @routes.get(UI_PREFIX + "/api/node-candidates/{role}")
    async def _node_candidates(req):
        role = req.match_info["role"]
        refresh = req.query.get("refresh") == "true"
        try:
            oi = await bridge.object_info(refresh=refresh)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        return web.json_response({"role": role, "candidates": nodes.candidates(oi, role)})

    @routes.get(UI_PREFIX + "/api/pipeline-ports")
    async def _pipeline_ports(_req):
        try:
            oi = await bridge.object_info()
        except Exception:
            oi = None
        return web.json_response({"ports": nodes.pipeline_ports(oi)})

    @routes.get(UI_PREFIX + "/api/all-nodes")
    async def _all_nodes(_req):
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        return web.json_response({"nodes": nodes.all_nodes(oi)})

    @routes.get(UI_PREFIX + "/api/node/{cls}")
    async def _node(req):
        try:
            oi = await bridge.object_info()
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"object_info unavailable: {e}")
        spec = nodes.describe_node(oi, req.match_info["cls"])
        if spec is None:
            raise web.HTTPNotFound(reason="Unknown node class")
        return web.json_response(spec)

    @routes.post(UI_PREFIX + "/api/models/refresh")
    async def _models_refresh(_req):
        try:
            await bridge.object_info(refresh=True)
        except Exception as e:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"refresh failed: {e}")
        return web.json_response({"ok": True})

    @routes.get(UI_PREFIX + "/api/models")
    async def _models_get(_req):
        return web.json_response(nodes.load_models())

    @routes.put(UI_PREFIX + "/api/models")
    async def _models_put(req):
        body = await req.json()
        return web.json_response(nodes.save_models(body))

    # --- UI: static frontend (must be registered AFTER api routes) ---
    @routes.get(UI_PREFIX)
    async def _root_redirect(_req):
        raise web.HTTPFound(UI_PREFIX + "/")

    @routes.get(UI_PREFIX + "/")
    async def _index(_req):
        return _serve_static("index.html")

    @routes.get(UI_PREFIX + "/{tail:.*}")
    async def _static(req):
        return _serve_static(req.match_info["tail"])

    print(f"[FunPack] Movie Editor available at {UI_PREFIX}/")
