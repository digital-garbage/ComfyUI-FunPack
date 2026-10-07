"""Local prompt drafting from saved shortcuts and the user's reviewed edits."""

import asyncio

from . import engine

ID = "conditioning_prompt_composer"
TITLE = "Prompt composer"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"


def routes(table, base, web):
    @table.get(base + "/status")
    async def _status(_req):
        return web.json_response(engine.status())

    @table.get(base + "/analysis")
    async def _analysis(_req):
        return web.json_response(await asyncio.to_thread(engine.analysis))

    @table.post(base + "/draft")
    async def _draft(req):
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"why": "that is not JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"why": "send a prompt object"}, status=400)
        try:
            return web.json_response(await asyncio.to_thread(engine.draft, body))
        except Exception as exc:  # noqa: BLE001 -- keep an optional feature from taking down Composer
            return web.json_response({"why": str(exc)}, status=400)

    @table.post(base + "/learn")
    async def _learn(req):
        try:
            body = await req.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"why": "that is not JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"why": "send a prompt object"}, status=400)
        try:
            return web.json_response(await asyncio.to_thread(engine.learn, body))
        except (ValueError, OSError) as exc:
            return web.json_response({"why": str(exc)}, status=400)

    @table.post(base + "/capture")
    async def _capture(req):
        try:
            body = await req.json()
            return web.json_response(await asyncio.to_thread(
                engine.capture, body.get("prompt_id"), body.get("text")))
        except (ValueError, AttributeError, OSError) as exc:
            return web.json_response({"why": str(exc)}, status=400)

    @table.post(base + "/settings")
    async def _settings(req):
        try:
            body = await req.json()
            return web.json_response(engine.set_enabled(body.get("enabled")))
        except (ValueError, AttributeError, OSError) as exc:
            return web.json_response({"why": str(exc)}, status=400)

    @table.post(base + "/clear")
    async def _clear(_req):
        try:
            return web.json_response(engine.clear())
        except OSError as exc:
            return web.json_response({"why": str(exc)}, status=500)


def prompt_transform(prompt):
    return engine.transform(prompt)


def on_rating(prompt_id, rating, axis=None):
    return engine.rate(prompt_id, rating, axis)


PROVIDES = {"routes": routes, "prompt_transform": prompt_transform, "on_rating": on_rating}
