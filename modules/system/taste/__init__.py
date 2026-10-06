"""Taste keys: named stores of what you liked and disliked, for features that learn.

This module learns nothing itself. It names the key a run teaches, keeps each
learning feature's capture of the latest run until you rate it, and answers
"which way is liked?" per feature. Features reach it through the `taste_store`
capability, so turning this module off turns every learning feature off with a
message instead of breaking it.
"""

import asyncio
import os
import tempfile

from . import store, value
from ... import _core

ID = "taste"
TITLE = "Taste key"
MOUNT = "generation.sampling"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

SETTINGS = {
    "key": {
        "type": "text", "default": "",
        "label": "Taste key",
        "hint": "Name what your ratings teach, e.g. a style or a character. Empty = learning features stay off.",
    },
}

_OPTION = "funpack_taste_key"
_judges = {}


class Handle:
    """One key, as a learning feature sees it: capture now, ask for a direction later."""

    def __init__(self, key):
        self.key = key

    def capture(self, kind, rows, keep=store.MAX_ROWS, only=None, mixed=False):
        store.capture(self.key, kind, rows, keep=keep, only=only, mixed=mixed)

    def rows(self, kind, blind_to=None):
        """Every rated row of `kind`, oldest first: {"reward", "rows", "prompt_id"}.
        `blind_to` ("image"/"composition"): dislikes blamed on that axis alone read as
        neutral -- for a learner that cannot tell whether that half was the problem."""
        return store.blind(store.load(self.key, kind)["rows"], blind_to)

    def collect(self, patcher, key, kind, keep=store.MAX_ROWS, fresh=None, only=None, mixed=False):
        """A dict to fill during sampling; kept as `kind`'s capture when the
        sampling call finishes. An interrupted or failed run keeps nothing, so
        a half-finished clip can never be what a rating teaches.

        `fresh()` runs as each sampling starts. What was learned must be read
        THERE, not at install: ComfyUI caches the node that installs modifiers
        (only the seed changes between runs), so install-time reads would ignore
        every rating after the first run."""
        from comfy.patcher_extension import WrappersMP
        rows = {}

        def outer(executor, *args, **kwargs):
            rows.clear()
            if fresh is not None:
                fresh()
            out = executor(*args, **kwargs)
            self.capture(kind, rows, keep=keep, only=only, mixed=mixed)
            return out

        patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)
        return rows

    def direction(self, kind, name):
        return store.direction(self.key, kind, name)

    def directions(self, kind, names, strength):
        """-> ({name: unit vector}, one-line summary) for the names that have
        enough ratings, with the ones still waiting named in the summary."""
        found, steering, waiting = {}, [], []
        for name in names:
            d, liked, disliked = self.direction(kind, name)
            if d is None:
                waiting.append(f"{name} ({liked}/{disliked})")
            else:
                found[name] = d
                steering.append(f"{name} ({liked}/{disliked})")
        parts = []
        if steering:
            parts.append(f"steering {', '.join(steering)} at {strength:g}" if strength > 0
                         else f"strength 0: learning only at {', '.join(steering)}")
        if waiting:
            parts.append(f"learning, needs 2 liked + 2 disliked at {', '.join(waiting)}")
        return found, f"key {self.key!r}: " + "; ".join(parts or ["learning"])

    def describe(self, video):
        return value.describe(video)

    def judge(self, kind, name):
        """A Judge trained on this key's `kind` rows holding `name`, or None
        while there are too few. Retrained only when the rows change."""
        rows = [r for r in store.load(self.key, kind)["rows"] if str(name) in r["rows"]]
        slot = (self.key, kind, str(name))
        stamp = tuple((r.get("prompt_id"), r["reward"]) for r in rows)
        hit = _judges.get(slot)
        if hit is None or hit[0] != stamp:
            hit = _judges[slot] = (stamp, value.Judge.train_on(
                [r["rows"][str(name)] for r in rows], [r["reward"] for r in rows]))
        return hit[1]

    def counts(self, kind, name=None):
        return store.counts(self.key, kind, name)


def install(patcher, values, key):
    name = str(values.get("key") or "").strip()
    if not name:
        return None
    if not store.valid(name):
        raise ValueError(f"{name!r} can't be a key name: letters, digits, space, _ . - only")
    patcher.model_options[_OPTION] = name
    return f"teaching {name!r}"


def taste_store(patcher):
    """The key this model's run teaches, or None when no key is set."""
    name = (getattr(patcher, "model_options", None) or {}).get(_OPTION)
    return Handle(name) if name else None


class Kind:
    """One kind of one key, for a panel that reads or clears it with no run going."""

    def __init__(self, key, kind):
        if not store.valid(key):
            raise ValueError(f"{key!r} is not a usable taste key name")
        self.key, self.kind = key, kind

    def rows(self):
        return store.load(self.key, self.kind)["rows"]

    def path(self):
        return store.kind_path(self.key, self.kind)

    def clear(self):
        store.clear_kind(self.key, self.kind)


def taste_latest_key():
    return store.latest_key()


def taste_kind(key, kind):
    return Kind(key, kind)


def routes(table, base, web):
    @table.get(base + "/keys")
    async def _keys(_req):
        return web.json_response({"keys": store.keys()})

    @table.post(base + "/rate")
    async def _rate(req):
        try:
            body = await req.json()
            out = store.rate(body.get("prompt_id"), body.get("rating"), body.get("axis"))
        except (ValueError, AttributeError) as exc:
            return web.json_response({"why": str(exc)}, status=400)
        # any module that learns from ratings (provides "on_rating") hears of it; one failing never loses the rating
        for spec in _core.registry.current().specs.values():
            hear = spec.provides.get("on_rating")
            if hear:
                try:
                    hear(body.get("prompt_id"), body.get("rating"), body.get("axis"))
                except Exception as exc:
                    _core.log.once(f"on_rating:{spec.id}", _core.log.ALERT, spec.title, f"could not learn from this rating: {exc}")
        return web.json_response(out)

    @table.post(base + "/generation")
    async def _generation(req):
        """A Generate is starting: captures still waiting for a rating are forgotten, except the run it just queued
        (`{prompt_id}`): a short run can capture before this request lands."""
        body = await req.json() if req.can_read_body else {}
        keep = body.get("prompt_id") if isinstance(body, dict) and isinstance(body.get("prompt_id"), str) else None
        return web.json_response({"dropped": store.new_generation(keep=keep)})

    @table.delete(base + "/keys/{name}")
    async def _delete(req):
        name = req.match_info["name"]
        try:
            removed = len(list(store._dir(name).glob("*"))) if store._dir(name).is_dir() else 0
            store.delete(name)
        except ValueError as exc:
            return web.json_response({"why": str(exc)}, status=400)
        return web.json_response({"keys": store.keys(), "deleted": name, "removed": removed})

    @table.get(base + "/keys/{name}/export")
    async def _export(req):
        """A key as a zip, to carry to another machine. Written to a temp file off the loop."""
        name = req.match_info["name"]
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        try:
            await asyncio.to_thread(store.export_key, name, tmp)
        except ValueError as exc:
            os.remove(tmp)
            return web.json_response({"why": str(exc)}, status=400)
        try:
            with open(tmp, "rb") as fh:
                body = fh.read()
        finally:
            os.remove(tmp)
        return web.Response(body=body, content_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})

    @table.post(base + "/keys/import")
    async def _import(req):
        """Take an exported key (the zip, as the raw request body) under `?name=`. Streamed to a
        temp file: a key can be tens of MB, past what reading a body whole allows."""
        name = req.query.get("name", "")
        if not store.valid(name):
            return web.json_response({"why": f"{name!r} is not a usable taste key name"}, status=400)
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                async for chunk in req.content.iter_chunked(1 << 20):
                    size += len(chunk)
                    if size > store.MAX_BYTES:
                        return web.json_response({"why": "that file is larger than 4 GB"}, status=413)
                    out.write(chunk)
            try:
                count = await asyncio.to_thread(store.import_key, name, tmp, req.query.get("overwrite") == "1")
            except FileExistsError:
                return web.json_response({"why": f'A key named "{name}" already exists.', "exists": True, "key": name}, status=409)
            except ValueError as exc:
                return web.json_response({"why": str(exc)}, status=400)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return web.json_response({"imported": name, "kinds": count, "keys": store.keys()})


PROVIDES = {"modifier": install, "taste_store": taste_store, "taste_kind": taste_kind, "taste_latest_key": taste_latest_key, "routes": routes}
