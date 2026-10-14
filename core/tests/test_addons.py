"""Add-ons: what is offered, what a switch does, and what the next start installs."""

import asyncio
import json

import pytest

from core import addons, config, update


@pytest.fixture
def here(tmp_path, monkeypatch):
    """addons.json in a temp folder; `present` is the set of add-on ids whose packages 'are installed'."""
    monkeypatch.setattr(config, "ADDONS_FILE", tmp_path / "addons.json")
    present = set()
    monkeypatch.setattr(addons, "_present", lambda a: a["id"] in present)
    for a in addons.ADDONS:
        monkeypatch.setitem(a, "offered", lambda: True)
    return present


def state(aid):
    return next(a["state"] for a in addons.listing() if a["id"] == aid)


def test_nothing_is_wanted_until_asked(here):
    assert state("nvidia_vsr") == "available" and state("camera") == "available"
    assert not addons.enabled("camera")


def test_what_was_installed_before_add_ons_existed_stays_on(here):
    here.add("camera")
    assert state("camera") == "installed" and addons.enabled("camera")


def test_switching_on_waits_for_the_restart_and_off_keeps_the_packages(here):
    assert addons.set_wanted("nvidia_vsr", True)
    assert state("nvidia_vsr") == "pending" and not addons.enabled("nvidia_vsr")
    here.add("nvidia_vsr")                                   # the restart installed it
    assert state("nvidia_vsr") == "installed" and addons.enabled("nvidia_vsr")
    addons.set_wanted("nvidia_vsr", False)
    assert state("nvidia_vsr") == "disabled" and not addons.enabled("nvidia_vsr")


def test_an_add_on_this_machine_cannot_run_is_not_offered(here, monkeypatch):
    monkeypatch.setitem(addons._BY_ID["nvidia_vsr"], "offered", lambda: False)
    assert [a["id"] for a in addons.listing()] == ["camera"]
    assert addons.set_wanted("nvidia_vsr", True) is False
    assert addons.set_wanted("no-such-addon", True) is False


def test_the_start_installs_only_what_is_wanted_and_missing(here, monkeypatch):
    ran = []

    def fake(lines):
        ran.append(list(lines))
        here.add("nvidia_vsr")
        return {"ok": True, "detail": ""}
    monkeypatch.setattr(update, "install_lines", fake)
    addons.install_wanted()
    assert ran == [], "installed something nobody asked for"
    addons.set_wanted("nvidia_vsr", True)
    addons.install_wanted()
    assert ran == [["nvidia-vfx>=0.2"]]
    addons.install_wanted()
    assert len(ran) == 1, "installed again though it is already here"


def test_a_failed_install_is_said_and_not_retried_until_switched_off_and_on(here, monkeypatch):
    ran = []
    monkeypatch.setattr(update, "install_lines", lambda lines: ran.append(1) or {"ok": False, "detail": "no wheel for this Python"})
    addons.set_wanted("nvidia_vsr", True)
    addons.install_wanted()
    row = next(a for a in addons.listing() if a["id"] == "nvidia_vsr")
    assert row["failed"] == "no wheel for this Python" and row["state"] == "available", "a failed install must not read as waiting for a restart"
    addons.install_wanted()
    assert len(ran) == 1, "a broken install would be retried on every start"
    addons.set_wanted("nvidia_vsr", False)
    addons.set_wanted("nvidia_vsr", True)
    addons.install_wanted()
    assert len(ran) == 2


def test_install_lines_skips_what_is_present_and_moves_nothing(monkeypatch):
    seen = {}
    monkeypatch.setattr(update, "_installed_version", lambda n: "1" if n == "gguf" else None)
    monkeypatch.setattr(update, "_pip_freeze", lambda: {"torch": "2.10.0"})
    monkeypatch.setattr(update, "_install_each", lambda status, path, before, note, timeout, broken:
                        seen.update(lines=status["install"], pins=open(path).read()) or {"ok": True})
    update.install_lines(["gguf", "nvidia-vfx>=0.2"])
    assert seen["lines"] == ["nvidia-vfx>=0.2"]
    assert "torch==2.10.0" in seen["pins"]


def test_the_routes_list_and_switch(comfyui, here):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from core import routes

    async def go():
        app, table = web.Application(), web.RouteTableDef()
        routes.register(table, prefix="/funpack")
        app.add_routes(table)
        async with TestClient(TestServer(app)) as client:
            got = await (await client.get("/funpack/api/addons")).json()
            bad = await client.post("/funpack/api/addons", json={"id": "camera", "enabled": "yes"})
            ok = await (await client.post("/funpack/api/addons", json={"id": "camera", "enabled": True})).json()
            return got, bad.status, ok
    got, bad, ok = asyncio.run(go())
    assert {a["id"] for a in got["addons"]} == {"nvidia_vsr", "camera"}
    assert bad == 400
    assert next(a["state"] for a in ok["addons"] if a["id"] == "camera") == "pending"
    assert json.loads(config.ADDONS_FILE.read_text())["wanted"] == ["camera"]
