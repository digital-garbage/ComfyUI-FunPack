"""Render, export and import as the page calls them: start a job, poll it, read the file it made."""

import http.client
import json
import time

import pytest

pytest.importorskip("aiohttp")

from core import media, projects  # noqa: E402
from core.tests.test_routes_shortcuts import server  # noqa: E402,F401
from modules.output.render import jobs  # noqa: E402
from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg, probe  # noqa: E402,F401

BASE = "/funpack/api/m/render"


def call(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    payload = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(data or b"{}")
    except ValueError:
        return resp.status, data


def finish(port, path, job_id, tries=200):
    for _ in range(tries):
        status, job = call(port, "GET", f"{path}/{job_id}")
        assert status == 200
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def clip(name="a.mp4", dur=1.0, **extra):
    return {"filename": name, "subfolder": "", "type": "output", "in": 0, "dur": dur, "fx": {}, "fps": 25,
            "w": 160, "h": 120, "transition": "", "tdur": 0, "volume": 1.0, **extra}


@needs_ffmpeg
def test_render_is_a_job_that_ends_with_a_playable_file(comfy, server):
    make_clip(comfy.output / "a.mp4", 1.0)
    proj = projects.save(projects.Project(width=160, height=120))
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": [clip(), clip()]})
    assert status == 200
    job = finish(server, f"{BASE}/projects/{proj.id}/render", queued["job_id"])
    assert job["state"] == "done" and job["clips"] == 2
    assert probe(comfy.temp / job["media"]["filename"])[0] == pytest.approx(2.0, abs=0.2)


def test_a_render_that_cannot_be_made_ends_in_an_error_with_a_sentence(comfy, server):
    proj = projects.save(projects.Project())
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": [clip("gone.mp4")]})
    job = finish(server, f"{BASE}/projects/{proj.id}/render", queued["job_id"])
    assert job["state"] == "error" and "not on disk" in job["detail"]


def test_nothing_to_render_is_refused_before_a_job_starts(comfy, server):
    proj = projects.save(projects.Project())
    status, body = call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": []})
    assert status == 400 and "Nothing to render" in body["detail"]
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/render", {"clips": "x"})[0] == 400
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/render", [])[0] == 400


def test_unknown_project_and_unknown_job_are_404(comfy, server):
    assert call(server, "POST", f"{BASE}/projects/aaaaaaaaaaaa/render", {"clips": [clip()]})[0] == 404
    proj = projects.save(projects.Project())
    assert call(server, "GET", f"{BASE}/projects/{proj.id}/render/nope")[0] == 404
    assert call(server, "GET", f"{BASE}/projects/{proj.id}/export-clips/nope")[0] == 404


@needs_ffmpeg
def test_export_one_clip_answers_directly_and_several_as_a_job(comfy, server):
    make_clip(comfy.output / "a.mp4", 2.0)
    proj = projects.save(projects.Project())
    status, out = call(server, "POST", f"{BASE}/projects/{proj.id}/export-clip",
                       {"clip": {"filename": "a.mp4", "in": 0.5, "dur": 1.0}})
    assert status == 200 and probe(comfy.temp / out["media"]["filename"])[0] == pytest.approx(1.0, abs=0.15)
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/export-clip", {"clip": {}})[0] == 400
    status, queued = call(server, "POST", f"{BASE}/projects/{proj.id}/export-clips",
                          {"clips": [{"filename": "a.mp4", "in": 0, "dur": 1.0}, {"filename": "a.mp4", "in": 1, "dur": 1.0}]})
    job = finish(server, f"{BASE}/projects/{proj.id}/export-clips", queued["job_id"])
    assert job["state"] == "done" and job["clips"] == 2
    assert call(server, "POST", f"{BASE}/projects/{proj.id}/export-clips", {"clips": []})[0] == 400


@needs_ffmpeg
def test_a_clip_imports_into_the_media_library_trimmed_and_named(comfy, server):
    make_clip(comfy.output / "a.mp4", 3.0)
    status, out = call(server, "POST", f"{BASE}/import-clip",
                       {"clip": {"filename": "a.mp4", "in": 1.0, "dur": 1.0}, "name": "my clip"})
    assert status == 200 and out["media"]["name"] == "my clip.mp4" and out["media"]["kind"] == "video"
    assert probe(media.path_for(out["media"]["id"]))[0] == pytest.approx(1.0, abs=0.15)
    status, whole = call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "a.mp4"}})
    assert status == 200 and media.path_for(whole["media"]["id"]).read_bytes() == (comfy.output / "a.mp4").read_bytes()
    assert call(server, "POST", f"{BASE}/import-clip", {"clip": {}})[0] == 400
    status, body = call(server, "POST", f"{BASE}/import-clip", {"clip": {"filename": "gone.mp4"}})
    assert status == 400 and "not on disk" in body["detail"]


def test_the_effects_and_seams_the_timeline_offers(server):
    status, lib = call(server, "GET", f"{BASE}/library")
    assert status == 200
    assert {"reverse", "crop", "flip_h", "fill_frame", "reset"} <= {e["id"] for e in lib["effects"]}
    assert [t["id"] for t in lib["video_transitions"]] == ["crossfade", "fadeblack", "wipeleft", "wiperight"]


def test_finished_jobs_are_forgotten_past_the_cap(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_JOBS", 3)
    ids = [jobs.new() for _ in range(5)]
    assert jobs.get(ids[0]) is None and jobs.get(ids[4]) is not None
