"""Serving a render and its preview segments over real HTTP, against real ffmpeg files."""

import http.client
import subprocess

import pytest

pytest.importorskip("aiohttp")

from core.tests.test_routes_shortcuts import server  # noqa: E402,F401
from core import projects  # noqa: E402
from modules.output.render import files, serving  # noqa: E402
from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg, probe  # noqa: E402,F401

BASE = "/funpack/api/m/render"


def get(port, path, method="GET", headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    out = (resp.status, dict(resp.getheaders()), body)
    conn.close()
    return out


@pytest.fixture(autouse=True)
def _fresh_caches():
    serving._remuxed.clear(), serving._remux_failed.clear(), serving._segments.clear()


def moov_at_end(path, seconds=2.0):
    """What ComfyUI's own video saver writes: a complete mp4 whose index is AFTER the picture data."""
    make_clip(path, seconds, faststart=False)
    assert files.moov_position(str(path)) == "end"
    return path


@needs_ffmpeg
def test_a_seekable_file_is_served_with_range_support(comfy, server):
    make_clip(comfy.output / "a.mp4")
    status, headers, body = get(server, f"{BASE}/result?filename=a.mp4&type=output")
    assert status == 200 and headers["Content-Type"] == "video/mp4" and 'filename="a.mp4"' in headers["Content-Disposition"]
    assert body == (comfy.output / "a.mp4").read_bytes()
    status, headers, part = get(server, f"{BASE}/result?filename=a.mp4", headers={"Range": "bytes=0-99"})
    assert status == 206 and len(part) == 100


@needs_ffmpeg
def test_a_moov_at_end_render_is_served_remuxed_and_never_raw(comfy, server):
    moov_at_end(comfy.output / "vhs.mp4")
    status, _h, body = get(server, f"{BASE}/result?filename=vhs.mp4")
    assert status == 200 and body != (comfy.output / "vhs.mp4").read_bytes()
    served = comfy.root / "served.mp4"
    served.write_bytes(body)
    assert files.moov_position(str(served)) == "front"
    # the same URL keeps serving the same bytes
    assert get(server, f"{BASE}/result?filename=vhs.mp4")[2] == body


@needs_ffmpeg
def test_a_file_still_being_written_is_503_for_get_and_head_and_never_served(comfy, server):
    full = moov_at_end(comfy.output / "full.mp4").read_bytes()
    (comfy.output / "partial.mp4").write_bytes(full[:len(full) // 3])        # no moov yet
    assert files.moov_position(str(comfy.output / "partial.mp4")) == "none"
    for method in ("GET", "HEAD"):
        status, headers, _b = get(server, f"{BASE}/result?filename=partial.mp4", method=method)
        assert status == 503 and headers.get("Retry-After") == "2", method
    # ... and when the write finishes the SAME url becomes playable
    (comfy.output / "partial.mp4").write_bytes(full)
    assert get(server, f"{BASE}/result?filename=partial.mp4")[0] == 200


@needs_ffmpeg
def test_a_file_that_cannot_be_remuxed_is_502_and_not_retried_every_request(comfy, server, monkeypatch):
    def box(kind, payload):
        return (len(payload) + 8).to_bytes(4, "big") + kind + payload
    (comfy.output / "bad.mp4").write_bytes(box(b"ftyp", b"isom0000") + box(b"mdat", b"x" * 64) + box(b"moov", b"junk" * 8))
    assert files.moov_position(str(comfy.output / "bad.mp4")) == "end"
    calls = []
    real = files.remux_faststart
    monkeypatch.setattr(files, "remux_faststart", lambda *a: (calls.append(1), real(*a))[1])
    assert get(server, f"{BASE}/result?filename=bad.mp4")[0] == 502
    assert get(server, f"{BASE}/result?filename=bad.mp4")[0] == 502
    assert len(calls) == 1                                                      # rate-limited


def test_a_missing_file_and_a_path_out_of_the_folder_are_404(comfy, server):
    (comfy.root / "secret.mp4").write_bytes(b"x")
    assert get(server, f"{BASE}/result?filename=nope.mp4")[0] == 404
    assert get(server, f"{BASE}/result?filename=secret.mp4&subfolder=..")[0] == 404
    assert get(server, f"{BASE}/result?filename=../secret.mp4")[0] == 404       # the basename is what is read
    assert get(server, f"{BASE}/result")[0] == 404


def test_a_non_video_file_is_served_as_it_is(comfy, server):
    (comfy.output / "p.png").write_bytes(b"\x89PNG")
    status, headers, body = get(server, f"{BASE}/result?filename=p.png")
    assert (status, headers["Content-Type"], body) == (200, "image/png", b"\x89PNG")


# --- preview segments ---------------------------------------------------------------


def _project(comfy, **scene):
    proj = projects.Project(frame_rate=25, num_frames_per_scene=25, scenes=[projects.Scene(id="s1", **scene)],
                            scene_renders={"s1": {"media": {"filename": "a.mp4", "subfolder": "", "type": "output"},
                                                  "inSec": 0.5, "promptId": "p"}})
    projects.save(proj)
    return proj


@needs_ffmpeg
def test_a_scene_segment_is_exactly_the_window_the_export_would_use(comfy, server):
    moov_at_end(comfy.output / "a.mp4", 4.0)
    proj = _project(comfy, source_in=0.5, frames=50, frames_mode="timeline")     # 2 s at 25 fps, from 0.5 + 0.5
    status, headers, body = get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1")
    assert status == 200 and headers["Cache-Control"] == "private, max-age=3600"
    (comfy.root / "seg.mp4").write_bytes(body)
    assert files.moov_position(str(comfy.root / "seg.mp4")) == "front"
    assert probe(comfy.root / "seg.mp4")[0] == pytest.approx(2.0, abs=0.2)
    assert get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1")[2] == body        # cached: same bytes


@needs_ffmpeg
def test_a_ghost_has_no_scene_so_the_window_comes_from_the_query(comfy, server):
    make_clip(comfy.output / "a.mp4", 4.0)
    proj = _project(comfy)
    url = (f"{BASE}/projects/{proj.id}/preview-segment/gone?filename=a.mp4&subfolder=&type=output"
           "&render_in=1&dur=1.5")
    status, _h, body = get(server, url)
    assert status == 200
    (comfy.root / "g.mp4").write_bytes(body)
    assert probe(comfy.root / "g.mp4")[0] == pytest.approx(1.5, abs=0.2)
    assert get(server, f"{BASE}/projects/{proj.id}/preview-segment/gone")[0] == 404


@needs_ffmpeg
def test_reverse_is_part_of_the_segments_identity_and_a_long_one_is_refused(comfy, server):
    make_clip(comfy.output / "a.mp4", 2.0)
    proj = _project(comfy, source_dur=1.0)
    fwd = get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1")[2]
    rev = get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1?rev=1")[2]
    assert fwd != rev
    long = _project(comfy, source_dur=90.0)
    status, _h, body = get(server, f"{BASE}/projects/{long.id}/preview-segment/s1?rev=1")
    assert status == 400 and b"every frame in memory" in body


def test_segment_edge_cases_are_answered_not_crashed(comfy, server):
    proj = _project(comfy)
    assert get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1")[0] == 400          # source not on disk
    assert get(server, f"{BASE}/projects/aaaaaaaaaaaa/preview-segment/s1")[0] == 404       # no such project
    excluded = projects.Project(scenes=[projects.Scene(id="s1", excluded=True)],
                                scene_renders={"s1": {"media": {"filename": "a.mp4"}}})
    projects.save(excluded)
    assert get(server, f"{BASE}/projects/{excluded.id}/preview-segment/s1")[0] == 404


@needs_ffmpeg
def test_the_cache_deletes_the_oldest_files_past_its_cap(comfy, monkeypatch):
    monkeypatch.setattr(serving, "CACHE_MAX", 2)
    cache = serving._Cache()
    paths = []
    for i in range(3):
        p = comfy.temp / f"s{i}.mp4"
        p.write_bytes(b"x")
        paths.append(p)
        cache.put(i, (None, str(p)))
    assert not paths[0].exists() and paths[1].exists() and paths[2].exists() and len(cache) == 2


@needs_ffmpeg
def test_the_url_window_wins_over_a_project_that_has_not_been_saved_yet(comfy, server):
    moov_at_end(comfy.output / "a.mp4", 4.0)
    proj = _project(comfy, source_in=0.0, frames=25, frames_mode="timeline")      # saved: 1 s
    status, _h, body = get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1?dur=2.5&src_in=1")
    assert status == 200
    (comfy.root / "seg.mp4").write_bytes(body)
    assert probe(comfy.root / "seg.mp4")[0] == pytest.approx(2.5, abs=0.2)
    # a different head trim is a different segment, not the cached one
    other = get(server, f"{BASE}/projects/{proj.id}/preview-segment/s1?dur=2.5&src_in=0")[2]
    assert other != body


@needs_ffmpeg
def test_a_ghost_whose_scene_is_not_saved_yet_still_gets_its_in_point(comfy, server):
    make_clip(comfy.output / "a.mp4", 4.0)
    proj = _project(comfy)
    url = (f"{BASE}/projects/{proj.id}/preview-segment/new?filename=a.mp4&subfolder=&type=output"
           "&render_in=0&src_in=2&dur=1.5")
    status, _h, a = get(server, url)
    status2, _h, b = get(server, url.replace("src_in=2", "src_in=0"))
    assert status == status2 == 200 and a != b


@needs_ffmpeg
def test_a_window_past_the_end_of_the_render_is_refused_not_served_empty(comfy, server):
    make_clip(comfy.output / "a.mp4", 1.0)
    proj = _project(comfy)
    url = (f"{BASE}/projects/{proj.id}/preview-segment/gone?filename=a.mp4&subfolder=&type=output"
           "&render_in=5&dur=1")
    status, _h, body = get(server, url)
    assert status == 400 and b"past the end" in body


@needs_ffmpeg
def test_last_frame_is_a_picture_with_its_brightness(comfy):
    src = make_clip(comfy.output / "lf.mp4", 2.0, audio=False)
    out = str(comfy.temp / "lf.png")
    bright = files.last_frame(str(src), out, 0.0, 2.0, 25.0)
    assert (comfy.temp / "lf.png").stat().st_size > 0 and bright > 20


def _post(port, path, body):
    import json
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = (resp.status, json.loads(resp.read() or b"{}"))
    conn.close()
    return out


@needs_ffmpeg
def test_last_frame_route_saves_a_picture_to_the_bin_once_and_flags_a_fade_to_black(comfy, server):
    from core import media
    moov_at_end(comfy.output / "a.mp4", 4.0)
    proj = _project(comfy, frames=50, frames_mode="timeline")
    serving._frames.clear()
    status, a = _post(server, f"{BASE}/projects/{proj.id}/last-frame", {"scene_id": "s1"})
    assert status == 200 and a["dark"] is False and media.path_for(a["media_id"])
    assert _post(server, f"{BASE}/projects/{proj.id}/last-frame", {"scene_id": "s1"})[1]["media_id"] == a["media_id"]     # same window: same file
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=2:r=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(comfy.output / "black.mp4")], check=True, capture_output=True)
    dark = _project(comfy, frames=50, frames_mode="timeline")
    dark.scene_renders["s1"] = {"media": {"filename": "black.mp4", "subfolder": "", "type": "output"}, "inSec": 0}
    projects.save(dark)
    assert _post(server, f"{BASE}/projects/{dark.id}/last-frame", {"scene_id": "s1"})[1]["dark"] is True
    assert _post(server, f"{BASE}/projects/{dark.id}/last-frame", {"scene_id": "nope"})[0] == 404
    assert _post(server, f"{BASE}/projects/{dark.id}/last-frame", {})[0] == 400
