"""The rate route over real HTTP, mounted by core's module-routes hook."""

import pytest
import torch

pytest.importorskip("aiohttp")

from core.tests.test_routes_shortcuts import _request, server  # noqa: F401,E402
from modules.system.taste import store  # noqa: E402


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def test_rate_and_list(server):  # noqa: F811
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate",
                           {"prompt_id": "p1", "rating": "liked"})
    assert status == 200 and out["recorded"] == ["reins"], out
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate",
                           {"prompt_id": "old", "rating": "liked"})
    assert status == 200 and out["why"], out
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate",
                           {"prompt_id": "p1", "rating": "meh"})
    assert status == 400
    assert _request(server, "GET", "/funpack/api/m/taste/keys")[1] == {"keys": ["fox"]}
    assert _request(server, "DELETE", "/funpack/api/m/taste/keys/fox")[1] == {"keys": [], "deleted": "fox", "removed": 1}


def test_a_dislike_over_http_carries_its_axis(server):  # noqa: F811
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate",
                           {"prompt_id": "p1", "rating": "disliked", "axis": "image"})
    assert status == 200 and out["recorded"] == ["reins"]
    assert store.load("fox", "reins")["rows"][0]["axis"] == "image"
    status, _ = _request(server, "POST", "/funpack/api/m/taste/rate",
                         {"prompt_id": "p1", "rating": "liked", "axis": "image"})
    assert status == 400


def _raw(server, method, path, body=b""):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", server)
    conn.request(method, path, body=body)
    resp = conn.getresponse()
    data = resp.read()
    return resp.status, data


def test_a_key_exports_and_imports_back_under_another_name(server):  # noqa: F811
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")
    _request(server, "POST", "/funpack/api/m/taste/rate", {"prompt_id": "p1", "rating": "liked"})
    status, zipped = _raw(server, "GET", "/funpack/api/m/taste/keys/fox/export")
    assert status == 200 and zipped[:2] == b"PK"
    status, out = _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=wolf", zipped)
    assert status == 200 and b'"wolf"' in out
    assert store.load("wolf", "reins")["rows"][0]["reward"] == 1.0
    status, out = _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=wolf", zipped)
    assert status == 409 and b'"exists": true' in out
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=wolf&overwrite=1", zipped)[0] == 200


def test_an_import_refuses_what_is_not_a_key(server):  # noqa: F811
    import io
    import zipfile
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", b"not a zip")[0] == 400
    for name in ("../evil.pt", "sub/dir.pt", "notes.txt", "reins.pending.pt"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, b"x")
        assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", buf.getvalue())[0] == 400, name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("reins.pt", b"this is not a torch file")
    status, out = _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", buf.getvalue())
    assert status == 400 and b"readable" in out
    assert store.keys() == []                                    # nothing half-imported
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=..%2Fx", b"")[0] == 400


def _zip_of(files):
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _torch_bytes(obj):
    import io
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def test_an_import_with_rows_the_learners_cannot_read_is_refused(server):  # noqa: F811
    for bad in ({"rows": [1, 2]}, {"rows": [{"reward": "high", "rows": {}}]},
                {"rows": [{"reward": 1.0, "rows": {"1": 5}}]},
                {"rows": [{"reward": 1.0, "axis": "mood", "rows": {"1": torch.ones(1)}}]}):
        status, out = _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x",
                           _zip_of({"reins.pt": _torch_bytes(bad)}))
        assert status == 400 and b"shape FunPack reads" in out, bad
    good = {"rows": [{"prompt_id": "p", "reward": 1.0, "rows": {"1": torch.ones(2)}}]}
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", _zip_of({"reins.pt": _torch_bytes(good)}))[0] == 200


def test_overwriting_a_key_drops_the_waiting_capture_so_the_next_rating_says_so(server):  # noqa: F811
    good = _zip_of({"reins.pt": _torch_bytes({"rows": []})})
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=fox&overwrite=1", good)[0] == 200
    assert store.latest_key() is None


def test_a_damaged_zip_and_an_empty_key_are_sentences_not_500s(server):  # noqa: F811
    data = bytearray(_zip_of({"reins.pt": b"x" * 4000}))
    data[40:60] = b"\xff" * 20                       # inside the entry's bytes
    status, out = _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", bytes(data))
    assert status == 400, out
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")      # waiting capture only
    status, out = _raw(server, "GET", "/funpack/api/m/taste/keys/fox/export")
    assert status == 400 and b"nothing to export" in out
    assert not store.valid("a ")


def test_reserved_names_shapes_and_missing_prompt_ids_are_refused(server):  # noqa: F811
    good = {"rows": [{"prompt_id": "p", "reward": 1.0, "rows": {"1": torch.ones(2)}}]}
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=latest.json", _zip_of({"reins.pt": _torch_bytes(good)}))[0] == 400
    mixed = {"rows": [{"prompt_id": "p", "reward": 1.0, "rows": {"1": torch.ones(2)}},
                      {"prompt_id": "q", "reward": -1.0, "rows": {"1": torch.ones(3)}}]}
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", _zip_of({"reins.pt": _torch_bytes(mixed)}))[0] == 400
    nopid = {"rows": [{"reward": 1.0, "rows": {"1": torch.ones(2)}}]}
    assert _raw(server, "POST", "/funpack/api/m/taste/keys/import?name=x", _zip_of({"reins.pt": _torch_bytes(nopid)}))[0] == 400
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate", {"rating": "liked"})
    assert status == 400


def test_the_generation_route_forgets_unrated_runs_over_http(server):  # noqa: F811
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p1")
    store.capture("fox", "reins", {1: torch.ones(2)}, prompt_id="p2")
    status, out = _request(server, "POST", "/funpack/api/m/taste/generation", {})
    assert status == 200 and out == {"dropped": 2}
    status, out = _request(server, "POST", "/funpack/api/m/taste/rate", {"prompt_id": "p1", "rating": "liked"})
    assert status == 200 and out["recorded"] == [] and "new Generate" in out["why"]
