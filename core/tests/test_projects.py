"""The project store, and the same store over HTTP.

A project is the one thing in the app that has to survive a reload, so the tests
that matter here are about what happens to it when the input is not what the app
would have sent: a hand-edited file, a body that is not an object, an id that is
really a path.
"""

import json
import socket
import threading

import pytest

from core import config, projects


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    return tmp_path / "projects"


# ── the model ────────────────────────────────────────────────────────────


def test_a_new_project_can_be_typed_into(store):
    """One empty scene, not none: a timeline with nothing on it has nowhere to
    put a prompt, so an empty new project would need an Add before it was usable."""
    made = projects.create("First")
    assert made.name == "First"
    assert len(made.scenes) == 1
    assert made.scenes[0].text == ""


def test_a_project_survives_a_round_trip(store):
    made = projects.create("Round trip")
    made.scenes.append(projects.Scene(text="a wolf on a ridge"))
    projects.save(made)

    read = projects.get(made.id)
    assert [s.text for s in read.scenes] == ["", "a wolf on a ridge"]
    assert read.name == "Round trip"


def test_scenes_keep_their_order(store):
    made = projects.create("Ordered")
    made.scenes = [projects.Scene(text=str(i)) for i in range(6)]
    projects.save(made)
    assert [s.text for s in projects.get(made.id).scenes] == [str(i) for i in range(6)]


def test_a_hand_edited_file_does_not_crash_the_app(store):
    """The store reads files a person can edit. Every one of these used to be an
    attribute error on a dict that was a list, or a str that was None."""
    made = projects.create("Edited")
    path = store / f"{made.id}.json"

    for body in ('{"scenes": "not a list"}', '{"scenes": [null, 3, "x"]}',
                 '{"name": 42}', '[]', '"just a string"', '{}'):
        path.write_text(body, encoding="utf-8")
        read = projects.get(made.id)
        assert read is not None, body
        assert isinstance(read.scenes, list)
        assert isinstance(read.name, str)


def test_unreadable_json_is_not_a_project(store):
    made = projects.create("Broken")
    (store / f"{made.id}.json").write_text("{oh no", encoding="utf-8")
    assert projects.get(made.id) is None
    assert projects.listing() == []      # and it does not take the list down with it


def test_a_scene_without_an_id_gets_one(store):
    """Ids address a scene from the client. Two scenes sharing one, or a scene
    with none, is a selection that lands on the wrong row."""
    p = projects.Project.from_dict({"scenes": [{"text": "a"}, {"text": "b"}]})
    ids = [s.id for s in p.scenes]
    assert all(projects.is_id(i) for i in ids)
    assert len(set(ids)) == 2


def test_an_id_from_a_request_cannot_be_a_path(store):
    """The filename is a generated id or the call fails. A sanitiser that
    repairs "../../x" writes somewhere nobody asked for and says it saved."""
    for bad in ("../escape", "a/b", "", "..", "x" * 64, None, 5, "ABCDEF123456"):
        assert projects.is_id(bad) is False
        assert projects.get(bad) is None
        assert projects.delete(bad) is False


def test_the_listing_is_newest_first(store):
    a = projects.create("A")
    b = projects.create("B")
    a.name = "A again"
    projects.save(a)                     # touches updated_at
    assert [p["id"] for p in projects.listing()] == [a.id, b.id]
    assert [p["name"] for p in projects.listing()][0] == "A again"


def test_a_long_name_is_cut_not_refused(store):
    made = projects.create("x" * 500)
    assert len(made.name) == projects.MAX_NAME


def test_delete_removes_it(store):
    made = projects.create("Doomed")
    assert projects.delete(made.id) is True
    assert projects.get(made.id) is None
    assert projects.delete(made.id) is False


def test_a_save_is_atomic(store):
    """A reader must never see half a project. The temp file is what makes that
    true, and a leftover .tmp in the directory would be read as one."""
    made = projects.create("Atomic")
    projects.save(made)
    assert list(store.glob("*.tmp")) == []
    assert len(projects.listing()) == 1


# ── over HTTP ────────────────────────────────────────────────────────────

pytest.importorskip("aiohttp")


@pytest.fixture
def server(store):
    from aiohttp import web as aioweb
    from core import routes

    app = aioweb.Application()
    table = aioweb.RouteTableDef()
    routes.register(table, prefix="/funpack")
    app.add_routes(table)

    holder = {"ready": threading.Event()}

    def run():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = aioweb.AppRunner(app)
        loop.run_until_complete(runner.setup())
        site = aioweb.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        holder["port"] = site._server.sockets[0].getsockname()[1]
        holder["loop"] = loop
        holder["ready"].set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert holder["ready"].wait(20), "the test server did not start"
    yield holder["port"]
    holder["loop"].call_soon_threadsafe(holder["loop"].stop)


def _request(port, method, path, body=None):
    payload = b"" if body is None else json.dumps(body).encode()
    head = (f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        sock.sendall(head + payload)
        chunks = []
        while True:
            got = sock.recv(65536)
            if not got:
                break
            chunks.append(got)
    header, _, rest = b"".join(chunks).partition(b"\r\n\r\n")
    status = int(header.split()[1])
    try:
        return status, json.loads(rest.decode() or "{}")
    except ValueError:
        return status, {"raw": rest.decode(errors="replace")}


def test_create_list_read_save_delete(server):
    status, made = _request(server, "POST", "/funpack/api/projects", {"name": "HTTP"})
    assert status == 200 and made["name"] == "HTTP"
    pid = made["id"]

    status, listed = _request(server, "GET", "/funpack/api/projects")
    assert status == 200 and [p["id"] for p in listed["projects"]] == [pid]
    assert listed["projects"][0]["scenes"] == 1

    made["scenes"].append({"text": "a second scene"})
    status, saved = _request(server, "PUT", f"/funpack/api/projects/{pid}", made)
    assert status == 200 and len(saved["scenes"]) == 2

    status, read = _request(server, "GET", f"/funpack/api/projects/{pid}")
    assert status == 200 and read["scenes"][1]["text"] == "a second scene"

    assert _request(server, "DELETE", f"/funpack/api/projects/{pid}")[0] == 200
    assert _request(server, "GET", f"/funpack/api/projects/{pid}")[0] == 404


def test_a_put_cannot_rename_its_own_target(server):
    """The id comes from the URL. A body that could name a different project is
    a write over someone else's work, reported as a success."""
    _, first = _request(server, "POST", "/funpack/api/projects", {"name": "First"})
    _, second = _request(server, "POST", "/funpack/api/projects", {"name": "Second"})

    _request(server, "PUT", f"/funpack/api/projects/{first['id']}",
             {**first, "id": second["id"], "name": "Overwritten"})

    _, untouched = _request(server, "GET", f"/funpack/api/projects/{second['id']}")
    assert untouched["name"] == "Second"
    _, target = _request(server, "GET", f"/funpack/api/projects/{first['id']}")
    assert target["name"] == "Overwritten"


def test_a_body_that_is_not_an_object_is_refused(server):
    _, made = _request(server, "POST", "/funpack/api/projects", {"name": "Shape"})
    for body in ([], "text", 7):
        status, _ = _request(server, "PUT", f"/funpack/api/projects/{made['id']}", body)
        assert status == 400, body


def test_saving_something_that_is_not_there_is_a_404(server):
    status, _ = _request(server, "PUT", "/funpack/api/projects/0123456789ab", {"name": "Ghost"})
    assert status == 404


def test_a_path_shaped_id_is_refused_not_served(server):
    for pid in ("..", "%2e%2e", "a/b"):
        status, _ = _request(server, "GET", f"/funpack/api/projects/{pid}")
        assert status in (404, 400), pid


# --- moving a project between machines ----------------------------------------

def test_a_downloaded_project_imports_back_as_a_new_one(server):
    """The round trip the feature exists for: save it, move the file
    somewhere else, load it there."""
    _, made = _request(server, "POST", "/funpack/api/projects", {"name": "Portable"})
    made["scenes"].append({"text": "a second scene"})
    _request(server, "PUT", f"/funpack/api/projects/{made['id']}", made)

    status, downloaded = _request(server, "GET", f"/funpack/api/projects/{made['id']}/download")
    assert status == 200
    assert downloaded["name"] == "Portable"
    assert len(downloaded["scenes"]) == 2

    status, imported = _request(server, "POST", "/funpack/api/projects/import", downloaded)
    assert status == 200
    assert imported["name"] == "Portable"
    assert [s["text"] for s in imported["scenes"]] == ["", "a second scene"]

    status, listed = _request(server, "GET", "/funpack/api/projects")
    assert sorted(p["id"] for p in listed["projects"]) == sorted([made["id"], imported["id"]]), \
        "the import did not land as a project of its own"


def test_importing_a_project_never_reuses_its_old_id(server):
    """Importing the SAME file twice -- or importing it back on the machine it
    came from -- must not silently overwrite whatever already has that id."""
    _, made = _request(server, "POST", "/funpack/api/projects", {"name": "Original"})
    status, downloaded = _request(server, "GET", f"/funpack/api/projects/{made['id']}/download")
    assert downloaded["id"] == made["id"]

    status, imported = _request(server, "POST", "/funpack/api/projects/import", downloaded)
    assert status == 200
    assert imported["id"] != made["id"]

    # The original is untouched -- an id collision would have overwritten it.
    _, original = _request(server, "GET", f"/funpack/api/projects/{made['id']}")
    assert original["name"] == "Original"


def test_an_id_offered_in_the_body_does_not_win_either(server):
    """Not just the downloaded shape -- ANY id in an imported body is a
    liability, including a made-up one that could collide with something
    imported later."""
    status, imported = _request(server, "POST", "/funpack/api/projects/import",
                                {"id": "aaaaaaaaaaaa", "name": "Sneaky", "scenes": [{"text": ""}]})
    assert status == 200
    assert imported["id"] != "aaaaaaaaaaaa"


def test_something_that_does_not_look_like_a_project_is_refused(server):
    for body in ({"name": "not a project"}, {}):
        status, resp = _request(server, "POST", "/funpack/api/projects/import", body)
        assert status == 400, body
        assert "problems" in resp


def test_a_scenes_field_of_the_wrong_shape_degrades_to_empty_rather_than_erroring(server):
    """`scenes` present but not a list is the same "never trust a field" rule
    every other project load already follows (Project.from_dict), not a
    special case this route invents -- an import is not the one place a
    malformed file gets to crash instead of degrading."""
    status, imported = _request(server, "POST", "/funpack/api/projects/import",
                                {"name": "Odd", "scenes": "not a list"})
    assert status == 200, imported
    assert imported["scenes"] == []


def test_downloading_a_project_that_does_not_exist_is_a_404(server):
    status, _ = _request(server, "GET", "/funpack/api/projects/0123456789ab/download")
    assert status == 404


def test_a_disk_failure_during_import_is_not_blamed_on_the_file(server, monkeypatch):
    """from_dict() is lenient by design -- it cannot actually raise on a
    well-formed body. The only thing that can fail here is save()'s own I/O,
    which is not the file's fault and should not be reported as though it
    were."""
    from core import projects

    def broken_save(_project):
        raise OSError("No space left on device")
    monkeypatch.setattr(projects, "save", broken_save)

    status, resp = _request(server, "POST", "/funpack/api/projects/import",
                            {"name": "Fine", "scenes": [{"text": ""}]})
    assert status == 500, resp
    assert "No space left on device" in resp["problems"][0]
    assert "invalid" not in resp["problems"][0].lower()


# --- what the project is generated at ----------------------------------------

def test_video_settings_survive_a_round_trip(store):
    made = projects.create("With settings")
    made.video = {"width": 832, "height": 480, "length": 97}
    projects.save(made)

    back = projects.get(made.id)
    assert back.video == {"width": 832, "height": 480, "length": 97}


def test_a_project_file_is_never_trusted_for_a_setting():
    """The one thing in this app that outlives the code that wrote it."""
    project = projects.Project.from_dict({"video": {
        "width": "832",          # a number that arrived as text
        "height": 480,
        "length": 0,             # out of range
        "batch": -4,             # out of range
        "huge": 999_999,         # out of range
        "flag": True,            # bool is an int in Python
        "name": "wide",          # not a number at all
        7: 512,                  # not a name
    }})
    assert project.video == {"width": 832, "height": 480}


def test_video_is_an_object_or_it_is_nothing():
    for junk in ([512, 512], "832x480", 7, None):
        assert projects.Project.from_dict({"video": junk}).video == {}


def test_a_project_with_no_settings_has_an_empty_one(store):
    assert projects.Project.from_dict({}).video == {}
    assert projects.create("Fresh").video == {}


def test_a_scene_keeps_a_crop_and_a_rating(store):
    made = projects.create("Rated")
    made.scenes = [projects.Scene(text="a cat", length=48, rating="liked")]
    projects.save(made)

    back = projects.get(made.id).scenes[0]
    assert back.length == 48
    assert back.rating == "liked"


def test_a_scene_field_read_back_is_never_trusted():
    scene = projects.Scene.from_dict({
        "length": "48",           # a number that arrived as text
        "rating": 7,              # not a label
    })
    assert scene.length == 48
    assert scene.rating == ""

    for junk in (0, -4, True, None, "soon", 999_999):
        assert projects.Scene.from_dict({"length": junk}).length is None, junk


def test_a_scenes_source_image_and_references_survive_a_round_trip(store):
    made = projects.create("Referenced")
    made.scenes = [projects.Scene(text="a fox", source_image="abcdef012345",
                                  references=["111111111111", "222222222222"])]
    projects.save(made)

    back = projects.get(made.id).scenes[0]
    assert back.source_image == "abcdef012345"
    assert back.references == ["111111111111", "222222222222"]


def test_a_scenes_media_ids_that_are_not_shaped_like_media_ids_are_dropped():
    scene = projects.Scene.from_dict({
        "source_image": "not a real id",
        "references": ["111111111111", "also not one", 4, None],
    })
    assert scene.source_image is None
    assert scene.references == ["111111111111"]


def test_a_scenes_references_field_of_the_wrong_shape_degrades_to_empty():
    for junk in ("111111111111", 4, None, {"a": 1}):
        assert projects.Scene.from_dict({"references": junk}).references == [], junk


# --- the project-level prompt fields: negative, anchor, postfix --------------

def test_negative_survives_a_round_trip(store):
    made = projects.create("Negative")
    made.negative = "blurry, low quality"
    projects.save(made)
    assert projects.get(made.id).negative == "blurry, low quality"


def test_anchor_survives_a_round_trip(store):
    made = projects.create("Anchor")
    made.anchor = "a lone rider on a frost-glider"
    projects.save(made)
    assert projects.get(made.id).anchor == "a lone rider on a frost-glider"


def test_postfix_and_its_toggle_survive_a_round_trip(store):
    made = projects.create("Postfix")
    made.postfix = "cinematic, 35mm"
    made.postfix_enabled = False
    projects.save(made)

    back = projects.get(made.id)
    assert back.postfix == "cinematic, 35mm"
    assert back.postfix_enabled is False


def test_a_fresh_project_has_empty_text_fields_and_postfix_on(store):
    made = projects.create("Fresh")
    assert made.negative == made.anchor == made.postfix == ""
    assert made.postfix_enabled is True


def test_a_project_file_from_before_these_fields_existed_reads_as_the_v4_default():
    """postfix_enabled defaults ON, matching v4 -- a project saved before this
    field existed must not silently read as "postfix off" just because
    nothing was there to say "on"."""
    project = projects.Project.from_dict({"name": "Old"})
    assert project.postfix_enabled is True
    assert project.negative == project.anchor == project.postfix == ""


def test_the_text_fields_read_back_are_never_trusted():
    for junk in (0, -4, True, None, ["a", "list"], {"not": "a string"}):
        project = projects.Project.from_dict(
            {"negative": junk, "anchor": junk, "postfix": junk})
        assert project.negative == project.anchor == project.postfix == "", junk


def test_postfix_enabled_read_back_is_never_trusted():
    for junk in ("yes", 1, 0, None, [], "false"):
        # Anything that is not a real bool degrades to the v4 default (on),
        # the same rule every other field here follows: a file cannot say
        # something with a value core does not recognise as that thing.
        assert projects.Project.from_dict({"postfix_enabled": junk}).postfix_enabled is True, junk
    assert projects.Project.from_dict({"postfix_enabled": False}).postfix_enabled is False


def test_variables_survive_a_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path)
    p = projects.create("v")
    p.variables = [{"name": "subject", "value": "a fox"}]
    projects.save(p)
    back = projects.get(p.id)
    assert back.variables == [{"name": "subject", "value": "a fox"}]


def test_a_fresh_project_has_no_variables():
    assert projects.Project().variables == []


def test_a_variable_with_no_name_is_dropped():
    p = projects.Project.from_dict({"variables": [
        {"name": "", "value": "x"}, {"name": "  ", "value": "y"}, {"name": "ok", "value": "z"},
    ]})
    assert p.variables == [{"name": "ok", "value": "z"}]


def test_a_leading_dollar_in_a_saved_variable_name_is_stripped():
    p = projects.Project.from_dict({"variables": [{"name": "$subject", "value": "fox"}]})
    assert p.variables == [{"name": "subject", "value": "fox"}]


def test_a_variables_value_that_is_not_a_string_reads_back_empty():
    p = projects.Project.from_dict({"variables": [{"name": "a", "value": 5}, {"name": "b"}]})
    assert p.variables == [{"name": "a", "value": ""}, {"name": "b", "value": ""}]


def test_variables_that_are_not_a_list_read_back_empty():
    for junk in ("nope", {"a": "b"}, 5, None):
        assert projects.Project.from_dict({"variables": junk}).variables == []


def test_entries_in_variables_that_are_not_objects_are_dropped():
    p = projects.Project.from_dict({"variables": ["nope", 5, {"name": "ok", "value": "1"}]})
    assert p.variables == [{"name": "ok", "value": "1"}]


def test_templates_and_editor_settings_survive_a_save_and_bad_entries_do_not():
    from core.projects import Project
    p = Project.from_dict({
        "prompt_templates": [
            {"name": "A", "anchor": "x", "scenes": ["s1", 5], "variables": [{"name": "$v", "value": "1"}]},
            {"name": "  "}, "junk", {"name": "A", "scenes": ["later wins"]}, {"name": "V4", "prompt": "old"}],
        "active_prompt_template": "A", "editor_settings": {"autocomplete": False, 3: "x"}})
    d = p.to_dict()
    names = {t["name"]: t for t in d["prompt_templates"]}
    assert names["A"]["scenes"] == ["later wins"] and names["V4"]["prompt"] == "old"
    assert d["editor_settings"] == {"autocomplete": False} and d["active_prompt_template"] == "A"
    assert Project.from_dict(d).to_dict() == d


def test_a_template_with_no_text_is_dropped_not_kept_as_a_wipe():
    from core import projects
    kept = projects._clean_templates([{"name": "empty"}, {"name": "v4", "prompt": "x"},
                                      {"name": "new", "scenes": ["a"]}])
    assert [t["name"] for t in kept] == ["v4", "new"]


# ── the editor's whole saved state ───────────────────────────────────────


def _v4():
    path = __import__("pathlib").Path(__file__).parent / "fixtures" / "v4_project.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_v4_project_comes_back_whole():
    """Every field v4 wrote -- timeline edits, audio and overlay lanes, renders, the pipeline --
    survives a save and a read. A dropped field would be the editor silently losing work."""
    raw = _v4()
    back = projects.Project.from_dict(raw).to_dict()
    gone = {k for k in raw if k not in back}
    assert gone == {"global_prompt"}, gone           # the Story is derived from the scenes now
    for key in raw:
        if key in ("global_prompt", "id", "updated_at", "scenes", "editor_settings"):
            continue
        assert back[key] == raw[key], key
    assert back["editor_settings"] == raw["editor_settings"]
    for mine, theirs in zip(back["scenes"], raw["scenes"]):
        for key, want in theirs.items():
            assert mine[key] == want, (theirs["id"], key)


def test_a_clients_own_scene_ids_are_kept_so_renders_still_point_at_their_scenes():
    back = projects.Project.from_dict(_v4())
    assert [s.id for s in back.scenes] == ["s1abc", "s2xyz"]
    assert set(back.scene_renders) == {"s1abc"} and back.timeline_order == ["s2xyz", "s1abc"]


def test_a_scene_id_that_could_be_a_path_is_replaced():
    for bad in ("../x", "a/b", "a b", "", "x" * 65, None, 5):
        assert projects.is_token(projects.Scene.from_dict({"id": bad}).id), bad
    assert projects.safe_part("../../etc/passwd") == "etcpasswd"


def test_timeline_fields_are_checked_not_trusted():
    s = projects.Scene.from_dict({
        "audio_volume": "loud", "frames": -3, "fps": 0, "frames_mode": "sideways", "source_in": -1,
        "source_dur": float("nan"), "gap_after_sec": True, "transition_frames": 1.5e9,
        "effects": {"blur": 0.2, "x" * 99: 1, "nested": {"a": 1}, "ok": "fill"},
        "source": {"type": "nope", "guide_strength": 7, "media_ref": ["x"]},
        "guides": ["a", {"k": 1}, 3], "gen_unit_id": "../x", "excluded": "yes"})
    assert (s.audio_volume, s.frames, s.fps, s.frames_mode) == (1.0, None, None, "project")
    assert (s.source_in, s.source_dur, s.gap_after_sec, s.transition_frames) == (0.0, None, 0.0, None)
    assert s.effects == {"blur": 0.2, "ok": "fill"}
    assert s.source == {"type": "carry", "media_ref": None, "target": None, "frame_ref": None,
                        "guide_strength": None}
    assert s.guides == [{"k": 1}] and s.gen_unit_id is None and s.excluded is False


def test_free_form_project_fields_must_be_the_right_shape_and_size():
    p = projects.Project.from_dict({
        "audio_tracks": [{"id": "a"}, "x", 3], "overlay_tracks": "nope", "models": [1],
        "scene_renders": {"s": "x" * (5 * 1024 * 1024)}, "timeline_order": ["a", "../b", 4],
        "num_frames_per_scene": 0, "width": "wide", "generation_mode": "weird", "references": ["m1", "", 7]})
    assert p.audio_tracks == [{"id": "a"}] and p.overlay_tracks == []
    assert p.models == {"slots": []} and p.scene_renders == {} and p.timeline_order == ["a"]
    assert (p.num_frames_per_scene, p.width, p.generation_mode, p.references) == (97, 768, "i2v", ["m1"])
