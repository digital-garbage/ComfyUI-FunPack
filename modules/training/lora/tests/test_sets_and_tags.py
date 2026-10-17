"""Training sets, captions, rated results and the run record. No ComfyUI needed."""

import pytest

from modules.training.lora import captions, jobs, media_io, rated, store


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from core import config
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path / "training")


def out(name, **extra):
    return {"src": {"kind": "output", "type": "output", "subfolder": "", "filename": name}, **extra}


def test_tags_are_trimmed_and_unique():
    assert captions.split_tags(" a,B ,, a\nc ") == ["a", "B", "c"]
    assert captions.with_parts("red, tall", "zq", "end") == "zq, red, tall, end"
    assert captions.with_parts("red", "", "") == "red" and captions.with_parts("", "zq", "") == "zq"
    assert captions.with_parts("A fox runs.\nIt jumps, twice.", " zq ", None) == "zq, A fox runs.\nIt jumps, twice."
    assert captions.add_tags("a fox, running home.", ["A  Fox", "grass", "sky"]) == "a fox, running home., grass, sky"
    assert captions.add_tags("", ["x", "y"]) == "x, y" and captions.add_tags("a long prompt,", ["x"]) == "a long prompt, x"


def test_bulk_edits_only_the_chosen_items():
    items = [{"id": "x", "caption": "cat, dog"}, {"id": "y", "caption": "cat"}]
    assert captions.bulk(items, "remove", "cat", ids={"x"}) == 1
    assert items[0]["caption"] == "dog" and items[1]["caption"] == "cat"
    assert captions.bulk(items, "replace", "cat", "lion") == 1 and items[1]["caption"] == "lion"
    assert captions.bulk(items, "add", "") == 0


def test_frequency_counts_each_caption_once():
    assert captions.frequency(["a, b", "a, a", "c"]) == [("a", 2), ("b", 1), ("c", 1)]


def test_a_set_keeps_items_and_drops_what_cannot_be_resolved():
    store.create("one")
    doc, added = store.add("one", [out("a.png"), out("a.png"), out("../x.png"), {"src": {"kind": "output", "type": "output", "subfolder": "../..", "filename": "b.png"}},
                                   {"src": {"kind": "bin", "id": "not an id"}}, out("c.mp4", label="liked")])
    assert added == 2 and [i["media_kind"] for i in doc["items"]] == ["image", "video"] and doc["items"][1]["label"] == "liked"
    with pytest.raises(ValueError):
        store.create("one")


def test_resolve_stays_inside_its_folder(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.png").write_bytes(b"x")
    ok = {"kind": "output", "type": "output", "subfolder": "sub", "filename": "a.png"}
    assert store.resolve(ok, {"output": tmp_path}, lambda i: None) == (tmp_path / "sub" / "a.png").resolve()
    assert store.resolve({**ok, "filename": "gone.png"}, {"output": tmp_path}, lambda i: None) is None
    assert store.resolve({**ok, "type": "temp"}, {"output": tmp_path}, lambda i: None) is None


@pytest.mark.parametrize("rating,label", [("10", "liked"), ("1", "disliked"), ("Disliked: bad image", "bad_image"),
                                          ("Disliked: bad composition", "bad_composition"), ("-Just forget it-", None), ("", None)])
def test_ratings_become_labels(rating, label):
    assert rated.label_of(rating) == label


def test_rated_results_come_from_renders_and_earlier_takes_once_each():
    media = lambda n: {"filename": n, "subfolder": "", "type": "output"}
    doc = {"id": "p1", "name": "P", "scenes": [{"id": "s", "rating": "10", "rated": {"text": "a fox"}}],
           "scene_renders": {"s": {"media": media("a.mp4")}},
           "scene_variants": {"s": [{"media": media("a.mp4"), "rating": "10"}, {"media": media("b.mp4"), "rating": "Disliked: bad image"}, {"media": media("c.txt"), "rating": "10"}]}}
    found = rated.scan([doc])
    assert [(r["media"]["filename"], r["label"]) for r in found] == [("a.mp4", "liked"), ("b.mp4", "bad_image")]
    assert found[0]["caption"] == "a fox"


def test_a_run_record_is_kept_and_thinned():
    jobs.begin("one", "lora", 3, {"rank": 4})
    for v in (3.0, 2.0, 1.0):
        jobs.step(v)
    assert jobs.snapshot(1)["loss"] == [2.0, 1.0]
    jobs.finish("done", lora="lora.safetensors")
    runs = jobs.history()
    assert runs[0]["state"] == "done" and runs[0]["final_loss"] == 1.0
    assert jobs.record(runs[0]["run"])["loss"] == [3.0, 2.0, 1.0]
    assert jobs.forget(runs[0]["run"]) and jobs.history() == []
    assert len(jobs.thin(list(range(100000)))) == jobs.MAX_KEPT_LOSSES
    assert jobs.request_stop() is False


def test_buckets_share_one_shape_or_follow_the_picture():
    assert media_io.bucket_for(1.0, 512, 64) == (512, 512)
    w, h = media_io.bucket_for(16 / 9, 512, 32)
    assert w % 32 == 0 and h % 32 == 0 and w > h
    assert len(set(media_io.buckets_for([1.0, 2.0, 0.5], 512, 64, True))) == 1
    assert len(set(media_io.buckets_for([1.0, 2.0, 0.5], 512, 64, False))) == 3


def test_a_rated_prompt_becomes_a_caption_from_the_parts_asked_for():
    doc = {"anchor": "cinematic", "postfix": "35mm", "postfix_enabled": True, "variables": [{"name": "hero", "value": "Ann"}]}
    expand = lambda t: t.replace("fox", "red fox")
    resolve = lambda t, v: t.replace("$hero", "Ann")
    ask = lambda *include: rated.prompt_caption("  a fox\n runs $hero  home ", doc, include, expand, resolve)
    assert ask("main") == "a red fox runs $hero home", "variables stay as written unless asked for"
    assert ask("main", "variables") == "a red fox runs Ann home"
    assert ask("prefix", "main", "postfix", "variables") == "cinematic a red fox runs Ann home 35mm"
    assert ask() == ""
    assert rated.prompt_caption("x", {**doc, "postfix_enabled": False}, ("main", "postfix")) == "x"
    media = {"filename": "a.mp4", "subfolder": "", "type": "output"}
    project = {"id": "p", "anchor": "A", "scenes": [{"id": "s", "rating": "10", "rated": {"text": "a $hero fox"}}], "scene_renders": {"s": {"media": media}}}
    assert rated.scan([project], expand, resolve, ("prefix", "main"))[0]["caption"] == "A a $hero red fox"


def _clip(path, frames, fps=24):
    import av
    import numpy as np
    with av.open(str(path), "w") as box:
        s = box.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 3 % 255, dtype=np.uint8), format="rgb24")
            for packet in s.encode(frame):
                box.mux(packet)
        for packet in s.encode():
            box.mux(packet)


def test_every_clip_is_read_for_as_many_frames_as_the_shortest_one_gives(tmp_path):
    from modules.training.lora import train
    short, long = tmp_path / "s.mp4", tmp_path / "l.mp4"
    _clip(short, 45)
    _clip(long, 100)
    assert media_io.frames_at(short, 24) == 45 and media_io.frames_at(short, 12) == 23
    items = [({"media_kind": "video"}, short), ({"media_kind": "video"}, long), ({"media_kind": "image"}, tmp_path / "x.png")]
    h3 = {"fps": 24, "frames": lambda n: n, "fit": lambda a: 17 * ((a - 5) // 17) + 5}
    assert train._fit_frames({"fps": 0, "frames": 120}, h3, items)["frames"] == 39       # 17 x 2 + 5: the longest on the grid that fits 45
    assert train._fit_frames({"fps": 0, "frames": 33}, h3, items)["frames"] == 33        # nothing to lower
    with pytest.raises(ValueError, match="too short"):
        train._fit_frames({"fps": 0, "frames": 120}, {**h3, "fit": lambda a: 0}, items)
    with pytest.raises(ValueError, match="s.mp4"):
        media_io.clip_frames(short, 80, 24, "start", 64, 48)


def test_clip_info_is_none_for_a_file_that_is_not_a_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    assert media_io.clip_info(bad) is None
    assert media_io.clip_info(tmp_path / "missing.mp4") is None
