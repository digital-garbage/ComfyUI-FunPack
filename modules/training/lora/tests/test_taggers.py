"""The automatic tagger, run on a tiny ONNX model built here (no download), and the download path with a fake fetch."""

import time

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from modules.training.lora import taggers


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    from core import config
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path / "training")
    taggers._sessions.clear()
    taggers._state.clear()


def make_tagger(tid="me/tiny"):
    """Input (1,16,16,3), four outputs: tag 0 and 3 are always sure, tag 1 never, tag 2 is a rating."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.zeros((16 * 16 * 3, 4), dtype=np.float32), "w")
    b = numpy_helper.from_array(np.array([9, -9, 9, 9], dtype=np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["input"], ["flat"], axis=1), helper.make_node("MatMul", ["flat", "w"], ["m"]),
         helper.make_node("Add", ["m", "b"], ["a"]), helper.make_node("Sigmoid", ["a"], ["out"])],
        "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 16, 3])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    d = taggers.folder(tid)
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(model.SerializeToString())
    (d / "selected_tags.csv").write_text("tag_id,name,category,count\n1,red_square,0,1\n2,never,0,1\n3,rating_general,9,1\n4,some_name,4,1\n", encoding="utf-8")
    return tid


def test_tags_come_out_by_threshold_and_ratings_never_do():
    from PIL import Image
    tid = make_tagger()
    assert taggers.ready(tid)
    assert taggers.run(tid, Image.new("RGB", (40, 20))) == ["red square"]
    assert taggers.run(tid, Image.new("RGB", (40, 20)), characters=True, character=0.5) == ["red square", "some name"]
    assert taggers.run(tid, Image.new("RGB", (40, 20)), blocked=["Red Square"]) == []


def test_a_job_hands_results_over_per_item():
    from PIL import Image
    tid = make_tagger()
    assert taggers.autotag(tid, [("a", "a.png"), ("b", "missing.png")], {}, lambda p: Image.new("RGB", (8, 8))) is True
    for _ in range(100):
        if taggers.job["state"] != "running":
            break
        time.sleep(0.05)
    assert taggers.job["state"] == "done" and taggers.job["results"]["a"] == ["red square"]


def test_an_unknown_name_is_not_a_tagger_and_one_that_is_not_downloaded_says_so():
    assert taggers.spec_of("../etc") is None and taggers.spec_of("a/b")["repo"] == "a/b"
    with pytest.raises(ValueError, match="not downloaded"):
        taggers.session("wd_vit_v3")


def test_download_goes_through_part_files(monkeypatch):
    def fetch(url, dest, progress):
        dest.write_bytes(b"data")
    assert taggers.download("wd_vit_v3", fetch=fetch) is True
    for _ in range(100):
        if taggers.ready("wd_vit_v3"):
            break
        time.sleep(0.05)
    assert taggers.ready("wd_vit_v3") and not list(taggers.folder("wd_vit_v3").glob("*.part"))
    assert taggers.remove("wd_vit_v3") and not taggers.ready("wd_vit_v3")
