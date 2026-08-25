"""Conditioning tensors live in a binary sidecar, not as base64 inside the key's JSON.

One conditioning delta is 586x3584 floats — several megabytes of base64 in a document that
is parsed and rewritten in full on every generation. That is what makes a well-used key get
slower run after run.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, ".")


@pytest.fixture
def C():
    import conditioning
    return conditioning


def test_a_tensor_survives_the_round_trip(C):
    tree = {"global": {"liked_dir": {"direction": C.tensor_to_serializable(torch.arange(6.0))}}}
    store = {}
    out = C.externalize_tensors(tree, store)
    assert "state.global.liked_dir.direction" in store
    back = C.internalize_tensors(json.loads(json.dumps(out)), store)
    got = C.serializable_to_tensor(back["global"]["liked_dir"]["direction"])
    assert torch.equal(got.cpu().float(), torch.arange(6.0))


def test_the_json_keeps_a_readable_reference(C):
    tree = {"global": {"liked_dir": {"direction": C.tensor_to_serializable(torch.ones(4))}}}
    out = C.externalize_tensors(tree, {})
    ref = out["global"]["liked_dir"]["direction"]
    assert ref["__tensor_ref__"] == "state.global.liked_dir.direction"
    assert "data" not in ref                       # the base64 is gone
    assert ref["shape"] == [4]                     # still says what it points at


def test_no_tensors_means_no_sidecar(C):
    store = {}
    C.externalize_tensors({"global": {"phrase_memory": {"a": {"score": 1.0}}}}, store)
    assert store == {}


def test_a_missing_tensor_is_not_a_broken_run(C):
    """A reference the sidecar has lost reads as no tensor — a direction not applied."""
    tree = {"d": {"__tensor_ref__": "state.d", "shape": [4], "dtype": "float32"}}
    back = C.internalize_tensors(tree, {})
    assert back["d"] == tree["d"]
    with pytest.raises(Exception):
        C.serializable_to_tensor(back["d"])


def test_an_old_key_with_inline_blobs_still_reads(C):
    """No migration step: a key written before the sidecar existed is read as it stands."""
    blob = C.tensor_to_serializable(torch.arange(3.0))
    assert torch.equal(C.serializable_to_tensor(blob).cpu().float(), torch.arange(3.0))


def test_lists_are_walked_too(C):
    tree = {"anchors": [{"embed": C.tensor_to_serializable(torch.zeros(2))}]}
    store = {}
    out = C.externalize_tensors(tree, store)
    assert "state.anchors[0].embed" in store
    assert out["anchors"][0]["embed"]["__tensor_ref__"] == "state.anchors[0].embed"


def test_the_state_written_to_disk_is_json(C, tmp_path):
    """The whole point: what lands in the .json must contain no tensor payload."""
    from conditioning import FunPackVideoRefinerV2
    r = FunPackVideoRefinerV2()
    path = str(tmp_path / "key.json")
    tree = {"global": {"liked_dir": {"direction": C.tensor_to_serializable(torch.ones(1000))}}}
    payload = r._v2_store_state_tensors(path, tree)
    text = json.dumps(payload)
    assert "__tensor_ref__" in text
    assert len(text) < 500
    assert os.path.exists(path + C.TENSOR_SIDECAR_SUFFIX)
