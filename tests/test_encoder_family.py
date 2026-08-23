"""The refinement logic has to follow the encoder, not a guess about it.

LTX-2.3 encodes with Gemma3, LTX-2.5 with Gemma4. Concept steering works by finding a phrase's
token ids inside the prompt's token ids and treating the match position as a sequence position,
so a tokenizer from the wrong family still returns spans — pointing at the wrong tokens. Nothing
raises, and the only symptom is that steering stops working. These tests pin the three places
that decide which tokenizer is used and what learned state it may be applied to.
"""
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

from conditioning import FunPackVideoRefiner, FunPackVideoRefinerV2, _ClipTokenizerAdapter  # noqa: E402


class FakeInner:
    """Comfy's inner tokenizers: text in, ids out, and no other kwargs."""

    def __init__(self, table, start=2):
        self.table = table
        self.start = start

    def __call__(self, text):
        ids = [self.start]
        for word in text.split():
            ids.extend(self.table.get(word, [99]))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "|".join(str(i) for i in ids)


def make_clip(name, table=None, start=2):
    """A CLIP shaped like comfy's: one named sub-tokenizer, named after the encoder."""
    sub = types.SimpleNamespace(
        tokenizer=FakeInner(table or {}, start=start),
        start_token=start,
        embedding_key=name,
    )
    top = types.SimpleNamespace(clip=name)
    setattr(top, name, sub)
    return types.SimpleNamespace(tokenizer=top)


# ----------------------------------------------------------------- resolving the tokenizer

def test_family_reads_the_encoder_behind_the_clip():
    assert FunPackVideoRefiner._encoder_family(make_clip("gemma3_12b")) == "gemma3_12b"
    assert FunPackVideoRefiner._encoder_family(make_clip("gemma4")) == "gemma4"
    assert FunPackVideoRefiner._encoder_family(make_clip("umt5_xxl")) == "umt5_xxl"


def test_family_is_empty_without_a_clip():
    """No CLIP means no way to know, which must disable the check rather than guess."""
    assert FunPackVideoRefiner._encoder_family(None) == ""
    assert FunPackVideoRefiner._encoder_family(types.SimpleNamespace()) == ""


def test_a_connected_clip_supplies_the_tokenizer(monkeypatch):
    """The download is a fallback, not the default: it can only ever be one family."""
    def fail(*a, **k):
        raise AssertionError("downloaded a tokenizer while a CLIP was connected")

    monkeypatch.setattr("conditioning.AutoTokenizer.from_pretrained", fail)
    tok = FunPackVideoRefiner._get_tokenizer("ltx2", clip=make_clip("gemma4"))
    assert isinstance(tok, _ClipTokenizerAdapter)
    assert tok.name_or_path == "gemma4"


def test_download_still_serves_a_conditioning_only_run(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("conditioning.AutoTokenizer.from_pretrained", lambda *a, **k: sentinel)
    FunPackVideoRefiner._tokenizers.pop("ltx2", None)
    try:
        assert FunPackVideoRefiner._get_tokenizer("ltx2", clip=None) is sentinel
    finally:
        FunPackVideoRefiner._tokenizers.pop("ltx2", None)


# ----------------------------------------------------------------- the adapter itself

def test_phrase_ids_drop_the_start_token_so_spans_can_match():
    """_build_word_groups finds a phrase by matching its ids inside the prompt's ids. A start
    token on the phrase never matches, so every concept would silently go missing."""
    clip = make_clip("gemma4", {"red": [10], "car": [11, 12]})
    tok = FunPackVideoRefiner._get_tokenizer("ltx2", clip=clip)

    full = tok("red car", add_special_tokens=True)["input_ids"]
    phrase = tok("car", add_special_tokens=False)["input_ids"]

    assert full == [2, 10, 11, 12]
    assert phrase == [11, 12]
    assert full[2:4] == phrase


def test_truncation_matches_the_conditioning_length():
    clip = make_clip("gemma4", {"a": [10], "b": [11], "c": [12]})
    tok = FunPackVideoRefiner._get_tokenizer("ltx2", clip=clip)
    assert tok("a b c", truncation=True, max_length=3)["input_ids"] == [2, 10, 11]


def test_decode_reaches_the_inner_tokenizer():
    """_get_top_tokens reports learned tokens back to the user by decoding single ids."""
    clip = make_clip("gemma4")
    tok = FunPackVideoRefiner._get_tokenizer("ltx2", clip=clip)
    assert tok.decode([10, 11], skip_special_tokens=True) == "10|11"


# ----------------------------------------------------------------- carrying a key across

def test_a_key_does_not_survive_an_encoder_swap():
    """2.3's Gemma3 spans mean nothing under 2.5's Gemma4."""
    assert FunPackVideoRefiner._encoder_family_changed("gemma3_12b", "gemma4") is True


def test_unknown_never_resets_a_key():
    """Keys are expensive to retrain. An older session file with no stamp, or a run with no CLIP
    to read one off, must both leave the key alone."""
    assert FunPackVideoRefiner._encoder_family_changed("", "gemma4") is False
    assert FunPackVideoRefiner._encoder_family_changed("gemma3_12b", "") is False
    assert FunPackVideoRefiner._encoder_family_changed("gemma4", "gemma4") is False


# ----------------------------------------------------------------- V2 state, the Editor's path

def _v2_node(tmp_path, monkeypatch, stored):
    node = FunPackVideoRefinerV2()
    path = tmp_path / "state.json"
    path.write_text(json.dumps(dict(stored, version=2)), encoding="utf-8")
    monkeypatch.setattr(node, "_v2_state_path", lambda key: str(path))
    return node, path


def test_v2_state_survives_a_matching_encoder(tmp_path, monkeypatch):
    node, _ = _v2_node(tmp_path, monkeypatch, {"encoder_family": "gemma4", "global": {"total_iterations": 7}})
    state, status = node._v2_load_state("k", encoder_family="gemma4")
    assert status == "loaded"
    assert state["global"]["total_iterations"] == 7


def test_v2_state_is_cleared_when_the_encoder_changed(tmp_path, monkeypatch):
    """Not blanked in place: the value function and Absolute store hold the same stale tensors,
    so this has to go through Session Reset or they are orphaned."""
    node, _ = _v2_node(tmp_path, monkeypatch, {"encoder_family": "gemma3_12b", "global": {"total_iterations": 7}})
    cleared = []
    monkeypatch.setattr(node, "_v2_absolute_state_path", lambda: str(tmp_path / "abs.json"))
    monkeypatch.setattr(node, "_v2_absolute_vf_path", lambda: str(tmp_path / "abs.pt"))
    monkeypatch.setattr("conditioning.refinement_state_path",
                        lambda *a, **k: str(tmp_path / "vf.pt"))
    monkeypatch.setattr("conditioning.FunPackVideoRefinerV2._v2_empty_state",
                        lambda self, key: (cleared.append(key), {"version": 2, "global": {}})[1])

    state, status = node._v2_load_state("k", encoder_family="gemma4")
    assert status == "reset encoder changed"
    assert cleared == ["k"]
    assert state["global"] == {}


def test_v2_state_without_a_stamp_is_adopted_not_cleared(tmp_path, monkeypatch):
    """Every key predating this change has no stamp. None of them may be wiped on sight."""
    node, _ = _v2_node(tmp_path, monkeypatch, {"global": {"total_iterations": 7}})
    state, status = node._v2_load_state("k", encoder_family="gemma4")
    assert status == "loaded"
    assert state["global"]["total_iterations"] == 7
    assert state["encoder_family"] == "gemma4"


# ----------------------------------------------------------------- vision probe

def _vision_clip(submodule_name):
    transformer = types.SimpleNamespace(vision_model=object(), multi_modal_projector=object())
    outer = types.SimpleNamespace()
    setattr(outer, submodule_name, types.SimpleNamespace(transformer=transformer))
    return types.SimpleNamespace(cond_stage_model=outer)


def test_vision_is_found_whatever_the_submodule_is_called():
    """Comfy still calls it `gemma3_12b` on LTX-2.5 where it holds Gemma4. That name is checked
    first but not depended on."""
    assert FunPackVideoRefinerV2._encoder_has_vision(_vision_clip("gemma3_12b")) is True
    assert FunPackVideoRefinerV2._encoder_has_vision(_vision_clip("gemma4")) is True


def test_no_vision_when_the_encoder_has_none():
    outer = types.SimpleNamespace(gemma3_12b=types.SimpleNamespace(transformer=types.SimpleNamespace()))
    assert FunPackVideoRefinerV2._encoder_has_vision(types.SimpleNamespace(cond_stage_model=outer)) is False
    assert FunPackVideoRefinerV2._encoder_has_vision(types.SimpleNamespace()) is False


# ── the tokenizer must not reach the network mid-run ──────────────────────────
# `from_pretrained` calls HuggingFace with no token, no timeout and no progress, on
# ComfyUI's execution thread. Mid-generation that is an unbounded stall with nothing in the
# log to explain it — and unauthenticated requests are rate limited.


def _tok_calls(monkeypatch, results):
    """Record every from_pretrained call; `results` maps (id, local_only) -> value or Exception."""
    calls = []

    def fake(model_id, **kw):
        local = kw.get("local_files_only", False)
        calls.append((model_id, local))
        out = results.get((model_id, local), FileNotFoundError("not cached"))
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr("conditioning.AutoTokenizer.from_pretrained", fake)
    FunPackVideoRefinerV2._tokenizers.clear()
    return calls


def test_the_local_cache_is_tried_before_any_network_call(monkeypatch):
    sentinel = object()
    calls = _tok_calls(monkeypatch, {("DreamFast/gemma-3-12b-it-heretic-v2", True): sentinel})
    assert FunPackVideoRefinerV2._get_tokenizer("ltx2") is sentinel
    assert calls == [("DreamFast/gemma-3-12b-it-heretic-v2", True)]   # never went online


def test_it_falls_back_to_the_network_only_when_nothing_is_cached(monkeypatch, capsys):
    sentinel = object()
    calls = _tok_calls(monkeypatch, {("DreamFast/gemma-3-12b-it-heretic-v2", False): sentinel})
    assert FunPackVideoRefinerV2._get_tokenizer("ltx2") is sentinel
    assert [c[1] for c in calls] == [True, False]        # cache first, then network
    assert "DOWNLOADING" in capsys.readouterr().out      # and it says so


def test_minimax_h3_asks_for_a_qwen_tokenizer_not_gemma(monkeypatch):
    """H3 encodes with Qwen3-VL. Falling through to the ltx2 entry measured every token span
    with a Gemma vocabulary, and fetched a 12B model's tokenizer to do it."""
    sources = FunPackVideoRefinerV2._get_tokenizer_sources("minimax_h3")
    assert sources and all("qwen" in mid.lower() for mid, _ in sources)
    assert not any("gemma" in mid.lower() for mid, _ in sources)


def test_an_unknown_family_still_falls_back_to_ltx2():
    sources = FunPackVideoRefinerV2._get_tokenizer_sources("nonesuch")
    assert any("gemma" in mid.lower() for mid, _ in sources)
