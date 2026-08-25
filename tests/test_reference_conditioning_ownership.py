"""A wired positive CONDITIONING owns the prompt; Studio does not re-encode it.

CLIP used to win whenever both were connected, so Studio replaced the wired conditioning
with its own text-only encode. That is not an approximation of the same tensor. The node
that built the conditioning may have shown Qwen a reference image — and Qwen3-VL is causal,
so the vision block precedes the prompt and every prompt row is a hidden state that has
already read the picture. Studio cannot reproduce that: nothing in the editor pipeline wires
`h3_references` or `source_image`, so its encode never sees an image at all.
"""
import inspect
import sys
import types

import pytest
import torch

sys.path.insert(0, ".")
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))


@pytest.fixture
def studio(monkeypatch):
    import conditioning
    node = conditioning.FunPackVideoRefinerV2.__new__(conditioning.FunPackVideoRefinerV2)
    monkeypatch.setattr(node, "_v2_encode_prompt",
                        lambda *a, **k: (torch.ones(1, 4, 8), {}, "encoded from text"),
                        raising=False)
    monkeypatch.setattr(node, "_v2_extract_conditioning",
                        lambda c: (c[0][0], dict(c[0][1])), raising=False)
    monkeypatch.setattr(node, "_v2_text_tokenizer_status", lambda: "tokenizer ok",
                        raising=False)
    return node


def _wired(seq=6, dim=8, **meta):
    return [[torch.zeros(1, seq, dim), meta]]


def test_a_reference_carrying_conditioning_survives(studio):
    """The case that motivated the change: an r2v node's conditioning reaching the sampler
    intact instead of being replaced by a text-only re-encode."""
    wired = _wired(minimax_token_tags=torch.tensor([1, 1, 0, 0, 1, 1]))
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", wired)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])


def test_a_text_only_conditioning_survives_too(studio):
    """Not gated on detecting a vision block. An i2v checkpoint pins its anchor as a keyframe
    rather than tokenizing it, so its conditioning carries no vision block and a detector
    would have missed it — while the node still prepared something Studio cannot rebuild."""
    wired = _wired(minimax_token_tags=torch.tensor([1, 1, 1, 1, 1, 1]))
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", wired)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, wired[0][0])


def test_clip_keeps_every_other_job(studio):
    """Both stay connected: CLIP is still what encodes the negative and the references."""
    _cond, _meta, note, _owner = studio._v2_conditioning_source(object(), "a cat", _wired())
    assert "CLIP still encodes the negative and references" in note


def test_no_setting_selects_between_them(studio):
    """The wire IS the instruction. Wanting CLIP to own the prompt means not wiring a
    conditioning, so there is nothing here to configure."""
    import inspect
    params = inspect.signature(studio._v2_conditioning_source).parameters
    assert "clip_owns_prompt" not in params
    assert "prefer_wired" not in params


def test_no_conditioning_wired_still_uses_clip(studio):
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", None)
    assert owner == "CLIP-owned"
    assert torch.equal(cond, torch.ones(1, 4, 8))


def test_an_invalid_wired_conditioning_falls_back_to_clip(studio, monkeypatch, capsys):
    """Falling through in silence would look like the wire did nothing."""
    monkeypatch.setattr(studio, "_v2_extract_conditioning",
                        lambda c: ("not a tensor", {}), raising=False)
    cond, _meta, _note, owner = studio._v2_conditioning_source(object(), "a cat", _wired())
    assert owner == "CLIP-owned"
    assert torch.equal(cond, torch.ones(1, 4, 8))
    assert "invalid" in capsys.readouterr().out


# --- the scene split cannot rebuild what it did not encode -----------------

def test_the_split_is_skipped_when_the_conditioning_is_wired(monkeypatch, capsys):
    """The split re-encodes every scene from text and its entries replace the output
    WHOLESALE. It re-establishes H3's visual conditioning from source_image /
    h3_references — neither of which the editor pipeline ever sets — so a wired reference
    conditioning was replaced by text-only encodes and the character became someone else.

    With CLIP disconnected the split returns None and the wired conditioning survived, which
    is exactly why disconnecting CLIP 'fixed' it.
    """
    import conditioning
    node = conditioning.FunPackVideoRefinerV2()
    called = []
    monkeypatch.setattr(node, "_v2_transition_scene_conditionings",
                        lambda *a, **k: called.append(a) or None, raising=False)

    src = inspect.getsource(conditioning.FunPackVideoRefinerV2.refine_v2)
    # The guard is on ownership, not on CLIP being absent.
    assert 'split_by_transitions and conditioning_owner == "CONDITIONING-owned"' in src
    assert "elif split_by_transitions:" in src


def test_the_skip_is_reported_not_silent():
    """A silently unsplit multi-scene run looks like the scene texts were ignored."""
    import conditioning
    src = inspect.getsource(conditioning.FunPackVideoRefinerV2.refine_v2)
    assert "the scene split is SKIPPED" in src
    assert "Disconnect it to let Studio split and encode per scene." in src


# --- a multi-entry wired conditioning keeps all of its entries --------------

def test_entries_past_the_first_are_not_discarded():
    """An r2v node emits the reference block and the encoded prompt as SEPARATE entries.
    `_v2_extract_conditioning` reads entry 0 and Studio rebuilt a single-entry list around
    it, so the character survived and the prompt did not — a tensor holding a vision block
    and nothing to say. Symptom: the right face speaking invented syllables."""
    import conditioning
    src = inspect.getsource(conditioning.FunPackVideoRefinerV2.refine_v2)
    assert 'conditioning_owner == "CONDITIONING-owned" and isinstance(positive_conditioning, list)' in src
    # Kept AND tagged: the Chain Sampler counts one entry per scene, so an untagged extra
    # entry becomes a second scene instead of riding with the first.
    assert 'meta_copy["funpack_companion_conditioning"] = True' in src
    assert "output_conditioning + companions" in src


def test_the_extra_entries_are_reported():
    import conditioning
    src = inspect.getsource(conditioning.FunPackVideoRefinerV2.refine_v2)
    assert "pass through unchanged" in src


def test_the_tag_surplus_message_names_the_prompt_as_the_missing_half():
    """The first version of this message said the REFERENCE was missing. It is the opposite:
    a leading run that is almost all image marks IS the reference, and what was trimmed is
    the prompt."""
    import conditioning
    src = inspect.getsource(conditioning)
    assert "are the encoded PROMPT" in src
    assert "The prompt has been cut, not the tags." in src


# --- nothing re-encodes a conditioning it did not build --------------------

def test_a_wired_entry_is_marked_so_later_steps_can_tell(studio):
    _cond, meta, _note, _owner = studio._v2_conditioning_source(object(), "a cat", _wired())
    assert meta["funpack_conditioning_owner"] == "wired"


def test_bounded_attention_skips_a_wired_entry(studio, monkeypatch):
    """It replaces the tensor with a re-encode of the text — which has no reference in it,
    because this node was never given one. 512 positions became 197: the whole conditioning
    swapped for a text-only encode."""
    import conditioning
    monkeypatch.setattr(conditioning, "_log",
                        types.SimpleNamespace(failed=lambda *a, **k: None,
                                              note_on_change=lambda *a, **k: None),
                        raising=False)
    monkeypatch.setattr(studio, "_v2_bounded_attention_split_encode",
                        lambda *a, **k: (torch.ones(1, 197, 8), 40), raising=False)

    wired = [[torch.zeros(1, 512, 8),
              {"funpack_conditioning_owner": "wired", "funpack_scene_text": "One. Two."}]]
    out = studio._v2_apply_bounded_attention(wired, object())

    assert int(out[0][0].shape[1]) == 512
    assert "funpack_bound_split_tokens" not in out[0][1]


def test_bounded_attention_still_runs_on_studios_own_encode(studio, monkeypatch):
    """Skipping it everywhere would neuter the feature for the graphs it was built for."""
    monkeypatch.setattr(studio, "_v2_bounded_attention_split_encode",
                        lambda *a, **k: (torch.ones(1, 197, 8), 40), raising=False)

    own = [[torch.zeros(1, 512, 8), {"funpack_scene_text": "One. Two."}]]
    out = studio._v2_apply_bounded_attention(own, object())

    assert out[0][1]["funpack_bound_split_tokens"] == 40


def test_every_finalize_stage_is_length_checked():
    """A stage that rebuilds the tensor showed up only as a tag mismatch three functions
    away, which read like housekeeping."""
    import conditioning
    src = inspect.getsource(conditioning.FunPackVideoRefinerV2._v2_finalize_conditioning)
    for name in ("pulse temporal", "rapid temporal", "auto temporal", "bounded attention",
                 "relative steering", "absolute steering"):
        assert f'_step("{name}"' in src
    assert "changed the conditioning length" in src
