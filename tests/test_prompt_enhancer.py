"""The prompt enhancer: rewrite the prompt with an LLM BEFORE anything is generated from it.

Not the repair advisor. That one is triggered by a rating and fixes a named fault; this runs
on every generation and only elaborates, so it needs no refinement key, no rating and no
previous run. It sits after shortcut expansion and after `$variables` resolve, which is the
last moment the text is still text — so ComfyUI is fed the enhanced prompt and nothing
downstream has to know it happened.

The invariant everything here defends: a failure must return the ORIGINAL prompt. An enhancer
that empties the prompt on a bad generation would render a blank video and blame the model.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import conditioning as C


@pytest.fixture
def studio():
    return C.FunPackVideoRefinerV2.__new__(C.FunPackVideoRefinerV2)


class FakeClip:
    """A CLIP that can generate, the way ComfyUI's own TextGenerate node expects."""

    def __init__(self, reply="a detailed prompt", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [[1, 2, 3]]}

    def generate(self, tokens, **kwargs):
        if self.fail:
            raise RuntimeError("no lm_head")
        self.last_generate = kwargs
        return [[4, 5, 6]]

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class EncoderOnlyClip:
    """A text encoder with no generation lane at all — the common wiring mistake."""

    def tokenize(self, text, **kwargs):
        return {}


# ── the failure modes, which all mean "leave the prompt alone" ───────────────

def test_a_clip_that_cannot_generate_leaves_the_prompt_untouched(studio):
    out, status = studio._v2_enhance_prompt(EncoderOnlyClip(), "a cat", "sys")
    assert out == "a cat"
    assert "does not expose text generation" in status


def test_a_clip_that_cannot_generate_says_where_to_wire_one(studio):
    _out, status = studio._v2_enhance_prompt(EncoderOnlyClip(), "a cat", "sys")
    assert "advisor_clip" in status


def test_a_generation_failure_returns_the_original(studio):
    out, status = studio._v2_enhance_prompt(FakeClip(fail=True), "a cat", "sys")
    assert out == "a cat"
    assert "failed" in status


def test_an_empty_generation_returns_the_original(studio):
    """The one that would render a blank video if it were allowed through."""
    out, status = studio._v2_enhance_prompt(FakeClip(reply="   "), "a cat", "sys")
    assert out == "a cat"
    assert "unchanged" in status


def test_an_empty_prompt_is_not_sent_to_the_model(studio):
    clip = FakeClip()
    out, status = studio._v2_enhance_prompt(clip, "   ", "sys")
    assert clip.calls == []
    assert "empty" in status


# ── the happy path ──────────────────────────────────────────────────────────

def test_the_enhanced_text_replaces_the_prompt(studio):
    out, status = studio._v2_enhance_prompt(FakeClip(reply="a ginger cat on a sunlit sill"),
                                            "a cat", "sys")
    assert out == "a ginger cat on a sunlit sill"
    assert "->" in status


def test_the_model_is_handed_the_prompt_it_will_replace(studio):
    clip = FakeClip()
    studio._v2_enhance_prompt(clip, "a cat", "my instructions")
    sent = clip.calls[0]
    assert "a cat" in str(sent)


def test_the_sampling_parameters_reach_the_model(studio):
    clip = FakeClip()
    studio._v2_enhance_prompt(clip, "a cat", "sys", temperature=1.3, top_p=0.5, max_length=128)
    assert clip.last_generate["temperature"] == pytest.approx(1.3)
    assert clip.last_generate["top_p"] == pytest.approx(0.5)
    assert clip.last_generate["max_length"] == 128


def test_a_short_length_limit_is_honoured_not_floored(studio):
    """The repair advisor floors at 128 tokens; a prompt enhancer must be allowed to be brief."""
    clip = FakeClip()
    studio._v2_enhance_prompt(clip, "a cat", "sys", max_length=48)
    assert clip.last_generate["max_length"] == 48


# ── the cache: one generation per distinct text, not per scene ───────────────

def test_the_same_text_is_only_generated_once(studio):
    """A multi-scene chain repeats the anchor line across scenes; each call is a full LLM run."""
    clip = FakeClip()
    cache = {}
    first, _ = studio._v2_enhance_prompt(clip, "a cat", "sys", cache=cache)
    second, status = studio._v2_enhance_prompt(clip, "a cat", "sys", cache=cache)
    assert first == second
    assert len(clip.calls) == 1
    assert "reused" in status


def test_different_instructions_are_not_shared(studio):
    clip = FakeClip()
    cache = {}
    studio._v2_enhance_prompt(clip, "a cat", "sys one", cache=cache)
    studio._v2_enhance_prompt(clip, "a cat", "sys two", cache=cache)
    assert len(clip.calls) == 2


def test_a_failed_generation_is_not_cached(studio):
    """Caching the fallback would make one transient failure stick for the whole run."""
    clip = FakeClip(reply="")
    cache = {}
    studio._v2_enhance_prompt(clip, "a cat", "sys", cache=cache)
    assert cache == {}


# ── cleaning what a chat model wraps around the answer ───────────────────────

def test_thinking_traces_are_stripped():
    out = C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt(
        "<think>the user wants a cat</think>A ginger cat sits on a sill.")
    assert out == "A ginger cat sits on a sill."


def test_an_unterminated_thinking_trace_does_not_become_the_prompt():
    """A model that runs out of budget mid-reasoning must not have its reasoning rendered."""
    out = C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt("A cat.<think>hmm, maybe I should")
    assert out == "A cat."


def test_a_leading_preamble_is_dropped():
    out = C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt(
        "Here is the enhanced prompt: A ginger cat sits on a sill.")
    assert out == "A ginger cat sits on a sill."


def test_a_bare_output_label_is_dropped():
    assert C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt("Output: A cat.") == "A cat."


def test_code_fences_are_dropped():
    out = C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt("```\nA cat sits.\n```")
    assert out == "A cat sits."


def test_a_fully_quoted_answer_is_unwrapped():
    assert C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt('"A cat sits."') == "A cat sits."


def test_quotes_inside_the_prompt_survive():
    """Speech is quoted on purpose — unwrapping must not eat it."""
    out = C.FunPackVideoRefinerV2._v2_clean_enhanced_prompt(
        'She says "hello" and waves.')
    assert out == 'She says "hello" and waves.'


# ── where it sits in refine_v2 ──────────────────────────────────────────────

def test_it_runs_before_the_prompt_is_encoded():
    """The whole point: ComfyUI must be fed the enhanced text, so the call has to come before
    _v2_conditioning_source, not after it and not somewhere downstream."""
    import inspect
    src = inspect.getsource(C.FunPackVideoRefinerV2.refine_v2)
    assert src.index("_v2_enhance_prompt(") < src.index("_v2_conditioning_source(")


def test_it_runs_after_variables_resolve():
    """`fully expanded` means shortcuts AND $variables — otherwise the model is handed a $name
    it cannot know the value of."""
    import inspect
    src = inspect.getsource(C.FunPackVideoRefinerV2.refine_v2)
    assert src.index("_resolve_variables(prompt_to_encode") < src.index("_v2_enhance_prompt(")


def test_multi_scene_enhances_each_scene_separately():
    """The editor's scene list is authoritative on COUNT. Rewriting the scenes as one
    paragraph and re-splitting could return a different number and desync every anchor."""
    import inspect
    src = inspect.getsource(C.FunPackVideoRefinerV2.refine_v2)
    assert "for t in split_scene_texts" in src


def test_it_is_off_by_default():
    import inspect
    sig = inspect.signature(C.FunPackVideoRefinerV2.refine_v2)
    assert sig.parameters["prompt_enhance"].default is False


def test_the_built_in_instructions_forbid_inventing_speech():
    """A video model will say whatever the prompt quotes, so an enhancer that invents dialogue
    puts words in a character's mouth the user never asked for."""
    text = C.V2_PROMPT_ENHANCER_SYSTEM_PROMPT.lower()
    assert "do not invent" in text and "speech" in text


# ── the model that will not stop ────────────────────────────────────────────
#
# Reported from a rental: with the cap raised to 8192 the enhancer used all of it. The H3
# encoder has no trained decoder head, so it may never emit a stop token — `max_length` stops
# being a ceiling and becomes the actual length. Nothing downstream would notice: it would
# just encode 8000 tokens of rambling as the prompt.

def _trim(text, **kw):
    return C.FunPackVideoRefinerV2._v2_trim_runaway_prompt(text, **kw)


def test_a_looping_model_is_cut_at_the_repeat(studio):
    """Repetition, not length, is what distinguishes a stuck model from a detailed one."""
    out, cut = _trim("A ginger cat sits on the sunlit sill. " * 40)
    assert cut is True
    assert out.count("A ginger cat sits on the sunlit sill.") < 3


def test_a_long_prompt_that_never_repeats_survives(studio):
    """H3 has no padding — every prompt token is a real sequence row, so a long prompt is a
    legitimate thing to want and must not be cut for being long."""
    text = " ".join(f"Shot {i}: a different thing happens in this room." for i in range(200))
    out, cut = _trim(text)
    assert cut is False and out == text


def test_length_alone_never_cuts_without_a_token_limit(studio):
    out, cut = _trim("Unique sentence number %d here. " % 0 + "".join(
        f"Sentence {i} describes something new. " for i in range(500)))
    assert cut is False


def test_the_length_bound_comes_from_the_users_token_limit(studio):
    """The limit the user set IS the control; no character cap is invented next to it."""
    text = "".join(f"Sentence {i} describes something new and distinct. " for i in range(500))
    out, cut = _trim(text, max_length=64)
    assert cut is True
    assert len(out) <= 64 * C.FunPackVideoRefinerV2.V2_ENHANCED_PROMPT_CHARS_PER_TOKEN


def test_a_generous_token_limit_keeps_a_generous_prompt(studio):
    text = "".join(f"Sentence {i} describes something new and distinct. " for i in range(200))
    _out, cut = _trim(text, max_length=8192)
    assert cut is False


def test_the_cut_lands_on_a_finished_sentence(studio):
    out, _cut = _trim("A ginger cat sits on the sunlit sill. " * 40)
    assert out.endswith(".")


def test_a_normal_prompt_is_not_cut(studio):
    text = "A ginger cat sits on a sunlit sill."
    out, cut = _trim(text, max_length=400)
    assert out == text and cut is False


def test_a_short_recurring_line_is_not_treated_as_a_loop(studio):
    """"She smiles." legitimately recurs in a long shot list."""
    text = " ".join(f"The {w} moves across the room. She smiles." for w in
                    ("cat", "dog", "bird", "fox", "owl"))
    _out, cut = _trim(text)
    assert cut is False


def test_a_runaway_never_ends_mid_word(studio):
    out, _cut = _trim("supercalifragilistic " * 500, max_length=64)
    assert not out.endswith("supercalifragilis")
    assert out == out.strip()


def test_the_user_is_told_the_model_repeated_itself(studio):
    clip = FakeClip(reply="A ginger cat sits on the sunlit sill. " * 40)
    _out, status = studio._v2_enhance_prompt(clip, "a cat", "sys")
    assert "repeated itself" in status


# ── where it ran ────────────────────────────────────────────────────────────

def test_the_status_says_which_device_generated(studio):
    """A 32B encoder sharing a card with the DiT can execute on the CPU — minutes per call,
    and nothing in the returned text says so."""
    clip = FakeClip()
    clip.patcher = types.SimpleNamespace(load_device="cuda:0")
    _out, status = studio._v2_enhance_prompt(clip, "a cat", "sys")
    assert "cuda:0" in status


def test_a_wrapper_device_is_found_too(studio):
    clip = FakeClip()
    clip._device = "cpu"
    _out, status = studio._v2_enhance_prompt(clip, "a cat", "sys")
    assert "cpu" in status


def test_an_unreported_device_says_so_rather_than_guessing(studio):
    _out, status = studio._v2_enhance_prompt(FakeClip(), "a cat", "sys")
    assert "unreported" in status


def test_the_status_reports_the_cap_that_was_applied(studio):
    _out, status = studio._v2_enhance_prompt(FakeClip(), "a cat", "sys", max_length=256)
    assert "256" in status
