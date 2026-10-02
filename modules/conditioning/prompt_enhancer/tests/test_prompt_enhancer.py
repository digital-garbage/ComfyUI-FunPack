"""The prompt enhancer: what the model is asked, how its answer is cleaned, and
that every failure hands the ORIGINAL prompt on."""

import json

import pytest

from core import config, shortcuts


@pytest.fixture(autouse=True)
def _library(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SHORTCUTS_FILE", tmp_path / "s.json")
    monkeypatch.setattr(config, "SHORTCUT_CATEGORIES_FILE", tmp_path / "c.json")
    monkeypatch.setattr(config, "ROOT", tmp_path)


def en():
    from modules.conditioning.prompt_enhancer import enhance
    return enhance


class FakeClip:
    """ComfyUI's generate/decode pair, recording what it was asked."""

    def __init__(self, answer="A fox walks.", fail=None):
        self.answer, self.fail, self.seen = answer, fail, {}

    def tokenize(self, text, **kw):
        self.seen = {"text": text, **kw}
        return "tok"

    def generate(self, tokens, **kw):
        if self.fail:
            raise self.fail
        self.seen["gen"] = kw
        return "ids"

    def decode(self, ids, skip_special_tokens=True):
        return self.answer


# --- cleaning ---------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("<think>hmm</think>A fox.", "A fox."),
    ("<think>never ends and eats the answer", ""),
    ("```text\nA fox.\n```", "A fox."),
    ("Here is the prompt: A fox.", "A fox."),
    ('"A fox says \\"hi\\" and runs."', 'A fox says \\"hi\\" and runs.'),
    ('A fox says "hi" and runs.', 'A fox says "hi" and runs.'),
])
def test_clean(raw, want):
    assert en().clean(raw) == want


def test_thinking_is_pulled_out_even_unterminated_or_headless():
    e = en()
    assert e.extract_thinking("<think>a</think>b") == "a"
    assert e.extract_thinking("<think>still going") == "still going"
    assert e.extract_thinking("opened by template</think>answer") == "opened by template"
    assert e.extract_thinking("plain") == ""


def test_a_looping_model_is_cut_at_a_sentence_and_a_long_one_by_the_users_limit():
    e = en()
    loop = "A fox walks across the snowy field. " * 5
    out, cut = e.trim_runaway("Intro sentence here. " + loop)
    assert cut
    assert out.count("A fox walks across the snowy field.") == 2
    long_text = "A sentence that goes on a bit. " * 40
    out, cut = e.trim_runaway(long_text, max_length=16)         # cap 64 chars
    assert cut and len(out) <= 64 and out.endswith(".")
    assert e.trim_runaway("short", max_length=400) == ("short", False)


# --- chat -------------------------------------------------------------------

def test_chat_shows_only_the_latest_rewrite_and_every_comment():
    e = en()
    rounds = [{"rewrites": {"whole": "v1"}, "comment": "warmer"},
              {"rewrites": {"whole": "v2"}, "comment": "no rain"}]
    block = e.chat_block(rounds)
    assert "v2" in block and "v1" not in block and "- warmer\n- no rain" in block
    assert e.chat_block([{"comment": "  "}, "junk", None]) == ""
    assert e.chat_block([{"rewrites": {"3": "only scene 3"}, "comment": "x"}], scene=3).count("only scene 3") == 1


def test_an_echoed_conversation_is_cut():
    e = en()
    assert e.cut_chat_echo("The fox runs.\n\nUSER FEEDBACK (oldest first):\n- warmer") == "The fox runs."
    assert e.cut_chat_echo("YOUR LATEST REWRITE: The fox runs.") == "The fox runs."


# --- reference --------------------------------------------------------------

def test_reference_uses_what_the_prompt_mentions_else_the_whole_group():
    e = en()
    shortcuts.save({"name": "Rain city", "triggers": ["rc"], "replacements": ["heavy neon rain on asphalt"]})
    shortcuts.save({"name": "Fox", "triggers": ["fx"], "replacements": ["a red fox"]})
    groups = e.sources(["Rain city", "Fox", "Missing"], [], say=lambda m: None)
    assert len(groups) == 1 and len(groups[0]) == 2
    only_rain = e.reference("a walk in the neon street", groups)
    assert "[Shortcut] Rain city" in only_rain and "[Shortcut] Fox" not in only_rain      # a word from its content
    both = e.reference("nothing related", groups)
    assert "Rain city" in both and "Fox" in both                                         # none mentioned: all
    assert e.reference("x", []) == ""


def test_a_missing_shortcut_or_file_is_said_not_silent():
    said = []
    assert en().sources(["Nope"], ["/no/such/file.json"], say=said.append) == []
    assert len(said) == 2 and "Nope" in said[0] and "/no/such/file.json" in said[1]


def test_lorebooks_match_by_keyword_whole_word_and_constants_ride_along(tmp_path):
    e = en()
    book = tmp_path / "lore.json"
    book.write_text(json.dumps({"entries": [
        {"keys": ["sea"], "content": "The sea is cold.", "comment": "Sea"},
        {"keys": ["/ra+in/"], "content": "Rain is soft."},
        {"keys": ["x"], "content": "Always true.", "constant": True}]}))
    groups = e.sources([], [str(book)])
    got = e.reference("a season of rain", groups)          # "sea" is NOT in "season"
    assert "Rain is soft." in got and "The sea is cold." not in got and "Always true." in got


# --- enhance ----------------------------------------------------------------

def test_the_model_gets_instructions_then_the_prompt_and_the_answer_is_returned():
    clip = FakeClip("Here is the prompt: A red fox walks through snow.")
    out, status, info = en().enhance(clip, "a fox", seed=5, max_length=100, do_sample=False)
    assert out == "A red fox walks through snow." and info["ok"]
    assert clip.seen["text"].startswith(en().SYSTEM_PROMPT) and clip.seen["text"].endswith("a fox")
    assert clip.seen["gen"]["seed"] == 5 and clip.seen["gen"]["do_sample"] is False
    assert clip.seen["skip_template"] is False and clip.seen["min_length"] == 1


@pytest.mark.parametrize("clip", [FakeClip(fail=RuntimeError("boom")), FakeClip(""), FakeClip("<think>x</think>"),
                                  None, object()])
def test_every_failure_returns_the_original_prompt(clip):
    out, status, info = en().enhance(clip, "a fox")
    assert out == "a fox" and not info["ok"] and status


def test_an_empty_prompt_is_skipped_not_sent():
    clip = FakeClip()
    out, status, _ = en().enhance(clip, "   ")
    assert out == "   " and "empty" in status and clip.seen == {}


def test_a_clip_that_does_not_take_the_newer_arguments_still_works():
    class Old(FakeClip):
        def tokenize(self, text, image=None):
            self.seen = {"text": text}
            return "tok"

        def generate(self, tokens, **kw):
            if "no_repeat_ngram_size" in kw:
                raise TypeError("nope")
            return "ids"
    assert en().enhance(Old("A fox."), "fox")[0] == "A fox."


def test_chat_revision_goes_in_the_system_turn_and_an_echo_is_cut():
    e = en()
    clip = FakeClip("The fox runs.\nUSER FEEDBACK: warmer")
    chat = e.chat_block([{"rewrites": {"whole": "old"}, "comment": "warmer"}])
    out, _, info = e.enhance(clip, "a fox", chat=chat)
    assert out == "The fox runs." and e.CHAT_SYSTEM in clip.seen["text"]
    assert "ORIGINAL PROMPT:\na fox" in clip.seen["text"] and info["sent"]


# --- the node ---------------------------------------------------------------

@pytest.fixture
def node(comfyui):
    from modules.conditioning.prompt_enhancer import nodes
    nodes.RUNS.clear()
    return nodes


def test_the_node_passes_through_when_off_and_never_touches_the_model(node):
    clip = FakeClip()
    out = node.FunPackEnhancePrompt.execute(clip, "a fox", enabled=False)
    assert tuple(out.args) == ("a fox", "off") and clip.seen == {}


def test_the_node_enhances_records_the_run_and_warns_on_failure(node, caplog):
    out = node.FunPackEnhancePrompt.execute(FakeClip("A red fox."), "a fox", enabled=True, seed=3)
    assert out.args[0] == "A red fox." and node.RUNS[-1]["after"] == "A red fox."
    from core import log
    log._reset()
    bad = node.FunPackEnhancePrompt.execute(FakeClip(fail=RuntimeError("x")), "a fox", enabled=True)
    assert bad.args[0] == "a fox" and node.RUNS[-1]["ok"] is False
    assert any("original prompt" in r["message"] for r in log.history())


def test_seed_zero_is_never_cached_a_set_seed_is(node):
    f = node.FunPackEnhancePrompt.fingerprint_inputs
    assert isinstance(f(seed=0, enabled=True), float)
    assert f(seed=7, enabled=True) == "" and f(seed=0, enabled=False) == ""


def test_the_readout_routes_are_provided_and_answer():
    from modules.conditioning import prompt_enhancer as mod
    assert mod.PROVIDES["routes"] is mod.routes
    table = {}

    class T:
        def get(self, path):
            return lambda fn: table.setdefault(path, fn)

    class W:
        @staticmethod
        def json_response(d):
            return d
    mod.routes(T(), "/m", W)
    assert set(table) == {"/m/defaults", "/m/runs"}
