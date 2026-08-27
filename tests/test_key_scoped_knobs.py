"""The learned render strengths belong to the refinement key, not to the project file.

Reported twice from a rental: the H3 gain and taste-bias values could not be cleared. They
were learned from ratings and stored in the key's own state — but the Editor ALSO wrote them
into the project's sampler_inputs, which made a second source of truth that outlived the key
it came from. Deleting every refinement key left the old values still applying.

The first attempt at this stripped them inside `Project.from_dict`, which was wrong in a way
no unit test caught: `from_dict` parses the SAVE payload as well as the file on disk, so the
editor sent a value, got a project back without it, and redrew the default. Every one of
these dials — including the mode that reaches them — was unsettable. Reported from the
rental as "mode automatically switches back to learned".

The rule now, in three parts:
  1. Values are dropped when a project is READ FROM DISK, not on the save round-trip.
  2. `h3_gain_mode` is NOT in the set. A mode is a deliberate choice, so clearing it
     discards user intent.
  3. In MANUAL mode nothing is dropped: there is no learned state to reset, because nothing
     learned it — those are hand-typed settings like any other.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_editor.backend.timeline import (  # noqa: E402
    KEY_SCOPED_SAMPLER_INPUTS, Project, without_key_scoped)

FRONTEND = Path(__file__).resolve().parents[1] / "movie_editor" / "frontend"


def test_only_the_rating_learned_values_are_key_scoped():
    assert KEY_SCOPED_SAMPLER_INPUTS == {
        "h3_gain_video", "h3_gain_prompt", "h3_gain_audio",
        "h3_prompt_scale", "h3_taste_bias", "h3_video_detail",
    }


def test_the_mode_itself_is_never_stripped():
    """Stripping it made manual mode unreachable: pick it, save, and the round trip handed
    back `learned` again."""
    assert "h3_gain_mode" not in KEY_SCOPED_SAMPLER_INPUTS
    assert without_key_scoped({"h3_gain_mode": "manual"})["h3_gain_mode"] == "manual"


# ── what happens on the way off disk ────────────────────────────────────────

def test_a_project_carrying_learned_values_is_cleaned():
    """An existing project has to heal itself; the user cannot be asked to hand-edit JSON."""
    assert without_key_scoped(
        {"h3_gain_video": 0.7, "h3_taste_bias": -0.2, "cfg": 1.0}) == {"cfg": 1.0}


def test_unrelated_sampler_inputs_survive():
    assert without_key_scoped(
        {"h3_audio_clock": True, "frame_overlap": 8, "h3_gain_audio": 1.2}) == {
            "h3_audio_clock": True, "frame_overlap": 8}


def test_a_missing_block_is_fine():
    assert without_key_scoped(None) == {}


def test_manual_mode_keeps_every_value_it_was_given():
    """They are hand-typed there. Clearing them would delete work, not stale learned state."""
    stored = {"h3_gain_mode": "manual", "h3_gain_video": 0.7, "h3_taste_bias": -0.2}
    assert without_key_scoped(stored) == stored


def test_learned_is_the_assumption_when_the_mode_is_not_recorded():
    """An older project has no mode key. Learned is the default, so treat it as learned."""
    assert without_key_scoped({"h3_gain_video": 0.7}) == {}


def test_it_reads_the_disk_load_path():
    import inspect
    from movie_editor.backend import projects
    assert "without_key_scoped" in inspect.getsource(projects.get)


# ── and what must NOT happen on save ────────────────────────────────────────

def test_parsing_a_save_payload_keeps_the_values():
    """`from_dict` runs on the browser's save as well as on the file. This is the exact
    regression: strip here and the Editor can never set any of these at all."""
    project = Project.from_dict({"sampler_inputs": {"h3_gain_video": 0.7, "cfg": 1.0}})
    assert project.sampler_inputs == {"h3_gain_video": 0.7, "cfg": 1.0}


def test_the_mode_survives_a_save_round_trip():
    saved = Project.from_dict({"sampler_inputs": {"h3_gain_mode": "manual"}}).to_dict()
    assert saved["sampler_inputs"]["h3_gain_mode"] == "manual"


@pytest.mark.parametrize("knob", sorted(KEY_SCOPED_SAMPLER_INPUTS))
def test_every_learned_dial_can_be_set_within_a_session(knob):
    """Applying only on the next open is the whole point: a dial moved now must take effect
    now, or manual mode does nothing."""
    saved = Project.from_dict({"sampler_inputs": {knob: 0.5}}).to_dict()
    assert saved["sampler_inputs"][knob] == 0.5


# ── the Editor keeps offering them ──────────────────────────────────────────

def _engine_settings():
    return (FRONTEND / "engine_settings.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("knob", sorted(KEY_SCOPED_SAMPLER_INPUTS | {"h3_gain_mode"}))
def test_every_render_dial_is_still_offered_in_settings(knob):
    """Untying a value from the project file is a storage change, not a reason to take the
    dial away."""
    import re
    assert re.search(r'\{\s*name:\s*"%s"' % re.escape(knob), _engine_settings())


def test_the_settings_hint_says_where_the_values_live():
    assert "live in the refinement key" in _engine_settings()


def test_they_are_still_widgets_on_the_node():
    """A raw ComfyUI graph with no Refiner in it still drives them by hand — that is what
    h3_gain_mode 'manual' is for."""
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    for knob in KEY_SCOPED_SAMPLER_INPUTS | {"h3_gain_mode"}:
        assert knob in fields


def test_no_key_means_neutral_not_a_remembered_value():
    """With the key gone there is nothing to read, so the run must fall back to the model's
    trained strengths rather than to whatever was last used."""
    import samplers
    import torch
    node = samplers.FunPackLTXAVSceneChainSampler.__new__(
        samplers.FunPackLTXAVSceneChainSampler)
    node._h3_gain_mode = "learned"
    out = node._h3_render_gains([[torch.zeros(1, 4, 8), {}]])
    assert out == node.H3_GAIN_NEUTRAL
