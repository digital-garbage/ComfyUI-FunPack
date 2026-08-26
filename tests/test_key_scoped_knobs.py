"""The learned render knobs belong to the refinement key, not to the project file.

Reported twice from a rental: the H3 gain and taste-bias values could not be cleared. They
were learned from ratings and stored in the key's own state — but the Editor ALSO wrote them
into the project's sampler_inputs, which made a second source of truth that outlived the key
it came from. Deleting every refinement key left the old values still applying, with no way
to clear them but typing the neutral value back in by hand.

The rule now: these knobs are key-scoped. Not in the project, stripped from projects that
already carry them, and neutral when no key is active — so deleting the key files resets the
behaviour, which is what a user deleting them plainly means.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_editor.backend.timeline import KEY_SCOPED_SAMPLER_INPUTS, Project

FRONTEND = Path(__file__).resolve().parents[1] / "movie_editor" / "frontend"


def test_every_learned_render_knob_is_key_scoped():
    assert KEY_SCOPED_SAMPLER_INPUTS == {
        "h3_gain_mode", "h3_gain_video", "h3_gain_prompt", "h3_gain_audio",
        "h3_prompt_scale", "h3_taste_bias",
    }


def test_a_project_carrying_them_is_cleaned_on_load():
    """An existing project has to heal itself; the user cannot be asked to hand-edit JSON."""
    project = Project.from_dict({
        "sampler_inputs": {"h3_gain_video": 0.7, "h3_taste_bias": -0.2, "cfg": 1.0},
    })
    assert project.sampler_inputs == {"cfg": 1.0}


def test_unrelated_sampler_inputs_survive():
    project = Project.from_dict({
        "sampler_inputs": {"h3_audio_clock": True, "frame_overlap": 8, "h3_gain_audio": 1.2},
    })
    assert project.sampler_inputs == {"h3_audio_clock": True, "frame_overlap": 8}


def test_a_project_without_them_is_unchanged():
    project = Project.from_dict({"sampler_inputs": {"cfg": 1.0}})
    assert project.sampler_inputs == {"cfg": 1.0}


def test_a_missing_sampler_inputs_block_is_fine():
    assert Project.from_dict({}).sampler_inputs == {}


def test_the_cleaning_survives_a_round_trip():
    """Load-strip-save is what actually removes the values from the file on disk."""
    project = Project.from_dict({"sampler_inputs": {"h3_gain_prompt": 0.5, "seed": 3}})
    assert "h3_gain_prompt" not in project.to_dict()["sampler_inputs"]


# ── the Editor must not put them back ───────────────────────────────────────

def _engine_settings():
    return (FRONTEND / "engine_settings.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("knob", sorted(KEY_SCOPED_SAMPLER_INPUTS))
def test_no_key_scoped_knob_is_offered_as_a_project_setting(knob):
    """Offering one writes it into sampler_inputs the moment it is touched, which is exactly
    how the second copy got there."""
    import re
    source = _engine_settings()
    assert not re.search(r'\{\s*name:\s*"%s"' % re.escape(knob), source)


def test_the_settings_hint_says_where_the_values_live():
    assert "live in the refinement key" in _engine_settings()


def test_they_are_still_widgets_on_the_node():
    """A raw ComfyUI graph with no Refiner in it still drives them by hand — that is what
    h3_gain_mode 'manual' is for. Key-scoped means not in the PROJECT, not removed."""
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    for knob in KEY_SCOPED_SAMPLER_INPUTS:
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
