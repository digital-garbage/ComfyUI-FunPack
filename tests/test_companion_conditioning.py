"""A wired r2v conditioning is several entries describing ONE generation, not several scenes.

The Chain Sampler reads one conditioning entry per SCENE. When Studio stopped discarding the
entries past the first of a wired CONDITIONING — the fix for "the right face speaking invented
syllables" — those entries started arriving at the sampler untagged, and it counted the
reference block as a second scene. From outside that looks like the model sampling the shot
twice for no reason anyone asked for.
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")


@pytest.fixture
def chain():
    from samplers import FunPackLTXAVSceneChainSampler
    return FunPackLTXAVSceneChainSampler


def _entry(companion=False, text="scene"):
    meta = {"funpack_scene_text": text}
    if companion:
        meta["funpack_companion_conditioning"] = True
    return [torch.zeros(1, 4, 8), meta]


def test_a_tagged_entry_is_a_companion(chain):
    assert chain._is_companion_conditioning(_entry(companion=True)) is True


def test_an_ordinary_entry_is_a_scene(chain):
    assert chain._is_companion_conditioning(_entry()) is False


@pytest.mark.parametrize("junk", [None, [], [torch.zeros(1)], "text", 7, [torch.zeros(1), None]])
def test_a_malformed_entry_is_not_mistaken_for_a_companion(chain, junk):
    """Guessing wrong here drops a real scene, so anything unreadable counts as a scene."""
    assert chain._is_companion_conditioning(junk) is False


def test_companions_do_not_add_scenes(chain):
    """The regression itself: one scene plus one reference companion is ONE scene."""
    positive = [_entry(), _entry(companion=True)]
    scenes = [c for c in positive if not chain._is_companion_conditioning(c)]
    assert len(scenes) == 1


def test_real_scenes_still_count(chain):
    positive = [_entry(text="a"), _entry(text="b"), _entry(companion=True)]
    scenes = [c for c in positive if not chain._is_companion_conditioning(c)]
    assert len(scenes) == 2


def test_studio_tags_what_it_passes_through():
    """Studio is what applies the tag; without it the sampler cannot tell the two apart."""
    import inspect

    from conditioning import FunPackVideoRefinerV2
    src = inspect.getsource(FunPackVideoRefinerV2.refine_v2)
    assert 'meta_copy["funpack_companion_conditioning"] = True' in src
