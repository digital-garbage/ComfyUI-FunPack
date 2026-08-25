"""Per-block LoRA application is driven by the adapter's deltas, not by the architecture.

It used to run only for image_model in {ltxv, ltxav} and only for keys named
`transformer_blocks.N.`, so every other model got the global fallback without anything ever
looking at its weights. MiniMax H3 names its stack `blocks.N.`, which the old pattern did
not match at all — per-block would have been a silent no-op there even with the gate open.
"""
import sys

import pytest
import torch

sys.path.insert(0, ".")


@pytest.fixture
def mm():
    import model_management
    return model_management


@pytest.mark.parametrize("key,expected", [
    ("diffusion_model.transformer_blocks.7.attn.to_q.weight", 7),      # LTX
    ("diffusion_model.blocks.7.attn.qkv.weight", 7),                   # MiniMax H3
    ("diffusion_model.double_blocks.3.img_attn.qkv.weight", 3),
    ("diffusion_model.single_blocks.11.linear1.weight", 11),
    ("model.layers.4.self_attn.q_proj.weight", 4),
    ("diffusion_model.final_layer.weight", None),
])
def test_a_numbered_block_is_found_whatever_the_stack_is_called(mm, key, expected):
    assert mm.transformer_block_index(key) == expected


def test_transformer_blocks_is_not_read_as_blocks(mm):
    """`transformer_blocks` must match as itself, not as a suffix of the generic name."""
    assert mm.block_container_name(
        "diffusion_model.transformer_blocks.2.ff.net.0.proj.weight") == "transformer_blocks"


def test_ltx_anchors_are_unchanged_on_a_28_block_model(mm):
    """PAG/STG's 14 and 19 were measured on LTX's 28 blocks. Expressing them as positions
    must resolve back to exactly those two blocks, or this generalisation silently retunes
    the model it was derived from."""
    assert mm.semantic_anchor_blocks(range(28)) == frozenset({14, 19})


def test_anchors_scale_to_a_deeper_stack(mm):
    """A 48-block model gets the proportionally equivalent blocks, not 14 and 19."""
    anchors = sorted(mm.semantic_anchor_blocks(range(48)))
    assert anchors == [round(14 / 27 * 47), round(19 / 27 * 47)]
    assert anchors != [14, 19]


def test_a_stack_too_short_to_compare_yields_what_it_has(mm):
    assert mm.semantic_anchor_blocks([0]) == frozenset({0})
    assert mm.semantic_anchor_blocks([]) == frozenset()


class _Entry(dict):
    pass


def test_per_block_is_no_longer_gated_on_the_model(mm):
    """The gate is 'did the user ask for it'. Whether it can actually run is discovered
    from the deltas afterwards."""
    loader = mm.FunPackLoraLoader()
    stack = {"mode": "minimax_h3", "per_block": True, "loras": []}
    assert loader._per_block_supported(object(), stack, _Entry()) is True
    assert loader._per_block_supported(object(), {"mode": "ltx2", "per_block": False},
                                       _Entry()) is False


def _deltas(prefix, n, energy=1.0):
    return {f"diffusion_model.{prefix}.{i}.attn.qkv.weight": torch.full((4, 4), energy * (i + 1))
            for i in range(n)}


def test_an_h3_style_stack_gets_a_real_block_profile(mm):
    """The end of the change that matters: `blocks.N.` deltas now produce per-block scales
    instead of falling through to global."""
    loader = mm.FunPackLoraLoader()
    template = loader._block_profile_template(_deltas("blocks", 8))
    assert template is not None
    assert template["block_count"] == 8
    assert len(template["base_scales"]) == 8


def test_two_block_stacks_decline_rather_than_collide(mm, capsys):
    """double_blocks.0 and single_blocks.0 are different blocks with the same index."""
    loader = mm.FunPackLoraLoader()
    mixed = {**_deltas("double_blocks", 4), **_deltas("single_blocks", 4)}
    assert loader._block_profile_template(mixed) is None
    assert "per-block declined" in capsys.readouterr().out
