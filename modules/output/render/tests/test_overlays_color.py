from modules.output.render.overlays import _rgba


def test_short_and_alpha_hex_colours():
    assert _rgba("#f00", 1) == (255, 0, 0, 255)
    assert _rgba("#ff000080", 1) == (255, 0, 0, 128)
    assert _rgba("#00ff00", 0.5) == (0, 255, 0, 127)
    assert _rgba("nonsense", 1) == (255, 255, 255, 255)
