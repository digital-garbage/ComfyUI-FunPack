"""What the loaders present to the graph, and to the person wiring it.

These are seam tests, not unit tests: a loader that "loads" is worth nothing if
what it offers cannot be filled in, or if what it produces is not what the next
stage reads. No model weights are needed for any of this.
"""

import pytest


@pytest.fixture(scope="module")
def schemas(comfyui):
    from modules.loaders.clip.nodes import FunPackCLIPLoader
    from modules.loaders.diffusion_model.nodes import FunPackDiffusionModelLoader
    from modules.loaders.lora.nodes import FunPackLoraLoader
    from modules.loaders.vae.nodes import FunPackVAELoader
    return [FunPackCLIPLoader, FunPackDiffusionModelLoader,
            FunPackLoraLoader, FunPackVAELoader]


def test_a_file_picker_is_a_widget_not_a_socket(schemas):
    """The failure this exists for: ComfyUI's autogrow input sets
    force_input=True on any WidgetInput, so a list of Combos renders as a row of
    sockets and the file picker stops being a picker. A dropdown nobody can drop
    down is not a loader."""
    for node in schemas:
        spec = node.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, (kind, options) in (spec.get(section) or {}).items():
                if kind != "COMBO":
                    continue
                assert not options.get("forceInput"), (
                    f"{node.__name__}.{name} is a COMBO forced to be a socket; "
                    f"nobody can pick a file from it"
                )


def test_every_combo_offers_something_to_pick(schemas):
    for node in schemas:
        spec = node.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, (kind, options) in (spec.get(section) or {}).items():
                if kind != "COMBO":
                    continue
                assert "options" in options, f"{node.__name__}.{name} has no options at all"


def test_a_default_is_actually_one_of_the_options(schemas):
    """A default outside its own list makes ComfyUI refuse to queue the whole
    prompt with 'Value not in list' -- and the node looks configured."""
    for node in schemas:
        spec = node.INPUT_TYPES()
        for section in ("required", "optional"):
            for name, (kind, options) in (spec.get(section) or {}).items():
                if kind != "COMBO" or "default" not in options:
                    continue
                assert options["default"] in options["options"], (
                    f"{node.__name__}.{name} defaults to {options['default']!r}, "
                    f"which is not one of its options"
                )
