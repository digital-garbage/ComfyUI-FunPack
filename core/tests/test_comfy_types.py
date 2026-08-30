"""How ComfyUI spells a type.

Every declaration below was copied from a real registry, not written from
memory. They came out of a survey run after an old bug report said a node
"showed only its LoRA selection": on this machine a stock install has 140
dynamic combos and 23 union-typed inputs, and neither shape was being read.

The registry-wide sweep lives in test_routes_pipeline.py, which is the one place
that loads ComfyUI's extras -- these are the shapes themselves, tested where they
cost nothing to run.
"""

import enum

from core import comfy_types as t


class ResizeType(enum.Enum):
    SCALE_DIMENSIONS = "scale dimensions"
    LONGEST_SIDE = "longest side"


# Verbatim shapes, as INPUT_TYPES() hands them over.
OLD_COMBO = (["euler", "dpmpp_2m"], {"default": "euler"})
V3_COMBO = ("COMBO", {"default": "default", "multiselect": False,
                      "options": ["default", "fp16", "bf16"]})
DYNAMIC_COMBO = ("COMFY_DYNAMICCOMBO_V3", {"options": [
    {"key": ResizeType.SCALE_DIMENSIONS,
     "inputs": {"required": {"width": ("INT", {"default": 512})}}},
    {"key": ResizeType.LONGEST_SIDE},
]})
MULTITYPE_WIDGET = ("FLOAT,INT", {"widgetType": "FLOAT", "default": 25})
MULTITYPE_SOCKET = ("SAM3_TRACK_DATA,MASK", {})
PLAIN_SOCKET = ("MODEL", {})


def widget_of(entry):
    kind, options = t.declared(entry)
    return t.widget_type(kind, options)


# --- what is a widget ------------------------------------------------------

def test_all_three_spellings_of_a_combo_are_a_combo():
    assert widget_of(OLD_COMBO) == t.COMBO
    assert widget_of(V3_COMBO) == t.COMBO
    # The one that was missed. Matching the literal "COMBO" left 140 inputs
    # classified as wires, so their parameters pane offered nothing to edit and
    # claimed they were fed by something.
    assert widget_of(DYNAMIC_COMBO) == t.COMBO


def test_a_multitype_wrapped_around_a_number_is_a_number():
    """Read as a socket it demands a source for a field nobody wires."""
    assert widget_of(MULTITYPE_WIDGET) == "FLOAT"


def test_a_union_of_socket_types_stays_a_socket():
    assert widget_of(MULTITYPE_SOCKET) is None
    assert widget_of(PLAIN_SOCKET) is None


# --- what a combo offers ---------------------------------------------------

def test_choices_are_read_from_wherever_they_were_written():
    assert t.choices(t.declared(V3_COMBO)[1]) == ["default", "fp16", "bf16"]
    assert t.choices({"choices": ["a", "b"]}) == ["a", "b"]
    assert t.choices({}) == []


def test_an_enum_choice_becomes_the_value_comfyui_wants():
    """str() on an Enum member is "ResizeType.SCALE_DIMENSIONS", which ComfyUI
    refuses -- what it wants is the value behind it."""
    assert t.choices(t.declared(DYNAMIC_COMBO)[1]) == ["scale dimensions", "longest side"]


def test_a_choice_that_brings_more_inputs_is_flagged():
    """Nothing renders those inputs yet, and dropping them silently offers an
    incomplete node as a complete one."""
    assert t.reveals(t.declared(DYNAMIC_COMBO)[1]) is True
    assert t.reveals(t.declared(V3_COMBO)[1]) is False


# --- what may feed what ----------------------------------------------------

def test_a_union_accepts_any_of_its_members():
    """An exact string comparison refused a MASK feeding "IMAGE,MASK" -- a legal
    wire refused, which is as bad a failure as an illegal one accepted."""
    assert t.accepts("IMAGE,MASK", "MASK")
    assert t.accepts("IMAGE,MASK", "IMAGE")
    assert t.accepts("MASK", "IMAGE,MASK")
    assert t.accepts("IMAGE,MASK", "SAM3_TRACK_DATA,MASK")


def test_a_union_still_refuses_a_type_it_does_not_name():
    assert not t.accepts("IMAGE,MASK", "MODEL")
    assert not t.accepts("MODEL", "CLIP")


def test_the_wildcard_takes_anything_either_way():
    assert t.accepts("*", "MODEL")
    assert t.accepts("MODEL", "*")
