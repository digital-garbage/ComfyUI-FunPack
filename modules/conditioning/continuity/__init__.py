"""Continuity: a shot that follows another starts from where the other ended.

A scene whose Source is "From generated frame" starts from the last picture of the clip before it, so
the shot carries on instead of starting from nothing. The picture is taken from the finished render
(never invented), saved in the media bin, and sent through the pipeline's own start-picture input; this
module only holds the two choices. The work is the editor's: it asks the render module for the frame.

A clip that ends in a fade to black has no picture to continue from (v4's i2i lesson: 3 of 4 anchors were
solid black), so by default such a shot starts without one and says so, rather than starting black.
"""

ID = "continuity"
TITLE = "Continuity"
MOUNT = "settings.general"
STAGE = "conditioning"
CATEGORY = "continuity"
STATUS = "experimental"
ROLES = ["assets.source_image"]       # a pipeline with no start-picture input has nothing to continue through

SETTINGS = {
    "carry": {
        "type": "bool", "default": True,
        "label": "Continue from the previous clip",
        "hint": "A scene set to 'From generated frame' starts from the last picture of the clip before it. Off: such a scene starts without a picture.",
    },
    "dark_guard": {
        "type": "bool", "default": True,
        "label": "Don't continue from a fade to black",
        "hint": "A clip that ends dark gives the next one a black start. With this on, that shot starts without a picture and the run says so.",
        "when": {"carry": True},
    },
}
