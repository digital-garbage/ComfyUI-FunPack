"""Appearance: theme and grid density.

The first real vertical slice, and deliberately the smallest one that exercises
the whole path -- announcement, schema, panel, values -- with no model, no
sampler and nothing that needs a GPU to be seen working.
"""

ID = "appearance"
TITLE = "Appearance"
MOUNT = "settings.general"
STAGE = "load"
STATUS = "proven"

SETTINGS = {
    "theme": {
        "type": "enum", "default": "auto",
        "label": "Colour scheme", "ui": "segmented",
        "options": [
            {"value": "dark", "label": "Dark"},
            {"value": "light", "label": "Light"},
            {"value": "auto", "label": "Auto"},
        ],
    },
    "density": {
        "type": "int", "default": 0, "min": 0, "max": 4, "step": 1,
        "label": "Media columns",
        "hint": "0 fits as many as the panel allows.",
        "ui": "stepper",
    },
}


def apply(values, context=None):
    """Nothing to do at generation time: appearance is the browser's business.

    Declared anyway so the module has the same shape as every other one -- a
    module core has to special-case is a module that will be forgotten.
    """
    return context
