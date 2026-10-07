"""Shot camera: cut times, views and camera moves for the [Shot N] blocks of an H3 prompt, chosen by rules.

MiniMax H3 reads shots written into the prompt ("[Shot 2] At 00:03.000, the camera cuts to... The camera pushes in toward the
lamp."). This writes them for you, with no language model: a small spaCy tagger finds what each shot is about, a few words
pick the move. Three independent, off-by-default switches, each with a chance:

* cut times: shots after the first open with their time; a shot whose point changes between two of your shortcuts is split
* views: your own trusted list (side view, from above, POV...), never on shot 1, never the same twice in a row
* camera moves: one to three per shot, aimed at the most specific thing in it ("<Subject 1>'s hand", not "<Subject 1>")

Deterministic: the same prompt gives the same shots; "Variation" re-rolls. Ratings teach it: which views and moves are liked
(a dislike that blames the picture alone teaches nothing here), and the words your prompts keep returning to are favoured.
`[Shot N]` is H3's own shot, never a timeline scene. Unvalidated on a GPU.

    text -> Shot Camera -> text -> (Prompt Markup -> encode)
"""

from . import memory
from .nodes import FunPackShotCamera

ID = "conditioning_shot_camera"
TITLE = "Shot camera"
MOUNT = "settings.general"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"

NODES = [FunPackShotCamera]

SETTINGS = {
    "camera_moves": {"type": "bool", "default": False, "label": "Camera moves",
                     "hint": "Adds a camera move, aimed at the most specific thing in the shot, to every [Shot N] that has none."},
    "camera_moves_chance": {"type": "float", "default": 0.7, "min": 0.05, "max": 1.0, "step": 0.05, "label": "Chance a shot gets a move",
                            "when": {"camera_moves": True}},
    "shot_cuts": {"type": "bool", "default": False, "label": "Shot cut times",
                  "hint": "Shots after the first open with their cut time; a shot whose point changes between two shortcuts is split in two."},
    "shot_cuts_chance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Split inside a shot",
                         "hint": "Chance a shot is cut in two where it moves from one shortcut to a different one. 1 = always, 0 = never (cuts only between your shots). Learns from ratings in between.",
                         "when": {"shot_cuts": True}},
    "shot_views": {"type": "bool", "default": False, "label": "Shot views",
                   "hint": "Later shots open on a view (side, from above, POV…), never the same twice in a row."},
    "shot_views_chance": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Chance a shot gets a view",
                          "when": {"shot_views": True}},
    "variation": {"type": "int", "default": 0, "min": 0, "max": 9999, "label": "Variation",
                  "hint": "The same prompt always gets the same shots. Change this number to draw different ones."},
}


def on_rating(prompt_id, rating, axis=None):
    return memory.on_rating(prompt_id, rating, axis)


def routes(table, base, web):
    """What the camera has learned, to read and prune from Settings."""
    @table.get(base + "/memory")
    async def _memory(_req):
        return web.json_response(memory.summary())

    @table.post(base + "/forget")
    async def _forget(req):
        try:
            body = await req.json()
            gone = memory.forget(body.get("kind"), body.get("name"))
        except (ValueError, AttributeError, TypeError) as exc:
            return web.json_response({"why": str(exc)}, status=400)
        return web.json_response({"forgotten": gone})


PROVIDES = {"on_rating": on_rating, "routes": routes}
