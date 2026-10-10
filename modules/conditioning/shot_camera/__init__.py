"""Shot camera: cut times, views and camera moves for the [Shot N] blocks of an H3 prompt, chosen by rules.

MiniMax H3 reads shots written into the prompt ("[Shot 2] At 00:03.000, the camera cuts to... The camera pushes in toward the
lamp."). This writes them for you, with no language model: a small spaCy tagger finds what each shot is about, a few words
pick the move. Off by default:

* cut times: shots after the first open with their time
* cut within the same shot: "[Shot 1] DANCE KISS" and "[Shot 1] DANCE DANCE" are cut between the shortcuts, so each part gets a time, a view and a focus. Its chance slider is hidden until this is on
* reframe within the shot: at a boundary that was not cut, the camera moves to a new focus with no cut, in one continuous take (the view may stay)
* views: your own trusted list (side view, from above, POV...), never on shot 1, never the same twice in a row
* camera moves: one to three per shot, aimed at the most specific thing in it ("<Subject 1>'s hand", not "<Subject 1>")
* liked details: a short detail you liked ("detailed lips") may be added, once, at the end of a later shot that already names that body part. Its chance slider is hidden until this is on

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
                  "hint": "Shots after the first open with their cut time, spread across the length of the video."},
    "cut_same_shot": {"type": "bool", "default": False, "label": "Cut within the same shot",
                      "hint": "Cut a shot apart where one shortcut ends and the next begins, including the same shortcut used again. Each part then gets its own time, view and focus."},
    "shot_cuts_chance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Chance of a cut within the shot",
                         "hint": "1 = always cut between those shortcuts, 0 = never. Ratings nudge the values in between.",
                         "when": {"cut_same_shot": True}},
    "reframe_same_shot": {"type": "bool", "default": False, "label": "Reframe within the shot",
                          "hint": "Where one shortcut ends and the next begins inside a shot that is not cut, the camera moves on to what the next action is about, without a cut: one continuous take, the actions follow on. The view may turn or stay. A shot that reframes gets no extra camera move."},
    "reframe_chance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Chance of a reframe",
                       "hint": "1 = always reframe between those shortcuts, 0 = never. Ratings nudge the values in between.",
                       "when": {"reframe_same_shot": True}},
    "shot_views": {"type": "bool", "default": False, "label": "Shot views",
                   "hint": "Later shots open on a view (side, from above, POV…), never the same twice in a row."},
    "shot_views_chance": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Chance a shot gets a view",
                          "when": {"shot_views": True}},
    "detail_notes": {"type": "bool", "default": False, "label": "Liked details",
                     "hint": "When a shot already names a body part, a short detail you liked of that part (such as \"detailed lips\") may be added once, as its own sentence at the end of the shot."},
    "detail_notes_chance": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "label": "Chance of a liked detail",
                            "hint": "1 = always add a matching detail, 0 = never. Ratings nudge the values in between. A detail that keeps coming back bad is dropped.",
                            "when": {"detail_notes": True}},
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
