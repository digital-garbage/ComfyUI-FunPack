# Writing a module

A module is one folder of Python that adds one feature. This guide builds a small one, then explains each part. Every field and capability is listed in the [Module reference](module-reference.md). Read [How FunPack is built](architecture.md) first if you have not.

## Where a module lives

```
modules/<group>/<name>/
    __init__.py        the announcement (required)
    nodes.py           ComfyUI nodes, if the module has any
    tests/test_*.py    tests (required for anything you want merged)
```

- Exactly two levels: a group, then a module. Groups today: `conditioning`, `latent`, `loaders`, `media`, `models`, `output`, `sampling`, `system`. A new group is just a new folder.
- A folder that starts with `.` or `_`, or has no `__init__.py`, is not scanned.
- A folder whose `__init__.py` has no `ID` is a helper package, not an error.
- A plain `.py` file at group level (`modules/loaders/common.py`) is a shared helper. It announces nothing and may not import a module.
- Core finds modules by scanning. There is no list to edit.

## A small modifier, end to end

This module scales the model's prediction down a little on every step. It is a teaching example (not shipped).

`modules/sampling/calm/__init__.py`

```python
"""Calm: scales the model's prediction down a little on every step."""

from comfy.patcher_extension import WrappersMP

from ..._core import log

ID = "calm"
TITLE = "Calm"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Calm the picture",
        "hint": "Scales the model's prediction down a little on every step.",
    },
    "strength": {
        "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
        "label": "How much", "ui": "slider",
        "hint": "0 = no change. Higher scales the prediction down further.",
        "when": {"enabled": True},
    },
}


def install(patcher, values, key):
    strength = float(values["strength"])
    if not values["enabled"] or strength <= 0.0:
        return None                                   # off: say nothing, install nothing

    def per_call(executor, *args, **kwargs):
        video, audio = executor(*args, **kwargs)
        log.once(f"{ID}:ran", log.INFO, "FunPack Calm", f"scaled the picture by {1.0 - strength:g}")
        return video * (1.0 - strength), audio        # the picture only; sound is never touched

    patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, per_call)
    return f"strength {strength:g}"                   # shown in the run's status


PROVIDES = {"modifier": install}
```

`modules/sampling/calm/tests/test_calm.py`

```python
import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    return FunPackLoadModifiers.execute(tiny.patcher, {"calm": values}).result


def test_it_is_off_until_switched_on(tiny_h3):
    _patched, status = _load(tiny_h3)
    assert "calm" not in status.splitlines()[0]


def test_it_scales_the_picture_and_never_the_sound(tiny_h3):
    base_video, base_audio = tiny_h3.run()
    patched, status = _load(tiny_h3, enabled=True, strength=0.5)
    assert "calm: strength 0.5" in status
    video, audio = tiny_h3.run(patched)
    assert torch.allclose(video, base_video * 0.5, atol=1e-5)
    assert torch.equal(audio, base_audio)


def test_it_leaves_the_original_model_alone(tiny_h3):
    _load(tiny_h3, enabled=True, strength=0.5)
    base_video, _ = tiny_h3.run()
    assert torch.equal(base_video, tiny_h3.run(tiny_h3.patcher)[0])
```

Run it with `pytest modules/sampling/calm` (see [Testing](testing.md) for the ComfyUI setup). That is the whole module. It needed no UI code, no registration and no list entry: the settings appear in **Settings > Engine > Guidance**, the node `FunPack Load Modifiers` installs it, and the Log names it when it runs.

## The announcement

Constants at the top of `__init__.py`, uppercase because they are declarations, not behaviour. Core reads them without running anything.

| Constant | Needed | Meaning |
|---|---|---|
| `ID` | yes | Unique, lowercase, no `.`. Saved projects refer to it, so never rename one lightly. Write it as `ID = "name"` on one line (tests read it that way). |
| `TITLE` | yes | The name people see. |
| `MOUNT` | when it has `SETTINGS` | A non-empty string. Use `"generation.sampling"`. See the note below. |
| `STAGE` | no, default `sampling` | Coarse run order: `load`, `conditioning`, `latent`, `guidance`, `sampling`, `post`. |
| `CATEGORY` | no | Where its settings are grouped: `continuity`, `guidance`, `conditioning`, `sampling`, `post`, `system`. |
| `STATUS` | no, default `experimental` | `proven` or `experimental`. Be honest, see below. |
| `SETTINGS` | no | What people can set. |
| `PROVIDES` | no | Things this module offers other modules, and how it plugs into a run. |
| `NODES` | no | ComfyUI node classes. |
| `REQUIRES` | no | Model traits it needs. |
| `USES`, `USES_WHEN` | no | Capabilities it asks other modules for. |
| `ROLES` | no | Pipeline inputs (`at` names) it acts through. |
| `AFTER`, `BEFORE` | no | Order among modules of the same stage. |
| `HOOKS` | no | Hook points a sampler must offer. |
| `TRAITS` | no | A model module's way of teaching the system what a model is. |

A declaration that is wrong does not load, and the Log says exactly what is wrong. There is no best-effort mode.

**About `MOUNT`.** Core requires it when a module has settings, so give it. In the app as shipped, though, module settings are drawn by Settings > Engine and the Sampler window, which group by `CATEGORY`, not `MOUNT`. `MOUNT` only matters to the optional `ui.js` path described at the end.

## Settings

A setting is a **preference**, a plain value. Data (an image, a mask, a clip) is never a setting: it travels as a ComfyUI socket.

```python
"name": {
    "type": "float",            # bool, int, float, enum, text, multiline, path, color
    "default": 0.5,             # required; must fit the type and the range
    "min": 0.0, "max": 1.0, "step": 0.05,
    "label": "Short name",
    "hint": "What it does, in one plain sentence.",
    "unit": "x",
    "ui": "slider",
    "when": {"enabled": True},
}
```

What is checked when the module loads (`core/schema.py`):

- `type` is one of the eight. `default` is present and fits (an `int` default is an `int`, a `bool` is a `bool`, an enum default is one of its options, a colour is `#rgb`, `#rrggbb` or `#rrggbbaa`).
- Numbers are finite and the default sits inside `min` and `max`. `min` is not above `max`. `step` is positive.
- An enum has `options`: a non-empty list of `{"value": scalar, "label": text}` with no repeated values.
- `ui`, if given, is a renderer allowed for that type. See [what the app really draws](#what-the-app-draws) below.
- `label`, `hint`, `unit` and `placeholder` are strings.
- `when` names other settings of the same module, and (for an enum) values it can actually hold. A condition that could never be true is refused instead of silently hiding the row.

What is checked on every run: each saved value is checked against the declaration. A wrong value (out of range, wrong type, NaN) refuses the run and names the setting. A key the module no longer declares is skipped with one Log line, so old projects keep working.

What the tests add (`core/tests/test_shipped_hints.py`), so treat them as rules:

- Every setting has a non-empty `hint`.
- No two settings of one module share a `label`.
- No hint tells people to look in "the inspector" (there is no such control for settings).

### Conventions that make a module feel like the others

- A feature gets a bool `enabled`, default `False`, and every other row has `"when": {"enabled": True}`. A module that `USES` a capability only counts as working when its `enabled` is true, which is what keeps the provider's settings (the taste key) hidden until a learning feature is on.
- A strength of `0` means off, or "learn only" if the module learns. Say which in the hint.
- Chance values: `1` is always, `0` is never. Do not call a probabilistic feature "always".
- Write labels and hints in plain words, one sentence, what it does for the clip and not how it works inside. "Makes the motion smoother" beats a description of the mechanism.
- Every control that changes the output must be a setting, so it is visible and switchable in the app. Do not hide a knob in code or only on a node.
- Never make a user's value inert. If a value cannot act, the module says so in the Log or refuses the run.

### What the app draws

Two code paths can draw a setting, and they differ:

- **Settings > Engine and the Sampler window** (`app/ui/features/settings/field.js`). This is what people use. It draws by `type`: `bool` as a toggle; `int` and `float` as a slider when `ui` is absent or `"slider"` and both `min` and `max` are set, otherwise as a number box; `enum` as a select; `color` as a swatch; `multiline` as a text area; `text` and `path` as a text box. A `ui` of `segmented`, `wheel`, `radioGroup`, `filterList` and so on is accepted by the schema but this page still draws the default for the type. `placeholder` is not used here.
- **`composer/panel.js`** honours every `ui` hint in `UI_HINTS`. It is used by the catalogue and the tests, and by the optional `ui.js` route.

So: choose `ui` only between `slider` and `number` for numbers, and design for the select and toggle everywhere else.

### Which settings are shown

The app hides a module's settings when nothing would read them (`usefulness` in `pipeline_state.js`):

- its `NODES` are not in the current pipeline;
- its `ROLES` (pipeline input `at` names) are not filled by any slot;
- it provides a capability that other modules `USES`, and none of those users is on (the taste key is hidden until a learning feature is on);
- the model lacks a trait in `REQUIRES`.

Hide, do not warn: a feature that cannot work here is removed from the screen, not explained. The one exception is a switched-on setting that cannot act, which should warn beside Generate.

## Modifiers: changing how a model runs

A modifier is the usual module. It provides `"modifier"`, a function `install(patcher, values, key)`.

- `patcher` is a **clone** of the model's patcher, wrapped so every hook you add is guarded (see below). The original model is never touched.
- `values` is a dict with every declared setting, defaults filled in and checked. Read `values["x"]` directly.
- `key` is `"funpack.<your id>"`. Use it as the hook key, or `f"{key}.suffix"` for a second hook.
- Return `None` if the module decided it is off. Return a short string if it installed: it appears in the run's status as `<id>: <string>`. Say what it did, with the numbers ("strength 0.3, blocks 29-46").
- Raising makes the module off for the run, logs it and quarantines it (see failure model in the architecture page).

### Hooks

Install through ComfyUI's own mechanisms on the patcher you were handed:

| You want | Use |
|---|---|
| To wrap one model call | `patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, fn)` or `DIFFUSION_MODEL` |
| To wrap a whole sample or its setup | `WrappersMP.OUTER_SAMPLE`, `PREPARE_SAMPLING`, `SAMPLER_SAMPLE` |
| To change guidance | `patcher.set_model_sampler_pre_cfg_function(fn)` or `..._post_cfg_function` |
| To replace a transformer block or attention | `core/dit_hooks.py`: `add_block_hook(patcher, key, index, hook)`, `add_attention_override(patcher, key, override)` |

Rules that exist because they were learned the hard way:

- **Tag what you install.** `add_wrapper_with_key` with your `key` is removable by key. Anonymous hooks (the `set_model_*` lists, `add_object_patch`) must be tagged: `patching.tag(fn, key)`. Before every install, `FunPack Load Modifiers` strips everything tagged `funpack.*`, so a model run through twice is not patched twice. An untagged hook cannot be removed and doubles every time.
- **Chain, do not assign.** Block replacement and attention override are single slots. `dit_hooks` chains them so your hook can call `extra["original_block"](args)` and never knows other modules exist.
- **Gate on step position.** `dit_hooks.current_step(transformer_options)` gives `(index, total)`. Never gate on a sigma threshold: schedules differ and a gate can be open on none of the steps. `late_half`, `last_step` and `weak_branch` are ready-made.
- **Ignore probe calls.** `dit_hooks.probing(to)` is true for throwaway model calls (a candidate scored and discarded). Learn and capture nothing from one.
- **Expect more than one call per step.** Second-order samplers call the model twice. If you must act once per step, track it and say so once when you cannot (see `velocity_bias`).
- **The picture and the sound.** `streams.video_of(x, named)` returns the picture and a function to put a changed picture back, whatever the model packs around it. A picture modifier never changes the sound.
- **Reset your state at the start of every run.** ComfyUI caches a node whose inputs have not changed, and between most generations only the seed changes, so `install` does not run again and a closure variable lives on into the next clip. Clear per-run state in an `OUTER_SAMPLE` wrapper (see `momentum`'s `forget`), and clear anything the size of a latent when the run ends. If your install depends on something outside the node's inputs (a switch in Settings), also provide `"cache_key"`: a function returning a value that changes with it, so ComfyUI re-runs the node.
- **Read call arguments by name.** `streams.model_args(args, kwargs)` returns `c_crossattn`, `transformer_options` and so on whether ComfyUI passed them positionally or by name.

### Guards

You do not opt in to being safe. The patcher you receive guards each hook: if it raises, it is dropped for the rest of the run, the run goes on without it, and the Log carries the traceback once. What the executor itself raises (an interrupt, running out of memory) passes through untouched and is not blamed on you.

One mode removes the guards: the **Full control** module, for people who want the raw behaviour. Do not write code that only works with guards off.

If a module cannot run on this machine (an older ComfyUI, a missing package), raise `control.Unavailable`. It is said in the Log but the module is not quarantined.

## Sampler modifiers

For effects that need the step index, the schedule, or to run a step twice, a module can provide `"sampler_modifier"` instead: a function `make(values, accepts)` returning an object with `active(ctx)` and `process(ctx, latent)`, plus `HOOKS = [...]` naming hook points the sampler must offer (today `"latent"`). `core/chain.py` runs the members in order after each step; a member that raises is dropped for the run. No shipped module uses this yet; prefer `"modifier"` unless you need it.

## Nodes

A module may contribute ComfyUI nodes with `NODES = [MyNode]`.

- Write them against `comfy_api.latest.io` (the V3 `io.ComfyNode`): `define_schema()` and `execute()`.
- `node_id` must start with `FunPack`. Node ids are global in ComfyUI and a clash silently overwrites, so a node without the prefix, or reusing an id, is left out and said.
- Category `FunPack/<Area>`. Give every input a `tooltip` that says what it does.
- Return a status string as the last output when it helps ("applied", "off") and use `log` for the reasons.
- A node whose result depends on something outside its inputs (a switch in Settings) should answer `fingerprint_inputs` so ComfyUI re-runs it instead of handing back a cached result.
- A **node widget** describes a file or how it is read (a loader's dtype). A **setting** is a feature knob (a strength, a window). Do not put feature knobs on loader nodes.

A module with only nodes needs no `MOUNT` and no `SETTINGS`.

## Model modules

Support for a model is a module in `modules/models/`. Core names no model.

- `detect(keys) -> bool` and `probe_traits(keys) -> list` read a safetensors header (tensor names) so the app can recognise a file before loading it.
- `TRAITS = traits` where `traits(model)` returns extra trait names for a loaded model.
- `empty_latent(model, width, height, length, batch_size)` and `decode(latent, model=..., vae=..., audio_vae=..., tile_size=...)` return `None` for models that are not yours. Claim by identity, never by shape. If you claim the model and fail, raise: falling back would feed a wrong shape to another model's code.
- `pipeline_presets()` returns ready-made pipelines (lists of slots). `optional_nodes()` says which slots can be passed over when their module is missing.
- Other capabilities teach shared code about your model: `video_stream`, `text_rows`, `prompt_rows`, `video_time_rows`, `query_rotation`, `target_rows`, `anchor_pin`, `latent_upscaler`, `rescale_conditioning`.
- A model you have not run for real stays labelled **UNTESTED** in its presets, its module docstring and the Log (see `modules/models/ltx`).

### Traits

Compatibility is by trait, never by model name. A model says what it **is**, a module says what it **needs** (`REQUIRES`). Core reads universal traits off the loaded model: `temporal_latent`, `spatial_latent`, `waveform_latent`, `temporal_compression`, `predict_<type>`. Model modules add their own (`audio_stream`, `dit_block_hooks`, `adaln_modalities`, `ltx_av`). A module that declares no traits works everywhere, and declaring one is how it leaves models it cannot work on. To test a structure, use `traits.has_block(model, "pkg.module.ClassName")` with a dotted name: bare names like `Attention` collide across models.

## Order

`STAGE` is the coarse order. `AFTER` and `BEFORE` list module ids for finer order inside a stage. There are no priority numbers, on purpose: a number is a global namespace. A relation naming a module that is absent is ignored. A relation that tries to cross stages is dropped and said. A cycle drops the modules on it, and says which.

## Talking to other modules

Never import another module at the top of a file (a test fails the build). Instead:

- **Offer**: `PROVIDES = {"name": callable}`. Anyone can ask for it.
- **Ask**: `registry.current().ask("name", *args)` returns the first provider's non-`None` answer, in id order, or `None`. A provider that raises is skipped and said once. `registry.current().providers("name")` lists every provider.
- **Declare**: `USES = ["name"]` tells the app you depend on a capability, so a provider's settings (the taste key) are hidden while no user is on. `USES_WHEN = {"mode": "learned"}` says you only ask while your own settings hold those values.
- If you need another module's code, import it inside the function that needs it, guarded.

Core never defines a capability name; the contract is between the module that provides it and the ones that ask. The names in use are in the [Module reference](module-reference.md#capabilities).

### Routes

A module that needs endpoints provides `"routes": routes(table, base, web)`. `base` is `/funpack/api/m/<id>`, so two modules never collide and removing a module removes its routes. Return JSON; on a bad request return `{"why": "..."}` with status 400, and say what to fix.

### Learning from ratings

Modules that learn from a person's ratings use the taste key (`modules/system/taste`). The pattern, as `velocity_bias` does it:

1. `USES = ["taste_store"]`.
2. In `install`, `taste = registry.current().ask("taste_store", patcher)`. If it is `None`, say "set a Taste key first" once and return `None`.
3. `taste.collect(patcher, key, KIND, ...)` captures this run's rows, `taste.rows(KIND)` reads the rated rows, and a rating arrives through the taste module.
4. A module that must hear every rating itself provides `"on_rating": fn(prompt_id, rating, axis)`.

Ratings are odds, not bans. Say in the hint how many liked (and disliked) clips it needs before it acts, and say so in the Log when it does not have enough.

## Saying what happened

`log` (`from ..._core import log`) has four levels. The difference between the middle two matters.

| Level | Use it when |
|---|---|
| `log.info` | It worked, and this is what it actually did. "applied: the average of 3 liked clips" is how people learn that a setting did nothing. |
| `log.alert` | Something is unset or unconnected and the run continues without it. |
| `log.warning` | Something will not do what its name says. Never swallow a failure and carry on silently. |
| `log.critical` | The run cannot proceed. |

`log.once(key, level, source, message)` says it the first time per run. Use it for anything inside a per-step hook. Thirty copies of one line say nothing the first did not. `log.broke(what, exc, doing)` and `log.failed(what, exc)` are for exceptions. Use `FunPack <Title>` as the source.

Research or debug data does not go in the console: put a toggle, a Download button and a Clear button in Settings (see Refinement & Taste).

## Honest labels

- `STATUS = "proven"` means it has been run on a real model and judged good. Everything else is `experimental`. The [CHANGELOG](../CHANGELOG.md) lists the two apart, and the list must stay true.
- Do not describe a feature as a guarantee. A feature that depends on chance or on learning says so.
- Keep overhead small. A per-step feature that adds a third to the step time is already costly; one that doubles it is not acceptable. Say the cost in the hint if it is noticeable.
- Fewest words in docstrings, hints and tooltips. No backstory and no future plans in the repo.
- Nothing NSFW in code, tests, comments, commits or docs. Build test data from constructed fixtures.

## Optional `ui.js`

A module may have `ui.js` next to its `__init__.py`. Core serves it at `/funpack/modules/<group>/<name>/ui.js` (only `.js` files are served from `modules/`, so a module cannot ship CSS) and puts the path in its manifest. The contract in `app/ui/shell/panels.js`:

```js
export function setup({ values, on, shell }) {
  // values.get() / values.set(key, value)   this module's settings only
  // on((key, value) => ...)                 hears this module's changes only
  // shell.theme, shell.density              the small list of app services
  return () => { /* teardown */ };
}
```

A module's `ui.js` gets **no element factory and no DOM**. It cannot style, build or inject anything: `tools/kit_lint` fails the build on inline styles, class changes, `document`, `window`, `globalThis`, `innerHTML`, hardcoded colours or pixel sizes, stylesheets and `eval`. The settings panel is drawn for it from the declaration.

**The running app ignores `ui.js` on purpose.** `boot.js` never calls `mountAll`; only the catalogue and the tests do, and no shipped module has a `ui.js`. A module does not get to draw anything. It declares what it wants in `SETTINGS` (type, `ui` hint, range, hint text) and the UI kit builds the control from that declaration. Anything else a person needs to see in the app is a [UI feature](writing-a-ui-feature.md), which is app code and not a module. Do not add a `ui.js`.

## Before you open a pull request

- [ ] Folder in `modules/<group>/<name>/`, unique `ID`, honest `STATUS`.
- [ ] Reaches core only with `from ..._core import ...`; imports no other module at load.
- [ ] Every setting has a default, a plain `label` (unique in the module) and a one-sentence `hint`.
- [ ] Every control that changes the output is a setting.
- [ ] `install` returns `None` when off, a status string when on, and a hook key that starts with `funpack.`.
- [ ] Anonymous hooks are tagged. Nothing leaks from one run to the next.
- [ ] Tests on a real tiny model (`tiny_h3`, `tiny_ltx`) or a real object, not a stand-in: off by default, the effect, nothing leaks, bad input.
- [ ] The Log says what happened, once, at the right level.
- [ ] `pytest` and `cd app && npm test` pass.
- [ ] CHANGELOG has the feature under Proven or Experimental.
