# How FunPack is built

Read this first. The other guides assume it.

## The idea

FunPack is a browser app (`/funpack/`) on top of ComfyUI. Everything it does is a **module that announces itself**. Core keeps no list of features, so adding a feature is adding a folder, and removing one is deleting it.

The same idea runs on both sides:

| Side | Things that announce themselves | Where they announce |
|---|---|---|
| Backend (Python) | modules | constants in `modules/<group>/<name>/__init__.py` |
| App (JavaScript) | features | `export default { id, mount, setup }` listed in `app/ui/modules.js` |
| App kit | UI elements | `define(group, variant, factory)` in `app/ui/composer/elements/` |

Anything that is announced can be absent. A module that fails to import, fails validation, or breaks during a run is left out, the Log says why, and everything else carries on. The UI follows the same rule: a feature that cannot load is not drawn at all (no grey button, no warning chip).

## The layers

```
browser  app/ui/            shell + kit + features (plain ES modules, no build step)
   |
aiohttp  core/routes.py     routes mounted on ComfyUI's own server under /funpack/
   |
core/    Python             finds modules, validates them, builds the ComfyUI prompt
   |
modules/ Python             features, grouped by domain
   |
ComfyUI  nodes + sampler    executes the prompt
```

- **core/** knows how to serve the app, load modules and build a ComfyUI request. It never names a feature.
- **modules/** is every feature: loaders, sampling techniques, model support, output, learning from ratings.
- **app/ui/shell/** is the app's own plumbing: the page frame, project and pipeline state, the feature loader, the API client.
- **app/ui/composer/** is the UI kit (called the Composer): tokens, internals, elements. It is the only code that styles anything.
- **app/ui/features/** is every screen feature: timeline, media bin, preview, settings sections, Generate.

## How a setting travels

1. A module declares `SETTINGS` (see [Writing a module](writing-a-module.md)).
2. `GET /funpack/api/modules` sends the browser the validated declarations, ordered by stage and relations, with the model's traits applied.
3. **Settings > Engine** (and the Sampler window) draw one row per setting. Nothing is hand-built per module.
4. An edit is saved with the project. The values travel as one JSON object, `{module_id: {key: value}}`, in the `settings` input of the node a module named as its `settings_sink` (today `FunPack Modifier Settings`).
5. At run time `FunPack Modifier Settings` checks every value against its declaration and `FunPack Load Modifiers` calls each enabled module's `install(patcher, values, key)` in order.

The screen is the truth: Generate reads the live UI state, and a module never reads the UI.

## The pipeline is data

The pipeline is a list of slots (`core/graph.py`): each names a node and what feeds it. Nothing is a fixed chain, so a slot can be pointed at another node or removed, and the server refuses what would not build and says why.

- A slot's `roles` put one of its inputs on the app's surface (the prompt box, width and height, steps, seed). A role's `at` names the place (`generation.prompt`, `project.video`, ...).
- A slot's `group` is how the pipeline reads to a person (the cards in Models & Pipeline).
- Model modules offer ready-made pipelines through `pipeline_presets`.

## Failure model

The rules below are enforced by code, not by habit. Each one names where.

| What goes wrong | What happens | Where |
|---|---|---|
| A module raises on import | Absent. One Warning in the Log. | `core/registry.py` |
| A declaration is invalid | Not loaded. The Log names the setting and the rule. | `core/schema.py` |
| Two modules share an id or a node id | The second is left out and said. | `registry.add`, `core/nodes.py` |
| Saved settings name a module or key that no longer exists | Skipped with one Log line. Old projects keep working. | `schema.check_values`, modifier nodes |
| A setting value is wrong | The run is refused, naming it (unless Full control is on). | `schema.refuse_or_warn` |
| `install` raises | That module is off for the run and quarantined. | `core/control.py` |
| A hook raises on step 3 of 30 | The hook is dropped for the rest of the run, the run continues. | `core/patching.py` |
| A module was quarantined | It stays off until its code changes (and ComfyUI restarts) or the person turns it on in Settings > Modules. | `core/control.py` |
| A relation cannot be satisfied (cycle) | The modules on the cycle are left out, and said. | `core/relations.py` |
| A model lacks a trait a module requires | The module is not offered for that model. | `core/traits.py` |

Out-of-memory errors and interrupts are not a module's fault and never quarantine it.

## Independence

- A module reaches core only through `modules/_core.py` (`from ..._core import log, registry`). A direct `import core` works in tests and breaks inside ComfyUI, so a test forbids it.
- A module never imports another module when the file loads. It asks by capability: `registry.current().ask("name", ...)`. If the provider is missing or raises, the answer is `None` and the asking module copes. A test forbids the import.
- A feature may not break the baseline, and the baseline does not depend on a feature on top of it.
- The same holds in the browser: a feature never imports another feature. It talks through `app` (see [Writing a UI feature](writing-a-ui-feature.md)).

## Who may touch what

There are three trust levels in the app code. Know which one you are writing.

| You are writing | May style or build DOM? | Checked by |
|---|---|---|
| A **backend module's settings** (Python only) | No JavaScript at all. The kit draws the rows. | `core/schema.py`, `core/tests/test_shipped_hints.py` |
| A **UI feature** in `app/ui/features/` | It builds with kit elements. A few features touch `document` or add `fp-` classes in `frame.css`; new work should not. | review; the kit rules are enforced on `composer/` and `shell/` |
| The **kit** itself | Yes. It is the only place that styles. | `tools/kit_lint` `KIT_RULES`, tests in `composer/tests/` |

Be exact about what the lint is. `tools/kit_lint` is a discipline check, not a security boundary. Modules run only in Python and are never served to the browser; first-party modules are reviewed.

## Where things live

| Path | What |
|---|---|
| `__init__.py` | ComfyUI entry point. Mounts the routes and hands ComfyUI the nodes modules announced. |
| `core/` | Registry, schema, traits, relations, chain, control, patching, graph, routes, run, log. |
| `modules/_core.py` | The one place a module reaches core from. |
| `modules/<group>/<name>/` | One module. `__init__.py` is the announcement. |
| `modules/<group>/*.py` | Helpers shared inside a group. Announce nothing. May not import a module. |
| `app/ui/boot.js` | Starts the frame, the core services, then loads features. Names no feature. |
| `app/ui/modules.js` | The one list of UI features. |
| `app/ui/shell/` | Frame, mounts, project and pipeline state, API client, settings window. |
| `app/ui/composer/` | The kit: `tokens/`, `internals/`, `elements/`, `panel.js`, `tests/`. |
| `app/ui/features/<name>/` | One feature. |
| `app/ui/catalogue/` | A page that shows every kit element and token, in both themes. |
| `app/assets/fonts/` | Bundled fonts (Inter, Inter Tight, IBM Plex Mono, all OFL). |
| `tools/devserver.py` | Serves the app and the real routes without ComfyUI. |
| `tools/kit_lint/` | The rule scanner. |
| `docs/` | These documents. |

Python tests sit beside what they test (`core/tests/`, `modules/<group>/<name>/tests/`). JavaScript tests sit beside the code (`*.test.js`) or in `composer/tests/` and `shell/tests/`.

## Running it locally

- **With ComfyUI**: clone into `ComfyUI/custom_nodes`, `pip install -r requirements.txt`, start ComfyUI, open `/funpack/`.
- **Without ComfyUI**: `python tools/devserver.py 8199` serves the app on the real routes. There is no queue behind it, so Generate says so. Set `COMFYUI_ROOT` to a ComfyUI checkout, or most modules will be absent (they import `comfy`).
- **The catalogue**: `/funpack/app/ui/catalogue/index.html` (tokens), `elements.html` (every element, both themes), `hide.html` (what happens when a module is broken), `log.html`.

Tests are covered in [Testing and verification](testing.md).
