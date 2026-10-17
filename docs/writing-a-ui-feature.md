# Writing a UI feature

A UI feature is one folder of JavaScript that adds one thing to the screen: a button, a panel, a timeline lane, a Settings section, a menu entry. Features are plain ES modules. There is no build step, no framework and no bundler, so what you write is what the browser runs.

Before you start, decide whether you need one. A **module setting** (see [Writing a module](writing-a-module.md)) is drawn for you in Settings > Engine and needs no JavaScript. Write a feature only when the screen needs something a setting cannot say. Read [UI design guidelines](ui-design-guidelines.md) too: they apply to every feature.

## A feature in 30 lines

`app/ui/features/scene-count/index.js`

```js
// The number of scenes in the project, beside the dock buttons.
import { composer as c } from "../../composer/composer.js";

export default {
  id: "scene-count",
  mount: "timeline.status",
  needs: ["project"],
  setup({ host, app }) {
    const label = c.text.xs({ text: "" });
    const draw = () => {
      const n = app.project.scenes.length;
      label.setText(`${n} ${n === 1 ? "scene" : "scenes"}`);
    };
    draw();
    host.append(label.node);
    const off = app.on(draw);
    return () => { off(); label.destroy(); };
  },
};
```

`app/ui/features/scene-count/hub.test.js`

```js
import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { offer, _reset } from "../../shell/mounts.js";
import { loadFeatures } from "../../shell/features.js";

let hub;
test.before(async () => { setupDom(); hub = (await import("./index.js")).default; });
test.after(() => teardownDom());

test("shows the number of scenes, follows edits, and leaves nothing behind", async () => {
  _reset();
  const host = document.createElement("div");
  offer("timeline.status", host);
  const heard = new Set();
  const app = { project: { scenes: [{ id: "a" }] }, on: (fn) => { heard.add(fn); return () => heard.delete(fn); } };
  const { mounted, hidden } = await loadFeatures(["x"], { load: async () => ({ default: hub }), app, has: (n) => n in app });
  assert.deepEqual(hidden, []);
  assert.match(host.textContent, /1 scene$/);

  app.project.scenes.push({ id: "b" });
  heard.forEach((fn) => fn("change"));
  assert.match(host.textContent, /2 scenes$/);

  mounted[0].teardown();
  assert.equal(host.textContent, "");
  assert.equal(heard.size, 0);
});

test("without a project service the feature is hidden, not broken", async () => {
  _reset();
  offer("timeline.status", document.createElement("div"));
  const { mounted, hidden } = await loadFeatures(["x"], { load: async () => ({ default: hub }), app: {}, has: (n) => n in {} });
  assert.equal(mounted.length, 0);
  assert.match(hidden[0].why, /needs project/);
});
```

Then add one line to `app/ui/modules.js`:

```js
"./features/scene-count/index.js",
```

Run it with `cd app && npm test`. That is the whole job.

## The hub

A feature file's default export is one hub, or an array of hubs.

| Field | Needed | Meaning |
|---|---|---|
| `id` | yes | Unique name, shown in the console when the feature is hidden. |
| `mount` | yes | The region to put it in. See the table below. |
| `setup({ host, app })` | yes | Builds the feature. `host` is the region's element (or, for `settings`, an object with `add`). May return a teardown function. |
| `needs` | no | Names that must exist on `app` (`["project", "api"]`). |
| `modules` | no | Backend module ids that must be installed (`["render", "continuity"]`). A button whose backend did not load would only fail when pressed. |

The order in `modules.js` is the order features are set up, so a feature that reads something another registers (`app.sceneSections`, `app.mediaMenu`) should come after it.

### When a feature is hidden

Nothing of it reaches the screen. The reason is written once to the browser console (`[FunPack] <id> hidden: <why>`) and returned by the loader. A feature is hidden when:

- its file does not load (syntax error, bad import);
- `mount` names a region nobody offers (the message lists what is offered);
- anything in `needs` is missing from `app`;
- anything in `modules` is not installed on the server (if the server cannot be asked, nothing is hidden for this reason);
- `setup` throws. Whatever it appended to `host` and whatever it subscribed to through `app.on` is removed again.

The rest of the app loads normally in every case. This is the "hide, don't warn" rule at work: do not draw a disabled button with an apology.

### What you must clean up yourself

The loader only undoes `host` children and `app.on` subscriptions, and only when `setup` throws. Everything else is yours, in the function `setup` returns:

- `document` and `window` listeners, timers, `ResizeObserver`s;
- entries you pushed onto `app` lists (see below), keys you bound, actions you offered;
- classes you put on `<html>`;
- overlays, popovers and modals you opened (`handle.destroy()` or `.close()`).

Do the steps that can throw first and the registrations last, so a `setup` that fails halfway leaves nothing registered.

## Regions (`mount`)

The shell offers these. A region is announced by the code that creates it, so the list is whatever `shell/frame.js` and `shell/settings.js` offer.

| `mount` | Where |
|---|---|
| `menubar.menus` | the menu bar, after the brand. File, Edit, View style menus go here. |
| `menubar.right` | the far end of the menu bar |
| `assets` zone: `assets.actions`, `assets.status`, `assets.projects`, `assets.media` | the Assets panel: header actions, footer, the project list, the media bin |
| `preview`, `preview.actions`, `preview.status` | the monitor panel: its body, header actions, footer |
| `inspector`, `inspector.actions`, `inspector.status` | the Properties panel |
| `timeline` | the timeline stage |
| `timeline.toolbar`, `timeline.toolbar.end` | the row over the stage: what acts on the cut, and zoom/hints |
| `timeline.actions`, `timeline.status` | the Timeline panel's header and footer |
| `settings` | the Settings window. Not a DOM region: `host.add(section)` |

A body region (`preview`, `inspector`, `timeline`, the two `assets.*`) is a stack that takes children. Other regions are toolbars.

### A Settings section

```js
export default {
  id: "my-settings",
  mount: "settings",
  needs: ["api"],
  setup({ host, app }) {
    host.add({
      id: "mine", group: "System", title: "My thing", subtitle: "One sentence.",
      icon: "◎", tone: "neutral", keywords: "search words",
      mount({ setFooter, close }) { return { node: someHandle.node, destroy() {} }; },
    });
  },
};
```

`group` is one of `""`, `"Generation"`, `"Learning"`, `"System"`, in that order. The window searches `title` and `keywords`. `mount` returns a handle (`node`, `destroy`); `setFooter({ actions, note })` puts buttons below the scrolling body.

A small preference belongs on the **Editor** page instead: push a function that returns a kit handle onto `app.editorSettings` (see `features/autosave`).

## What `app` offers

`app` is built in `app/ui/boot.js`. Only a name listed in `needs` is guaranteed; check the table for what each is. Where a method is not listed here, read its file header: each one starts with a comment that says what it is for.

| Name | What | File |
|---|---|---|
| `project` | The open project: `scenes`, `selectedId`, `edit(fn)`, `undo()`, `redo()`, `pref(key, fallback)`, `setPref(key, value)`, `flush()`, `unsaved`. Saved to the server. | `shell/project.js` |
| `selection` | Which scenes are picked: `ids`, `focus`, `pick`, `on(fn)` | `shell/selection.js` |
| `playhead` | `at` (seconds), `set(sec)`, `on(fn)`, `playing` | `shell/playhead.js` |
| `pipeline` | The live pipeline and module values: `slots()`, `activeModules()`, `currentValues()`, `setModuleValue(id, key, value)`, `usefulness()`, `subscribe(fn)`, `ensureLoaded()`, `allOff()` | `shell/pipeline_state.js` |
| `generate`, `runner` | Starting runs. `runner.units(ids, times)` is set by the Generate feature. | `shell/generate.js` |
| `api` | Every server call, one method each. Use it; do not call `fetch`. | `shell/api.js` |
| `keys` | `bind("mod+z", fn)` returns an unbind. Return `false` to let the key through. Typing in a field is never intercepted. | `shell/keys.js` |
| `dock` | Which of Assets, Preview, Properties show | `shell/dock.js` |
| `theme` | `window.ComposerTheme` (`apply`, `get`, `resolved`) | `shell/theme.js` |
| `maintenance` | Update, switch branch, restart. Ends in a page reload. | `shell/maintenance.js` |
| `on(fn)`, `say(name)` | The app's event line. `fn` hears `"open"` (another project was opened), `"change"` (an edit), `"select"`, and named events other features `say` (`"zoom.in"`, `"play.toggle"`, `"timeline.lanes"`, `"media.peek"`). It returns an unsubscribe. | `boot.js` |
| `hasModule(id)` | Whether the server has that backend module. `true` for everything if the server is unreachable. | `boot.js` |
| `openSettings(id)` | Opens Settings at a section | `shell/settings.js` |
| `title(mount, text)` | Sets a zone's title (`mount` is the zone's name: `assets`, `preview`, `inspector`, `timeline`) | `shell/frame.js` |
| `resetLayout()` | Restores the default layout | `boot.js` |

A feature never imports another feature. It reaches the others through `app`, and the rest of the app is reached through a service. (Two existing imports break this: `sampler` uses `settings/field.js` and `settings/models.js`, and `settings/modules.js` uses `addons`. Do not add more: move the shared helper into `shell/` instead.)

### Lists other features read

A feature that adds to the app pushes onto one of these lists and removes its entry in teardown. The reader wraps every entry in a `try`, so one broken entry does not stop the rest.

| List | An entry is | Read by |
|---|---|---|
| `app.editorSettings` | `() => handle` for the Editor settings page | `settings/editor.js` |
| `app.timelineLanes` | `{ owns(id), lanes(project), select/move/trim(id, ...) }`; `app.say("timeline.lanes")` after a change | `timeline` |
| `app.mediaMenu` | `(item, raw) => [{ id, label, run }]` entries for a media tile's menu | `media` |
| `app.mediaPeek` | an object: `media` sets `item` and says `"media.peek"`, `preview/peek.js` opens it | `preview` |
| `app.sceneSections` | `{ rows(scene, project), key(scene, project) }` rows under a scene's own in Properties | `inspector` |
| `app.clipTags` | `(scene) => tags` shown on a clip | `timeline` |
| `app.actions` | `{ id, label, icon, run, when?, optIn? }`. Add with `offer(app, action)` from `shell/actions.js`; it returns the remover. The wheel and menus list them. | `wheel` |
| `app.inputHooks` | `async ({ project, scene, slots }) => ({ inputs, notes })`: put values into the run's inputs; `notes` are shown | `generate` |
| `app.promptPrefix` | `(scene, project) => [text, ...]` lines put before the scene's prompt at run time (never written into the prompt box) | `generate`, `composer` |
| `app.beforeRun` | `async () => {}` awaited before a run is built, so an edit still on its way lands first | `generate` |

To add a new list, add it in `boot.js` and name it in this table. Keep the entry shape small.

## Pairing with a backend module

A feature that exists only because a backend module exists names it: `modules: ["continuity"]`. If the module is not installed or failed to load, the feature is not drawn. Pair the two ids in a test the way `core/tests/test_ui_module_ids.py` does: it fails the build when a feature names a module id that does not exist.

When the backend module offers routes, call them through `app.api` (add the method in `shell/api.js`, and test it in `shell/tests/api.test.js`). The path is `/funpack/api/m/<module id>/...`.

## Building the screen

- **Use kit elements.** `composer.button.md({ label, onClick })`. Every element returns a handle `{ node, destroy, ...setters }`. Put `handle.node` in the page and call `handle.destroy()` when done. Containers take **handles**, not nodes.
- **Content is text.** Pass strings. The kit sets them as text, never as markup, and throws on anything else.
- **Do not style.** No `style`, no new classes, no colours or pixel sizes. If the kit lacks what you need, add an element to the kit (see [UI kit reference](ui-kit-reference.md#adding-an-element)) and use that.
- **A redraw replaces content, not controls.** `region.stack` has `set(children)`; it keeps the caret in a field being typed in. Do not redraw while a field has focus (the Engine page skips redraws while typing).
- **The screen is the truth.** Generate reads live UI state and never waits for saves. Autosave must not disturb typing. Do not cache generated content across requests.
- **Per-browser preferences** (a collapsed section, the dock) go in `localStorage`, wrapped in `try`/`catch`, and the feature must work when storage throws. Anything that should follow a project goes in `app.project.setPref`.

### Elements that are not in the kit

If you ask for one that does not exist (`composer.hologram`), it throws `UnknownElement` naming what does exist. In `setup` that makes the feature hidden, which is the intended result: better nothing than something that looks like it works.

## Tests

Put `hub.test.js` (or `<name>.test.js`) beside the code. `npm test` finds every `*.test.js` under `app/`.

- Use jsdom through `composer/tests/_dom.js` (`setupDom`, `teardownDom`, `fire`, `key`). jsdom has no layout, so anything that depends on measured size belongs in the browser, not here.
- Load the feature through `loadFeatures` with a fake `app` that has only what `needs` names. That tests the real hiding rules too.
- Test teardown: nothing left in `host`, no listener left in `app`.
- Use real objects (a real `project` store, a real `composer`) over hand-made stand-ins where you can.

## Before you open a pull request

- [ ] One folder under `app/ui/features/`, one line in `app/ui/modules.js`.
- [ ] `needs` and `modules` name everything it depends on.
- [ ] Built from kit elements only. No style, no raw HTML, no colour, no pixel size.
- [ ] Teardown removes every listener, timer, list entry and class.
- [ ] A control that changes the output is visible and switchable in the app in the same change.
- [ ] Works in light and dark, and with keyboard only.
- [ ] A test for the hub: mounts, follows changes, leaves nothing behind, hides when a need is missing.
- [ ] `cd app && npm test` passes.
