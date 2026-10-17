# UI kit reference

The kit is called the Composer: `app/ui/composer/`. It is the only code that styles or builds raw DOM. For when to use which element, read [UI design guidelines](ui-design-guidelines.md). This page is the reference.

## How elements work

```js
import { composer } from "../../composer/composer.js";

const save = composer.button.md({ label: "Save", tone: "primary", onClick: save });
host.appendChild(save.node);
save.setBusy(true);
// later
save.destroy();
```

- An element is `composer.<group>.<variant>(props)`. It returns a **handle**: `{ node, destroy, ...methods }`.
- `node` is the DOM node to place. `destroy()` removes listeners and the node. A feature that creates elements after `setup` returns must destroy them itself.
- Methods differ by element. The common ones are `setValue`, `get`, `setText`, `setItems`, `setDisabled`, `setBusy`, `setPressed`, `open` and `close`. Read the factory in `composer/elements/<file>.js` for the exact set.
- **Content is text.** Every string prop goes through `asText` (`internals/text.js`) and is shown as text, never parsed. A prop that would take HTML does not exist.
- **Containers take handles**, not nodes or markup: `panel.default({ children: [handleA, handleB] })`.
- **Unknown elements throw `UnknownElement`.** `composer.button.huge(...)` fails with the list of known variants. A feature that asks for an element that does not exist therefore hides itself instead of showing something approximate.
- The kit's own CSS classes start with `cx-`. App frame classes start with `fp-` (`app/ui/frame.css`) and belong to the shell. Never use either from a feature; the feature rule test fails on a `cx-` or `fp-` string.

### Where each group lives

| File | Groups |
|---|---|
| `text.js` | `brand`, `code`, `header`, `hint` (`low: true` hides it first when the row runs short), `label`, `text` |
| `button.js` | `button`, `buttonGroup`, `iconButton` |
| `input.js` | `filterList`, `input`, `number`, `search`, `stepper`, `textarea` |
| `choice.js` | `checkbox`, `checkboxRow`, `checklist`, `color`, `radioGroup`, `segmented`, `select`, `toggle` |
| `slider.js` | `range`, `slider` |
| `layout.js` | `actionBar`, `card`, `collapsible`, `field`, `frame`, `group`, `hero`, `list`, `panel`, `region`, `settingsRow`, `sidebar`, `splitPane`, `tabs`, `toolbar`, `workspace` |
| `tile.js` | `grid`, `tile` |
| `stage.js` | `stage` |
| `chart.js` | `chart` |
| `tags.js` | `tags` |
| `captioned.js` | `captioned` |
| `status.js` | `banner`, `chip`, `dot`, `dropzone`, `emptyState`, `inlineError`, `progress`, `skeleton`, `spinner`, `toast` |
| `modal.js` | `modal` |
| `popover.js` | `autocomplete`, `menu`, `popover`, `splitButton`, `tooltip` |
| `floating.js` | `floating` (`window`, `corner`), `overlay`, `slideOver` |
| `gallery.js` | `gallery`, `ruler`, `viewer` (`media`, `cover`) |
| `track.js`, `nle.js` | `track`, `timeline` |
| `wheel.js` | `wheel` |

## Elements

Variants are listed after the group. Props shown are the ones every use needs; the catalogue (`catalogue/elements.html`) and `catalogue/demos.js` show each element with a working set.

### Text

| Element | Variants | Main props |
|---|---|---|
| `header` | `display`, `xl`, `lg`, `md`, `sm` | `text` |
| `text` | `lg`, `md`, `sm`, `xs` | `text`. Has `setText`. |
| `hint` | `default` | `text` |
| `label` | `field`, `section` | `text`, `for` |
| `code` | `inline`, `block` | `text`, `label` (block) |
| `brand` | `default` | `name` |

### Actions

| Element | Variants | Main props |
|---|---|---|
| `button` | `xl`, `lg`, `md`, `sm`, `menu` | `label`, `onClick`, `tone` (`neutral`, `primary`, `danger`, `ghost`, `render`), `icon`, `disabled`, `busy`, `type`, `title`, `pressed`. Methods `setBusy`, `setDisabled`, `setPressed`, `setLabel`. |
| `iconButton` | `md`, `sm`, `micro` | `icon`, `label` (required), `onClick`, `tone`, `disabled`, `pressed` |
| `buttonGroup` | `md` | `label`, `value`, `items` |
| `splitButton` | `md` | `label`, `items`, `onClick` |

### Inputs and choices

| Element | Variants | Main props |
|---|---|---|
| `input` | `md`, `sm` | `value`, `placeholder`, `label`, `onCommit` |
| `search` | `md` | `placeholder`, `label` |
| `textarea` | `md` | `value`, `rows`, `label` |
| `number` | `md` | `value`, `min`, `max`, `step`, `precision`, `unit`, `snap`, `onChange`, `disabled`, `id`, `label` |
| `stepper` | `md` | `value`, `min`, `max`, `step`, `label` |
| `slider` | `md`, `sm`, `readout`, `macro` | `value`, `min`, `max`, `step`, `label`, `onChange`, `onCommit`. `.setMax(n)` changes the top and redraws the fill. |
| `range` | `md` | `from`, `to`, `min`, `max`, `step`, `label` |
| `toggle` | `default` | `label`, `hint`, `checked`, `onChange` |
| `checkbox` | `default` | `checked`, `label` |
| `checkboxRow` | `default` | `label`, `hint`, `checked` |
| `checklist` | `default` | `label`, `values`, `items` |
| `radioGroup` | `default` | `label`, `value`, `options` |
| `segmented` | `md`, `sm` | `label`, `value`, `options`, `onChange` |
| `select` | `md`, `sm` | `label`, `value`, `options`, `onChange` |
| `filterList` | `md` | `value`, `items` |
| `autocomplete` | `default` | anchored to an input; sits above modals |
| `color` | `swatch` | `label`, `value` |
| `wheel` | `picker`, `half` | `edge`, `items` |
| `dropzone` | `default` | `label`, `hint` |

`options` is a list of `{ value, label }`. Changes are reported as the value, not as an event.

### Layout

| Element | Variants | Notes |
|---|---|---|
| `frame` | `app` | The page frame. Owned by the shell. |
| `workspace` | `docked` | Docked areas around a centre. |
| `splitPane` | `h`, `v` | Two children with a draggable divider. |
| `region` | `stack` | `children`, `gap`, `label`, `fill`. `.set(children)` replaces them. |
| `panel` | `default`, `zone` | A titled card. |
| `group` | `default` | A quiet sub-block. |
| `field` | `default`, `row` | A label, a control and a hint. |
| `settingsRow` | `default` | The row the Settings window uses. `.setHint(text)` changes its hint later. |
| `card` | `default` | `children`. Related fields on one rounded card (a `region.stack` with a background). |
| `grid` | `pair`, `cards` | `children`. Tiles in two columns, or as many columns as fit (at least 190 wide). |
| `tile` | `action`, `node`, `add` | `title`, `sub`, `onClick`. `node` also takes `state` (`ready`, `needs`, `skipped`) and draws a dot. `add` is the dashed empty slot. |
| `hero` | `default` | `title`, `hint`. The mark, a big title and one line (the top of About). |
| `collapsible` | `default` | |
| `tabs` | `underline`, `dock` | `value`, `tabs`. |
| `sidebar` | `rail` | `value`, `items`. |
| `toolbar`, `actionBar` | `default`, `sticky` | Toolbars wrap; an action bar sticks to the bottom of its scroll area. |
| `list` | `rows` | Grouped rows. |

### Status and feedback

| Element | Variants | Main props |
|---|---|---|
| `banner` | `info`, `warn`, `danger` | `text` |
| `chip` | `neutral`, `good`, `warn`, `danger`, `info` | `label` |
| `dot` | `default` | `tone`, `label` |
| `inlineError` | `default` | `text` |
| `toast` | `info`, `good`, `warn`, `danger` | `text`, `action`, `duration` (default 3200 ms) |
| `progress` | `bar`, `indeterminate`, `inline` | `value`, `max`, `label`. `inline` is a short bar with a line of text beside it: `.setValue`, `.setText`, `.setHidden`. |
| `spinner` | `md`, `sm` | `label` |
| `skeleton` | `line`, `block`, `grid` | `count`, `label` |
| `emptyState` | `default` | `icon`, `title`, `hint` |

### Overlays

| Element | Variants | Main props |
|---|---|---|
| `modal` | `generic`, `stacked`, `dialogue`, `prompt`, `choice` | `generic`: `title`, `subtitle`, `size` (`sm`, `md`, `lg`, `xl`), `body`, `actions`, `onClose`, `closeOnOutside`, `closeOnEsc`. `dialogue`: `title`, `message`, `tone`, `confirmLabel`, `cancelLabel`, `onResult`, and a `.result` promise. |
| `popover` | `anchored` | Positioned by `internals/anchor.js`, flips at an edge. |
| `menu` | `dropdown`, `context` | A context menu is anchored to a point, not an element. |
| `tooltip` | `default` | |
| `floating` | `window`, `corner` | `window` is a draggable window (the Settings and Sampler windows). `corner` is a small button pinned inside the bottom-right corner of `anchor` (a handle or a node); it follows the box and calls `onGone` when the box leaves. |
| `slideOver` | `md` | A panel that slides in from the side. |
| `overlay` | `blocking` | `message`. Blocks the page while something runs. |

All overlays use the portal, the z-ladder and the dismiss stack.

### Charts, tags and captioned media

| Element | Variants | Main props |
|---|---|---|
| `chart` | `line` | `series` (`{ name, values }` or `{ name, points: [{x, y}] }`, plus `tone` `accent`/`good`/`warn`/`muted`, `faint` for a raw noisy line under a smoothed one, `dashed`), `xLabel`, `yLabel`, `height` (`sm`, `md`, `lg`), `logY`, `empty`, `label`. Points that are not numbers are skipped; long series are cut to a few hundred drawn points keeping highs and lows. Hover reads the nearest values. Methods `setSeries`, `setEmpty`. |
| `tags` | `editor` | `value` (list of strings), `suggestions`, `placeholder`, `label`, `maxTags`, `onChange(list)`. Enter or comma adds, Backspace on an empty box removes the last, chips drag to reorder, Tab takes the first suggestion. Methods `get`, `setValue` (does not call `onChange`), `focus`. |
| `tags` | `cloud` | `items` (`{ tag, count }`), `selected`, `onPick(tag)`, `onRemove(tag)` (adds a remove button), `label`, `empty`. Draws the first 300 and says how many more. Methods `setItems`, `setSelected`. |
| `captioned` | `cell`, `side` | A picture or clip with its caption: preview, name, a badge and a text box for the caption in one card. `cell` stacks the picture over the caption (for grids), `side` puts them side by side (for one at a time). `src` (a picture URL; a clip passes its still), `kind` (`image` or `video`, a clip gets a play mark), `title`, `caption` (free text, kept as typed), `badge` and `tone` (`neutral`, `good`, `bad`, `warn`), `selected` and `onSelect(on)` (adds a tick box), `onChange(caption)` (as it is typed), `onOpen()` (makes the picture a button). Methods `getCaption`, `setCaption` (does not call `onChange`), `setSelected`, `setBadge`, `setPicture`, `focus`. |

### Media and time

| Element | Variants | Main props |
|---|---|---|
| `gallery` | `adaptive`, `icons`, `list`, `rows`, `cards`, `strip` | `id`, `cols`, `items`. Density through `internals/density.js`. |
| `viewer` | `media`, `cover` | `media`: `src`, `empty`; `.setLook({ transform, filter, fit, opacity })` changes how the picture is drawn (kept across sources). `cover` is a layer over its parent that shows a picture, clip or sound with a close button: `.show({ src, kind })`, `.close()`, `onClose`. |
| `stage` | `overlay` | A transparent layer over a picture holding text and pictures that can be dragged and resized. Gives data (`items`: position as 0..1, size in picture pixels, colours, text); `onMove(id, {x, y})` and `onResize(id, factor)` report numbers, never styles. `.fit(w, h)` sizes it to the picture inside its parent, `.setItems(items)` draws, `.busy` is true during a gesture. |
| `ruler` | `default` | `label`, `total`, `marks` |
| `track` | `default` | `label`, `pxPerSecond`, `playhead`, `selection`, `items` |
| `timeline` | `stage` | `pxPerSecond`, `lanes`. Methods `setLanes`, `setPlayhead`, `setZoom`. |

## Internals

Used by elements. A feature does not import them. They exist in `composer/internals/`:

| File | Does |
|---|---|
| `register.js` | `define`, `UnknownElement`, lookups. |
| `el.js`, `text.js` | Safe DOM building, `asText`, `setText`. |
| `ids.js` | Unique ids for label and control pairs. |
| `zlayer.js` | The z-ladder. The only place a `z-index` is written. |
| `portal.js` | Where overlays are drawn. |
| `dismiss.js` | One Escape and outside-click stack. |
| `focus.js` | `trap` and `roving`. |
| `anchor.js` | Pure placement (measure, flip, clamp) plus `anchorTo`. |
| `drag.js`, `dragReorder.js`, `snap.js`, `swapping.js` | Dragging, reordering and snapping. |
| `density.js` | Grid columns 0 to 4, saved per id. |
| `motion.js` | Reduced motion check and shared timings. |

## Adding an element

1. **Write it** in the right file under `composer/elements/` (or a new file). Call `define(group, variant, factory)` once per variant. The factory takes props and returns a handle. Use only `el`, `asText` and the internals. Never write `innerHTML`.
2. **Register the file.** A new file is imported in `elements/index.js` and its stylesheet is added to `composer.css` before `polish.css`.
3. **Style it** with tokens only, using `cx-` class names. Cover both themes and reduced motion.
4. **Add a demo** to `catalogue/demos.js` under `"<group>.<variant>"`. A registered element with no demo fails the suite. A demo is a props object, or a function that receives `composer` when the props need other handles.
5. **Teach the content test** any new string prop name. `composer/tests/rule_content_is_data.test.js` attacks every string under the names in `CONTENT_KEYS` with a markup payload and proves it comes back as text. If your prop name is not in the set (for example `caption` or `empty`), add it, or the test fails with a message naming the demo.
6. **Test behaviour** in `composer/tests/elements_behaviour.test.js` or a new `*.test.js` beside it: what the element does when used, not only that it renders.
7. **Look at it** in `catalogue/elements.html` in dark and light.

## Rule tests that run over the whole kit

| Test | Proves |
|---|---|
| `rule_content_is_data.test.js` | Every string prop of every demo is shown as text. |
| `rule_unknown_element_hides.test.js` | Asking for a missing element throws and hides the feature. |
| `rule_features_use_the_kit.test.js` | Every file in `app/ui/features/` passes `FEATURE_RULES` (no `document`, `window`, style, class names, literal colours or sizes). |
| `rule_kit_never_interprets.test.js` | `tools/kit_lint` finds no markup parsing or `eval` in the kit and the shell, and still catches a bad example. |
| `register.test.js` | Duplicate definitions are refused. |
| `overlays.test.js`, `dismiss.test.js`, `zlayer.test.js`, `portal.test.js` | Layering, Escape and portal behaviour. |
| `panel_schema.test.js` | `panel.js` draws every setting type. |

## `panel.js` and the catalogue

`composer/panel.js` renders a module's `SETTINGS` as a panel with the allowed `ui` renderers. The live app does not use it for Settings (it uses `features/settings/field.js`); the catalogue and the tests do. So a `ui` value changes what the catalogue shows, not what Settings > Engine draws. See [Module reference](module-reference.md#renderers).

## The catalogue

Open `/funpack/app/ui/catalogue/` with the devserver running (`python tools/devserver.py 8199`).

| Page | Shows |
|---|---|
| `index.html` | Tokens: type, colour, spacing, radius, motion. |
| `elements.html` | Every registered element in both themes. |
| `hide.html` | What happens when a module is broken or missing. |
| `log.html` | The Log view. |
