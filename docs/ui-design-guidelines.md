# UI design guidelines

How FunPack looks and behaves, and why. Read this before you add a screen, a setting or a kit element. For the element list see [UI kit reference](ui-kit-reference.md). For where a feature plugs in see [Writing a UI feature](writing-a-ui-feature.md).

## Principles

1. **Hide, do not warn.** If a feature cannot work (its module is missing, the model lacks a trait, it failed to load), it is not drawn. No grey button, no "unavailable" chip. A person never sees a control that cannot be used. The one place that explains absence is the Log.
2. **The screen is the truth.** Generate reads the live UI state. A module never reads the UI, and the UI never keeps a second copy of what the project already stores.
3. **Every control that changes the output is visible.** Nothing that alters a result hides behind a mode, a tab you did not open or a default you cannot see. Advanced does not mean secret.
4. **Never disrupt input.** A value the person is typing is not replaced under their hands. A panel redraw keeps focus and the caret. A toast never steals focus.
5. **Content is data.** Every string is shown as text. Markup in a name or a prompt is displayed, never run. The kit has no way to render HTML on purpose.
6. **Say only what is true.** A label describes what the control does today. Do not call something "always" if it is a chance, and do not call a trial feature proven.
7. **Calm by default.** One primary action per area. Quiet fills, no outlines for the sake of outlines, no row of loud buttons.

## Where styling lives

Only the kit styles anything: `app/ui/composer/` (tokens, internals, elements). A feature builds with kit elements and names no colour, size or class.

- A rule test fails the build if kit code parses content as markup or compiles code (`tools/kit_lint` `KIT_RULES`).
- The same scanner is run over module `ui.js` files (the running app does not load them) with stricter rules (`MODULE_RULES`): no inline style, no `classList`, no `document`, no raw hex colour, no `px` size.
- A few older features touch `document` or add `fp-` classes in `shell/frame.css`. Do not copy them. New work uses the kit, and if the kit lacks the element, add the element (see [Adding an element](ui-kit-reference.md#adding-an-element)).

## Tokens

Never write a literal colour, size, radius, duration or z-index. Use a token. Tokens live in `composer/tokens/tokens.css`; `polish.css` is loaded last and retunes some of them (fonts, weights, control heights, card radius), so read the final values in the catalogue (`catalogue/index.html`), not only in `tokens.css`.

| Family | Tokens | Use |
|---|---|---|
| Type size | `--fs-1` .. `--fs-8`, display `--fs-d1` .. `--fs-d3` | Body text is `--fs-4`. Use the `text` and `header` elements instead of setting sizes. |
| Spacing | `--sp-0` .. `--sp-11` (2px base) | Gaps and padding. Pick the step, do not compute pixels. |
| Radius | `--r-1` .. `--r-5`, `--r-pill`, `--card-r` | Small controls low, cards high, pills for chips. |
| Control height | `--ctl-h-sm`, `--ctl-h-md`, `--ctl-h-lg` | Every button, input and select in a row shares one of these. |
| Motion | `--dur-1` .. `--dur-3`, `--ease-standard` | All become zero under reduced motion. |
| Density | `--cell-thumb`, `--cell-card`, `--cell-panel`, `--cell-icon`, `--gutter`, `--control-col`, `--menubar-h` | Media grid cells and layout columns. |
| Surface | `--elev-0` .. `--elev-3` | Page, panel, raised row, floating. Higher is closer to the person. |
| Lines and states | `--line`, `--hover`, `--active`, `--focus-ring` | Hairlines, hover and pressed fills, focus outline. |
| Text | `--text`, `--muted`, `--faint`, `--on-accent` | Main, secondary, tertiary, text on an accent fill. |
| Meaning | `--accent*`, `--teal`, `--violet`, `--good`, `--warn`, `--danger`, `--pink` | See the next section. |
| Depth | `--shadow-1` .. `--shadow-3`, `--grad-*`, `--void` | Derived. Do not redefine them. |

Fonts are bundled (Inter, Inter Tight, IBM Plex Mono) and declared in `composer/tokens/fonts.css`. Use `--font-ui`, `--font-display` and `--font-mono`. Do not link a font from the web: the app must render offline.

### Semantic colour

Colour carries one meaning each. Do not reuse a colour for decoration.

| Colour | Means |
|---|---|
| Accent (amber) | The one primary action, the selection, the focus of attention. |
| Teal | Rendering and audio. |
| Violet | Learned or taste-driven things. |
| Good | Done, saved, on target. |
| Warn | Needs a look, but nothing is broken. |
| Danger | Failed or destructive. |
| Muted, faint | Help, units, secondary detail. |

Never carry meaning by colour alone. A chip says its state in words, an error has text, and a toggle has a visible on position.

### Themes

`shell/theme.js` sets `data-theme` (`dark` or `light`) on `<html>`. The choice behind it is `data-theme-pref` (`auto`, `dark`, `light`) and is stored in `localStorage` under `funpack_theme`. `auto` follows the system and is resolved before first paint so there is no wrong-theme flash.

- Token values are scoped to any `[data-theme]` element, so a single pane can show the other theme (the catalogue does this).
- Test every new element in both themes in the catalogue.
- A custom property used but never defined resolves to nothing without any error, so check that every `var(--x)` you write exists.

## Layout and density

- Build layout with `region.stack`, `splitPane`, `panel`, `toolbar`, `group` and `workspace`, not with ad hoc CSS.
- Align on the spacing scale. Related things sit close (`--sp-1`, `--sp-2`), separate groups sit apart (`--sp-4` and up).
- Media grids use `gallery` and its density control (`internals/density.js`, 0 to 4 columns, remembered per gallery id in `localStorage` as `cx.density.<id>`, always wrapped in try/catch because storage can be blocked).
- Text never forces the layout wider. Long names truncate with the full name in the title. Numbers that change use tabular figures.
- Support narrow windows: a panel scrolls inside itself, the page does not scroll sideways.

## Which element for what

| You need | Use | Not |
|---|---|---|
| One action | `button` (`md`; one `primary` per area) | A link styled as a button |
| An action with only an icon | `iconButton` (the `label` is required and becomes the tooltip and the accessible name) | An icon with no name |
| An on/off setting | `toggle`, or `checkboxRow` in a list of options | A select with Yes and No |
| A number | `number`; add `slider` when the range is small and exploring helps | A text box for a number |
| One of 2 to 4 short options | `segmented` | A select |
| One of many options | `select`; `filterList` when the list is long | A very long segmented row |
| Short text | `input` | `textarea` |
| A prompt or notes | `textarea` | A one-line input |
| A colour | `color.swatch` | A hex text box |
| A message that stays | `banner` | A toast |
| A message that fades | `toast` | A banner that never leaves |
| A small state word | `chip` | Coloured plain text |
| Ask before a risky step | `modal.dialogue` with `tone: "danger"` | `window.confirm` |
| A list of things | `list.rows` or `gallery` | Hand-made divs |
| Group of related settings | `panel`, `collapsible` or `settingsRow` | A bare run of inputs |
| Wait with no known length | `progress.indeterminate`, `spinner`, `skeleton` | Frozen controls with no sign |

Never call `alert`, `confirm` or `prompt`. They block the page, cannot be themed and cannot be tested.

## Interaction

- **Click targets** are at least the control height for their size. Dense toolbars use `iconButton.sm`; `micro` is for a row that already has its own hit area.
- **Commit rules.** Text and number inputs commit on blur or Enter (`onCommit`). Sliders report live movement through `onChange` and the settled value through `onCommit`. Do not start a run or write the project on every keystroke.
- **Focus.** The focus ring (`--focus-ring`) is always visible for keyboard use and is never removed. Overlays trap focus (`internals/focus.js` `trap`) and hand it back on close. Lists and menus use `roving` (arrow keys move, Tab leaves).
- **Escape.** Overlays register with the dismiss stack (`internals/dismiss.js`). Escape closes only the top entry, a click outside dismisses the top entry only if it was opened with `closeOnOutside`. `closeOnEsc` works the same way. Do not add your own key handler for this.
- **Busy.** A button doing work shows `busy` and ignores further clicks. It does not change its label to "Please wait".
- **Undo beats confirm** when the step can be undone. Confirm only for what cannot.
- **Drag.** Dragging uses `internals/drag.js` and `dragReorder.js`. Every drag has a keyboard or button way to do the same thing.

## Overlays and layers

`z-index` is written in one place only: `internals/zlayer.js`. Elements ask for a layer; they never pick a number.

Order, low to high: `local`, `dropline`, `sticky`, `menubar`, `backdrop`, `wizard`, `floatingWindow`, `modal`, `popover`, `toast`, `autocomplete`, `tour`. Each rung owns a block of 100 slots, so peers stack inside a rung and never cross into the next.

- A modal is above every floating window, because nothing behind it may be used.
- Autocomplete is above a modal on purpose: it opens from a field that is often inside one.
- Overlays are drawn into the portal root (`internals/portal.js`). The portal root must not create a stacking context, or the ladder stops working.
- An anchored popover positions itself with `internals/anchor.js` and flips to stay on screen.

## Accessibility

- Every control has an accessible name. Icon-only controls get it from `label`.
- State is exposed (`pressed`, `disabled`, `busy`) and not only drawn.
- Contrast: text on a surface uses `--text` or `--muted`. `--faint` is for non-essential detail only. Check both themes.
- Reduced motion: `--dur-*` become zero and `internals/motion.js` checks `prefers-reduced-motion`. A new animation must go through them. Do not use a raw `transition: 200ms`.
- Everything reachable by mouse is reachable by keyboard.

## Words in the UI

Dex's rule is plain words. Apply it to every label, hint, error and toast.

- Short and plain. A label is a noun or a short verb phrase ("Steps", "Save project").
- A hint is one sentence that says what the setting does for the result, not how it works inside ("Keeps the first picture's look through the clip").
- Sentence case. No ending full stop on a label. A hint ends with one.
- A plain hyphen. No em dash or en dash anywhere in the repo, docs and code strings included. Existing code comments still contain some; do not add more.
- An error says what happened and what to do next ("The seed must be a whole number"). It never shows a stack trace or an internal name.
- Do not promise what is only likely. Not "always", not "guaranteed". Say "usually" or describe the setting.
- Do not describe a feature by its mechanism first. Lead with what the person gets.
- Use the same word for the same thing everywhere. If the app says "Project", do not say "Session" in a hint.
- No jargon a person would need a manual for. If a technical word is unavoidable, the hint explains it in a few plain words.

## Checklist before you open a pull request

- [ ] Only kit elements; no literal colour, size or z-index; no `document` in new feature code.
- [ ] Every string is passed as text. A name containing `<b>` shows as `<b>`.
- [ ] Looks right in dark and light, at a narrow width, with reduced motion on.
- [ ] Usable by keyboard alone; Escape closes the top overlay only.
- [ ] If the feature cannot load, nothing is drawn and the Log says why.
- [ ] Text follows the word rules above, and has no long dashes.
- [ ] Nothing a person is typing is overwritten by a redraw.
- [ ] `npm test` passes (see [Testing](testing.md)); a new element has a demo in the catalogue.
