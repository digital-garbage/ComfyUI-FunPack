// The pipeline, opened up and edited.
//
// One wide modal, a card per group, and a group that opens into a node list on
// the left with that node's parameters on the right, saved a group at a time.
//
// Two kinds of edit, and they behave differently on purpose:
//
// * A VALUE is drafted. Changing steps from 20 to 30 cannot make a graph
//   illegal, so it waits in the draft until Save, and Cancel puts it back.
// * A STRUCTURE change -- adding a node, removing one, pointing a slot at a
//   different node, moving it to another group -- goes to the server as it
//   happens, because only the server can say whether the result still wires up,
//   and a refusal shown ten edits later is a refusal nobody can act on.
//
// Nothing here composes a graph, and nothing here makes an element. Both would
// be a second answer to a question that already has one -- what runs, and what a
// control looks like.

import { composer } from "../composer/composer.js";
import { rendererFor, rendererNameFor, SELF_LABELLING } from "../composer/panel.js";
import { settingFor, label as humanLabel, whyNotEditable } from "./widgets.js";

const UNGROUPED = "Other";

// The value of a list nobody has chosen from. Not "" -- an empty string is a
// value a combo could legitimately offer.
const NOT_SET = "\u0000not-set";

/** Groups in the order they first appear, plus any the user has just made. */
export function groupsOf(slots, extra = []) {
  const order = [];
  const byGroup = new Map();
  const put = (name) => {
    if (!byGroup.has(name)) { byGroup.set(name, []); order.push(name); }
    return byGroup.get(name);
  };
  for (const slot of slots) put(slot.group || UNGROUPED).push(slot);
  for (const name of extra) put(name);
  return { order, byGroup };
}

const isLink = (value) => Array.isArray(value) && value.length === 2
  && typeof value[0] === "string" && typeof value[1] === "number";

const countOf = (n) => (n === 0 ? "empty" : `${n} node${n === 1 ? "" : "s"}`);

/**
 * open({ load, describe, check, search, onApply }) -> handle
 *
 * Every server call is injected. The window is then testable without a server
 * and -- more to the point -- cannot reach for one that is not there.
 */
export function open({ load, describe, check, search, onApply } = {}) {
  let slots = [];
  const nodes = new Map();          // class name -> description | null
  const extraGroups = [];           // made in this window, no slot in them yet
  let draft = null;                 // slot id -> { input: value }, while editing
  let editing = null;               // group name
  let selected = null;              // slot id
  let notes = { refused: [], incomplete: [] };

  const body = composer.region.stack({ gap: "md", label: "Pipeline", fill: true });
  const modal = composer.modal.generic({
    title: "Models and pipeline",
    subtitle: "What the run is made of.",
    size: "xl",
    body,
    // A half-finished group edit must not vanish because a click landed on the
    // backdrop: the draft is the only copy of it.
    closeOnOutside: false,
  });

  // ---------------------------------------------------------------- server

  async function refresh() {
    const answer = await load();
    slots = answer.slots;
    notes = { refused: [], incomplete: answer.incomplete || [] };
    await learn(slots.map((slot) => slot.node));
  }

  /** Descriptions for classes we have not seen. Asked once per class. */
  async function learn(classes) {
    const unknown = [...new Set(classes)].filter((c) => c && !nodes.has(c));
    if (!unknown.length) return;
    const described = await describe(unknown);
    // Every class asked about gets an entry, `null` included: without it an
    // absent node is asked for again on every redraw, and reads on screen as a
    // node still being looked up rather than one that is not installed.
    for (const name of unknown) nodes.set(name, described[name] ?? null);
  }

  /** Send the pipeline as it now stands and keep what the server says of it. */
  async function commit(next, action = {}) {
    let answer;
    try {
      answer = await check({ slots: next, ...action });
    } catch (err) {
      notes = { refused: [err.message], incomplete: notes.incomplete };
      draw();
      return false;
    }
    // A refused edit did not happen, so what is on screen must stay what it
    // was: showing the refusal beside the change it refused is how someone
    // comes to believe an edit landed when it did not.
    if (answer.refused.length) {
      notes = { refused: answer.refused, incomplete: notes.incomplete };
      draw();
      return false;
    }
    slots = answer.slots;
    notes = { refused: [], incomplete: answer.incomplete };
    await learn(slots.map((slot) => slot.node));
    if (onApply) onApply(slots);
    draw();
    return true;
  }

  // ----------------------------------------------------------------- draw

  function draw() {
    if (editing === null) {
      body.set(index());
      // Nothing to save from the index: the cards are a way in, not an edit.
      modal.setFooter({});
      return;
    }
    body.set(group(editing));
    modal.setFooter(footer());
  }

  /**
   * What to say, and where.
   *
   * A refusal is about the edit just attempted, so it belongs wherever that
   * edit was made. What is still unfilled is a property of the WHOLE pipeline,
   * and repeating all of it above every group put three warnings about the
   * loaders over the sampler's settings, where none of them were about anything
   * on screen. Inside a group the unfilled control says so itself -- it reads
   * "not set" -- which is the same information in the place it can be acted on.
   */
  function messages({ everything = false } = {}) {
    return [
      ...notes.refused.map((text) => composer.banner.danger({ text })),
      // Not an error: a fresh install has no model picked, and colouring that
      // red says something went wrong when nothing has yet.
      ...(everything ? notes.incomplete.map((text) => composer.banner.warn({ text })) : []),
    ];
  }

  // ---- the index: one card per group

  function index() {
    const { order, byGroup } = groupsOf(slots, extraGroups);
    return [
      ...messages({ everything: true }),
      composer.gallery.cards({
        items: order.map((name) => ({
          id: name, label: name, hint: countOf(byGroup.get(name).length),
        })),
        onActivate: (item) => enter(item.id),
      }),
      composer.toolbar.default({ label: "Groups", items: [newGroupButton()] }),
    ];
  }

  function newGroupButton() {
    return composer.button.md({
      label: "New group",
      onClick: async () => {
        const picked = await composer.modal.prompt({
          title: "New group",
          label: "Name",
          placeholder: "Upscaling",
          validate: (name) => {
            if (!name.trim()) return "A group needs a name.";
            if (groupsOf(slots, extraGroups).byGroup.has(name.trim())) {
              return "There is already a group called that.";
            }
            return null;
          },
        }).result;
        if (picked === null) return;
        // Held only for as long as this window is open. A group with no node in
        // it is not part of the pipeline, so there is nothing to save it on --
        // said in the card rather than left to be discovered after a reload.
        extraGroups.push(picked.trim());
        draw();
      },
    });
  }

  // ---- one group: nodes on the left, the selected node's parameters right

  function enter(name) {
    editing = name;
    draft = new Map();
    selected = (groupsOf(slots, extraGroups).byGroup.get(name) || [])[0]?.id ?? null;
    draw();
  }

  function leave() {
    editing = null;
    draft = null;
    selected = null;
    draw();
  }

  function group(name) {
    const mine = groupsOf(slots, extraGroups).byGroup.get(name) || [];
    if (selected && !mine.some((slot) => slot.id === selected)) selected = mine[0]?.id ?? null;

    const left = composer.region.stack({ gap: "sm", label: "Nodes", children: [
      composer.filterList.md({
        items: mine.map((slot) => ({ id: slot.id, label: slot.id, hint: titleOf(slot.node) })),
        value: selected,
        placeholder: "Find a node",
        empty: "No nodes in this group yet",
        onChange: (id) => { selected = id; draw(); },
      }),
      composer.toolbar.default({ label: "Nodes", items: [
        composer.button.sm({ label: "Add node", onClick: () => pickNode(name) }),
        composer.button.sm({
          label: "Remove", tone: "ghost", disabled: !selected,
          onClick: () => remove(selected),
        }),
      ] }),
    ] });

    const right = composer.region.stack({ gap: "sm", label: "Parameters",
      children: selected ? parameters(slotBy(selected)) : [
        composer.emptyState.default({
          icon: "◇", title: "Nothing selected",
          hint: "Pick a node on the left, or add one.",
        }),
      ] });

    return [
      composer.toolbar.default({ label: "Group", items: [
        composer.button.sm({ label: "← All groups", tone: "ghost", onClick: leave }),
        composer.text.sm({ text: name }),
      ] }),
      ...messages(),
      composer.splitPane.h({ panes: [left, right], size: 32, label: "Nodes and parameters" }),
    ];
  }

  /**
   * Save and Cancel, in the modal's own footer rather than in the body.
   *
   * The footer is the one row outside the scrolling body. A bar built inside it
   * sticks to the bottom of the SCROLLER, which is above the card's own edge by
   * the body's padding -- so a strip of the settings showed through underneath
   * the buttons and scrolled as the settings scrolled.
   */
  function footer() {
    const pending = draft ? draft.size : 0;
    return {
      note: pending
        ? `${pending} node${pending === 1 ? "" : "s"} edited — not saved yet`
        : "No changes to save",
      actions: [
        composer.button.lg({ label: "Cancel", tone: "neutral", onClick: leave }),
        composer.button.lg({
          label: "Save", tone: "primary",
          onClick: async () => { if (await saveDraft()) leave(); },
        }),
      ],
    };
  }

  const slotBy = (id) => slots.find((slot) => slot.id === id) || null;
  const titleOf = (className) => (nodes.get(className) || {}).title || className;

  // ---- the parameters of one node

  function parameters(slot) {
    if (!slot) return [];
    const described = nodes.get(slot.node);

    const rows = [
      composer.settingsRow.default({
        label: titleOf(slot.node),
        hint: slot.node,
        control: composer.button.sm({
          label: "Change node",
          onClick: () => pickNode(slot.group || UNGROUPED, slot.id),
        }),
      }),
      composer.settingsRow.default({
        label: "Group",
        hint: "Which card this node appears under.",
        control: composer.select.md({
          options: groupsOf(slots, extraGroups).order.map((g) => ({ value: g, label: g })),
          value: slot.group || UNGROUPED,
          onChange: (to) => moveTo(slot.id, to),
        }),
      }),
    ];

    if (!described) {
      // The slot names a node this install does not have. Said here, where its
      // parameters would be: an empty pane reads as "no settings".
      rows.push(composer.banner.danger({
        text: `${slot.node} is not installed here, so there is nothing to edit and the `
            + "pipeline will not run until this slot points somewhere else.",
      }));
      return rows;
    }

    for (const widget of described.widgets) {
      let setting = settingFor(widget);
      if (setting && unset(slot, widget)) setting = asUnset(setting);
      if (!setting) {
        rows.push(composer.settingsRow.default({
          label: humanLabel(widget.name), hint: whyNotEditable(widget),
        }));
        continue;
      }
      const current = valueOf(slot, widget.name, setting.default);
      const control = rendererFor(setting)(setting, current,
        (next) => { if (next !== NOT_SET) edit(slot.id, widget.name, next); });
      // A checkbox row draws its own label and hint. Wrapping it in a settings
      // row printed both of them twice, one above the other.
      rows.push(SELF_LABELLING.has(rendererNameFor(setting))
        ? control
        : composer.settingsRow.default({
            label: setting.label, hint: setting.hint, control,
          }));
    }

    for (const socket of described.sockets) {
      const wired = (slot.inputs || {})[socket.name];
      rows.push(composer.settingsRow.default({
        label: humanLabel(socket.name),
        // Said outright, because there is no control here to look for. Wiring
        // is not something this window does yet, and an input reading only
        // "nothing is wired to it" reads as a thing the user failed to find.
        hint: isLink(wired)
          ? `fed by ${wired[0]}`
          : `${socket.type} — nothing feeds it, and this window cannot wire it yet`,
      }));
    }

    return rows;
  }

  /** Has this input been given a value -- by the pipeline, or in this draft? */
  function unset(slot, widget) {
    const edited = draft && draft.get(slot.id);
    if (edited && Object.prototype.hasOwnProperty.call(edited, widget.name)) return false;
    return (slot.inputs || {})[widget.name] === undefined;
  }

  /**
   * A choice that has not been made, shown as one.
   *
   * A select rendered with the first choice looks like a value that was picked:
   * the pipeline says "model_name is not filled" while the box beside it reads
   * `ltx2_3_video_diffusion_fp8.safetensors`, and picking that same entry fires
   * no change event, so the one click that looks like the fix does nothing.
   * Only for lists -- a text box showing "" is already showing what it holds.
   */
  function asUnset(setting) {
    if (setting.type !== "enum") return setting;
    return {
      ...setting,
      default: NOT_SET,
      options: [{ value: NOT_SET, label: "— not set —" }, ...setting.options],
    };
  }

  /** The drafted value if this node has been edited, else what the slot holds. */
  function valueOf(slot, name, fallback) {
    const edited = draft && draft.get(slot.id);
    if (edited && Object.prototype.hasOwnProperty.call(edited, name)) return edited[name];
    const held = (slot.inputs || {})[name];
    // A wired input is not a value: rendering the link as text would put
    // "model,0" in a box and offer to save it as one.
    if (held === undefined || isLink(held)) return fallback;
    return held;
  }

  function edit(slotId, name, value) {
    draft.set(slotId, { ...(draft.get(slotId) || {}), [name]: value });
    // The footer, not the body. Redrawing the body would rebuild the control
    // that fired this and take the cursor out of it mid-edit; leaving the
    // footer alone left it reading "No changes to save" over an edit that had
    // just been made.
    modal.setFooter(footer());
  }

  async function saveDraft() {
    if (!draft.size) return true;
    const next = slots.map((slot) => (draft.has(slot.id)
      ? { ...slot, inputs: { ...(slot.inputs || {}), ...draft.get(slot.id) } }
      : slot));
    const ok = await commit(next);
    if (ok) draft = new Map();
    return ok;
  }

  // ---- structure

  async function moveTo(slotId, to) {
    await commit(slots.map((slot) => (slot.id === slotId ? { ...slot, group: to } : slot)));
  }

  async function remove(slotId) {
    await commit(slots, { action: "remove", slot: slotId });
  }

  /** Add a node to `group`, or -- with `replacing` -- point that slot elsewhere. */
  function pickNode(group, replacing = null) {
    const results = composer.region.stack({ gap: "sm", label: "Results" });
    let latest = 0;
    let typing = null;

    async function run(query) {
      const mine = ++latest;
      let answer;
      try {
        answer = await search(query);
      } catch (err) {
        results.set([composer.inlineError.default({ text: err.message })]);
        return;
      }
      // An older search finishing after a newer one would overwrite the newer
      // results with an answer to a question nobody is asking any more.
      if (mine !== latest) return;
      results.set([
        composer.filterList.md({
          items: answer.nodes.map((n) => ({
            id: n.node, label: n.title, hint: n.category || n.node,
          })),
          placeholder: "Filter these results",
          empty: "Nothing matches",
          onChange: (className) => { picker.close("picked"); chose(className); },
        }),
        answer.total > answer.nodes.length
          ? composer.hint.default({
              text: `Showing ${answer.nodes.length} of ${answer.total} — `
                  + "type to narrow it down.",
            })
          : null,
      ].filter(Boolean));
    }

    // As you type, not on Enter: a picker that answers only when committed to
    // makes you guess the whole name before it says whether it exists.
    // Debounced, because otherwise every keystroke is a request.
    const box = composer.search.md({
      placeholder: "Search installed nodes",
      label: "Node",
      onInput: (query) => { clearTimeout(typing); typing = setTimeout(() => run(query), 150); },
      onCommit: (query) => { clearTimeout(typing); run(query); },
    });

    const picker = composer.modal.stacked({
      title: replacing ? `Change ${replacing}` : `Add a node to ${group}`,
      size: "md",
      body: composer.region.stack({ gap: "sm", children: [box, results] }),
      // A pending search must not land on a closed picker, and a search whose
      // answer is thrown away is a request nobody needed.
      onClose: () => { clearTimeout(typing); latest += 1; },
    });
    run("");

    async function chose(className) {
      await learn([className]);
      if (replacing) {
        await commit(slots, { action: "replace", slot: replacing, node: className });
        return;
      }
      // No inputs: what feeds a new node is a wiring decision, and guessing one
      // is how a node ends up quietly reading the wrong thing.
      const id = freeId(className);
      if (await commit([...slots, { id, group, node: className, inputs: {} }])) {
        selected = id;
        draw();
      }
    }
  }

  /** A slot id nobody is using, derived from the node's name. */
  function freeId(className) {
    const base = className.replace(/[^A-Za-z0-9]+/g, "_").toLowerCase();
    const taken = new Set(slots.map((slot) => slot.id));
    if (!taken.has(base)) return base;
    for (let n = 2; ; n += 1) if (!taken.has(`${base}_${n}`)) return `${base}_${n}`;
  }

  // --------------------------------------------------------------- startup

  body.set([composer.spinner.md({ label: "Reading the pipeline" })]);
  const ready = refresh().then(draw).catch((err) => {
    body.set([composer.emptyState.default({
      icon: "▲",
      title: "The pipeline could not be read",
      hint: err.message,
    })]);
  });

  return {
    node: modal.node,
    close: (reason) => modal.close(reason),
    ready,
    // The window's own view of things, so a test does not have to infer state
    // by reading the DOM back.
    get slots() { return slots; },
    get editing() { return editing; },
    get selected() { return selected; },
    get pending() { return draft ? draft.size : 0; },
    enter,
    leave,
  };
}
