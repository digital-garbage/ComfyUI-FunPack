// Models & Pipeline: every node in the live pipeline by group, its values editable, and its shape: swap, add, remove, rewire, regroup.
// The server refuses what would not build (a type that does not fit, a loop) and says why; the reason is shown and nothing changes.
import { composer as c } from "../../composer/composer.js";
import { pickNode } from "./nodepick.js";
import { labelOf, shownValues, sourcesFor } from "./wiring.js";

const SEP = "\0", NEW_GROUP = "\0new";
const fed = (v) => Array.isArray(v);

const widgetControl = (w, current, set) => {
  const label = w.name;
  if (w.type === "COMBO") {
    const choices = [...(w.choices || [])];
    if (current != null && !choices.includes(current)) choices.unshift(current);       // a file gone from disk still shows, so saving something else cannot swap it out
    return choices.length ? c.select.md({ label, value: current ?? choices[0], options: choices.map((v) => ({ value: v, label: String(v) })), onChange: set })
      : c.hint.default({ text: "No choices available — nothing found in the models folder." });
  }
  if (w.type === "INT" || w.type === "FLOAT") {
    return c.number.md({ label, value: current ?? w.default ?? 0, min: w.min, max: w.max, step: w.step ?? (w.type === "INT" ? 1 : undefined),
      precision: w.type === "INT" ? 0 : undefined, onChange: set });
  }
  return w.multiline ? c.textarea.md({ label, value: current ?? "", rows: 4, onCommit: set }) : c.input.md({ label, value: current ?? "", onCommit: set });
};

export const models = (app) => function mount() {
  const ps = app.pipeline, api = app.api;
  const page = c.region.stack({ gap: "md" });
  let specs = {}, note = "", group = "", off = null, opened = null, gone = false, openId = null;
  const slotsNow = () => ps.slots() || [];
  const say = (text) => { note = text; draw(); };
  const label = (slot) => labelOf(slot, specs, slotsNow());

  async function describe() {
    const missing = [...new Set(slotsNow().map((s) => s.node))].filter((n) => !(n in specs));
    if (!missing.length) return;
    try { Object.assign(specs, (await api.describeNodes(missing)).nodes || {}); } catch (err) { note = err.message; }
    missing.forEach((n) => { if (!(n in specs)) specs[n] = null; });
  }
  const setInput = (slot, name) => async (value) => {
    if (JSON.stringify((slot.inputs || {})[name]) === JSON.stringify(value)) return;          // a blur that changed nothing saves nothing
    await ps.save({ inputs: { [slot.id]: { [name]: value } } });
    say([...(ps.refused() || []), ...ps.saveNotes()].join(" "));
  };
  async function structural(body) {
    const res = await ps.edit(body);
    await describe();
    if (!res.refused.length) {
      const target = body.action === "add" ? slotsNow().slice(-1)[0] : body.action === "replace" || body.action === "unwire" ? slotsNow().find((s) => s.id === body.slot) : null;
      const values = target && shownValues(target, specs[target.node], body.action === "unwire" ? body.input : undefined);
      if (values && Object.keys(values).length) await ps.save({ inputs: { [target.id]: values } });
    }
    say(res.refused.length ? `Not changed: ${res.refused.join(" ")}` : "");
    return !res.refused.length;
  }
  const confirm = (title, message, tone) => c.modal.dialogue({ title, message, tone, confirmLabel: title.split(" ")[0] }).result;

  // A select over what could feed an input: "a value" (or not connected), or an output of another node.
  const wiring = (slot, input, type, current, valueLabel) => {
    const options = [{ value: "", label: valueLabel }];
    const sources = sourcesFor(slot, type, slotsNow(), specs), now = fed(current) ? `${current[0]}${SEP}${current[1]}` : "";
    if (now && !sources.some((s) => s.value === now)) options.push({ value: now, label: `${current[0]} · output ${current[1]}` });
    sources.forEach((s) => options.push({ value: s.value, label: `Fed by ${s.label}` }));
    return c.select.md({ label: input, value: now, options, onChange: (v) => {
      if (!v) structural({ action: "unwire", slot: slot.id, input });
      else { const [from, idx] = v.split(SEP); structural({ action: "wire", slot: slot.id, input, from_slot: from, from_output: Number(idx) }); }
    } });
  };

  const groups = () => [...new Set(slotsNow().map((s) => s.group || "Other"))];
  const structure = (slot) => {
    const here = slot.group || "Other";
    return c.field.default({ label: "Group", hint: "Which tab this node sits under. Any node can live in any group.", control: c.toolbar.default({
      items: [c.select.md({ label: "Group", value: here, options: [...groups().map((g) => ({ value: g, label: g })), { value: NEW_GROUP, label: "New group…" }], onChange: async (v) => {
        let name = v;
        if (v === NEW_GROUP) { name = ((await c.modal.prompt({ title: "New group", label: "Name", confirmLabel: "Create" }).result) || "").trim(); if (!name) return draw(); }
        await ps.setGroup(slot.id, name); group = name; draw();
      } })],
      trailing: [c.button.sm({ label: "Swap node…", tone: "ghost", title: "Use a different node here. It must still produce what the nodes after it read.", onClick: async () => {
        const cls = await pickNode(api, `Swap ${label(slot)} for…`);
        if (!cls) return;
        const driven = (slot.roles || []).map((r) => r.label || r.input).filter(Boolean);
        if (await confirm("Swap node", `Swap ${label(slot)} for ${cls}? Everything set on this node (files, values, connections into it) is cleared, and the new node starts empty.${driven.length ? ` The main window drives this node's ${driven.join(", ")}: those controls keep working only for inputs the new node also has, by the same name.` : ""}`, "danger")) structural({ action: "replace", slot: slot.id, node: cls });
      } }), c.button.sm({ label: "Remove", tone: "danger", title: "Take this node out; what it fed is rewired to what fed it, when that is unambiguous.", onClick: async () => {
        if (await confirm("Remove node", `Remove ${label(slot)} from the pipeline?`, "danger")) structural({ action: "remove", slot: slot.id });
      } })] }) });
  };

  // Which pack gives a missing node (from ComfyUI-Manager's map, when it is installed), and a button to clone it.
  const providers = {};
  let managerHere = true;
  const installRow = (cls) => {
    const row = c.region.stack({ gap: "sm" });
    const show = () => {
      const url = providers[cls];
      row.set(url === undefined ? [c.hint.default({ text: "Looking for the pack that provides it…" })]
        : url ? [c.settingsRow.default({ label: url.replace(/^https?:\/\/(www\.)?github\.com\//, ""), hint: "Provides this node. Installs into custom_nodes; restart ComfyUI afterwards.", control: c.button.sm({ label: "Install", tone: "primary", onClick: async () => {
          try { await api.pack("install", { url }); c.toast.good({ text: "Installed. Restart ComfyUI for the node to appear." }); } catch (err) { c.toast.warn({ text: err.message }); }
        } }) })]
        : [c.hint.default({ text: managerHere ? "No known pack provides it. Add one by its git URL in Settings ▸ Custom Nodes." : "ComfyUI-Manager is not installed here, so there is no list to look it up in. Add the pack by its git URL in Settings ▸ Custom Nodes." })]);
    };
    show();
    if (!(cls in providers)) api.packProviders([cls]).then((r) => { managerHere = r.manager !== false; providers[cls] = (r.providers || {})[cls] || null; show(); }).catch(() => { providers[cls] = null; show(); });
    return row;
  };

  const slotBlock = (slot) => {
    const spec = specs[slot.node];
    if (spec === undefined) return [c.hint.default({ text: "Loading…" })];
    const head = c.header.sm({ text: label(slot) });
    if (spec === null) return [head, structure(slot), c.banner.warn({ text: `${slot.node} is not installed — this slot can't be edited or run. Install the pack that provides it, swap it for another node, or remove it.` }), installRow(slot.node)];
    const sockets = (spec.sockets || []).map((s) => c.field.default({ label: s.name, hint: s.required ? `${s.type} · needed` : s.type, control: wiring(slot, s.name, s.type, slot.inputs && slot.inputs[s.name], "Not connected") }));
    const rows = (spec.widgets || []).flatMap((w) => {
      const cur = slot.inputs && slot.inputs[w.name];
      if (fed(cur)) return [c.field.default({ label: w.name, hint: "Fed by another node: change it there, or pick “A value” to type one here.", control: wiring(slot, w.name, w.type, cur, "A value (typed here)") })];
      const value = cur !== undefined ? cur : w.default;
      const row = w.type === "BOOLEAN" ? c.toggle.default({ label: w.name, hint: w.tooltip || "", checked: Boolean(value), onChange: setInput(slot, w.name) })
        : c.settingsRow.default({ label: w.name, hint: w.tooltip || "", control: widgetControl(w, value, setInput(slot, w.name)) });
      const appOwned = (slot.roles || []).some((r) => r.input === w.name && r.at !== "project.video");       // the app writes it: wiring it elsewhere would make every Generate refuse
      const canTake = !appOwned && (w.type !== "COMBO" || (w.choices || []).length) && sourcesFor(slot, w.type, slotsNow(), specs).length;
      return canTake ? [row, c.field.default({ label: "↳ or take it from", hint: "Another node's output can drive this value; one Primitive can drive several inputs.", control: wiring(slot, w.name, w.type, null, "A value (typed here)") })] : [row];
    });
    return [head, structure(slot), ...sockets, ...rows, ...(!rows.length && !sockets.length ? [c.hint.default({ text: "Nothing to configure on this node." })] : [])];
  };

  // One node as a card: what it is, the one or two values that matter, and whether it needs attention.
  const summary = (slot) => {
    const spec = specs[slot.node];
    if (spec === undefined) return "Loading…";
    if (spec === null) return "Not installed";
    const first = (spec.widgets || [])[0], shown = (spec.widgets || []).map((w) => (slot.inputs || {})[w.name]).filter((v) => typeof v === "string" && v && v !== "default" && v !== "disabled" && !fed(v)).slice(0, 2);
    if (first && first.type === "COMBO" && !(slot.inputs || {})[first.name]) return `${first.name} not set`;
    return shown.length ? shown.join(" · ") : `${(spec.widgets || []).length} setting${(spec.widgets || []).length === 1 ? "" : "s"}`;
  };
  const nodeCard = (slot) => {
    const node = document.createElement("button");
    node.type = "button"; node.className = "fp-node-card cx-focusable";
    const needs = specs[slot.node] === null || ps.incomplete().some((t) => t.startsWith(`${slot.id}:`));
    const head = Object.assign(document.createElement("span"), { className: "fp-node-title", textContent: label(slot) });
    const dot = Object.assign(document.createElement("span"), { className: `fp-node-state${needs ? " fp-needs" : ""}`, title: needs ? "Needs attention" : "Ready" });
    const sub = Object.assign(document.createElement("span"), { className: "fp-node-sub", textContent: summary(slot) });
    node.append(dot, head, sub);
    node.addEventListener("click", () => { openId = slot.id; draw(); });
    return { node };
  };
  const addCard = () => {
    const node = document.createElement("button");
    node.type = "button"; node.className = "fp-node-card fp-node-add cx-focusable";
    node.append(Object.assign(document.createElement("span"), { className: "fp-node-title", textContent: "＋ Add a node" }),
      Object.assign(document.createElement("span"), { className: "fp-node-sub", textContent: `to ${group}` }));
    node.addEventListener("click", async () => { const cls = await pickNode(api, `Add a node to ${group}`); if (cls) structural({ action: "add", node: cls, group }); });
    return { node };
  };

  function status(only) {
    const all = ps.incomplete(), refused = ps.refused(), notes = ps.saveNotes();
    const incomplete = only ? all.filter((t) => t.startsWith(`${only}:`)) : all;
    return [refused.length ? c.banner.warn({ text: `Could not save: ${refused.join(" ")}` }) : null, notes.length ? c.banner.info({ text: notes.join(" ") }) : null,
      incomplete.length ? c.banner.warn({ text: only ? incomplete.join(" ") : `Not ready to generate yet: ${incomplete.length} node${incomplete.length === 1 ? "" : "s"} still need something (the orange dots). Open one to see what.` }) : ps.queueable() && !only ? c.hint.default({ text: "Every slot is filled — this pipeline is ready to generate." }) : null];
  }

  const tools = () => c.toolbar.default({ items: [c.button.sm({ label: "Start from…", tone: "ghost", title: "Replace this pipeline with a model's own starting point", onClick: presets }),
    c.button.sm({ label: "Import workflow…", tone: "ghost", title: "Use a ComfyUI workflow (UI or API export) as this pipeline", onClick: importWorkflow }),
    c.button.sm({ label: "Revert", tone: "ghost", disabled: !opened, title: "Put back what was here when this page was opened", onClick: async () => { const r = await ps.restore(opened); say(r.refused.length ? r.refused[0] : ""); } })],
    trailing: [c.button.sm({ label: "🖼 Export settings…", tone: "ghost", title: "Render this pipeline as a PNG: loaders, typed-in values, torch / CUDA / attention", onClick: card })] });

  async function presets() {
    let found;
    try { found = (await api.pipelinePresets()).presets || []; } catch (err) { return say(err.message); }
    if (!found.length) return say("No module offers a starting point.");
    const id = await c.modal.choice({ title: "Start from…", items: found.map((p) => ({ id: p.id, label: p.title, hint: p.module })) }).result;
    const pick = found.find((p) => p.id === id);
    if (!pick || !(await confirm("Replace pipeline", `Replace the whole pipeline with “${pick.title}”? Every file and value you set here is cleared. Revert puts it back until you leave this page.`, "danger"))) return;
    const r = await ps.restore({ slots: pick.slots, removed: [], unwired: {} });
    specs = {}; await describe(); say(r.refused.length ? `Not changed: ${r.refused[0]}` : "");
  }
  async function importWorkflow() {
    const file = await new Promise((resolve) => { const i = Object.assign(document.createElement("input"), { type: "file", accept: ".json,application/json" }); i.onchange = () => resolve(i.files[0]); i.oncancel = () => resolve(null); i.click(); });
    if (!file) return;
    let got;
    try { got = await api.importWorkflow(JSON.parse(await file.text())); } catch (err) { return say(err instanceof SyntaxError ? "That file is not valid JSON." : err.message); }
    const names = { prompt: "Prompt", negative: "Negative prompt", seed: "Seed", width: "Width", height: "Height", frames: "Length", fps: "FPS", image: "Start picture" };
    const lines = [`${got.slots.length} nodes.`, ...Object.entries(names).map(([k, label]) => `${label}: ${got.bound[k] ? "→ " + got.bound[k].join(", ") : "not connected (set on the node itself)"}`), ...got.notes];
    if (!(await confirm("Use this workflow", `${lines.join("\n")}\n\nReplace the whole pipeline with it? Revert puts the old one back until you leave this page.`, "danger"))) return;
    const r = await ps.restore({ slots: got.slots, removed: [], unwired: {} });
    specs = {}; await describe(); say(r.refused.length ? `Not changed: ${r.refused[0]}` : "");
  }
  async function card() {
    let url;
    try { url = URL.createObjectURL(await api.settingsCard(slotsNow(), app.project.project && app.project.project.name, app.theme && app.theme.get ? app.theme.get() : "dark")); } catch (err) { return say(`Could not render the card: ${err.message}`); }
    const img = Object.assign(document.createElement("img"), { src: url, alt: "Pipeline settings", style: "max-width:100%" });
    const win = c.modal.generic({ title: "Pipeline settings", size: "lg", body: { node: img }, onClose: () => URL.revokeObjectURL(url) });
    win.setFooter({ actions: [c.button.sm({ label: "Save PNG", tone: "primary", onClick: () => Object.assign(document.createElement("a"), { href: url, download: "funpack-settings.png" }).click() })] });
  }

  function draw() {
    if (gone) return;
    if (ps.loading()) return page.set([c.hint.default({ text: "Loading…" })]);
    if (ps.loadError()) return page.set([c.banner.warn({ text: `Could not load models & pipeline: ${ps.loadError()}` })]);
    const all = groups(), slots = slotsNow();
    if (!all.includes(group)) group = all[0] || "";
    const open = openId && slots.find((x) => x.id === openId);
    if (openId && !open) openId = null;                       // removed or swapped away
    if (open) {
      const back = c.button.sm({ label: `‹ ${open.group || "Other"}`, tone: "ghost", onClick: () => { openId = null; draw(); } });
      const detail = c.region.stack({ gap: "md", children: slotBlock(open) });
      detail.node.classList.add("fp-card");
      return page.set([back, note ? c.banner.warn({ text: note }) : null, ...status(open.id), detail].filter(Boolean));
    }
    const grid = c.region.stack({ gap: "none", children: [...slots.filter((x) => (x.group || "Other") === group).map(nodeCard), group ? addCard() : null].filter(Boolean) });
    grid.node.classList.add("fp-node-grid");
    page.set([tools(), note ? c.banner.warn({ text: note }) : null, ...status(),
      !slots.length ? c.emptyState.default({ icon: "⬡", title: "No pipeline loaded", hint: "Is ComfyUI reachable? Or start from a model's starting point above." }) : null,
      all.length ? c.tabs.underline({ label: "Group", tabs: all.map((g) => ({ value: g, label: g })), value: group, onChange: (g) => { group = g; draw(); } }) : null,
      grid].filter(Boolean));
  }
  // A redraw while a box has focus would eat what is being typed: it waits for the focus to leave.
  const typing = () => page.node.contains(document.activeElement) && /^(input|textarea)$/i.test(document.activeElement.tagName);
  page.node.addEventListener("focusout", () => setTimeout(() => { if (!typing()) draw(); }));
  draw();
  ps.ensureLoaded().then(async () => { opened = ps.snapshot(); await describe(); draw(); });
  off = ps.subscribe(() => { if (!typing()) { describe().then(draw); } });
  return { node: page.node, destroy: () => { gone = true; off && off(); page.node.remove(); } };
};
