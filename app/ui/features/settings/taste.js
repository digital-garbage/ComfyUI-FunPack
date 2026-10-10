// Refinement & Taste: the refinement keys FunPack has learned from your ratings. Keys are files: carry one to another machine
// as a zip, or delete one to start that taste over.
import { composer as c } from "../../composer/composer.js";

export const taste = (app) => function mount() {
  const page = c.region.stack({ gap: "md" });
  const tell = (text) => c.toast.warn({ text });
  let keys = null, error = "";

  async function refresh() {
    try { keys = (await app.api.tasteKeys()).keys || []; error = ""; } catch (err) { keys = []; error = err.message; }
    draw();
  }
  const remove = async (name) => {
    if (!(await c.modal.dialogue({ title: "Delete refinement key", message: `Delete “${name}” and everything it learned? This cannot be undone.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
    try { await app.api.deleteTasteKey(name); } catch (err) { tell(err.message); }
    refresh();
  };
  const exportKey = async (name) => {         // fetched first, so a refusal is said instead of navigating the page to its JSON
    try {
      const r = await fetch(app.api.tasteKeyUrl(name));
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).why || `The server refused (${r.status}).`);
      const link = Object.assign(document.createElement("a"), { download: `${name}.zip`, href: URL.createObjectURL(await r.blob()) });
      link.click(); URL.revokeObjectURL(link.href);
    } catch (err) { tell(err.message); }
  };
  async function bring([file]) {
    const name = await c.modal.prompt({ title: "Import refinement key", label: "Name", value: file.name.replace(/\.zip$/i, ""), confirmLabel: "Import" }).result;
    if (!name || !name.trim()) return;
    try {
      let r = await app.api.importTasteKey(name.trim(), file, false);
      if (r.exists && await c.modal.dialogue({ title: "Replace key", message: `A key named “${name.trim()}” already exists. Replace it?`, tone: "danger", confirmLabel: "Replace" }).result) r = await app.api.importTasteKey(name.trim(), file, true);
      if (!r.exists) c.toast.good({ text: `Imported “${name.trim()}”.` });
    } catch (err) { tell(err.message); }
    refresh();
  }
  function draw() {
    page.set([
      error ? c.banner.warn({ text: `Could not read the keys: ${error}` }) : null,
      keys && !keys.length && !error ? c.emptyState.default({ icon: "✦", title: "No keys yet", hint: "A key appears once you rate a render with learning on." }) : null,
      ...(keys || []).map((name) => c.settingsRow.default({ label: name, hint: "Refinement key", control: c.toolbar.default({ items: [
        c.button.sm({ label: "⤓ Export", tone: "ghost", onClick: () => exportKey(name) }),
        c.button.sm({ label: "Delete", tone: "danger", onClick: () => remove(name) })] }) })),
      c.label.section({ text: "Bring a key here" }),
      c.dropzone.default({ label: "Drop or choose an exported key (.zip)", hint: "from another machine", accept: ".zip,application/zip", multiple: false, onFiles: bring }),
    ].filter(Boolean));
  }
  const exportInfluence = async (key) => {
    try {
      const r = await fetch(app.api.blockInfluenceExportUrl(key));
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).why || `The server refused (${r.status}).`);
      const link = Object.assign(document.createElement("a"), { download: `${key}.block_influence.pt`, href: URL.createObjectURL(await r.blob()) });
      link.click(); URL.revokeObjectURL(link.href);
    } catch (err) { tell(err.message); }
  };
  const clearInfluence = async (key) => {
    if (!(await c.modal.dialogue({ title: "Clear block influence", message: `Clear everything recorded for “${key}”? This cannot be undone.`, tone: "danger", confirmLabel: "Clear" }).result)) return;
    try { await app.api.clearBlockInfluence(key); } catch (err) { tell(err.message); }
    refreshResearch();
  };
  // Block influence: research recording, read here. Its numbers say whether the four groups point apart; no clip changes.
  const research = c.region.stack({ gap: "sm" });
  let influence = null;
  const pct = (x) => (x >= 0 ? "+" : "") + x.toFixed(2);
  async function refreshResearch() {
    try {
      // Separately: a failed report must not hide the switch, or the user cannot turn recording off.
      const state = await app.api.blockInfluence("default").catch((err) => ({ error: err.message }));
      const four = await app.api.blockInfluenceGroups("default").catch((err) => ({ error: err.message }));
      influence = { state: state.error ? null : state, four: four.error ? null : four, error: [state.error, four.error].filter(Boolean).join("; ") };
    } catch (err) { influence = { state: null, four: null, error: err.message }; }
    drawResearch();
  }
  function drawResearch() {
    const rows = [c.label.section({ text: "Block influence (research)" })];
    if (influence && influence.error) rows.push(c.banner.warn({ text: `Could not read the recording: ${influence.error}` }));
    // The switch is always here, even when the report could not be read: a failed read must not take away the way to turn recording off.
    rows.push(c.toggle.default({ label: "Record block influence", hint: "On: each run on a Taste key is measured, and kept once you rate it. Off: nothing is recorded.",
      checked: influence && influence.state ? !!influence.state.enabled : true,
      onChange: async (on) => { try { await app.api.setBlockInfluence(on); } catch (err) { tell(err.message); } refreshResearch(); } }));
    if (influence && influence.state) {
      const { state, four } = influence;
      if (state.problem) rows.push(c.banner.warn({ text: state.problem }));
      rows.push(c.settingsRow.default({ label: "Rated runs kept", hint: `Runs on this key that have a rating: ${state.runs}, of which ${state.skipped} had no usable picture rows`,
        control: c.hint.default({ text: `${state.used} kept` }) }));
      if (state.used) rows.push(c.settingsRow.default({ label: "Flatness", hint: "How evenly the blocks move the picture. Near 0 means every block moves it equally, so there is nothing to aim at.",
        control: c.hint.default({ text: state.flatness == null ? "n/a" : state.flatness.toFixed(3) }) }));
      if (state.used && state.mean_novelty != null) rows.push(c.settingsRow.default({ label: "Novelty", hint: "Near 1: blocks amplify what came before. Near 0: they add something new.",
        control: c.hint.default({ text: state.mean_novelty.toFixed(3) }) }));
      rows.push(c.settingsRow.default({ label: "Recording data", hint: "Download the stored data for this key, or clear it.", control: c.toolbar.default({ items: [
        c.button.sm({ label: "⤓ Download", tone: "ghost", onClick: () => exportInfluence(state.key) }),
        c.button.sm({ label: "Clear", tone: "danger", onClick: () => clearInfluence(state.key) })] }) }));
      if (four) rows.push(c.settingsRow.default({ label: "Clips rated", hint: `Key: ${four.key || state.key || "none yet"}`,
        control: c.hint.default({ text: four.used ? Object.entries(four.counts || {}).map(([g, n]) => `${g} ${n}`).join(", ") : "none yet" }) }));
      if (four && four.note) rows.push(c.hint.default({ text: four.note }));
      if (four && four.used && Object.values(four.counts || {}).some((n) => n < 5))
        rows.push(c.hint.default({ text: "Few clips in a group: a difference this size can be chance. Rate more before reading it." }));
      for (const [pair, cos] of Object.entries((four && four.cos) || {})) {
        const c0 = four.chance[pair];
        rows.push(c.settingsRow.default({ label: pair, hint: `Chance from shuffled labels: mean ${pct(c0.mean)}, 95th ${pct(c0.p95)}`,
          control: c.hint.default({ text: pct(cos) }) }));
      }
    }
    research.set(rows);
  }
  draw(); refresh(); refreshResearch();
  const node = document.createElement("div");
  node.append(page.node, research.node);
  return { node, destroy: () => node.remove() };
};
