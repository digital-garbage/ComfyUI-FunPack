// Composer > Enhance and Composer > Chat: the prompt enhancer's settings, the
// last run's rewrite, and the comments that steer the next one.
//
// The settings are the INPUTS of the pipeline's FunPackEnhancePrompt node, edited
// through the same PipelineState every other panel uses -- there is no second copy.
// The Chat comments are the project's own (project.editor_settings.enhance_chat) and
// reach the node at Generate (store.js), so they travel with the project.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const PS = window.PipelineState;
  const M = "/funpack/api/m/conditioning_prompt_enhancer";
  const NODE = "FunPackEnhancePrompt";

  let defaults = null;       // what an empty instructions / intro field means
  let last = null;           // the newest run the server remembers
  let rerender = () => {};

  async function get(path) {
    const res = await fetch(M + path);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
  const slot = () => (PS.slots() || []).find((s) => s.node === NODE) || null;
  const set = (name, value) => {
    const sl = slot(); if (!sl) return;
    sl.inputs = { ...(sl.inputs || {}), [name]: value };
    return PS.save({ inputs: { [sl.id]: { [name]: value } } });
  };

  // What a run remembers, newest last. Fetched whenever a tab opens, and again after a run.
  async function refreshLast() {
    try { const runs = (await get("/runs")).runs || []; last = runs[runs.length - 1] || null; }
    catch (_) { /* no module (an older pack): nothing to show */ }
  }
  let lastState = "";
  window.addEventListener("funpack-gen-progress", (e) => {
    const now = (e.detail && e.detail.state) || "";
    const finished = ["queuing", "running"].includes(lastState) && !["queuing", "running"].includes(now);
    lastState = now;
    if (finished) refreshLast().then(() => rerender());
  });

  function field(label, ctrl, hint) {
    const l = el("label", "lib-field"); l.append(el("span", null, label)); l.append(ctrl);
    if (hint) l.append(el("div", "insp-hint", hint));
    return l;
  }
  function check(label, value, onChange, hint) {
    const row = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = !!value;
    cb.onchange = () => onChange(cb.checked);
    row.append(cb, el("span", null, label));
    if (hint) row.title = hint;
    return row;
  }
  function number(value, min, max, step, onChange) {
    const i = el("input", "lib-in"); i.type = "number"; i.min = min; i.max = max; i.step = step; i.value = value;
    i.onchange = () => { const v = Number(i.value); if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v))); };
    return i;
  }
  function text(value, rows, placeholder, onChange) {
    const t = el("textarea", "lib-in"); t.rows = rows; t.value = value || ""; t.placeholder = placeholder || "";
    t.onchange = () => onChange(t.value);
    return t;
  }

  function readout(wrap) {
    const box = el("div", "enh-readout");
    box.append(el("div", "lib-form-title", "Last run"));
    if (!last) { box.append(el("div", "insp-hint", "Nothing yet. The rewrite of a run shows here.")); wrap.append(box); return; }
    box.append(el("div", "insp-hint", last.status || ""));
    const part = (title, body) => {
      if (!body) return;
      const d = el("details", "pv-raw"); d.append(el("summary", null, title)); d.append(el("pre", null, body)); box.append(d);
    };
    part("Before", last.before);
    const after = el("div", "enh-after"); after.append(el("b", null, "After: "), el("span", null, last.after || "")); box.append(after);
    part("Model's thinking", last.thinking);
    part("Sent to the model (chat run)", last.sent);
    wrap.append(box);
  }

  function enhanceTab(host) {
    rerender = host;
    const wrap = el("div", "bin enh-bin");
    const sl = slot();
    if (!sl) {
      wrap.append(el("div", "insp-hint",
        "The open pipeline has no prompt enhancer. Pick a preset that has one (MiniMax H3 · Reference to Video), "
        + "or add the \"FunPack Enhance Prompt\" node in Models & Pipeline."));
      return wrap;
    }
    const v = sl.inputs || {};
    const val = (name, dflt) => (v[name] === undefined ? dflt : v[name]);
    wrap.append(el("div", "lib-form-title", "Prompt enhancer"));
    wrap.append(el("div", "insp-hint",
      "A language model rewrites the finished prompt before it is encoded. Off, or if it fails, your prompt is used as typed. "
      + "(word:1.5) and [a@2-4] markup does not survive a rewrite."));
    wrap.append(check("Enhance prompt", val("enabled", false), (c) => set("enabled", c).then(host)));
    if (!val("enabled", false)) { readout(wrap); return wrap; }

    wrap.append(field("Instructions", text(val("instructions", ""), 6,
      defaults ? defaults.instructions : "(built-in instructions)", (t) => set("instructions", t)),
      "Empty uses the built-in instructions shown faintly here."));
    wrap.append(field("Max tokens", number(val("max_length", 400), 32, 4096, 8, (n) => set("max_length", n))));
    wrap.append(check("Use the picture", val("use_image", true), (c) => set("use_image", c),
      "Sends the scene's picture to the model along with the prompt."));
    wrap.append(check("Thinking", val("thinking", false), (c) => set("thinking", c),
      "Let a thinking model reason first. Its reasoning eats the token budget."));
    wrap.append(check("Greedy (always the likeliest word)", val("greedy", false), (c) => set("greedy", c).then(host)));
    if (!val("greedy", false)) {
      const dials = [["temperature", 0.7, 0, 2, 0.05], ["top_p", 0.92, 0, 1, 0.01], ["top_k", 50, 1, 1000, 1],
        ["min_p", 0.05, 0, 1, 0.01], ["repetition_penalty", 1.3, 1, 2, 0.05], ["presence_penalty", 0, 0, 2, 0.05]];
      dials.forEach(([n, d, lo, hi, st]) => wrap.append(field(n, number(val(n, d), lo, hi, st, (x) => set(n, x)))));
    }
    wrap.append(field("Seed", number(val("seed", 0), 0, 2 ** 53, 1, (n) => set("seed", n)),
      "0 = a fresh rewrite every Generate. Any other number repeats the same rewrite."));

    // Reference: shortcuts picked from the library, and lorebook / shortcut files by path.
    wrap.append(el("div", "lib-form-title", "Reference"));
    wrap.append(el("div", "insp-hint",
      "Handed to the model after your prompt: the entries your prompt mentions, or all of a source when it mentions none."));
    const picked = new Set(String(val("shortcuts", "")).split("\n").map((s) => s.trim()).filter(Boolean));
    const list = el("div", "enh-shortcuts");
    (S.get().shortcuts || []).forEach((sc) => {
      list.append(check(sc.name, picked.has(sc.name), (c) => {
        c ? picked.add(sc.name) : picked.delete(sc.name);
        set("shortcuts", [...picked].join("\n"));
      }));
    });
    if (!(S.get().shortcuts || []).length) list.append(el("div", "insp-hint", "No shortcuts in the library."));
    wrap.append(list);
    wrap.append(field("Lorebook / shortcut files", text(val("lorebooks", ""), 2, "one path per line",
      (t) => set("lorebooks", t)), "SillyTavern lorebooks or FunPack shortcut files, on the machine running ComfyUI."));
    wrap.append(field("Reference intro", text(val("reference_intro", ""), 2,
      defaults ? defaults.reference_intro : "", (t) => set("reference_intro", t))));
    readout(wrap);
    return wrap;
  }

  // ── chat ────────────────────────────────────────────────────────────────────
  const rounds = () => {
    const p = S.get().project;
    const r = p && p.editor_settings && p.editor_settings.enhance_chat;
    return Array.isArray(r) ? r : [];
  };
  // Written onto the project directly: setEditorSetting also keeps a per-browser copy,
  // which a NEW project would then inherit.
  const saveRounds = (list) => {
    const p = S.get().project;
    if (p) S.patchProjectQuiet({ editor_settings: { ...(p.editor_settings || {}), enhance_chat: list } });
  };

  function chatTab(host) {
    rerender = host;
    const wrap = el("div", "bin enh-bin");
    wrap.append(el("div", "lib-form-title", "Chat"));
    wrap.append(el("div", "insp-hint",
      "Comment on the last rewrite. The next Generate rewrites again with every comment still applying. Reset starts over."));
    if (!S.get().project) { wrap.append(el("div", "insp-hint", "Open a project first.")); return wrap; }
    const log = el("div", "enh-chat");
    const list = rounds();
    list.forEach((r) => {
      const rw = Object.values(r.rewrites || {})[0];
      if (rw) log.append(el("div", "enh-bubble enh-model", rw));
      log.append(el("div", "enh-bubble enh-user", r.comment));
    });
    if (last && last.ok) {
      log.append(el("div", "insp-hint", "Last rewrite of: " + String(last.before || "").slice(0, 80)));
      log.append(el("div", "enh-bubble enh-model enh-last", last.after));
    }
    if (!list.length && !(last && last.ok)) log.append(el("div", "insp-hint", "Generate once with the enhancer on, then comment here."));
    wrap.append(log);

    const box = el("textarea", "lib-in"); box.rows = 2; box.placeholder = "e.g. warmer light, no rain";
    const send = el("button", "btn primary tiny", "Send");
    send.onclick = () => {
      const comment = box.value.trim();
      if (!comment) return;
      // The rewrite this comment is about; {} when nothing new ran since the last comment.
      const prev = list.length ? Object.values(list[list.length - 1].rewrites || {})[0] : null;
      const fresh = last && last.ok && last.after !== prev;
      const rewrites = fresh ? { whole: last.after } : {};
      // The prompt the rewrite answered: comments only apply to that same prompt.
      const tail = list.length ? list[list.length - 1].original : undefined;
      const original = fresh ? last.before : tail;
      saveRounds([...list, original === undefined ? { rewrites, comment } : { rewrites, comment, original }]);
      host();
    };
    const reset = el("button", "btn ghost tiny", "Reset");
    reset.disabled = !list.length;
    reset.onclick = () => { saveRounds([]); host(); };
    const row = el("div", "lib-form-actions"); row.append(send, reset);
    wrap.append(box, row);
    return wrap;
  }

  async function ready() {
    try { await PS.ensureLoaded(); } catch (_) { /* the tab says there is no enhancer */ }
    if (!defaults) { try { defaults = await get("/defaults"); } catch (_) { defaults = null; } }
    await refreshLast();
  }

  window.EnhancerUI = { enhanceTab, chatTab, ready, rounds };
})();
