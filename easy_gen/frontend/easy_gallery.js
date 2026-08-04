// "Gallery" picker: choose an already-uploaded media-bin file as the i2v anchor,
// without re-uploading it. Standalone modal (same modal-overlay/.modal shell as
// project_menu.js) since Easy Gen has no Media Bin panel of its own.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;
  const S = window.Store;
  let overlay = null;

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  // Continuity pin, right on the gallery card — same setting as Engine settings ▸
  // Continuity ▸ Identity pin, just quicker to reach (mirrors mediabrowser.js's card
  // pin button in the Editor). Images only, chain-sampler pipelines only.
  const pinEls = []; // { id, btn, badge } for every image cell in the currently-open grid

  function refreshPins() {
    const cs = (S.get().project || {}).continuity_settings || {};
    pinEls.forEach(({ id, btn, badge }) => {
      const isPin = cs.identity_pin_ref === id;
      btn.classList.toggle("active", isPin);
      btn.title = isPin
        ? "Continuity pin — click to unpin"
        : "Pin as identity guide for every scene (continuity pin)"
          + (cs.auto_enabled === false ? " — also turns Auto continuity back on" : "");
      badge.hidden = !isPin;
    });
  }

  function togglePin(m) {
    const cs = (S.get().project || {}).continuity_settings || {};
    const isPin = cs.identity_pin_ref === m.id;
    const patch = { ...cs, identity_pin_ref: isPin ? null : m.id };
    if (!isPin && patch.auto_enabled === false) patch.auto_enabled = true;
    S.patchProject({ continuity_settings: patch });
    refreshPins();
  }

  // Reference marks — the same project list the Editor's Media Bin writes. Any number of
  // items can carry one and their ORDER is the numbering shown, so they can be told apart
  // when wired to node inputs in Models & Pipeline.
  const refEls = []; // { id, btn, badge } per cell in the open grid
  let clearBtn = null;

  function refs() {
    return (S.get().project || {}).references || [];
  }

  function refreshRefs() {
    const list = refs();
    refEls.forEach(({ id, btn, badge }) => {
      const idx = list.indexOf(id);
      btn.classList.toggle("active", idx >= 0);
      btn.title = idx >= 0
        ? `Reference R${idx + 1} — click to unmark`
        : "Mark as a reference — wireable into node inputs in Models & Pipeline";
      badge.textContent = idx >= 0 ? `R${idx + 1}` : "";
      badge.hidden = idx < 0;
    });
    if (clearBtn) {
      clearBtn.hidden = !list.length;
      clearBtn.textContent = `Clear references (${list.length})`;
    }
  }

  function toggleRef(m) {
    const list = refs();
    const isRef = list.includes(m.id);
    S.patchProject({ references: isRef ? list.filter((r) => r !== m.id) : [...list, m.id] });
    refreshRefs();
  }

  function cellFor(m) {
    const cell = el("div", "gal-cell");
    cell.title = m.name || m.id;
    const thumb = el("div", "gal-thumb");
    const url = API.mediaUrl(m.id);
    if (m.kind === "video") {
      const v = el("video");
      v.src = url; v.muted = true; v.preload = "metadata";
      thumb.append(v);
    } else {
      const img = el("img");
      img.src = url; img.loading = "lazy";
      thumb.append(img);
    }
    if (m.kind === "image" && window.PipelineCaps?.usesChainSampler(S.get())) {
      const btn = el("button", "gal-pin-btn", "📌");
      btn.type = "button";
      btn.onclick = (e) => { e.stopPropagation(); togglePin(m); };
      thumb.append(btn);
      const badge = el("span", "gal-pin-badge", "📌");
      badge.hidden = true;
      thumb.append(badge);
      pinEls.push({ id: m.id, btn, badge });
    }
    const refBtn = el("button", "gal-ref-btn", "R");
    refBtn.type = "button";
    refBtn.onclick = (e) => { e.stopPropagation(); toggleRef(m); };
    thumb.append(refBtn);
    const refBadge = el("span", "gal-ref-badge");
    refBadge.hidden = true;
    thumb.append(refBadge);
    refEls.push({ id: m.id, btn: refBtn, badge: refBadge });
    cell.append(thumb);
    cell.append(el("div", "gal-name", m.name || m.id));
    cell.onclick = async () => {
      S.setSceneMedia(m.id, m.kind === "video" ? "video" : "image");
      await S.save();
      close();
    };
    return cell;
  }

  async function open() {
    if (!S.get().project) return;
    close();
    pinEls.length = 0;
    refEls.length = 0;
    overlay = el("div", "modal-overlay");
    const box = el("div", "modal modal-wide");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Gallery — choose an anchor"));
    clearBtn = el("button", "btn ghost tiny", "Clear references (0)");
    clearBtn.type = "button";
    clearBtn.title = "Remove the R mark from every item";
    clearBtn.hidden = true;
    clearBtn.onclick = () => { S.patchProject({ references: [] }); refreshRefs(); };
    const x = el("button", "btn ghost tiny", "✕");
    x.onclick = close;
    head.append(el("div", "modal-head-right"), clearBtn, x);
    box.append(head);

    const content = el("div", "modal-content gal-grid");
    content.append(el("div", "sw-hint", "Loading…"));
    box.append(content);

    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
    });
    document.body.append(overlay);

    try {
      const r = await API.listMedia();
      const items = (r.media || []).filter((m) => m.kind === "image" || m.kind === "video");
      clear(content);
      if (!items.length) {
        content.append(el("div", "sw-hint", "No uploaded media yet — use Upload first."));
        return;
      }
      items.forEach((m) => content.append(cellFor(m)));
      refreshPins();
      refreshRefs();
    } catch (e) {
      clear(content);
      content.append(el("div", "sw-hint", "Could not load media: " + (e.message || e)));
    }
  }

  window.EasyGallery = { open, close };
})();
