// Visual media picker — thumbnail grid instead of <select> dropdowns for image refs.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  function create(opts) {
    const filter = opts.filter || ((m) => m.kind === "image");
    const noneLabel = opts.noneLabel || "— none —";
    let selected = opts.value || null;
    let open = !!opts.startOpen;
    let onChange = opts.onChange || (() => {});

    const wrap = el("div", "media-picker" + (opts.compact ? " compact" : ""));
    const selRow = el("div", "mpk-sel");
    const thumb = el("div", "mpk-thumb");
    const meta = el("div", "mpk-meta");
    const nameEl = el("div", "mpk-name");
    const toggle = el("button", "btn ghost tiny mpk-toggle", open ? "Hide" : "Browse");
    toggle.type = "button";
    meta.append(nameEl);
    const actions = el("div", "mpk-actions");
    const clearBtn = el("button", "btn ghost tiny mpk-clear", "✕");
    clearBtn.type = "button";
    clearBtn.title = "Clear selection";
    actions.append(toggle, clearBtn);
    selRow.append(thumb, meta, actions);

    const gridWrap = el("div", "mpk-grid-wrap" + (open ? " open" : ""));
    const grid = el("div", "media-grid mpk-grid");
    gridWrap.append(grid);
    wrap.append(selRow, gridWrap);

    function items(bin) {
      return (bin || []).filter(filter);
    }

    function asset(bin, id) {
      return id ? (bin || []).find((m) => m.id === id) : null;
    }

    function pick(id) {
      selected = id || null;
      onChange(selected);
      drawSel(opts.mediaBin);
      drawGrid(opts.mediaBin);
    }

    function drawSel(bin) {
      clear(thumb);
      const a = asset(bin, selected);
      if (a && a.kind === "image") {
        const img = el("img");
        img.src = API.mediaUrl(a.id);
        img.loading = "lazy";
        thumb.append(img);
        nameEl.textContent = a.name;
        clearBtn.style.display = "";
      } else if (a) {
        thumb.append(el("span", "media-icon", a.kind === "video" ? "▶" : "◆"));
        nameEl.textContent = a.name;
        clearBtn.style.display = "";
      } else {
        thumb.append(el("span", "media-icon", "◇"));
        nameEl.textContent = noneLabel;
        clearBtn.style.display = "none";
      }
    }

    function drawGrid(bin) {
      clear(grid);
      const noneCard = el("div", "media-card mpk-none" + (!selected ? " picked" : ""));
      noneCard.append(el("div", "media-thumb", el("span", "media-icon", "∅")));
      noneCard.append(el("div", "media-name", noneLabel));
      noneCard.onclick = () => pick(null);
      grid.append(noneCard);

      items(bin).forEach((m) => {
        const card = el("div", "media-card" + (selected === m.id ? " picked" : ""));
        const t = el("div", "media-thumb");
        if (m.kind === "image") {
          const img = el("img");
          img.src = API.mediaUrl(m.id);
          img.loading = "lazy";
          t.append(img);
        } else t.append(el("span", "media-icon", m.kind === "video" ? "▶" : "◆"));
        card.append(t);
        card.append(el("div", "media-name", m.name));
        card.onclick = () => { pick(m.id); open = false; gridWrap.classList.remove("open"); toggle.textContent = "Browse"; };
        grid.append(card);
      });
      if (!items(bin).length) {
        grid.append(el("div", "pj-meta mpk-empty", "No images in the Media bin yet."));
      }
    }

    toggle.onclick = () => {
      open = !open;
      gridWrap.classList.toggle("open", open);
      toggle.textContent = open ? "Hide" : "Browse";
      if (open) drawGrid(opts.mediaBin);
    };
    clearBtn.onclick = () => pick(null);

    wrap.setValue = (v) => { selected = v || null; drawSel(opts.mediaBin); drawGrid(opts.mediaBin); };
    wrap.setMediaBin = (bin) => { opts.mediaBin = bin; drawSel(bin); if (open) drawGrid(bin); };
    wrap.setOnChange = (fn) => { onChange = fn || (() => {}); };

    drawSel(opts.mediaBin);
    if (open) drawGrid(opts.mediaBin);
    return wrap;
  }

  window.MediaPicker = { create };
})();