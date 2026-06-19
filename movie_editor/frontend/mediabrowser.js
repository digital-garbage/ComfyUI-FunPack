// Left zone (Assets): projects + media bin only. Characters/Shortcuts/Splits live in the Composer;
// per-clip effects & transitions live in the Properties inspector.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("media-body");

  const mediaSelected = new Set();   // media bin multi-select
  let mediaSelectMode = false;     // when off, click = preview; when on, click = toggle selection
  const MF_KEY = "fp_media_filter";
  const MS_KEY = "fp_media_sort";
  const MC_KEY = "fp_media_grid_cols";
  let mediaFilter = localStorage.getItem(MF_KEY) || "all";
  let mediaSort = localStorage.getItem(MS_KEY) || "name_asc";
  let mediaGridCols = parseInt(localStorage.getItem(MC_KEY) || "0", 10);
  if (![0, 1, 2, 3, 4].includes(mediaGridCols)) mediaGridCols = 0;

  const MEDIA_FILTERS = [
    { id: "all", label: "All" },
    { id: "video", label: "Video" },
    { id: "audio", label: "Audio" },
    { id: "image", label: "Images" },
  ];
  const MEDIA_SORTS = [
    { id: "name_asc", label: "Name A-Z" },
    { id: "name_desc", label: "Name Z-A" },
    { id: "type", label: "Type" },
    { id: "date_desc", label: "Date added" },
  ];
  const MEDIA_GRID_COLS = [
    { id: 0, label: "Auto" },
    { id: 1, label: "1×" },
    { id: 2, label: "2×" },
    { id: 3, label: "3×" },
    { id: 4, label: "4×" },
  ];
  const MEDIA_KIND_ORDER = { image: 0, video: 1, audio: 2, other: 3 };
  const MEDIA_GROUP_LABELS = { image: "Images", video: "Video", audio: "Audio", other: "Other" };

  // ── projects ───────────────────────────────────────────────────────────────────
  function projectsSection(st) {
    const sec = el("div", "mb-section");
    const head = el("div", "mb-section-title");
    head.append(el("span", null, "Projects"));
    const add = el("button", "btn ghost tiny", "＋ New");
    add.onclick = () => {
      if (window.SlotPicker?.openPrompt) {
        window.SlotPicker.openPrompt({
          title: "New project",
          value: "Untitled montage",
          placeholder: "Project name",
          onPick: (name) => S.newProject(name),
        });
      } else {
        S.newProject(prompt("Project name:", "Untitled montage"));
      }
    };
    head.append(add);
    sec.append(head);
    (st.projects || []).forEach((p) => {
      const row = el("div", "project-row" + (st.project && st.project.id === p.id ? " active" : ""));
      row.onclick = () => S.loadProject(p.id);
      row.append(el("span", "pj-name", p.name));
      row.append(el("span", "pj-meta", `${p.scene_count}▦`));
      sec.append(row);
    });
    if (!(st.projects || []).length) sec.append(el("div", "pj-meta", "No projects yet."));
    return sec;
  }

  // ── media bin ──────────────────────────────────────────────────────────────────
  const MEDIA_KIND_META = {
    image: { icon: "◐", label: "Image", short: "IMG", cls: "kind-image" },
    video: { icon: "▶", label: "Video", short: "VID", cls: "kind-video" },
    audio: { icon: "♪", label: "Audio", short: "AUD", cls: "kind-audio" },
    other: { icon: "◆", label: "File", short: "FILE", cls: "kind-other" },
  };

  function _mediaKindMeta(kind) {
    return MEDIA_KIND_META[kind] || MEDIA_KIND_META.other;
  }

  function _appendMediaThumb(thumb, m) {
    const url = API.mediaUrl(m.id);
    if (m.kind === "image") {
      const img = el("img");
      img.src = url;
      img.loading = "lazy";
      thumb.append(img);
      return;
    }
    if (m.kind === "video") {
      const ph = el("span", "media-icon media-vid-ph", "▶");
      thumb.append(ph);
      const vid = document.createElement("video");
      vid.className = "media-vid-thumb";
      vid.muted = true;
      vid.preload = "metadata";
      vid.playsInline = true;
      vid.src = url;
      vid.onloadeddata = () => {
        if (!thumb.isConnected) return;
        ph.remove();
        if (!thumb.querySelector("video")) thumb.append(vid);
      };
      return;
    }
    if (m.kind === "audio") {
      thumb.append(el("span", "media-icon media-aud-ph", "♪"));
      return;
    }
    thumb.append(el("span", "media-icon", "◆"));
  }

  function _appendMediaKindBadge(thumb, m) {
    const meta = _mediaKindMeta(m.kind);
    const badge = el("span", "media-kind-badge " + meta.cls);
    badge.title = meta.label;
    badge.append(el("span", "media-kind-ico", meta.icon), el("span", "media-kind-lbl", meta.short));
    thumb.append(badge);
  }

  function _pruneMediaSelection(bin) {
    const ids = new Set((bin || []).map((m) => m.id));
    for (const id of mediaSelected) {
      if (!ids.has(id)) mediaSelected.delete(id);
    }
  }

  function _setMediaSelectMode(on) {
    mediaSelectMode = !!on;
    if (!mediaSelectMode) mediaSelected.clear();
  }

  function _filterMediaBin(bin) {
    if (mediaFilter === "all") return bin;
    if (mediaFilter === "image") return bin.filter((m) => m.kind === "image");
    if (mediaFilter === "video") return bin.filter((m) => m.kind === "video");
    if (mediaFilter === "audio") return bin.filter((m) => m.kind === "audio");
    return bin;
  }

  function _sortMediaBin(items) {
    const arr = [...items];
    const byName = (a, b) => (a.name || "").localeCompare(b.name || "", undefined, { sensitivity: "base" });
    switch (mediaSort) {
      case "name_desc":
        return arr.sort((a, b) => byName(b, a));
      case "type":
        return arr.sort((a, b) => {
          const ka = MEDIA_KIND_ORDER[a.kind] ?? 99;
          const kb = MEDIA_KIND_ORDER[b.kind] ?? 99;
          if (ka !== kb) return ka - kb;
          return byName(a, b);
        });
      case "date_desc":
        return arr.sort((a, b) => (b.added || 0) - (a.added || 0));
      default:
        return arr.sort(byName);
    }
  }

  function _exportableMediaId(st, items) {
    let id = null;
    if (mediaSelectMode && mediaSelected.size === 1) id = [...mediaSelected][0];
    else if (st.mediaPreviewId) id = st.mediaPreviewId;
    if (!id) return null;
    const m = items.find((x) => x.id === id);
    return m && (m.kind === "image" || m.kind === "video") ? id : null;
  }

  function _mediaFilterSortControls(st, shown, total) {
    const controls = el("div", "media-bin-controls");
    const typeRow = el("div", "insp-switch media-bin-type");
    MEDIA_FILTERS.forEach((f) => {
      const btn = el("button", "insp-seg" + (mediaFilter === f.id ? " active" : ""), f.label);
      btn.onclick = () => {
        mediaFilter = f.id;
        localStorage.setItem(MF_KEY, mediaFilter);
        render(S.get());
      };
      typeRow.append(btn);
    });
    controls.append(typeRow);

    const sortRow = el("div", "media-bin-sort");
    sortRow.append(el("span", "media-bin-sort-lbl", "Sort by"));
    const sortSel = el("select", "lib-in media-bin-sort-sel");
    MEDIA_SORTS.forEach((s) => {
      const op = el("option", null, s.label);
      op.value = s.id;
      if (mediaSort === s.id) op.selected = true;
      sortSel.append(op);
    });
    sortSel.onchange = () => {
      mediaSort = sortSel.value;
      localStorage.setItem(MS_KEY, mediaSort);
      render(S.get());
    };
    sortRow.append(sortSel);
    controls.append(sortRow);

    const colsRow = el("div", "media-bin-cols");
    colsRow.append(el("span", "media-bin-sort-lbl", "Grid"));
    const colsSeg = el("div", "insp-switch media-bin-cols-switch");
    MEDIA_GRID_COLS.forEach((c) => {
      const btn = el("button", "insp-seg" + (mediaGridCols === c.id ? " active" : ""), c.label);
      btn.title = c.id === 0
        ? "Column count follows panel width"
        : `${c.id} column${c.id > 1 ? "s" : ""} (× rows)`;
      btn.onclick = () => {
        mediaGridCols = c.id;
        localStorage.setItem(MC_KEY, String(c.id));
        render(S.get());
      };
      colsSeg.append(btn);
    });
    colsRow.append(colsSeg);
    controls.append(colsRow);

    if (total > 0 && shown !== total) {
      controls.append(el("div", "pj-meta media-bin-count", `${shown} of ${total} shown`));
    }
    return controls;
  }

  function _beginMediaRename(m, nameEl) {
    const inp = el("input", "media-name-edit");
    inp.type = "text";
    inp.value = m.name || "";
    nameEl.replaceWith(inp);
    inp.focus();
    inp.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      if (!save) { render(S.get()); return; }
      const next = inp.value.trim();
      if (!next) { alert("Name cannot be empty."); render(S.get()); return; }
      if (next === m.name) { render(S.get()); return; }
      await S.renameMedia(m.id, next);
      render(S.get());
    };
    inp.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      if (e.key === "Escape") { e.preventDefault(); finish(false); }
    };
    inp.onclick = (e) => e.stopPropagation();
    inp.onblur = () => { finish(true); };
  }

  function _mediaCard(m, st) {
    const picked = mediaSelectMode && mediaSelected.has(m.id);
    const onTimeline = m.kind === "video" && (st.project?.scenes || []).some(
      (s) => !s.excluded && S.isVideoClip?.(s) && s.source?.media_ref === m.id,
    );
    const selScene = st.selectedSceneId ? S.scene(st.selectedSceneId) : null;
    const previewing = st.mediaPreviewId === m.id
      || (onTimeline && selScene?.source?.media_ref === m.id);
    const card = el("div", "media-card"
      + (previewing ? " previewing" : "")
      + (picked ? " picked" : ""));
    card.draggable = true;
    card.addEventListener("dragstart", (e) => { e.dataTransfer.setData("application/funpack-media", m.id); e.dataTransfer.effectAllowed = "copy"; });
    card.title = mediaSelectMode
      ? `${m.name}\nClick to toggle selection`
      : m.kind === "image"
        ? `${m.name}\nImage · click to preview · drag onto timeline to add a clip, or onto a clip to set its anchor`
        : m.kind === "audio"
          ? `${m.name}\nAudio · click to preview · add via timeline + Add → Audio`
          : m.kind === "video"
            ? `${m.name}\nVideo · click to preview · drag onto timeline or + Add → Video`
            : `${m.name}\nDrag onto a clip to set anchor`;
    const thumb = el("div", "media-thumb");
    _appendMediaThumb(thumb, m);
    _appendMediaKindBadge(thumb, m);
    card.append(thumb);
    const nameRow = el("div", "media-name-row");
    const nameEl = el("div", "media-name", m.name);
    nameEl.title = "Double-click to rename";
    nameEl.ondblclick = (e) => { e.stopPropagation(); _beginMediaRename(m, nameEl); };
    nameRow.append(nameEl);
    const actions = el("div", "media-name-actions");
    const ren = el("button", "media-act media-ren", "✎");
    ren.title = "Rename";
    ren.onclick = (e) => {
      e.stopPropagation();
      const cur = card.querySelector(".media-name");
      if (cur) _beginMediaRename(m, cur);
    };
    actions.append(ren);
    if (m.kind === "image" || m.kind === "video") {
      const exp = el("button", "media-act media-exp", "⤓");
      exp.title = "Export to disk";
      exp.onclick = (e) => { e.stopPropagation(); S.exportMediaAsset(m.id); };
      actions.append(exp);
    }
    const del = el("button", "media-act media-del", "✕");
    del.title = "Delete asset";
    del.onclick = (e) => { e.stopPropagation(); if (confirm(`Delete "${m.name}"?`)) S.deleteMedia(m.id); };
    actions.append(del);
    nameRow.append(actions);
    card.append(nameRow);
    card.onclick = (e) => {
      if (e.target.closest(".media-name-actions, .media-act, .media-name input")) return;
      if (mediaSelectMode) {
        if (picked) mediaSelected.delete(m.id);
        else mediaSelected.add(m.id);
        render(S.get());
        return;
      }
      if (m.kind === "image" || m.kind === "audio" || m.kind === "video") {
        S.previewMedia(m.id);
      }
    };
    return card;
  }

  function _appendMediaGrid(grid, items, st) {
    const showGroups = mediaSort === "type" && mediaFilter === "all";
    if (!showGroups) {
      items.forEach((m) => grid.append(_mediaCard(m, st)));
      return;
    }
    let lastKind = null;
    items.forEach((m) => {
      if (m.kind !== lastKind) {
        grid.append(el("div", "media-group-hdr", MEDIA_GROUP_LABELS[m.kind] || MEDIA_GROUP_LABELS.other));
        lastKind = m.kind;
      }
      grid.append(_mediaCard(m, st));
    });
  }

  function mediaTab(st) {
    const bin = st.mediaBin || [];
    _pruneMediaSelection(bin);
    const total = bin.length;
    const items = _sortMediaBin(_filterMediaBin(bin));
    const shown = items.length;
    const selN = mediaSelected.size;
    const exportId = _exportableMediaId(st, items);

    const wrap = el("div", "bin" + (mediaSelectMode ? " media-select-mode" : ""));
    const drop = el("div", "mediabin");
    drop.append(el("div", "big", "🎞"));
    drop.append(el("div", null, "Drop images, video & audio here"));
    drop.append(el("div", "pj-meta", "or click to browse · drag onto a clip to set its anchor"));
    const file = el("input"); file.type = "file"; file.accept = "image/*,video/*,audio/*"; file.multiple = true; file.style.display = "none";
    file.onchange = () => { if (file.files.length) S.uploadMedia([...file.files]); file.value = ""; };
    drop.onclick = () => file.click();
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", (e) => { const fs = [...(e.dataTransfer?.files || [])]; if (fs.length) S.uploadMedia(fs); });
    wrap.append(drop); wrap.append(file);

    if (total > 0) wrap.append(_mediaFilterSortControls(st, shown, total));

    if ((mediaSelectMode && selN > 0) || exportId) {
      const actions = el("div", "media-bin-actions");
      if (exportId) {
        const expBtn = el("button", "btn ghost tiny", "⤓ Export");
        expBtn.title = "Save this image or video to your computer";
        expBtn.onclick = () => S.exportMediaAsset(exportId);
        actions.append(expBtn);
      }
      if (mediaSelectMode && selN > 0) {
        const rem = el("button", "btn ghost tiny danger", `Remove selected (${selN})`);
        rem.onclick = async () => {
          if (!confirm(`Delete ${selN} selected item${selN > 1 ? "s" : ""}?`)) return;
          await S.deleteMediaMany([...mediaSelected]);
          mediaSelected.clear();
          render(S.get());
        };
        actions.append(rem);
      }
      wrap.append(actions);
    }

    const grid = el("div", "media-grid");
    grid.dataset.cols = String(mediaGridCols);
    if (items.length) _appendMediaGrid(grid, items, st);
    else if (total) grid.append(el("div", "pj-meta media-grid-empty", "No items match this filter."));
    else grid.append(el("div", "pj-meta media-grid-empty", "No media yet."));
    wrap.append(grid);

    if (total > 0) {
      const footer = el("div", "media-bin-footer");
      const selectBtn = el("button", "btn ghost tiny media-select-btn" + (mediaSelectMode ? " active" : ""), "Select");
      selectBtn.title = mediaSelectMode
        ? "Selection mode on — click assets to toggle, then Remove selected"
        : "Turn on to select multiple assets for bulk remove";
      selectBtn.onclick = () => {
        _setMediaSelectMode(!mediaSelectMode);
        render(S.get());
      };
      footer.append(selectBtn);
      if (mediaSelectMode) {
        const hasSelection = selN > 0;
        const bulk = el("button", "btn ghost tiny", hasSelection ? `Deselect all (${selN})` : `Select all (${shown})`);
        bulk.onclick = () => {
          if (hasSelection) mediaSelected.clear();
          else items.forEach((m) => mediaSelected.add(m.id));
          render(S.get());
        };
        footer.append(bulk);
      }
      wrap.append(footer);
    }
    return wrap;
  }

  function render(st) {
    clear(body);
    body.append(projectsSection(st));
    const sec = el("div", "mb-section mb-bin-shell");
    const scroll = el("div", "mb-bin-scroll");
    scroll.append(mediaTab(st));
    sec.append(scroll);
    body.append(sec);
  }

  if (window.ViewBus) window.ViewBus.subscribeMediabrowser(render);
  else S.subscribe(render);
})();
