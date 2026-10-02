// Left zone (Assets): projects + media bin only. Shortcuts/Splits live in the Composer;
// per-clip effects & transitions live in the Properties inspector.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("media-body");

  // Every store change that touches this panel (e.g. toggling a reference mark) rebuilds
  // the whole grid from scratch — cheap for the buttons/badges, but a fresh <img>/<video>
  // per card meant every thumbnail visibly reloaded and every video re-decoded its first
  // frame, just because an unrelated R button was clicked. Cache the already-built preview
  // node per media id and re-append (move, not recreate) it into the new card instead.
  const thumbCache = new Map();   // media id -> {url, node}
  function _pruneThumbCache(bin) {
    const ids = new Set(bin.map((m) => m.id));
    for (const id of thumbCache.keys()) if (!ids.has(id)) thumbCache.delete(id);
  }

  // Drop-zone upload bar, patched in place on funpack-media-upload-progress (below) instead
  // of rebuilding — a full render only happens once per file, at the fpMediabrowser fingerprint
  // in view_bus.js's granularity (see mediaTab). Cleared whenever the zone isn't currently the
  // "uploading" one, so a stale reference from a since-discarded render never gets written to.
  let _uploadFillEl = null, _uploadPctEl = null;
  window.addEventListener("funpack-media-upload-progress", (e) => {
    if (!_uploadFillEl || !_uploadFillEl.isConnected) return;
    const up = (e && e.detail) || {};
    const pct = up.size > 0 ? Math.min(100, Math.round((up.loaded / up.size) * 100)) : null;
    _uploadFillEl.style.width = (pct == null ? 0 : pct) + "%";
    if (_uploadPctEl) _uploadPctEl.textContent = pct == null ? "uploading…" : `${pct}%`;
  });

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

  // Long edge of a cached thumbnail bitmap, in px. The grid cell itself never exceeds 240px
  // (#media-zone .media-grid's --mb-cell clamp) -- this leaves headroom for retina without
  // caching a full-resolution decode of every image/video forever, which is what a bin of
  // real photos/4K clips was actually doing before (each `<img>`/video-frame canvas held its
  // native-resolution bitmap in memory for the rest of the session, per card).
  const THUMB_MAX_DIM = 320;

  function _thumbScale(srcW, srcH, max) {
    srcW = Math.max(1, srcW); srcH = Math.max(1, srcH);
    if (srcW <= max && srcH <= max) return { w: srcW, h: srcH };
    const scale = max / Math.max(srcW, srcH);
    return { w: Math.max(1, Math.round(srcW * scale)), h: Math.max(1, Math.round(srcH * scale)) };
  }

  function _appendMediaThumb(thumb, m) {
    if (m.kind === "image" || m.kind === "video") {
      // The server now generates and caches a small JPEG per media id (an ffmpeg frame-grab
      // for video, a Pillow resize for images) instead of this loading the FULL original
      // file just to show a 56-96px square -- was true even for a 4K photo or clip. A plain
      // <img> would work now that the source is already small, but this still draws it to a
      // canvas: a <canvas> isn't natively draggable, so the card's own drag-and-drop handling
      // just works, instead of needing img.draggable=false to stop the browser's own image
      // drag from hijacking it.
      const url = API.mediaThumbUrl(m.id);
      const cached = thumbCache.get(m.id);
      if (cached && cached.url === url) { thumb.append(cached.node); return; }
      const ph = el("span", "media-icon" + (m.kind === "video" ? " media-vid-ph" : ""),
        m.kind === "video" ? "▶" : "◆");
      thumb.append(ph);
      const img = new Image();
      img.onload = () => {
        if (!thumb.isConnected) return;
        try {
          const { w, h } = _thumbScale(img.naturalWidth, img.naturalHeight, THUMB_MAX_DIM);
          const canvas = document.createElement("canvas");
          canvas.className = "media-vid-thumb";
          canvas.width = w; canvas.height = h;
          canvas.getContext("2d").drawImage(img, 0, 0, w, h);
          ph.remove();
          thumb.append(canvas);
          thumbCache.set(m.id, { url, node: canvas });
        } catch (_) {}
      };
      // On failure (no ffmpeg, corrupt source, unsupported codec) the placeholder icon
      // above just stays put -- no fallback to the full original file: that was the whole
      // cost this endpoint exists to avoid, and a broken thumbnail is not worth paying it.
      img.src = url;
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
    card.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("application/funpack-media", m.id);
      e.dataTransfer.effectAllowed = "copy";
      // Drag the card's own thumbnail as the ghost: the default is a snapshot of the whole
      // card, which for a full-resolution source image is a large rasterize on every drag.
      const th = card.querySelector(".media-thumb");
      if (th) { try { e.dataTransfer.setDragImage(th, 20, 20); } catch (_) {} }
    });
    card.title = mediaSelectMode
      ? `${m.name}\nClick to toggle selection`
      : m.kind === "image"
        ? `${m.name}\nImage · click to preview · double-click to set it as the selected clip's anchor · drag onto the timeline to add a clip`
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
    // Continuity pin, right on the gallery card — same setting as Engine settings ▸
    // Continuity ▸ Identity pin, just quicker to reach. Images only, chain-sampler
    // pipelines only (custom pipelines have no auto-continuity machinery).
    if (m.kind === "image" && st.project && window.PipelineCaps?.usesChainSampler(st)) {
      const cs = st.project.continuity_settings || {};
      const isPin = cs.identity_pin_ref === m.id;
      const pinBtn = el("button", "media-act media-pin" + (isPin ? " active" : ""), "📌");
      pinBtn.title = isPin
        ? "Continuity pin — click to unpin"
        : "Pin as identity guide for every scene (continuity pin)"
          + (cs.auto_enabled === false ? " — also turns Auto continuity back on" : "");
      pinBtn.onclick = (e) => {
        e.stopPropagation();
        const patch = { ...cs, identity_pin_ref: isPin ? null : m.id };
        // A pin with auto continuity off does nothing — pinning means "use it", so
        // re-enable (auto_enabled is on by default; only flip an explicit false).
        if (!isPin && patch.auto_enabled === false) patch.auto_enabled = true;
        S.patchProject({ continuity_settings: patch });
      };
      actions.append(pinBtn);
      if (isPin) {
        const badge = el("span", "media-pin-badge", "📌");
        badge.title = "Continuity pin — identity guide for every scene";
        thumb.append(badge);
      }
    }
    // Reference mark. Unlike the single continuity pin, any number of items can carry one,
    // and their ORDER within their own KIND is the numbering the badge shows (R1, R2, …) —
    // that order is what distinguishes two references of the same kind when wiring them to
    // node inputs, and it is what a "Reference video 1" slot resolves against. See
    // references.js for why the count is per kind rather than across every mark.
    if (st.project && m.kind !== "other") {
      const refs = st.project.references || [];
      const idx = refs.indexOf(m.id);
      const isRef = idx >= 0;
      const refN = window.References.referenceNumber(refs, st.mediaBin || [], m.id);
      const kindLabel = m.kind || "image";
      const refBtn = el("button", "media-act media-ref" + (isRef ? " active" : ""), "R");
      refBtn.title = isRef
        ? `Reference ${kindLabel} ${refN} — click to unmark`
        : "Mark as a reference — wireable into node inputs in Models & Pipeline";
      refBtn.onclick = (e) => {
        e.stopPropagation();
        S.patchProject({ references: isRef ? refs.filter((r) => r !== m.id) : [...refs, m.id] });
      };
      actions.append(refBtn);
      if (isRef) {
        const badge = el("span", "media-ref-badge", `R${refN}`);
        badge.title = `Reference ${kindLabel} ${refN} of `
          + window.References.referenceCountOfKind(refs, st.mediaBin || [], kindLabel);
        thumb.append(badge);
      }
    }
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
    // Double-click = set this image as the selected clip's i2v anchor: the same assignment
    // dragging it onto the clip performs. Drag was the ONLY way in, and a high-resolution
    // image can refuse to drag at all (the browser builds a full-size drag image first), so
    // the feature was unreachable for exactly the pictures most worth using as an anchor.
    card.ondblclick = (e) => {
      if (e.target.closest(".media-name-actions, .media-act, .media-name")) return;
      if (mediaSelectMode || m.kind !== "image") return;
      e.preventDefault(); e.stopPropagation();
      const cur = S.get();
      const sc = cur.selectedSceneId ? S.scene(cur.selectedSceneId) : null;
      if (!sc || !S.isGenerativeScene(sc)) {
        alert("Select a generated clip on the timeline first — the anchor is set on that clip.");
        return;
      }
      S.assignMediaToScene(sc.id, m.id);
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
    _pruneThumbCache(bin);
    const total = bin.length;
    const items = _sortMediaBin(_filterMediaBin(bin));
    const shown = items.length;
    const selN = mediaSelected.size;
    const exportId = _exportableMediaId(st, items);

    const wrap = el("div", "bin" + (mediaSelectMode ? " media-select-mode" : ""));
    const up = st.mediaUpload;
    const drop = el("div", "mediabin" + (up ? " uploading" : ""));
    if (up) {
      // Was silent before: uploadMedia's own loop ran with no visible state at all, so a
      // batch of photos or one big video over a slow connection just looked like the drop
      // zone had eaten the files and done nothing.
      const pct = up.size > 0 ? Math.min(100, Math.round((up.loaded / up.size) * 100)) : null;
      drop.append(el("div", "big", "⬆"));
      drop.append(el("div", null, up.total > 1
        ? `Uploading ${up.current}/${up.total}: ${up.name}`
        : `Uploading ${up.name}`));
      const barWrap = el("div", "media-upload-bar");
      const bar = el("div", "media-upload-bar-fill");
      bar.style.width = (pct == null ? 0 : pct) + "%";
      barWrap.append(bar);
      drop.append(barWrap);
      const pctEl = el("div", "pj-meta", pct == null ? "uploading…" : `${pct}%`);
      drop.append(pctEl);
      // Live-patched by the funpack-media-upload-progress listener above for every tick
      // within this file — this render only happens once per file (see fpMediabrowser).
      _uploadFillEl = bar; _uploadPctEl = pctEl;
    } else {
      _uploadFillEl = null; _uploadPctEl = null;
      drop.append(el("div", "big", "🎞"));
      drop.append(el("div", null, "Drop images, video & audio here"));
      drop.append(el("div", "pj-meta", "or click to browse · drag onto a clip to set its anchor"));
    }
    const file = el("input"); file.type = "file"; file.accept = "image/*,video/*,audio/*"; file.multiple = true; file.style.display = "none";
    file.onchange = () => { if (file.files.length) S.uploadMedia([...file.files]); file.value = ""; };
    drop.onclick = () => file.click();
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", (e) => { const fs = [...(e.dataTransfer?.files || [])]; if (fs.length) S.uploadMedia(fs); });
    wrap.append(drop); wrap.append(file);

    if (total > 0) wrap.append(_mediaFilterSortControls(st, shown, total));

    const refIds = (st.project?.references || []).filter((id) => bin.some((m) => m.id === id));
    const selRefN = mediaSelectMode ? refIds.filter((id) => mediaSelected.has(id)).length : 0;
    if ((mediaSelectMode && selN > 0) || exportId || refIds.length) {
      const actions = el("div", "media-bin-actions");
      if (mediaSelectMode && selN > 0) {
        // Marking a whole selection at once — the point of turning select mode on for
        // references. Appends in the order shown, so the R numbering is predictable.
        const unmarked = items.filter((m) => mediaSelected.has(m.id) && m.kind !== "other"
          && !refIds.includes(m.id));
        if (unmarked.length) {
          const mark = el("button", "btn ghost tiny", `Mark as reference (${unmarked.length})`);
          mark.title = "Add these to the reference list, in the order shown";
          mark.onclick = () => S.patchProject({ references: [...refIds, ...unmarked.map((m) => m.id)] });
          actions.append(mark);
        }
      }
      if (refIds.length) {
        const n = selRefN || refIds.length;
        const clear = el("button", "btn ghost tiny", `Clear ${selRefN ? "selected " : ""}references (${n})`);
        clear.title = selRefN
          ? "Remove the R mark from the selected items"
          : "Remove the R mark from every item in the bin";
        clear.onclick = () => S.patchProject({
          references: selRefN ? refIds.filter((id) => !mediaSelected.has(id)) : [],
        });
        actions.append(clear);
      }
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
