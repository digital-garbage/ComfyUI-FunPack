// Left zone: projects + tabbed library bins — Media, Characters, Shortcuts, Split markers, Effects, Transitions.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("media-body");

  let tab = "Media";                 // active bin
  let q = { Characters: "", Shortcuts: "", Splits: "", Effects: "", Transitions: "" };
  let editCharacter = null;          // character item being edited ({} = new), or null
  let _libModal = null;
  const mediaSelected = new Set();   // media bin multi-select
  let mediaSelectMode = false;     // when off, click = preview; when on, click = toggle selection
  const MF_KEY = "fp_media_filter";
  const MS_KEY = "fp_media_sort";
  let mediaFilter = localStorage.getItem(MF_KEY) || "all";
  let mediaSort = localStorage.getItem(MS_KEY) || "name_asc";

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
  const MEDIA_KIND_ORDER = { image: 0, video: 1, audio: 2, other: 3 };
  const MEDIA_GROUP_LABELS = { image: "Images", video: "Video", audio: "Audio", other: "Other" };

  // ── library edit helpers ───────────────────────────────────────────────────────
  function closeLibModal() {
    if (_libModal) { _libModal.remove(); _libModal = null; }
  }

  function openLibModal(title, buildContent) {
    closeLibModal();
    const overlay = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", title));
    const headRight = el("div", "modal-head-right");
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.onclick = closeLibModal;
    headRight.append(closeBtn);
    head.append(headRight);
    box.append(head);
    const content = el("div", "modal-content");
    buildContent(content, closeLibModal);
    box.append(content);
    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeLibModal(); });
    document.body.append(overlay);
    _libModal = overlay;
  }
  const splitLines = (v) => String(v || "").split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
  function labeled(label, ctrl) {
    const l = el("label", "lib-field"); l.append(el("span", null, label)); l.append(ctrl); return l;
  }
  function checkRow(label, checked) {
    const row = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = checked;
    row.append(cb); row.append(el("span", null, label)); row._cb = cb; return row;
  }

  function openShortcutEditor(item) {
    const isNew = !item.name;
    openLibModal(isNew ? "New shortcut" : `Edit “${item.name}”`, (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
      const trig = el("textarea", "lib-in"); trig.rows = 3; trig.placeholder = "Triggers — one per line or comma-separated"; trig.value = (item.triggers || []).join("\n");
      const reps = el("textarea", "lib-in"); reps.rows = 4; reps.placeholder = "Replacement(s) — one per line; multiple = random pick"; reps.value = (item.replacements || []).join("\n");
      const en = checkRow("Enabled", item.enabled !== false);
      box.append(labeled("Name", name), labeled("Triggers", trig), labeled("Replacements", reps), en);
      const actions = el("div", "lib-form-actions");
      const save = el("button", "btn primary tiny", "Save");
      save.onclick = async () => {
        const triggers = splitLines(trig.value);
        if (!triggers.length) { alert("At least one trigger is required."); return; }
        await S.saveShortcut({
          name: name.value.trim() || triggers[0], triggers,
          replacements: splitLines(reps.value), enabled: en._cb.checked,
          original_name: item.name || undefined,
        });
        close(); render(S.get());
      };
      const cancel = el("button", "btn ghost tiny", "Cancel");
      cancel.onclick = close;
      actions.append(save, cancel); box.append(actions);
      content.append(box);
    });
  }

  const PLACEMENTS = ["global", "start", "end", "silent"];
  function selectFrom(opts, value) {
    const sel = el("select", "lib-in");
    opts.forEach((o) => { const op = el("option", null, o); op.value = o; if (o === value) op.selected = true; sel.append(op); });
    return sel;
  }
  function openSplitMarkerEditor(item) {
    const isNew = !item.name && !item.trigger;
    openLibModal(isNew ? "New split marker" : `Edit “${item.name || item.trigger}”`, (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
      const trig = el("input", "lib-in"); trig.placeholder = "Trigger phrase (what appears in the prompt)"; trig.value = item.trigger || "";
      const place = selectFrom(PLACEMENTS, item.placement || "global");
      const en = checkRow("Enabled", item.enabled !== false);
      box.append(labeled("Name", name), labeled("Trigger", trig), labeled("Placement", place), en);
      box.append(el("div", "insp-hint", "Splits the generation prompt only — not a video dissolve on the timeline."));
      const actions = el("div", "lib-form-actions");
      const save = el("button", "btn primary tiny", "Save");
      save.onclick = async () => {
        const trigger = trig.value.trim();
        if (!trigger) { alert("A trigger phrase is required."); return; }
        await S.saveTransition({
          name: name.value.trim() || trigger, trigger,
          placement: place.value, enabled: en._cb.checked,
          original_name: item.name || undefined,
        });
        close(); render(S.get());
      };
      const cancel = el("button", "btn ghost tiny", "Cancel");
      cancel.onclick = close;
      actions.append(save, cancel); box.append(actions);
      content.append(box);
    });
  }

  function mediaRefPicker(st, value, onChange, opts) {
    return window.MediaPicker.create({
      value,
      mediaBin: st.mediaBin,
      onChange,
      compact: true,
      startOpen: !!(opts && opts.startOpen),
    });
  }

  function characterForm(st, item) {
    const isNew = !item.id;
    const box = el("div", "lib-form");
    box.append(el("div", "lib-form-title", isNew ? "New character" : `Edit “${item.name || "character"}”`));
    const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
    const appearance = el("textarea", "lib-in"); appearance.rows = 2; appearance.placeholder = "Face, hair, eyes, skin…"; appearance.value = item.appearance || "";
    const body = el("textarea", "lib-in"); body.rows = 2; body.placeholder = "Build, height, proportions…"; body.value = item.body || "";
    const wardrobe = el("textarea", "lib-in"); wardrobe.rows = 2; wardrobe.placeholder = "Default outfit or style…"; wardrobe.value = item.wardrobe || "";
    const always = el("textarea", "lib-in"); always.rows = 2; always.placeholder = "Tags always included in this character's prompt"; always.value = item.always_include || "";
    const never = el("textarea", "lib-in"); never.rows = 2; never.placeholder = "Appended to negative when this character is on a scene"; never.value = item.never_include || "";
    let faceRef = item.face_ref || null;
    let bodyRef = item.body_ref || null;
    let detailRef = item.detail_ref || null;
    box.append(
      labeled("Name", name),
      labeled("Appearance", appearance),
      labeled("Body", body),
      labeled("Wardrobe", wardrobe),
      labeled("Always include", always),
      labeled("Never include", never),
      labeled("Face ref (identity pin)", mediaRefPicker(st, faceRef, (v) => { faceRef = v; }, { startOpen: true })),
      labeled("Body ref", mediaRefPicker(st, bodyRef, (v) => { bodyRef = v; }, { startOpen: true })),
      labeled("Detail ref", mediaRefPicker(st, detailRef, (v) => { detailRef = v; }, { startOpen: true })),
    );
    const actions = el("div", "lib-form-actions");
    const save = el("button", "btn primary tiny", "Save");
    save.onclick = async () => {
      const nm = name.value.trim();
      if (!nm) { alert("Character name is required."); return; }
      await S.saveCharacter({
        id: item.id || undefined,
        original_id: item.id || undefined,
        name: nm,
        appearance: appearance.value,
        body: body.value,
        wardrobe: wardrobe.value,
        always_include: always.value,
        never_include: never.value,
        face_ref: faceRef,
        body_ref: bodyRef,
        detail_ref: detailRef,
      });
      editCharacter = null; render(S.get());
    };
    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = () => { editCharacter = null; render(S.get()); };
    actions.append(save, cancel); box.append(actions);
    return box;
  }

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
    if (mediaSelected.size === 1) id = [...mediaSelected][0];
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
    const picked = mediaSelected.has(m.id);
    const card = el("div", "media-card"
      + (st.mediaPreviewId === m.id ? " previewing" : "")
      + (picked ? " picked" : ""));
    card.draggable = true;
    card.addEventListener("dragstart", (e) => { e.dataTransfer.setData("application/funpack-media", m.id); e.dataTransfer.effectAllowed = "copy"; });
    card.title = mediaSelectMode
      ? `${m.name}\nClick to toggle selection`
      : m.kind === "image"
        ? `${m.name}\nImage · click to preview · drag onto a clip to set anchor`
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
    card.onclick = () => {
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

    if (selN > 0 || exportId) {
      const actions = el("div", "media-bin-actions");
      if (exportId) {
        const expBtn = el("button", "btn ghost tiny", "⤓ Export");
        expBtn.title = "Save this image or video to your computer";
        expBtn.onclick = () => S.exportMediaAsset(exportId);
        actions.append(expBtn);
      }
      if (selN > 0) {
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
        mediaSelectMode = !mediaSelectMode;
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

  // ── characters bin ───────────────────────────────────────────────────────────────
  function charactersTab(st) {
    if (editCharacter) return characterForm(st, editCharacter);
    const wrap = el("div", "bin");
    wrap.append(searchRow("Characters", "Filter characters…", () => render(S.get())));
    const hint = el("div", "pj-meta char-hint");
    hint.textContent = st.selectedSceneId
      ? "Click a character to add/remove it on the selected scene."
      : "Select a scene, then click characters to assign them.";
    wrap.append(hint);
    const head = el("div", "lib-head");
    const add = el("button", "btn ghost tiny", "＋ New");
    add.onclick = () => { editCharacter = {}; render(S.get()); };
    head.append(add); wrap.append(head);

    const assigned = st.selectedSceneId ? new Set(S.sceneCharacterIds(st.selectedSceneId)) : null;
    const list = el("div", "char-list");
    const items = filtered(st.characters || [], q.Characters, (c) =>
      `${c.name} ${c.appearance || ""} ${c.body || ""} ${c.wardrobe || ""}`);
    items.forEach((c) => {
      const onScene = assigned && assigned.has(c.id);
      const row = el("div", "char-row" + (onScene ? " on-scene" : ""));
      const prev = el("div", "char-thumb");
      if (c.face_ref) {
        const img = el("img"); img.src = API.mediaUrl(c.face_ref); img.loading = "lazy"; prev.append(img);
      } else prev.append(el("span", null, "◎"));
      row.append(prev);
      const info = el("div", "char-info");
      info.append(el("div", "char-name", c.name || "(unnamed)"));
      const sub = [c.appearance, c.wardrobe].filter(Boolean).join(" · ");
      if (sub) info.append(el("div", "char-sub", sub));
      row.append(info);
      if (onScene) row.append(el("span", "char-badge", "on scene"));
      row.onclick = () => {
        if (st.selectedSceneId) S.toggleSceneCharacter(st.selectedSceneId, c.id);
        else { editCharacter = { ...c }; render(S.get()); }
      };
      const edit = el("button", "btn ghost tiny", "✎");
      edit.title = "Edit character";
      edit.onclick = (e) => { e.stopPropagation(); editCharacter = { ...c }; render(S.get()); };
      const del = el("button", "btn ghost tiny danger", "✕");
      del.title = "Delete character";
      del.onclick = async (e) => {
        e.stopPropagation();
        if (confirm(`Delete character “${c.name}”?`)) await S.deleteCharacter(c.id);
      };
      const actions = el("div", "char-actions");
      actions.append(edit, del);
      row.append(actions);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.characters || []).length ? "No match." : "No characters yet — create one."));
    wrap.append(list);
    return wrap;
  }

  // ── shortcuts bin ────────────────────────────────────────────────────────────────
  function shortcutsTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Shortcuts", "Filter shortcuts…", () => render(S.get())));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add");
    addBtn.onclick = () => openShortcutEditor({});
    const expBtn = el("a", "btn ghost tiny", "↓ Export");
    expBtn.href = API.exportShortcutsUrl(); expBtn.download = "funpack_shortcuts.json"; expBtn.title = "Download shortcuts as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importShortcuts(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} shortcut(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↑ Import"); impBtn.title = "Import shortcuts from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn); toolbar.append(expBtn); toolbar.append(impBtn); toolbar.append(impFile);
    wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.shortcuts || [], q.Shortcuts, (s) => `${s.name} ${(s.triggers || []).join(" ")} ${(s.replacements || []).join(" ")}`);
    items.forEach((s) => {
      const trig = (s.triggers || [])[0] || s.name;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", s.name + (s.enabled === false ? " (off)" : "")));
      const rep = (s.replacements || []).join(" / ");
      if (rep) main.append(el("div", "lib-sub", "→ " + rep));
      row.append(main);
      const ins = el("button", "btn ghost tiny", "insert"); ins.title = "Append to selected scene's prompt";
      ins.onclick = () => { if (!S.insertShortcutIntoSelection(trig)) alert("Select a scene first."); };
      const edit = el("button", "ic-btn", "✎"); edit.title = "Edit shortcut";
      edit.onclick = () => openShortcutEditor(s);
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete shortcut";
      del.onclick = () => { if (confirm(`Delete shortcut "${s.name}"?`)) S.deleteShortcut(s.name); };
      row.append(ins); row.append(edit); row.append(del);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.shortcuts || []).length ? "No match." : "No shortcuts yet."));
    wrap.append(list);
    return wrap;
  }

  // ── split markers bin (generation prompt splits) ───────────────────────────────
  function splitMarkersTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Splits", "Filter split markers…", () => render(S.get())));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add");
    addBtn.onclick = () => openSplitMarkerEditor({});
    const expBtn = el("a", "btn ghost tiny", "↓ Export");
    expBtn.href = API.exportTransitionsUrl(); expBtn.download = "funpack_promptsplit.json"; expBtn.title = "Download split markers as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importTransitions(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} split marker(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↑ Import"); impBtn.title = "Import split markers from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn); toolbar.append(expBtn); toolbar.append(impBtn); toolbar.append(impFile);
    wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.transitions || [], q.Splits, (t) => `${t.name || ""} ${t.trigger || ""} ${t.placement || ""}`);
    items.forEach((t) => {
      const trig = t.trigger || t.name || t.key;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", (t.name || trig) + (t.enabled === false ? " (off)" : "")));
      const sub = [t.trigger && t.trigger !== (t.name || "") ? `"${t.trigger}"` : "",
                   t.placement && t.placement !== "global" ? t.placement : ""].filter(Boolean).join(" · ");
      if (sub) main.append(el("div", "lib-sub", sub));
      row.append(main);
      const apply = el("button", "btn ghost tiny", "apply"); apply.title = "Set as split marker before the selected scene (generation prompt)";
      apply.onclick = () => { if (!S.applySplitMarkerToSelection(trig)) alert("Select a scene first."); };
      const edit = el("button", "ic-btn", "✎"); edit.title = "Edit split marker";
      edit.onclick = () => openSplitMarkerEditor(t);
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete split marker";
      del.onclick = () => { if (confirm(`Delete split marker "${t.name || trig}"?`)) S.deleteTransition(t.name || trig); };
      row.append(apply); row.append(edit); row.append(del);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.transitions || []).length ? "No match." : "No split markers yet."));
    wrap.append(list);
    return wrap;
  }

  function nlePresetTab(st, kind) {
    const isEffect = kind === "effect";
    const items = isEffect ? (st.nleEffects || []) : (st.nleVideoTransitions || []);
    const qKey = isEffect ? "Effects" : "Transitions";
    const wrap = el("div", "bin");
    wrap.append(searchRow(qKey, `Filter ${qKey.toLowerCase()}…`, () => render(S.get())));
    wrap.append(el("div", "insp-hint", isEffect
      ? "Clip effects applied on the timeline after render (zoom, blur, fades)."
      : "Video blends between clips on the timeline — not prompt split markers."));
    const list = el("div", "lib-list");
    const filteredItems = filtered(items, q[qKey], (x) => `${x.name || ""} ${x.id || ""} ${x.description || ""}`);
    filteredItems.forEach((item) => {
      const row = el("div", "lib-row lib-row-nle");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", item.name || item.id));
      const sub = item.description || (item.param ? `${item.param.label} default ${item.param.default}` : "");
      if (sub) main.append(el("div", "lib-sub", sub));
      row.append(main);
      const apply = el("button", "btn ghost tiny", "apply");
      apply.title = isEffect ? "Add effect to selected clip" : "Set video transition on selected clip";
      apply.onclick = () => {
        if (!st.selectedSceneId) { alert("Select a clip first."); return; }
        if (isEffect) {
          const v = item.param ? item.param.default : null;
          if (!S.applyNleEffect(item.id, v)) alert("Could not apply effect.");
        } else if (!S.applyNleVideoTransition(item.id, item.param?.default)) {
          alert("Could not apply transition.");
        }
      };
      row.append(apply);
      list.append(row);
    });
    if (!filteredItems.length) list.append(el("div", "pj-meta", items.length ? "No match." : "No presets loaded."));
    wrap.append(list);
    return wrap;
  }

  // ── helpers ──────────────────────────────────────────────────────────────────────
  function filtered(arr, query, textOf) {
    const s = (query || "").trim().toLowerCase();
    return s ? arr.filter((x) => textOf(x).toLowerCase().includes(s)) : arr;
  }
  function searchRow(key, placeholder, onInput) {
    const inp = el("input", "lib-search"); inp.type = "text"; inp.placeholder = placeholder; inp.value = q[key] || "";
    inp.oninput = () => { q[key] = inp.value; onInput(); };
    return inp;
  }

  function render(st) {
    clear(body);
    body.append(projectsSection(st));

    const sec = el("div", "mb-section mb-bin-shell");
    const tabs = el("div", "bin-tabs bin-tabs-sticky");
    ["Media", "Characters", "Shortcuts", "Splits", "Effects", "Transitions"].forEach((name) => {
      const b = el("button", "bin-tab" + (tab === name ? " active" : ""), name);
      b.title = name === "Splits" ? "Split markers (generation prompt)" : name;
      b.onclick = () => {
        tab = name;
        editCharacter = null;
        if (name !== "Media") mediaSelectMode = false;
        closeLibModal();
        render(S.get());
      };
      tabs.append(b);
    });
    sec.append(tabs);
    const scroll = el("div", "mb-bin-scroll");
    scroll.append(
      tab === "Media" ? mediaTab(st)
        : tab === "Characters" ? charactersTab(st)
          : tab === "Shortcuts" ? shortcutsTab(st)
            : tab === "Splits" ? splitMarkersTab(st)
              : tab === "Effects" ? nlePresetTab(st, "effect")
                : nlePresetTab(st, "transition"),
    );
    sec.append(scroll);
    body.append(sec);
  }

  if (window.ViewBus) window.ViewBus.subscribeMediabrowser(render);
  else S.subscribe(render);
})();
