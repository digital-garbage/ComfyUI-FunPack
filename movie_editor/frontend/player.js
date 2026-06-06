// Right-top zone: the program monitor (result video) + generation readout.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("preview-body");
  const statusEl = document.getElementById("preview-status");

  function render(st) {
    clear(body);
    const gen = st.gen || { state: "idle", media: [] };

    if (gen.media && gen.media.length) {
      const m = gen.media[0];
      const url = API.resultUrl(st.project.id, m);
      const isVideo = m.kind !== "images" || /\.(mp4|webm|mov)$/i.test(m.filename) ||
        (m.format || "").toLowerCase().includes("mp4");
      if (isVideo && !/\.gif$/i.test(m.filename)) {
        const v = el("video", "result-media"); v.src = url; v.controls = true; v.autoplay = true; v.loop = true;
        body.append(v);
      } else {
        const img = el("img", "result-media"); img.src = url; body.append(img);
      }
      body.append(el("div", "result-cap", m.filename));
    } else {
      const empty = el("div", "program-empty");
      empty.append(el("div", "reel", "🎬"));
      empty.append(el("div", null, st.project ? "No render yet" : "Open or create a project"));
      empty.append(el("div", "pj-meta", "Generate from the menu or timeline"));
      body.append(empty);
    }

    // readout overlay
    if (["queuing", "running", "pending", "error"].includes(gen.state)) {
      const ro = el("div", "gen-readout" + (gen.state === "error" ? " error" : ""));
      ro.append(el("span", "pulse"));
      ro.append(el("span", null, gen.msg || gen.state));
      body.append(ro);
    }

    // header status
    clear(statusEl);
    if (gen.promptId) statusEl.append(el("span", null, gen.state.toUpperCase()));
  }

  S.subscribe(render);
})();
