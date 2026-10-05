// The pipeline travels with the project: opening a project puts ITS pipeline over the session's, and
// every landed pipeline edit becomes the project's copy (project.models). One writer -- the pipeline state.
export function linkPipeline({ project, pipeline, onOpen, say = () => {} }) {
  // The id of the project the live pipeline belongs to, or null while one is being put in (or could not be): only
  // it is written. A project opened while an open or a save is still in flight must never receive the other's pipeline.
  // An id, not the object: Undo puts a copy of the same project in place.
  let owner = null;
  let retries = 0, opens = 0, retry = null, waiting = false;      // waiting: gave up because ComfyUI did not answer
  // Only a real change is written: opening a project must not rewrite its file.
  const live = (base, slots) => ({ ...base, slots: JSON.parse(JSON.stringify(slots || [])), removed: pipeline.removedIds(), unwired: pipeline.unwiredMap(), whole: pipeline.whole() });
  const store = (base, slots) => {
    const next = live(base, slots);
    if (JSON.stringify(next) !== JSON.stringify(project.project.models || {})) project.setField("models", next, { quiet: true });
  };

  async function adopt() {
    clearTimeout(retry);                    // a retry still waiting is for a project that may no longer be open
    waiting = false;
    if (pipeline.settled) await pipeline.settled();       // an edit on its way lands in the project it was made in first
    const here = project.project && project.project.models;
    // Undo / Redo of something else: the same project, its pipeline already live, nothing to put in (or to lose).
    if (owner && project.project && owner === project.project.id && here && pipeline.slots()
        && JSON.stringify(live(here, pipeline.slots())) === JSON.stringify(here)) return;
    owner = null;
    const mine = ++opens, target = project.project && project.project.id;
    const saved = project.project && project.project.models || {};
    let ok = false, timer;
    try {
      // Bounded: a hung request must not hold the project behind it.
      ok = await Promise.race([pipeline.adopt(saved.slots || [], saved.removed, saved.unwired, saved.whole, project.fresh), new Promise((r) => { timer = setTimeout(() => r(false), 20000); })]);
    } catch { ok = false; } finally { clearTimeout(timer); }
    if (mine !== opens) return;                                  // opened again meanwhile (another project, or this one): the latest open handles it
    if (ok && pipeline.slots()) {
      owner = target;
      retries = 0;
      // Opening writes nothing: what an update's defaults fill in is laid again on every open. A new project keeps what it took.
      if (project.fresh) store(saved, pipeline.slots());
    } else if (retries++ < 3) {
      retry = setTimeout(adopt, 5000);
    } else if (!pipeline.slots()) {
      // ComfyUI is down: keep asking, slowly; and if a panel's own load gets there first, its announcement puts this project's in.
      waiting = true;
      retry = setTimeout(adopt, 15000);
      if (retries === 4) say("ComfyUI is not answering, so this project's pipeline is not loaded yet: it goes in as soon as ComfyUI answers.");
    } else {
      say("This project's saved pipeline could not be loaded; the default is in use.");
    }
  }

  pipeline.subscribe((slots) => {
    if (waiting && owner === null) { retries = 0; adopt(); return; }
    if (!project.project || project.project.id !== owner) return;
    store(project.project.models || {}, slots);
  });
  onOpen(() => { retries = 0; return adopt(); });       // each project opened gets its own retries
  return { adopt };
}
