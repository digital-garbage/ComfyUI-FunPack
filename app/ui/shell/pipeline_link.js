// The pipeline travels with the project: opening a project puts ITS pipeline over the session's, and
// every landed pipeline edit becomes the project's copy (project.models). One writer -- the pipeline state.
export function linkPipeline({ project, pipeline, onOpen, say = () => {} }) {
  // The id of the project the live pipeline belongs to, or null while one is being put in (or could not be): only
  // it is written. A project opened while an open or a save is still in flight must never receive the other's pipeline.
  // An id, not the object: Undo puts a copy of the same project in place.
  let owner = null;
  let retries = 0, opens = 0;
  // Only a real change is written: opening a project must not rewrite its file.
  const store = (base, slots, opening) => {
    const next = { ...base, slots: JSON.parse(JSON.stringify(slots || [])), removed: pipeline.removedIds(), unwired: pipeline.unwiredMap(), whole: pipeline.whole() };
    if (opening && !("whole" in base)) delete next.whole;          // an older file is not rewritten just to record the guess; an edit records it
    if (JSON.stringify(next) !== JSON.stringify(project.project.models || {})) project.setField("models", next, { quiet: true });
  };

  async function adopt() {
    owner = null;
    const mine = ++opens, target = project.project && project.project.id;
    const saved = project.project && project.project.models || {};
    const own = (saved.slots || []).length > 0 || (saved.removed || []).length > 0;
    let ok = false, timer;
    try {
      // Bounded: a hung request must not hold the project behind it.
      ok = await Promise.race([pipeline.adopt(saved.slots || [], saved.removed, saved.unwired, saved.whole, project.fresh), new Promise((r) => { timer = setTimeout(() => r(false), 20000); })]);
    } catch { ok = false; } finally { clearTimeout(timer); }
    if (mine !== opens) return;                                  // opened again meanwhile (another project, or this one): the latest open handles it
    if (ok && pipeline.slots()) {
      owner = target;
      retries = 0;
      // An old project with nothing saved runs the default and stays as it is on disk; a new one keeps what it took.
      if (own || project.fresh) store(saved, pipeline.slots(), true);
    } else if (retries++ < 3) {
      setTimeout(adopt, 5000);
    } else {
      say("This project's saved pipeline could not be loaded; the default is in use.");
    }
  }

  pipeline.subscribe((slots) => {
    if (!project.project || project.project.id !== owner) return;
    store(project.project.models || {}, slots);
  });
  onOpen(adopt);
  return { adopt };
}
