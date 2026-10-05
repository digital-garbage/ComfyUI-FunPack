// The pipeline travels with the project: opening a project puts ITS pipeline over the session's, and
// every landed pipeline edit becomes the project's copy (project.models). One writer -- the pipeline state.
export function linkPipeline({ project, pipeline, onOpen, say = () => {} }) {
  let adopted = true;        // false while the open project's pipeline is not the live one: its copy is not rewritten
  let retries = 0;
  // Only a real change is written: opening a project must not rewrite its file.
  const store = (base, slots) => {
    const next = { ...base, slots: JSON.parse(JSON.stringify(slots || [])), removed: pipeline.removedIds(), unwired: pipeline.unwiredMap(), whole: pipeline.whole() };
    if (!("whole" in base) && !next.whole) delete next.whole;     // an older file reads as not whole anyway: no rewrite just to say so
    if (JSON.stringify(next) !== JSON.stringify(project.project.models || {})) project.setField("models", next, { quiet: true });
  };

  async function adopt() {
    const saved = project.project && project.project.models || {};
    const own = (saved.slots || []).length > 0 || (saved.removed || []).length > 0;
    let ok = false, timer;
    try {
      // Bounded: a hung request must not hold the project behind it.
      ok = await Promise.race([pipeline.adopt(saved.slots || [], saved.removed, saved.unwired, saved.whole, project.fresh), new Promise((r) => { timer = setTimeout(() => r(false), 20000); })]);
    } catch { ok = false; } finally { clearTimeout(timer); }
    adopted = Boolean(ok && pipeline.slots());
    if (adopted) {
      retries = 0;
      // An old project with nothing saved runs the default and stays as it is on disk; a new one keeps what it took.
      if (own || project.fresh) store(saved, pipeline.slots());
    } else if (retries++ < 3) {
      setTimeout(adopt, 5000);
    } else {
      say("This project's saved pipeline could not be loaded; the default is in use.");
    }
  }

  pipeline.subscribe((slots) => {
    if (!project.project || !adopted) return;
    store(project.project.models || {}, slots);
  });
  onOpen(adopt);
  return { adopt };
}
