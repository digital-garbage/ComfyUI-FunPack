// The pipeline travels with the project: opening a project puts ITS pipeline over the session's, and
// every landed pipeline edit becomes the project's copy (project.models). One writer -- the pipeline state.
export function linkPipeline({ project, pipeline, onOpen, say = () => {} }) {
  let adopted = true;        // false while the open project's pipeline is not the live one: its copy is not rewritten
  let retries = 0;

  async function adopt() {
    const saved = project.project && project.project.models || {};
    let ok = false;
    try {
      // Bounded: a hung request must not hold the project behind it.
      ok = await Promise.race([pipeline.adopt(saved.slots || [], saved.removed, saved.unwired), new Promise((r) => setTimeout(() => r(false), 20000))]);
    } catch { ok = false; }
    adopted = Boolean(ok && pipeline.slots());
    if (adopted) {
      retries = 0;
      project.setField("models", { ...saved, slots: JSON.parse(JSON.stringify(pipeline.slots())), removed: pipeline.removedIds(), unwired: pipeline.unwiredMap() }, { quiet: true });
    } else if (retries++ < 3) {
      setTimeout(adopt, 5000);
    } else {
      say("This project's saved pipeline could not be loaded; the default is in use.");
    }
  }

  pipeline.subscribe((slots) => {
    if (!project.project || !adopted) return;
    const had = project.project.models || {};
    project.setField("models", { ...had, slots: JSON.parse(JSON.stringify(slots || [])), removed: pipeline.removedIds(), unwired: pipeline.unwiredMap() }, { quiet: true });
  });
  onOpen(adopt);
  return { adopt };
}
