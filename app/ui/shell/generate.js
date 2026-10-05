// Queues runs on ComfyUI and reports where they are. A thin shell over run.js / session.js / pipeline.js:
// one real queue per run, each taken to a terminal state before the next starts. Nothing here knows what a
// scene is -- the caller says which scenes a run covers and what inputs it sends.
import { createRun, viewUrl, DONE, FAILED, CANCELLED } from "./run.js";
import { clientId, connect, queuedFor, finishedFor } from "./client.js";
import { wire, waitForTerminal } from "./session.js";
import { check } from "./pipeline.js";
import { wireReferences } from "./reference_wiring.js";

export function createGenerate({ pipeline }) {
  const id = clientId();
  const run = createRun({ clientId: id, connect });

  const listeners = { say: [], warn: [], hold: [], release: [], adopt: [] };
  let lastAdopt = null;      // replayed to a late listener: it carries the only copy of which scenes a found run belongs to
  const fire = (kind, arg) => {
    if (kind === "adopt") lastAdopt = arg;
    (listeners[kind] || []).forEach((fn) => { try { fn(arg); } catch { /* one listener must not stop the rest */ } });
  };
  // The hooks session.js calls on its page.transport. Nothing here draws.
  const transport = {
    generate: { setDisabled() {} }, draw() {},
    say: (msg) => fire("say", msg), warn: (msg) => fire("warn", msg),
    hold: (msg) => fire("hold", msg), release: (state) => fire("release", state),
  };

  // Which scenes/project the next run belongs to, read when it actually queues (and sent as extra_data so a
  // reload mid-run can find it again). `inputs` is {slotId: {input: value}}.
  let ranFor = null, ranForList = [], ranForProject = null, pendingInputs = null, pendingSlots = null;

  const session = wire({
    run, page: { transport }, check, id, queuedFor, finishedFor,
    slots: () => pendingSlots || pipeline.slots(),     // the pipeline as it was at the click, when the caller froze it
    inputs: async () => pendingInputs || {},
    extra: () => (ranForProject && ranFor
      ? { funpack_scene_id: ranFor, funpack_scene_ids: ranForList, funpack_project_id: ranForProject } : null),
    onAdopt: (sceneId, projectId, sceneIds) => {
      ranFor = sceneId; ranForProject = projectId;
      ranForList = (sceneIds && sceneIds.length) ? sceneIds : (sceneId ? [sceneId] : []);
      fire("adopt", { sceneId, projectId, sceneIds: ranForList });
    },
  });

  /** The slot+role pair a pipeline role points at, or null when nothing offers it. */
  const slotForRole = (at) => {
    for (const slot of pipeline.slots() || []) {
      for (const role of slot.roles || []) if (role.at === at) return { slot, role };
    }
    return null;
  };

  return {
    DONE, FAILED, CANCELLED, run, slotForRole, wireReferences,
    viewUrl: (image) => viewUrl(image),
    state: () => run.state,
    subscribe: (fn) => run.subscribe(fn),
    on(kind, fn) {
      (listeners[kind] ||= []).push(fn);
      if (kind === "adopt" && lastAdopt) { try { fn(lastAdopt); } catch { /* as above */ } }
      return () => { listeners[kind] = (listeners[kind] || []).filter((f) => f !== fn); };
    },
    cancel: () => run.cancel(),
    waitForTerminal: () => waitForTerminal(run),
    ready: session.ready,
    /** Queue one run; resolves true once it is on ComfyUI's queue. */
    async generate({ sceneId, sceneIds, projectId, inputs, slots } = {}) {
      pendingSlots = slots || null;
      ranFor = sceneId || null;
      ranForList = (sceneIds && sceneIds.length) ? sceneIds : (sceneId ? [sceneId] : []);
      ranForProject = projectId || null;
      pendingInputs = inputs || null;
      return session.generate();
    },
  };
}
