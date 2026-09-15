// Bridges v5's real check -> queue -> terminal generation flow into the classic-
// script v4 world. app/shell/run.js, session.js, pipeline.js and client.js are
// unmodified here -- they have no DOM/composer dependency (see the port plan's
// "Preserved and rewired" list) -- this file is only the small ES-module shim
// that hands their globals-free API to v4's global-script callers.
//
// One real queue per RUN, run to a terminal state before the next one starts
// -- v5 has no chain-sampler/montage stage (postponed, LTX-only -- see the
// port plan), so a run can cover one scene or every active scene at once
// (H3's own whole-timeline-in-one-call shape); store.js decides which and
// hands the scene list in, this only knows how to run what it's given.
import { createRun, viewUrl, DONE, FAILED, CANCELLED } from "../shell/run.js";
import { clientId, connect, queuedFor, finishedFor } from "../shell/client.js";
import { wire, waitForTerminal } from "../shell/session.js";
import { check } from "../shell/pipeline.js";
import { wireReferences } from "../shell/reference_wiring.js";

const PS = window.PipelineState;
const id = clientId();
const run = createRun({ clientId: id, connect });

// The five hooks session.js's wire()/createGenerator call on `page.transport`.
// Nothing here owns UI -- store.js already renders generation status through
// its own `state.gen` -- so these just fan out to whoever subscribes via
// GenerateBridge.on(...).
const listeners = { say: [], warn: [], hold: [], release: [], disabled: [], adopt: [] };
// The most recent "adopt" payload, replayed to a listener that registers
// AFTER it already fired. This module's own reattach (inside wire(), below)
// starts the moment this script evaluates -- before store.js's classic
// script has even run, let alone reached the point in its boot sequence
// where it calls GB.on("adopt", ...) -- so on a reload during a generation,
// adopt routinely fires and is gone before anyone is listening. Unlike
// "say"/"warn"/"hold"/"release" (transient status text, fine to miss), this
// event carries the ONLY copy of which scene(s) a reattached run belongs to;
// losing it doesn't just miss a message, it orphans that run's eventual
// result from ever being recorded onto its scene.
let lastAdopt = null;
function fire(kind, arg) {
  if (kind === "adopt") lastAdopt = arg;
  (listeners[kind] || []).forEach((fn) => { try { fn(arg); } catch (_) {} });
}
const transport = {
  generate: { setDisabled: (v) => fire("disabled", v) },
  say: (msg) => fire("say", msg),
  warn: (msg) => fire("warn", msg),
  draw: () => {},           // run.subscribe() below already covers this
  hold: (msg) => fire("hold", msg),
  release: (state) => fire("release", state),
};

// Which scene(s)/project a run belongs to. Set by generate() at click time;
// read by run.start() at the moment IT actually queues (see session.js's own
// comment on why this can't be read only after the fact). `ranFor` is the
// FIRST scene -- the single-scalar shape every other funpack_scene_id reader
// (client.js's queuedFor, session.js's finishedFor path) still expects --
// `ranForList` is every scene THIS run actually covers, sent alongside it so
// a reattach after reload can recover the whole list instead of only the
// first (see client.js's own funpack_scene_ids comment).
let ranFor = null;
let ranForList = [];
let ranForProject = null;
// The inputs the NEXT queued run should send -- built by the caller (store.js
// knows the scene text, anchor, postfix, variables; this file does not) and
// handed to generate() right before it runs, not assembled here.
let pendingInputs = null;

const session = wire({
  run, page: { transport }, check, id, queuedFor, finishedFor,
  // PipelineState (pipeline_state.js) is the one client-side copy of the live
  // pipeline, shared with Engine Settings and Models & Pipeline. A second copy
  // here would be the exact bug pipeline_state.js was extracted to fix.
  slots: () => PS.slots(),
  // Module settings are already IN `slots` by the time Generate runs -- every
  // Engine Settings edit is saved through PS.save(), which writes into the
  // sink node's input (see pipeline_state.js). `values` exists for the
  // composer's own role-mounted dials, which legacy does not use.
  values: () => ({}),
  inputs: async () => pendingInputs || {},
  extra: () => (ranForProject && ranFor
    ? { funpack_scene_id: ranFor, funpack_scene_ids: ranForList, funpack_project_id: ranForProject }
    : null),
  onAdopt: (sceneId, projectId, sceneIds) => {
    ranFor = sceneId; ranForProject = projectId;
    ranForList = (sceneIds && sceneIds.length) ? sceneIds : (sceneId ? [sceneId] : []);
    // A run this PAGE queued sets ranFor itself, at click time (see generate()
    // below). A run only found on reload -- nobody here called generate() for
    // it -- has no other way to tell store.js which scene(s) it belongs to.
    fire("adopt", { sceneId, projectId, sceneIds: ranForList });
  },
});

/** The slot+role pair a pipeline role points at, or null if nothing offers it. */
function slotForRole(at) {
  for (const slot of PS.slots() || []) {
    for (const role of slot.roles || []) {
      if (role.at === at) return { slot, role };
    }
  }
  return null;
}

window.GenerateBridge = {
  DONE, FAILED, CANCELLED,
  viewUrl: (image) => viewUrl(image),
  state: () => run.state,
  subscribe: (fn) => run.subscribe(fn),
  on: (kind, fn) => {
    (listeners[kind] || (listeners[kind] = [])).push(fn);
    if (kind === "adopt" && lastAdopt) { try { fn(lastAdopt); } catch (_) {} }
    return () => { listeners[kind] = (listeners[kind] || []).filter((f) => f !== fn); };
  },
  cancel: () => run.cancel(),
  waitForTerminal: () => waitForTerminal(run),
  ready: session.ready,
  slotForRole,
  wireReferences,
  /**
   * Queue one run. `inputs` is {slotId: {inputName: value}}. `sceneIds`, when
   * the run covers more than one scene (no chain sampler to split it into
   * several calls), is every scene it covers -- `sceneId` alone is kept as
   * the first of them, for reattach's single-scalar contract.
   */
  async generate({ sceneId, sceneIds, projectId, inputs } = {}) {
    ranFor = sceneId || null;
    ranForList = (sceneIds && sceneIds.length) ? sceneIds : (sceneId ? [sceneId] : []);
    ranForProject = projectId || null;
    pendingInputs = inputs || null;
    return session.generate();
  },
};
