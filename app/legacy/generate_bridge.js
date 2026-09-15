// Bridges v5's real check -> queue -> terminal generation flow into the classic-
// script v4 world. app/shell/run.js, session.js, pipeline.js and client.js are
// unmodified here -- they have no DOM/composer dependency (see the port plan's
// "Preserved and rewired" list) -- this file is only the small ES-module shim
// that hands their globals-free API to v4's global-script callers.
//
// Single scene at a time, on purpose: v5 has no chain-sampler/montage stage
// (postponed, LTX-only -- see the port plan), so "Generate" here means what
// app/boot.js's own generateCurrentScene/generateAll mean -- one real queue per
// scene, run to a terminal state before the next one starts. store.js supplies
// the scene loop; this only knows how to run ONE.
import { createRun, viewUrl, DONE, FAILED, CANCELLED } from "../shell/run.js";
import { clientId, connect, queuedFor, finishedFor } from "../shell/client.js";
import { wire, waitForTerminal } from "../shell/session.js";
import { check } from "../shell/pipeline.js";

const PS = window.PipelineState;
const id = clientId();
const run = createRun({ clientId: id, connect });

// The five hooks session.js's wire()/createGenerator call on `page.transport`.
// Nothing here owns UI -- store.js already renders generation status through
// its own `state.gen` -- so these just fan out to whoever subscribes via
// GenerateBridge.on(...).
const listeners = { say: [], warn: [], hold: [], release: [], disabled: [], adopt: [] };
function fire(kind, arg) { (listeners[kind] || []).forEach((fn) => { try { fn(arg); } catch (_) {} }); }
const transport = {
  generate: { setDisabled: (v) => fire("disabled", v) },
  say: (msg) => fire("say", msg),
  warn: (msg) => fire("warn", msg),
  draw: () => {},           // run.subscribe() below already covers this
  hold: (msg) => fire("hold", msg),
  release: (state) => fire("release", state),
};

// Which scene/project a run belongs to. Set by generate() at click time; read
// by run.start() at the moment IT actually queues (see session.js's own
// comment on why this can't be read only after the fact).
let ranFor = null;
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
    ? { funpack_scene_id: ranFor, funpack_project_id: ranForProject } : null),
  onAdopt: (sceneId, projectId) => {
    ranFor = sceneId; ranForProject = projectId;
    // A run this PAGE queued sets ranFor itself, at click time (see generate()
    // below). A run only found on reload -- nobody here called generate() for
    // it -- has no other way to tell store.js which scene it belongs to.
    fire("adopt", { sceneId, projectId });
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
    return () => { listeners[kind] = (listeners[kind] || []).filter((f) => f !== fn); };
  },
  cancel: () => run.cancel(),
  waitForTerminal: () => waitForTerminal(run),
  ready: session.ready,
  slotForRole,
  /** Queue one scene's generation. `inputs` is {slotId: {inputName: value}}. */
  async generate({ sceneId, projectId, inputs } = {}) {
    ranFor = sceneId || null;
    ranForProject = projectId || null;
    pendingInputs = inputs || null;
    return session.generate();
  },
};
