// Reattaching to a run that outlived the page.

import test from "node:test";
import assert from "node:assert/strict";

import { queuedFor, finishedFor } from "../client.js";

const answering = (body, ok = true) => async () => ({
  ok, status: ok ? 200 : 500, json: async () => body,
});

test("the run this browser queued is found by its client id", async () => {
  const found = await queuedFor("me", {
    fetch: answering({
      queue_running: [
        [1, "someone-elses", {}, { client_id: "them" }, []],
        [2, "mine", {}, { client_id: "me" }, []],
      ],
      queue_pending: [],
    }),
  });
  assert.deepEqual(found, { promptId: "mine", running: true, sceneId: null, projectId: null });
});

test("another browser's run is not adopted", async () => {
  const promptId = await queuedFor("me", {
    fetch: answering({
      queue_running: [[1, "theirs", {}, { client_id: "them" }, []]],
      queue_pending: [[2, "also-theirs", {}, { client_id: "them" }, []]],
    }),
  });
  assert.equal(promptId, null);
});

test("nothing running means nothing to reattach to", async () => {
  assert.equal(await queuedFor("me", {
    fetch: answering({ queue_running: [], queue_pending: [] }),
  }), null);
});

test("a queue that cannot be reached is not an error the app has to show", async () => {
  // The dev server has no queue behind it, which is the ordinary case while
  // working on the UI.
  const promptId = await queuedFor("me", {
    fetch: async () => { throw new TypeError("Failed to fetch"); },
  });
  assert.equal(promptId, null);
});

test("a malformed queue entry is skipped rather than crashing the load", async () => {
  const found = await queuedFor("me", {
    fetch: answering({
      queue_running: [null, "nonsense", [1], [2, "mine", {}, { client_id: "me" }, []]],
      queue_pending: [undefined, {}, [3, null, {}, { client_id: "me" }, []]],
    }),
  });
  assert.deepEqual(found, { promptId: "mine", running: true, sceneId: null, projectId: null });
});

test("where a run belongs travels with it, when it said so at queue time", async () => {
  const found = await queuedFor("me", {
    fetch: answering({
      queue_running: [[1, "mine", {}, { client_id: "me", funpack_scene_id: "s1",
                                        funpack_project_id: "p1" }, []]],
      queue_pending: [],
    }),
  });
  assert.deepEqual(found, { promptId: "mine", running: true, sceneId: "s1", projectId: "p1" });
});

test("a run queued before this existed has nowhere it said it belongs, and says so with null", async () => {
  const found = await queuedFor("me", {
    fetch: answering({
      queue_running: [[1, "mine", {}, { client_id: "me" }, []]],
      queue_pending: [],
    }),
  });
  assert.equal(found.sceneId, null);
  assert.equal(found.projectId, null);
});

test("a finished run is recognised as this browser's from its history entry", async () => {
  const asked = [];
  const promptId = await finishedFor("me", ["theirs", "mine"], {
    fetch: async (url) => {
      asked.push(url);
      const id = url.split("/").pop();
      const owner = id === "mine" ? "me" : "them";
      return { ok: true, status: 200, json: async () => ({
        [id]: { prompt: [1, id, {}, { client_id: owner }, []], outputs: {}, status: {} },
      }) };
    },
  });
  assert.equal(promptId, "mine");
  // Newest first: the run that just ended is the one being looked for.
  assert.equal(asked[0], "/history/mine");
});

test("onFound hands back the extra_data of a match, without changing what this returns", async () => {
  const found = [];
  const promptId = await finishedFor("me", ["mine"], {
    fetch: async () => ({ ok: true, status: 200, json: async () => ({
      mine: { prompt: [1, "mine", {}, { client_id: "me", funpack_scene_id: "s1",
                                        funpack_project_id: "p1" }, []] },
    }) }),
    onFound: (extra) => found.push(extra),
  });
  assert.equal(promptId, "mine");
  assert.deepEqual(found, [{ client_id: "me", funpack_scene_id: "s1", funpack_project_id: "p1" }]);
});

test("onFound is never called when nothing matches", async () => {
  let called = false;
  await finishedFor("me", ["theirs"], {
    fetch: async () => ({ ok: true, status: 200, json: async () => ({
      theirs: { prompt: [1, "theirs", {}, { client_id: "them" }, []] },
    }) }),
    onFound: () => { called = true; },
  });
  assert.equal(called, false);
});

test("a finished run belonging to someone else is not adopted", async () => {
  const promptId = await finishedFor("me", ["theirs"], {
    fetch: async (url) => ({ ok: true, status: 200, json: async () => ({
      theirs: { prompt: [1, "theirs", {}, { client_id: "them" }, []] },
    }) }),
  });
  assert.equal(promptId, null);
});

test("a history entry with nothing in it is skipped, not trusted", async () => {
  for (const body of [{}, { mine: {} }, { mine: { prompt: [] } }, { mine: { prompt: [1, "mine", {}] } }]) {
    const promptId = await finishedFor("me", ["mine"], {
      fetch: async () => ({ ok: true, status: 200, json: async () => body }),
    });
    assert.equal(promptId, null, JSON.stringify(body));
  }
});

test("no history to ask is not an error", async () => {
  const promptId = await finishedFor("me", ["mine"], {
    fetch: async () => { throw new TypeError("Failed to fetch"); },
  });
  assert.equal(promptId, null);
});

test("a run still waiting its turn is found too", async () => {
  // /queue answers with both halves, and a job sits in the pending one from the
  // moment /prompt returns until the worker picks it up -- always briefly, and
  // for as long as it takes whenever something is running ahead of it. A reload
  // in that window found nothing and let the same job be queued twice.
  const promptId = await queuedFor("me", {
    fetch: answering({
      queue_running: [[1, "theirs", {}, { client_id: "them" }, []]],
      queue_pending: [[2, "mine", {}, { client_id: "me" }, []]],
    }),
  });
  assert.deepEqual(promptId, { promptId: "mine", running: false, sceneId: null, projectId: null },
    "a job waiting its turn was reported as one already under way");
});

test("a run under way is preferred over one still waiting", async () => {
  const promptId = await queuedFor("me", {
    fetch: answering({
      queue_running: [[1, "under-way", {}, { client_id: "me" }, []]],
      queue_pending: [[2, "waiting", {}, { client_id: "me" }, []]],
    }),
  });
  assert.deepEqual(promptId, { promptId: "under-way", running: true, sceneId: null, projectId: null });
});

test("somebody else's pending run is not adopted either", async () => {
  const promptId = await queuedFor("me", {
    fetch: answering({
      queue_running: [],
      queue_pending: [[1, "theirs", {}, { client_id: "them" }, []]],
    }),
  });
  assert.equal(promptId, null);
});
