// Reattaching to a run that outlived the page.

import test from "node:test";
import assert from "node:assert/strict";

import { runningFor, finishedFor } from "../client.js";

const answering = (body, ok = true) => async () => ({
  ok, status: ok ? 200 : 500, json: async () => body,
});

test("the run this browser queued is found by its client id", async () => {
  const promptId = await runningFor("me", {
    fetch: answering({
      queue_running: [
        [1, "someone-elses", {}, { client_id: "them" }, []],
        [2, "mine", {}, { client_id: "me" }, []],
      ],
    }),
  });
  assert.equal(promptId, "mine");
});

test("another browser's run is not adopted", async () => {
  const promptId = await runningFor("me", {
    fetch: answering({ queue_running: [[1, "theirs", {}, { client_id: "them" }, []]] }),
  });
  assert.equal(promptId, null);
});

test("nothing running means nothing to reattach to", async () => {
  assert.equal(await runningFor("me", { fetch: answering({ queue_running: [] }) }), null);
});

test("a queue that cannot be reached is not an error the app has to show", async () => {
  // The dev server has no queue behind it, which is the ordinary case while
  // working on the UI.
  const promptId = await runningFor("me", {
    fetch: async () => { throw new TypeError("Failed to fetch"); },
  });
  assert.equal(promptId, null);
});

test("a malformed queue entry is skipped rather than crashing the load", async () => {
  const promptId = await runningFor("me", {
    fetch: answering({ queue_running: [null, "nonsense", [1], [2, "mine", {}, { client_id: "me" }, []]] }),
  });
  assert.equal(promptId, "mine");
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
