// A render-style job: start it, poll it until it stops. -> the final status ({state, detail, media}).
import { call } from "../../shell/api.js";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export async function runJob(path, body) {
  const { job_id: job } = await call("POST", path, body);
  let state;
  do { await wait(1500); state = await call("GET", `${path}/${job}`); } while (state.state === "queued" || state.state === "running");
  return state;
}
