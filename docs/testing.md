# Testing and verification

Every rule in these guides is held by a test or says who checks it. This page shows how to run the tests, how to write a good one, and what counts as proof.

## Python tests (core and modules)

```
pip install -r requirements.txt pytest
COMFYUI_ROOT=/path/to/ComfyUI python -m pytest
```

- `pyproject.toml` sets `--import-mode=importlib` and `testpaths = ["core/tests", "modules"]`. The root `__init__.py` is ComfyUI's entry point and cannot be imported on its own, which is why the mode matters.
- Tests sit beside what they test: `core/tests/` and `modules/<group>/<name>/tests/`.
- `COMFYUI_ROOT` points at a ComfyUI checkout (it must contain `comfy/supported_models.py`). Without it, the default is `~/Documents/ComfyUI`. With none found, tests that need ComfyUI skip, and a module that imports `comfy` at load is simply absent.
- ComfyUI needs `torch`. A CPU build is enough for the tests. If torch complains about CUDA, force the CPU device for the test run rather than editing a test.
- Tests that need `aiohttp` skip themselves with `importorskip` when it is missing.

### Fixtures (`conftest.py` at the repo root)

| Fixture | Gives you |
|---|---|
| `comfyui` | The ComfyUI path on `sys.path`, or skips the test. |
| `tiny_h3` | The real H3 model class, built tiny (random weights, CPU) with a patcher. Use it to prove a modifier or hook runs on the real forward pass. It says nothing about picture quality. |
| `tiny_ltx` | The real LTX model class, built tiny (random weights, CPU), once per run. It proves wiring only; LTX stays UNTESTED on a real model. Clone its patcher; never leave anything installed on the model itself. |
| autouse | A private addons file, a private quarantine file, and block-influence switched off, all in `tmp_path`, so a developer's own settings never change a result. |

`packed_av` and `unpacked` build and undo the flat audio and video packing H3 uses.

### What the suite already enforces

These run with the rest and fail the build if you break them.

| Test | Rule |
|---|---|
| `core/tests/test_module_imports.py` | A module reaches core only through `_core` and never imports another module. |
| `core/tests/test_registry.py`, `test_retired_modules.py` | Announcement, ids, absence on failure. |
| `core/tests/test_shipped_hints.py` | Every shipped setting has a plain one-sentence hint and a unique label. |
| `core/tests/test_loads_as_comfyui_does.py` | The pack loads the way ComfyUI loads it. |
| `core/tests/test_log_call_sites.py` | Log calls use the right level and shape. |
| `core/tests/test_control.py` | Quarantine and `Unavailable` behave as described in [Architecture](architecture.md). |

### Environment-dependent failures

A few tests depend on the machine: a real GPU, a git checkout on a named branch, or a subprocess probe. On a stripped CPU box or a detached checkout they can fail without your change being the cause. Before you blame your change, run the same test on a clean checkout of the base branch. Name any such failure in your pull request instead of hiding it.

## JavaScript tests (the app)

```
cd app
npm install     # dev only: jsdom and Playwright. Users never run npm.
npm test        # node --test --test-timeout=30000
```

- Plain ES modules, no build step, no bundler.
- Tests sit beside the code (`*.test.js`) or in `composer/tests/` and `shell/tests/`.
- `composer/tests/_dom.js` gives a jsdom `window`. It has a URL, because without one `localStorage` throws and anything that remembers a preference silently takes its fallback.
- jsdom is honest about structure and events and dishonest about layout. It has no box model. Anything that depends on measured geometry (placement, scrolling, wrapping) belongs in a real browser: check it in the catalogue, or in a Playwright pass.

### Testing a UI feature

The pattern is in [Writing a UI feature](writing-a-ui-feature.md): build a host, a fake `app`, call `setup`, assert on the handle's node. Also test the hide paths: a missing need, an unoffered mount and a throwing `setup` must each leave nothing drawn.

### Testing a kit element

See [Adding an element](ui-kit-reference.md#adding-an-element). The registry-wide rule tests apply to your element the moment it has a demo.

### `tools/kit_lint`

`scan.js` applies `MODULE_RULES` to module `ui.js` files and `KIT_RULES` to the kit. `rule_no_module_styling.test.js` runs it and also runs the files in `composer/tests/lint_fixtures`, so a rule that stops catching its own bad example fails. It is a discipline check, not a security boundary.

## Looking at it

```
python tools/devserver.py 8199
```

Open `http://localhost:8199/funpack/`. The real routes are served and nothing is queued, so Generate says there is no queue. Set `COMFYUI_ROOT`, or most modules will be absent. The catalogue pages are in [UI kit reference](ui-kit-reference.md#the-catalogue). Check any visual change in dark and light and at a narrow width.

## How to write a test that proves something

- **Use fixtures, not real projects.** Build the smallest input that shows the behaviour. A test must not depend on someone's saved project, models or addons file.
- **Prove the mechanism end to end.** A unit test of a helper does not show the feature works. For a modifier: install it on `tiny_h3` or `tiny_ltx`, run a step, and compare the output with the module off. For a route: call it through the real router. For a UI feature: mount it with the real kit.
- **Attack your own change.** Ask what a hostile or careless caller would do: a missing key, a value outside `min`..`max`, a hook that raises on step 3, a name containing markup, a module that is absent. Write the test for the one most likely to happen.
- **Test the failure path as hard as the happy path.** The promise of this project is that a broken piece is absent and the rest carries on. A test for a new module should show that.
- **Fail first.** Make sure the test fails without your change. A test that passes either way proves nothing.
- **Say what was and was not checked.** A pull request states what was run and what could not be (for example, no GPU, so no real render). Do not describe a probabilistic effect as certain.
- **Real renders are a separate step.** Judging quality needs a real model on a real GPU and a person looking at the result. The unit tests prove the wiring; they do not prove the picture is better.

## Before you push

- [ ] `python -m pytest` and `npm test` pass, or the failures are shown to be unrelated.
- [ ] New code has a test that fails without it.
- [ ] No test depends on your machine's paths, projects or models.
- [ ] No long dashes in new text. No `node_modules` or scratch files committed.
