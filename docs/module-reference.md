# Module reference

Facts for module authors, in tables. For the how and why, read [Writing a module](writing-a-module.md). Everything here is read from `core/contract.py`, `core/schema.py`, `core/registry.py` and the shipped modules. If this page and the code ever disagree, the code wins: please fix the page.

## Announcement constants

Read by `core/registry.py` from the module's `__init__.py` (the `FIELDS` table). A constant the registry does not know is ignored.

| Constant | Python type | Default | Rule |
|---|---|---|---|
| `ID` | `str` | none | Required, non-empty. Unique across modules (a second claimant is left out and said). Keep it free of `.`: a fault is traced back to a module from its hook key `funpack.<id>`. |
| `TITLE` | `str` | none | Required, non-empty. |
| `MOUNT` | `str` | `""` | Required and non-empty when the module has `SETTINGS`. |
| `SETTINGS` | `dict` | `{}` | `{key: setting}`. Must be a dict. |
| `STAGE` | `str` | `"sampling"` | One of `load`, `conditioning`, `latent`, `guidance`, `sampling`, `post`. |
| `CATEGORY` | `str` | `""` | `""` or one of `continuity`, `guidance`, `conditioning`, `sampling`, `post`, `system`. Empty shows under "Other". |
| `STATUS` | `str` | `"experimental"` | `proven` or `experimental`. |
| `REQUIRES` | `list[str]` | `[]` | Model traits that must all be present. |
| `USES` | `list[str]` | `[]` | Capabilities this module asks other modules for. |
| `USES_WHEN` | `dict` | `{}` | Same shape as `when`: own settings under which it asks. Keys must be its own settings. |
| `ROLES` | `list[str]` | `[]` | Pipeline role `at` names it acts through. Inert in a pipeline where none is filled. |
| `AFTER`, `BEFORE` | `list[str]` | `[]` | Module ids. Order within a stage. |
| `HOOKS` | `list[str]` | `[]` | Hook points a sampler must offer for a `sampler_modifier`. |
| `NODES` | `list[type]` | `[]` | Classes with `define_schema` and `execute` (V3 `io.ComfyNode`). |
| `TRAITS` | callable | none | `traits(model) -> iterable[str]`. |
| `PROVIDES` | `dict[str, callable]` | `{}` | Capability name to function. Names must be non-empty strings. |

`CONTRACT_VERSION` in `core/contract.py` is `1`. It is sent to the browser as `contract` in `/funpack/api/modules`. Core does not read a version from modules, so there is nothing for a module to declare.

## Setting keys

| Key | Applies to | Notes |
|---|---|---|
| `type` | all | `bool`, `int`, `float`, `enum`, `text`, `multiline`, `path`, `color`. |
| `default` | all | Required. `bool` is a bool; `int` is an int (not a float); `float`/`int` are finite and inside `min`..`max`; `enum` is one of the options; `color` is `#rgb`, `#rrggbb` or `#rrggbbaa`; the rest are strings. |
| `min`, `max` | `int`, `float` | Finite numbers, `min <= max`. A value outside is refused at run time. |
| `step` | `int`, `float` | Positive. |
| `options` | `enum` | Non-empty list of `{"value": scalar, "label": str}`. No repeated values. |
| `label` | all | String. Unique within the module (test). |
| `hint` | all | String. Required by test, one plain sentence. |
| `unit` | numbers | String. Shown after the hint. |
| `placeholder` | text types | String. Used by the panel renderer, not by the Engine page. |
| `ui` | all | A renderer allowed for the type (next table). |
| `when` | all | `{sibling_key: value or [values]}`. Row shown only while every condition holds. Sibling must exist; an enum value must be one of its options; no self-reference. |

The panel renderer also reads `precision`, `presets` (for `macroSlider`) and an option's `icon` and `hint`. The schema does not check them.

### Renderers

Allowed `ui` per type (the first is the default for `composer/panel.js`):

| Type | Allowed `ui` |
|---|---|
| `bool` | `checkboxRow`, `toggle` |
| `int`, `float` | `number`, `slider`, `stepper`, `macroSlider` |
| `enum` | `select`, `segmented`, `radioGroup`, `filterList`, `wheel` |
| `text` | `input`, `search` |
| `multiline` | `textarea`, `autoTextarea` |
| `path` | `filterList` |
| `color` | `swatch` |

What **Settings > Engine** and the **Sampler** window draw (`features/settings/field.js`), whatever `ui` says:

| Type | Drawn as |
|---|---|
| `bool` | toggle with label and hint |
| `int`, `float` | slider if `ui` is absent or `slider` and both `min` and `max` are set, else a number box |
| `enum` | select |
| `color` | swatch |
| `multiline` | text area (3 rows) |
| `text`, `path` | text box (commits on blur or Enter) |

## Stages and order

`STAGE` order is the list above. Within a stage modules run in id order unless `AFTER`/`BEFORE` say otherwise. Shipped modules today: `load` (loaders, models, pipeline, taste), `conditioning`, `latent`, `guidance` (taste-steered techniques), `sampling`, `post`.

## Categories

Used only to group settings in Settings > Engine and to pick what the Sampler window shows (`sampling`). Order on the page: Continuity, Guidance, Conditioning, Sampling, Post, System, Other.

## Pipeline roles

A slot in a pipeline lists `roles`: `{"at": ..., "input": ..., "label": ..., "drives": ...}`. `at` is where the input appears in the app. Names in use:

| `at` | Shown as |
|---|---|
| `generation.prompt` | the prompt |
| `project.negative` | the project's negative prompt |
| `project.video` | width, height, length, frame rate (`drives`: `frames` or `fps`) |
| `generation.sampling` | steps, sampler, scheduler (Sampler window) |
| `generation.seed` | the seed |
| `generation.model` | the model choice |
| `assets.source_image` | the start picture (what `continuity` needs) |

A role naming a place the app does not offer is simply not shown.

## Traits

Read off any loaded model by core (`traits.universal`):

| Trait | True when |
|---|---|
| `temporal_latent`, `spatial_latent`, `waveform_latent` | the latent has 3, 2 or 1 axes |
| `temporal_compression` | the latent format compresses time |
| `predict_<type>` | the model type, lowercased (`predict_eps`, `predict_flow`, ...) |

Announced by model modules: `audio_stream` (H3, LTX), `dit_block_hooks` (H3: honours block replacement and routes attention through the override), `adaln_modalities` (H3 with the AdaLN projection), `ltx_av` (LTX audio+video). Add new ones in the model's own module. Modality is deliberately not a trait.

## Capabilities

A capability is a name in `PROVIDES`. Core defines none: the contract is between the provider and the askers. All of these are in use today. `None` as an answer from a model-level provider means "not my model"; a raise means "it was mine and it broke".

### Plugging into a run

| Name | Provided by | Signature | Called by |
|---|---|---|---|
| `modifier` | most sampling and guidance modules | `install(patcher, values, key) -> str or None` | `FunPack Load Modifiers` |
| `sampler_modifier` | none yet | `make(values, accepts) -> object with active(ctx), process(ctx, latent)` or `None` | `FunPack Sampler` via `core/chain.py` |
| `settings_sink` | `sampling_modifiers` | `() -> {"node": node id, "input": input name}` | the manifest; the app writes settings there |
| `cache_key` | `phrase_probe` | `() -> value` that changes with its switch | `FunPack Load Modifiers` cache key |
| `guards_off` | `full_control` | `(settings) -> bool` | modifier nodes and the sampler |
| `routes` | several | `routes(table, base, web)` | `core/routes.py`, base `/funpack/api/m/<id>` |

### Pipelines

| Name | Provided by | Signature |
|---|---|---|
| `default_pipeline` | `pipeline` | `() -> list of slots` |
| `pipeline_presets` | each model module | `() -> [{"id", "title", "slots"}]` |
| `optional_nodes` | H3, LTX | `() -> {node class: {"pass": {output index: input name}}}` slots that may be skipped when their module is absent |
| `upgrade_slots` | `output_save` | `(slots, outputs) -> slots` migrates old saved pipelines |

### Knowing a model

| Name | Signature | Notes |
|---|---|---|
| `detect` | `(keys) -> bool` | keys are tensor names from a safetensors header. Never load to answer. |
| `probe_traits` | `(keys) -> list[str]` | traits knowable from the header alone |
| `empty_latent` | `(model, width, height, length, batch_size=1) -> {"samples": ...}` or `None` | claim by identity |
| `decode` | `(latent, model=None, vae=None, audio_vae=None, tile_size=0) -> result` or `None` | claim by identity |
| `video_stream` | `(x, named) -> (video, rebuild)` or `None` | the picture inside a packed latent; `core/streams.video_of` asks |
| `text_rows` | `(model_kwargs, length) -> bool mask` or `None` | which conditioning rows are the prompt's words |
| `text_tokenizer` | `(clip) -> tokenizer` or `None` | a tokenizer that returns offsets |
| `prompt_rows` | `(meta, cond_len, prompt_tokens) -> (start, end)` or `None` | |
| `video_time_rows` | `(samples, t0, t1) -> (rows, start, end)` or `None` | window in seconds |
| `target_rows` | `(mod_segments, seq_len, device, stream) -> bool mask` or `None` | `stream` is `"video"` or `"audio"` |
| `query_rotation` | `(rope_freqs, seq_len) -> rotate(x, rows, inverse=False)` or `None` | |
| `anchor_pin` | `(kwargs, transform) -> kwargs` or `None` | |
| `rescale_conditioning` | `(conditioning, height, width) -> (conditioning, count)` or `None` | |
| `latent_upscaler` | `(state_dict) -> object with funpack_latent_upscale(x, scale)` or `None` | |

### Learning

| Name | Provided by | Signature |
|---|---|---|
| `taste_store` | `taste` | `(patcher) -> handle` with `collect`, `rows`, `capture`, `direction`, `directions`, `describe`; `None` without a key |
| `taste_kind` | `taste` | `(key, kind) -> Kind` for one kind of one key |
| `taste_latest_key` | `taste` | `() -> the key the most recent capture went to, or None` |
| `on_rating` | `taste` calls every module that provides it | `(prompt_id, rating, axis) -> None` |

## Helpers a module may import from `_core`

`from ..._core import ...` offers exactly these: `addons`, `chain`, `config`, `contract`, `control`, `dit_hooks`, `input_steer`, `log`, `media`, `patching`, `projects`, `rated_dial`, `registry`, `relations`, `run`, `schema`, `shortcuts`, `streams`, `temp_files`, `traits`, `vae_tiles`. Adding one means editing `modules/_core.py` in both of its branches.

Most used:

| Helper | For |
|---|---|
| `log` | `info`, `alert`, `warning`, `critical`, `once`, `broke`, `failed` |
| `registry` | `registry.current().ask(...)`, `.providers(...)`, `.specs` |
| `patching` | `tag`, `strip`, `guard`, `Dropped` |
| `dit_hooks` | `add_block_hook`, `add_attention_override`, `current_step`, `late_half`, `last_step`, `probing`, `weak_branch`, `parse_blocks`, `block_count`, `publish` |
| `streams` | `video_of`, `av_video_stream`, `model_args`, `with_options` |
| `traits` | `has_block`, `traits_of` |
| `control` | `Unavailable` |

## Served paths

| Path | What |
|---|---|
| `/funpack/` | the app (`index.html`) |
| `/funpack/api/modules` | the manifest (`?traits=a,b` narrows to a model) |
| `/funpack/api/m/<id>/...` | a module's own routes |
| `/funpack/app/...` | the app. Allowed types: `.js .css .html .woff2 .svg`. |
