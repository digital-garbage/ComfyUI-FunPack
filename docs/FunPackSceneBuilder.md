# FunPack Scene Builder

`FunPack Scene Builder` stores named scene presets from phrases the user manually selects. Connected positive and negative prompts are memory sources only: every queue run collects their phrases into universal memory, but the node outputs only the selected scene phrases or a saved scene preset.

## Modes

**Manual** outputs the current Scene Builder GUI selection: selected positive phrases, selected negative phrases, scene LoRAs, mode, per-block setting, sigmas, and refinement key.

**Auto** scans `intent_prompt` for a saved scene name or alias. Exact matches are preferred, then a conservative word-match fallback is used. If nothing matches, the node falls back to Manual output.

## Inputs

**scene**: Saved scene dropdown.

**scene_name**: Name used when saving a scene.

**aliases**: Comma-separated trigger names for Auto mode.

**output_mode**: `Manual` or `Auto`.

**intent_prompt**: Text used for Auto scene detection. This can be the same intent prompt that feeds Refiner V2.

**positive_prompt** and **negative_prompt**: Universal memory sources. They do not override scene output.

**mode**: Stored model namespace, currently `ltx2` or `wan`.

**per_block**: Stored LoRA loader per-block preference.

**refinement_key** / **refinement_key_input**: Optional Refiner V2 key. When present, Scene Builder also tries to load matching stored positive conditioning.

**sigmas**: Optional sigma schedule stored with saved scenes.

## Outputs

**positive_prompt**: Selected positive scene phrases joined with commas.

**negative_prompt**: Selected negative scene phrases joined with commas.

**scene_name**: The manual scene name or the Auto-matched saved scene.

**sigmas**: Stored scene sigmas, current connected sigmas, or an empty sigma tensor.

**positive_conditioning**: Best available conditioning from the scene refinement key, if any.

**lora_stack**: A `FUNPACK_LORA_STACK` that can connect directly to `FunPack LoRA Loader`.

**refinement_key**: The scene refinement key.

**status**: Summary of selected mode, matched scene, phrase counts, LoRA count, and conditioning lookup.

## Workflow

1. Connect or type positive and negative prompts, then queue once to collect phrase memory.
2. Select phrase chips from the universal phrase bank.
3. Add scene LoRAs and set mode/per-block/refinement key as needed.
4. Enter a scene name and press **Save**.
5. Switch to **Auto** and write a saved scene name or alias into `intent_prompt` to apply that preset automatically.
