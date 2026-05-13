# FunPack Scene Builder

`FunPack Scene Builder` stores named scene presets from phrases the user manually selects. Connected positive and negative prompts are memory sources only: every queue run collects their phrases into universal memory, but the node outputs only the selected scene phrases or a saved scene preset.

## Modes

**Manual** outputs the current Scene Builder GUI selection: selected positive phrases, selected negative phrases, sigmas, refinement key, and the connected LoRA stack passed through unchanged.

**Auto** scans `intent_prompt` for a saved scene name or alias. Exact matches are preferred, then a conservative word-match fallback is used. If nothing matches, the node falls back to Manual output.

## Inputs

**scene**: Saved scene dropdown.

**scene_name**: Name used when saving a scene.

**aliases**: Comma-separated trigger names for Auto mode.

**output_mode**: `Manual` or `Auto`.

**intent_prompt**: Connection-only text used for Auto scene detection. This can be the same intent prompt that feeds Refiner V2.

**positive_prompt** and **negative_prompt**: Connection-only universal memory sources. They do not override scene output and do not appear as editable text fields on the node.

**refinement_key** / **refinement_key_input**: Optional key passed through with the scene.

**sigmas**: Optional sigma schedule stored with saved scenes.

**lora_stack**: Optional current LoRA stack. Scene Builder passes it through unchanged so Refiner can use the active LoRAs for suggestions.

## Outputs

**positive_prompt**: Selected positive scene phrases joined with commas.

**negative_prompt**: Selected negative scene phrases joined with commas.

**scene_name**: The manual scene name or the Auto-matched saved scene.

**sigmas**: Stored scene sigmas, current connected sigmas, or an empty sigma tensor.

**lora_stack**: The connected `FUNPACK_LORA_STACK`, passed through unchanged.

**refinement_key**: The scene refinement key.

**status**: Summary of selected mode, matched scene, phrase counts, and pass-through LoRA stack count.

## Workflow

1. Connect positive and negative prompt text, then queue once to collect phrase memory.
2. Select phrase chips from the universal phrase bank.
3. Connect the current LoRA stack if Refiner should receive active LoRAs for suggestions.
4. Enter a scene name and press **Save**.
5. Switch to **Auto** and write a saved scene name or alias into `intent_prompt` to apply that preset automatically.
