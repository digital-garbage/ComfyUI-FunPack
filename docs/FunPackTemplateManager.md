# FunPack Template Manager

`FunPack Template Manager` stores reusable generation concept presets. Normal workflow runs are passive: the node loads the selected template and outputs its stored values. Save, Update, Delete, Import, Export, and Refresh are explicit node buttons.

## Inputs

**template**: Saved template dropdown. Refresh updates this list.

**name**: New template name. This is only used by Save when the name is unique. If the name already exists or is empty, Save targets the selected dropdown template.

**mode**: Stored workflow namespace. Refiner V2 conditioning lookup ignores this value, while legacy refiner fallback still uses it.

**wildcard_seed**: Seed for resolving `{option A|option B}` groups in the stored positive prompt. `0` uses random choices on each execution.

**activation_word**: Optional trigger or concept word stored with the template.

**refinement_key**: Optional `FunPack Video Refiner V2` key. When stored, the node outputs this key and tries to load the best saved positive conditioning from the matching refiner state.

**positive_prompt**: Optional prompt stored with the template. Supports wildcard groups such as `{wide shot|close-up}`.

**negative_prompt**: Optional negative prompt stored with the template.

**sigmas**: Optional sigma schedule stored with the template.

**lora_stack**: Optional stack from `FunPack Apply LoRA Weights`.

## Buttons

**Save** creates a new template when `name` is unique. Otherwise, it fully replaces the selected template with the currently provided fields.

**Update** updates only fields currently provided to the node. Existing stored fields not provided in this run are left untouched.

**Delete** removes the selected template.

**Export** downloads all templates as `funpack_templates.json`.

**Import** loads a previously exported JSON file and merges its templates into the local store. Templates with the same name are replaced.

**Refresh** reloads the template dropdown. ComfyUI's regular refresh also updates the list.

## Outputs

**positive_prompt**: Stored positive prompt with wildcard groups resolved.

**negative_prompt**: Stored negative prompt.

**activation_word**: Stored activation word, or an empty string.

**sigmas**: Stored sigmas, or an empty sigma tensor.

**positive_conditioning**: Best available conditioning from the stored refinement key. The node prefers the refiner's liked average when present, then the best-rated history item, then the stored reference.

**lora_stack**: Stored FunPack LoRA stack, or no stack.

**refinement_key**: Stored refinement key, or an empty string. If no key is stored, conditioning is not loaded.

**status**: Summary of the selected action, stored fields, wildcard seed, and refiner-conditioning lookup.

## Notes

Save is the way to intentionally remove fields from an existing template: leave unwanted fields empty or disconnected, then press Save on the selected template.

Update is additive and conservative. Empty strings and disconnected optional inputs are ignored so existing template data is not accidentally cleared.
