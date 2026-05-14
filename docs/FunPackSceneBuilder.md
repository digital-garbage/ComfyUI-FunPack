# FunPack Scene Builder

`FunPack Scene Builder` stores named scene presets inside the selected refinement key, separate from Refiner's conditioning-delta learning. Connected positive and negative prompts are memory sources only: every queue run collects their phrases and useful words into universal memory, but the node outputs composed scene text or pass-through text in Learning mode.

## Visible Controls

The node surface is a compact launcher. Scene name, mode, aliases, prompt editors, saved scenes, and database controls live in the centered custom editor UI.

If no scene exists yet, opening the editor, prompt editor, or database first asks for a new scene name.

## Editor Menus

**Positive prompt**: Opens a multiline text composer. Typed text, commas, periods, and manual ordering are preserved exactly. Database chips can be clicked or dragged into the text field.

**Negative prompt**: Same composer behavior for the negative prompt.

**Database**: Add, edit, delete, search, categorize, and mark universal words and phrases as wildcards. Double-click a word or phrase to edit it inline, then use **OK** or **Cancel**.

The editor refreshes the selected scene database before opening. When `refinement_key_input` is connected to a Refinement Key Loader, the editor reads that linked key so prompt memory appears in the same database where queue runs stored it.

Each editor menu has **Back**, **Cancel**, and **Confirm** controls. Back always returns to the main editor. Cancel restores the menu snapshot; if the prompt/database editor was opened directly from the node, Cancel closes the dialog instead.

## Modes

**Manual** outputs the current composed positive and negative scene text, plus the connected LoRA stack passed through unchanged.

**Auto** scans `intent_prompt` for a saved scene name or alias. Exact matches are preferred, then a conservative word-match fallback is used. If nothing matches, the node falls back to Manual output.

**Learning** saves connected positive and negative prompt phrases and useful words into Scene Builder universal memory, then passes the connected positive prompt, negative prompt, and LoRA stack through unchanged.

## Connection-Only Inputs

**intent_prompt**: Text used for Auto scene detection. This can be the same intent prompt that feeds Refiner V2.

**positive_prompt** and **negative_prompt**: Universal memory sources. They do not override scene output except in Learning pass-through mode.

**refinement_key_input**: Optional key that owns this Scene Builder memory and saved scenes.

**lora_stack**: Optional current LoRA stack. Scene Builder passes it through unchanged so Refiner can use the active LoRAs for suggestions.

Scene Builder memory is stored in the selected refinement key under its own `scene_builder` section. Refiner reset clears conditioning-delta learning and prompt histories, but preserves Scene Builder universal memory and saved scenes.

## Outputs

**positive_prompt**: Composed positive scene text.

**negative_prompt**: Composed negative scene text.

**lora_stack**: The connected `FUNPACK_LORA_STACK`, passed through unchanged.

**status**: Summary of selected mode, matched scene, phrase counts, and pass-through LoRA stack count.

## Wildcards

Wildcard is a checkbox on each database word or phrase. If two or more wildcard-enabled entries appear next to each other in the runtime composed prompt, Scene Builder outputs one randomly selected entry from that adjacent run and omits the others. The saved prompt text is not rewritten.

## Workflow

1. Connect positive and negative prompt text, then queue once to collect phrase and word memory.
2. Open **Database** to clean up phrases, assign categories, or enable wildcard random choice.
3. Open **Positive prompt** or **Negative prompt** to type freely and insert chips from the database.
4. Enter a scene name and confirm or save the scene.
5. Use **Auto** with a scene name or alias in `intent_prompt` to apply a saved scene automatically.
6. Use **Learning** when generating changing prompts and you want Scene Builder to collect memory while leaving the generation prompt untouched.
