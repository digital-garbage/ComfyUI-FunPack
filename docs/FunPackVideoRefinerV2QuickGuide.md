# Refiner V2 Quick Guide

`FunPack Video Refiner V2` is a prompt memory node. It takes your positive prompt, encodes it through the connected CLIP, remembers how you rated the previous output, and slowly learns which actions, camera moves, details, quality cues, and style cues work for your workflows.

It does not look at the finished video pixels. It does not magically know what is inside your input image. It does not replace clear prompting. In image-to-video, it now treats character appearance, clothing, subject identity, and background as image-owned by default, so it should not inject old remembered outfits or characters into a new image unless you explicitly prompt them.

## How To Use It

1. Add `FunPack Video Refiner V2` before your sampler where positive conditioning is encoded.
2. Connect your `CLIP`.
3. Type a `refinement_key`, or use `FunPack Refinement Key Loader` and connect its output into `refinement_key_input`.
4. First run only seeds memory. Generate once, look at the result, then rate that result on the next run with the same key.
5. Use `Perfect` when the output is good and should be remembered.
6. Use `Missing action` when the requested motion did not happen.
7. Use `Missing details` when small non-identity details are weak, like smoke, reflections, particles, hand contact, object interaction, or secondary motion.
8. Use `Missing quality` when the idea is right but the result is messy or ugly.
9. Use `Wrong action`, `Wrong details`, or `Wrong details + action` when the video looks usable but did the wrong thing.
10. Use `Wrong appearance` when Refiner/Lucky brought in unwanted clothing, face, body, character, subject, or background memory.
11. Use `-Just forget it-` for broken runs, bad seeds, workflow mistakes, model failures, or anything you do not want learned.

## Lucky Mode

`I'm Feeling Lucky` composes a prompt from learned memory. It is useful when you want the node to suggest motion, camera, detail, quality, or style ideas from previous liked outputs.

Lucky should not auto-add remembered outfits, character traits, subjects, or backgrounds anymore. If you want an appearance item, write it directly in the prompt.

## Examples

If your prompt says `person smoking and exhaling smoke` but the video only shows a puff with no exhale, rate `Missing details` or `Missing action`, depending on what matters most. This teaches the key that smoke/exhale matters in that prompt context, without globally banning smoking.

If you liked `white tights` in older prompts but your new input image is only a portrait face, and Lucky tries to bring tights into the scene, rate `Wrong appearance`. That tells Refiner to suppress that appearance memory from auto-injection while still allowing `white tights` when you explicitly type it.

If the camera should push in, orbit, pan, or change framing and it stays static, rate `Missing action` or `Wrong action`. Camera movement belongs to the action/camera side of memory, so it remains safe for repair and Lucky.

## Key Loader

Use `FunPack Refinement Key Loader` when you want a reusable key dropdown or want to move keys between workflows/machines. Its `Import` and `Export` buttons save/load the key as JSON. Connect the loader to both Refiner V2 and `FunPack Apply LoRA Weights` so LoRA suggestions and prompt memory use the same key.
