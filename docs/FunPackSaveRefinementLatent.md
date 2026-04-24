# FunPack Save Refinement Latent

This node saves a video latent under a refinement key so `FunPack Video Refiner` can use it as a reference on later runs.

## Parameters

**latent**: Plain video latent to save. In LTX audio/video workflows, connect the video output from `LTXVSeparateAVLatent`, not the audio latent or combined AV latent.

**refinement_key**: Name of the refinement session. Use the same key in `FunPack Video Refiner`.

**mode**: Tokenizer mode namespace. Use the same mode as `FunPack Video Refiner`.

## Outputs

**latent**: Passthrough latent.

**status**: Save status with the stored tensor shape.

## Notes

The saved latent lives in FunPack's local `refinements` folder as a PyTorch tensor file. The refiner only uses it when the key and mode match and the `refined_latent` output is connected. If the refiner receives a latent and no saved reference exists, it saves the incoming latent automatically.

Use only the video latent side, then reconnect the refined result to `LTXVConcatAVLatent` as `video_latent`. If a separated LTX video latent still reports `type: audio`, it is accepted when its `samples` tensor has the 5D video latent shape.

## Common Mistakes

- Do not connect the sampler's combined AV latent directly to this node.
- Do not connect the audio output from `LTXVSeparateAVLatent`.
- Do not connect the output of `LTXVConcatAVLatent` back into this node.
- Do not expect a saved latent to apply across different `refinement_key` or `mode` values.
- A saved latent does not resize a new generation. If the shape changes, `FunPack Video Refiner` replaces the reference and passes the current latent through unchanged.
- This is not a general latent cache. It stores the reference used by the Video Refiner and may be replaced by reset/session behavior.
