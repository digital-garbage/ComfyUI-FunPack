# FunPack Save Refinement Latent

This node saves a latent tensor bundle under a refinement key so `FunPack Video Refiner` can use it as an optional latent refinement reference on later runs.

## Parameters

**latent**: Latent tensor bundle to save.

**refinement_key**: Name of the refinement session. Use the same key in `FunPack Video Refiner`.

**mode**: Tokenizer mode namespace. Use the same mode as `FunPack Video Refiner`.

## Outputs

**latent**: Passthrough latent.

**status**: Save status with the stored tensor shape.

## Notes

The saved latent lives in FunPack's local `refinements` folder as a PyTorch tensor file. The refiner only uses it when both the key and mode match and the `refined_latent` output is connected. If the refiner receives a latent and no saved reference exists, it saves the incoming latent automatically. Zero-valued latent positions are treated as intentional and are not refined.
