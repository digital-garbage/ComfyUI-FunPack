"""Compatibility by trait, never by model name.

A module listing model names has to be edited every time a model ships, and the
one nobody remembers to edit is the one that silently stops appearing. A model
says what it IS; a module says what it NEEDS.

Two sources, and the split matters:

* **Universal facts** are read straight off the loaded model. They need no
  knowledge of any particular model, so this file can produce them for anything
  ComfyUI can load -- including models that did not exist when it was written.

* **Everything else is announced.** Recognising that a model carries an audio
  stream, or exposes a particular block structure, is that model's business, so
  it lives in that model's own module and is contributed through `TRAITS`. Core
  therefore names no model, and supporting a new one is a new folder rather than
  an edit here.

Deliberately NOT asserted: modality. ComfyUI does not classify it -- there is no
such attribute on `supported_models_base.BASE` -- and it cannot be inferred from
the latent, because image models borrow video VAEs (QwenImage, JoyImage, Krea2
and HunyuanImage21Refiner all report latent_dimensions=3 while being image
models). A "video" trait would be FunPack guessing, so the traits below describe
the tensor and the module tree instead: they are true either way.
"""

from typing import Any, Iterable, List, Sequence, Set, Tuple

from . import log
from .contract import ModuleSpec

# How many axes the latent has. A statement about the tensor, not about what the
# model is for -- a model with a length-1 time axis still has one, and a modifier
# that walks it is correct there.
LATENT_RANK = {1: "waveform_latent", 2: "spatial_latent", 3: "temporal_latent"}


def split(specs: Iterable[ModuleSpec], available: Sequence[str]) -> Tuple[List[ModuleSpec], List[ModuleSpec]]:
    """(compatible, incompatible) for the traits currently on offer.

    Narrowing, not opting in: a module that declares no traits is compatible with
    everything. Declaring one is how a module excludes itself from models it
    genuinely cannot work on.
    """
    have: Set[str] = set(available or ())
    compatible, incompatible = [], []
    for spec in specs:
        (compatible if set(spec.requires) <= have else incompatible).append(spec)
    return compatible, incompatible


def missing_for(spec: ModuleSpec, available: Sequence[str]) -> List[str]:
    """Which traits a module wanted and did not get. For the modules dump."""
    return sorted(set(spec.requires) - set(available or ()))


def has_block(model: Any, class_name: str) -> bool:
    """Whether the model's module tree contains a block of this class.

    The structural probe model modules are expected to use. It asks "does this
    model have the thing I touch", which is the question a modifier actually has
    -- unlike "is this H3", it stays true for anything built the same way.

    Pass a DOTTED name to require the defining module too
    (`"comfy.ldm.minimax.model.AdalnProj"`, matched on the tail of the module
    path). A bare name matches on the class name alone, which is fine for
    something distinctive like `AdalnProj` and actively wrong for names as
    common as `Attention`, `MLP` or `Block` -- those collide across unrelated
    architectures, and a false positive here enables a modifier on a model it
    was never written for. When in doubt, qualify.
    """
    root = getattr(getattr(model, "model", None), "diffusion_model", None)
    if root is None:
        return False
    try:
        modules = root.modules()
    except Exception:                            # noqa: BLE001 -- not an nn.Module
        return False

    if "." in class_name:
        wanted_module, _, wanted = class_name.rpartition(".")
    else:
        wanted_module, wanted = "", class_name

    for sub in modules:
        kind = type(sub)
        if kind.__name__ != wanted:
            continue
        if not wanted_module:
            return True
        origin = getattr(kind, "__module__", "") or ""
        if origin == wanted_module or origin.endswith("." + wanted_module):
            return True
    return False


def universal(model: Any) -> Set[str]:
    """Facts readable off any loaded model, with no per-model knowledge."""
    found: Set[str] = set()
    inner = getattr(model, "model", None)
    if inner is None:
        return found

    config = getattr(inner, "model_config", None)
    latent = getattr(config, "latent_format", None)
    rank = getattr(latent, "latent_dimensions", None)
    if rank in LATENT_RANK:
        found.add(LATENT_RANK[rank])

    # Name it rather than deriving a bool, so a module can ask for the thing it
    # actually needs instead of re-deriving the rule.
    if getattr(latent, "temporal_downscale_ratio", 1) not in (None, 1):
        found.add("temporal_compression")

    kind = getattr(inner, "model_type", None)
    name = getattr(kind, "name", None)
    if isinstance(name, str) and name:
        found.add(f"predict_{name.lower()}")

    return found


def contributed(model: Any, specs: Iterable[ModuleSpec]) -> Set[str]:
    """Traits announced by model modules, each guarded on its own.

    A provider that raises contributes nothing and says so once. It must not stop
    the others: one unrecognised model would otherwise disable every module that
    declares any trait at all.
    """
    found: Set[str] = set()
    for spec in specs:
        provider = spec.traits
        if provider is None:
            continue
        try:
            offered = provider(model)
        except Exception as exc:                 # noqa: BLE001
            log.broke(f"{spec.id}.TRAITS", exc, "working out this model's traits")
            continue
        if not offered:
            continue
        for trait in offered:
            if isinstance(trait, str) and trait:
                found.add(trait)
    return found


def traits_of(model: Any, specs: Iterable[ModuleSpec] = ()) -> List[str]:
    """Everything true of this model, universal facts plus what modules asserted.

    An unrecognised model yields whatever CAN be read rather than nothing: an
    empty set would silently disable every module that declares a trait, which
    reads as "nothing is supported" instead of "this is unfamiliar".
    """
    return sorted(universal(model) | contributed(model, list(specs)))
