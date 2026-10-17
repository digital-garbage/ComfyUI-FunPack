"""Block swap: the frozen weights of most transformer blocks wait in system memory and come to the card one layer at a time.

Training a LoRA never changes the base weights, so they can live in (pinned) system memory. ComfyUI's operations already
copy a weight to the card for the moment a layer runs and drop that copy afterwards; they only do it for a layer that is
told to (`comfy_cast_weights`), so this moves the weights and tells every layer in those blocks. Costs speed (each swapped
weight crosses the bus twice per step: forward, then again when checkpointing recomputes the block), gains about a
block's weight in free card memory per swapped block.

The idea, and the number of blocks as the one knob, follow musubi-tuner's block swap for MiniMax H3 training
(kohya-ss, Apache License 2.0). This is an independent implementation on ComfyUI's own weight casting, not its code.
"""

from __future__ import annotations


def _host(tensor, pin: bool):
    out = tensor.detach().to("cpu")
    if pin:
        try:
            return out.pin_memory()
        except RuntimeError:                           # the host will not lock more memory: this weight stays pageable
            return out
    return out


def move(blocks, count: int, torch) -> dict:
    """Put the weights of the last `count` blocks in system memory. -> {"blocks", "bytes"} actually moved.

    A block already moved is left alone, so asking twice is harmless. Trainable weights stay where they are.
    """
    chosen = list(blocks)[max(0, len(blocks) - count):] if count > 0 else []
    moved, size, pin = 0, 0, True
    for block in chosen:
        if getattr(block, "_funpack_swapped", False):
            continue
        for module in block.modules():
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True       # this layer now fetches its weight for the moment it runs
            for name, param in list(module.named_parameters(recurse=False)):
                if param.requires_grad or param.device.type == "cpu":
                    continue
                size += param.numel() * param.element_size()
                param.data = _host(param.data, pin)
                pin = pin and param.data.is_pinned()
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.device.type != "cpu":
                    size += buf.numel() * buf.element_size()
                    module._buffers[name] = _host(buf, pin)
        block._funpack_swapped = True
        moved += 1
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"blocks": moved, "bytes": size}


def still_away(blocks, count: int) -> bool:
    """Whether the first weight of the last swapped block is still in system memory (ComfyUI may move a model back)."""
    for block in list(blocks)[max(0, len(blocks) - count):][-1:]:
        for param in block.parameters():
            return param.device.type == "cpu"
    return False
