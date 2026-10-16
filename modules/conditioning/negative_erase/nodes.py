from comfy_api.latest import io

from ..._core import log
from . import erase


class FunPackNegativeErase(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackNegativeErase",
            display_name="FunPack Negative Erase",
            category="FunPack/Conditioning",
            description="Give the negative prompt a job at CFG 1: take its direction out of the positive "
                        "prompt's words. Strength 0 passes the positive through untouched.",
            inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                    io.Float.Input("strength", default=0.0, min=0.0, max=2.0, step=0.05,
                                   tooltip="0 = off. How much of the negative prompt is taken out of the positive prompt's words (1 = all of it)."),
                    io.Combo.Input("mode", options=list(erase.MODES), default="project",
                                   tooltip="project: removes the part of each word that lies along the negative's direction. subtract: pulls every word away from the negative by a share of its own size."),
                    io.Boolean.Input("keep_size", default=True,
                                     tooltip="Put each word back on its original size, changing its direction only.")],
            outputs=[io.Conditioning.Output(display_name="positive"), io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, positive, negative, strength: float, mode: str, keep_size: bool = True) -> io.NodeOutput:
        out, note = erase.apply(positive, negative, strength, mode, keep_size)
        if note != "off":
            log.once(f"negative_erase:{note}", log.ALERT, "FunPack Negative Erase", note)
        return io.NodeOutput(out, note)
