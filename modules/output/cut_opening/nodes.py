"""FunPack Cut Opening. See the module docstring."""

from comfy_api.latest import io


class FunPackCutOpening(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackCutOpening",
            display_name="FunPack Cut Opening",
            category="FunPack/Output",
            description="Drop frames off the front of a clip, and the matching stretch of sound.",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("frames", default=0, min=0, max=4096,
                             tooltip="How many frames to cut off the front. 0 = keep everything."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01,
                               tooltip="The clip's frame rate, so the sound is cut by the same time."),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, images, frames: int, fps: float, audio=None) -> io.NodeOutput:
        total = int(images.shape[0])
        # Never an empty clip: that is an obscure crash downstream, where a clip
        # that is visibly too short says what happened.
        cut = max(0, min(int(frames), total - 1))
        if cut == 0:
            return io.NodeOutput(images, audio, "nothing cut")
        status = f"cut {cut} of {total} frames ({cut / fps:.2f}s)"
        if int(frames) > cut:
            status += f"; asked for {int(frames)}, but one frame has to stay"
        if audio is not None:
            wave, rate = audio["waveform"], int(audio["sample_rate"])
            samples = min(int(round(cut / float(fps) * rate)), int(wave.shape[-1]))
            audio = {**audio, "waveform": wave[..., samples:]}
        return io.NodeOutput(images[cut:], audio, status)
