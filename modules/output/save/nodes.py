"""Save a video as MP4 into ComfyUI's output folder, fast: H.264 on the GPU's encoder by default.

Core's SaveVideo encodes with libx264 through PyAV, and FFmpeg's library default is ONE thread -- on a
rental's CPU that is where a run's last half-minute went. Here the GPU's own encoder (NVENC) does it when
this PyAV build and the card have one. H.264 by default: it plays in every browser, Firefox included, and
NVENC makes it as fast as H.265. H.265 (smaller files, tagged hvc1) is an option for Chrome/Safari. With no
NVENC it is libx264 on every core: CPU H.265 is several times slower. Same names and metadata as SaveVideo. Takes the frames and sound directly: core's
CreateVideo step in between only bundled them.
"""

import json
import os
import time
from fractions import Fraction

import av
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io, ui

from ..._core import log

ENCODERS = ("auto", "gpu", "cpu")
CODECS = ("h264", "h265")
# NVENC's constant-quality mode at libx264's default number (crf 23).
NVENC = {"h265": ("hevc_nvenc", {"preset": "p5", "rc": "vbr", "cq": "23", "b:v": "0"}),
         "h264": ("h264_nvenc", {"preset": "p5", "rc": "vbr", "cq": "23", "b:v": "0"})}
X264 = ("libx264", {"crf": "23"})


def _write(path, frames, fps, audio, metadata, codec):
    name, options = codec
    # use_metadata_tags keeps the workflow in the file; faststart puts the index first, which the preview player needs to seek
    with av.open(path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as out:
        for key, value in (metadata or {}).items():
            out.metadata[key] = json.dumps(value)
        stream = out.add_stream(name, rate=Fraction(round(fps * 1000), 1000))
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = options
        stream.thread_type, stream.codec_context.thread_count = "AUTO", 0       # every core (FFmpeg's default is one)
        if "hevc" in name or "265" in name:
            stream.codec_context.codec_tag = "hvc1"            # hev1 (the default) is refused by Safari and Chrome on a Mac
        sound = None
        if audio is not None:                                  # both tracks before the first write: the header is fixed then
            rate, wave = int(audio["sample_rate"]), audio["waveform"]
            wave = wave[0, :, :int(rate * len(frames) / fps + 0.999)].float().cpu().contiguous().numpy()
            layout = {1: "mono", 2: "stereo"}.get(wave.shape[0], "stereo")
            sound = out.add_stream("aac", rate=rate, layout=layout)
        for frame in frames:
            out.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv420p")))
        out.mux(stream.encode(None))
        if sound is not None:
            chunk = av.AudioFrame.from_ndarray(wave, format="fltp", layout=layout)
            chunk.sample_rate, chunk.pts = rate, 0
            out.mux(sound.encode(chunk))
            out.mux(sound.encode(None))


def encode(path, frames, fps, audio=None, metadata=None, encoder="auto", codec="h264"):
    """-> which encoder wrote the file. `frames`: uint8 [T, H, W, 3]. GPU first unless told otherwise;
    a GPU attempt that fails leaves no half file and falls back to CPU H.264, unless "gpu" was demanded."""
    gpu = NVENC[codec]
    if encoder != "cpu" and gpu[0] in av.codecs_available:
        try:
            _write(path, frames, fps, audio, metadata, gpu)
            return f"{codec.upper()} on NVENC (GPU)"
        except Exception as exc:                               # noqa: BLE001 -- no encoder on this card, or a driver refusal
            if os.path.exists(path):
                os.remove(path)
            if encoder == "gpu":
                raise RuntimeError(f"the GPU encoder (NVENC) failed: {exc}") from exc
            log.once("save_nvenc", log.ALERT, "FunPack Save Video", f"GPU encoder unavailable ({exc}); using H.264 on the CPU")
    elif encoder == "gpu":
        raise RuntimeError(f"this PyAV build has no NVENC encoder ({gpu[0]}); pick auto or cpu")
    _write(path, frames, fps, audio, metadata, X264)
    if encoder == "auto":
        log.once("save_no_nvenc", log.ALERT, "FunPack Save Video", f"no NVENC in this PyAV build ({gpu[0]} missing): saved on the CPU")
    return "H.264 on libx264 (CPU, all cores)"


class FunPackSaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackSaveVideo",
            display_name="FunPack Save Video",
            category="FunPack/Output",
            description="Save as MP4 (H.264): on the GPU's encoder when there is one, otherwise on every CPU core.",
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio", optional=True),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01),
                io.String.Input("filename_prefix", default="FunPack"),
                io.Combo.Input("encoder", options=list(ENCODERS), default="auto", optional=True,
                               tooltip="auto: GPU (NVENC) if available, else H.264 on the CPU. gpu: fail rather than use the CPU. cpu: H.264 (libx264) on every core."),
                io.Combo.Input("codec", options=list(CODECS), default="h264", optional=True,
                               tooltip="For the GPU encoder. H.264: plays in every browser. H.265: smaller files, but Firefox may not play it."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
        )

    @classmethod
    def execute(cls, images, fps=24.0, filename_prefix="FunPack", audio=None, encoder="auto", codec="h264") -> io.NodeOutput:
        frames = (images * 255).clamp(0, 255).byte().cpu().numpy()
        width, height = frames.shape[2], frames.shape[1]
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        metadata = None
        if not args.disable_metadata:
            metadata = {**(cls.hidden.extra_pnginfo or {}), **({"prompt": cls.hidden.prompt} if cls.hidden.prompt is not None else {})} or None
        file = f"{filename}_{counter:05}_.mp4"
        t = time.perf_counter()
        used = encode(os.path.join(folder, file), frames, float(fps), audio, metadata, encoder, codec)
        log.info("FunPack Save Video", f"{file}: {len(frames)} frames in {time.perf_counter() - t:.1f}s with {used}")
        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
