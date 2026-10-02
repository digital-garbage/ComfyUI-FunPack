"""Post-render upscale: a finished video through a models/upscale_models model.

Frames stream through ffmpeg in small chunks (never the whole clip in RAM), each chunk
goes through ComfyUI's own ImageUpscaleWithModel (tiling, OOM back-off, device handling),
and the result is encoded with the source's frame rate and sound.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from . import files

CHUNK = 8                # frames per upscale call
SUBFOLDER = "funpack_upscaled"


def probe(path):
    """-> (width, height, fps string, has sound)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise files.ClipError("ffprobe was not found on PATH: install ffmpeg to upscale videos.")
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout).get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise files.ClipError(f"no video stream in {os.path.basename(path)}")
    return (int(video["width"]), int(video["height"]), video.get("r_frame_rate") or "25",
            any(s.get("codec_type") == "audio" for s in streams))


def _read(stream, n):
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            break
        buf += part
    return bytes(buf)


def upscale_file(src, dst, upscale, interrupted=lambda: None):
    """Stream `src` through `upscale` ([B,H,W,3] float 0-1 -> same, larger) into `dst`.
    Returns the frames written. `interrupted()` raises to stop between chunks."""
    import numpy as np
    import torch
    w, h, fps, has_sound = probe(src)
    frame = w * h * 3
    ffmpeg = files.ffmpeg()
    dec = subprocess.Popen([ffmpeg, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc, frames = None, 0
    try:
        while True:
            interrupted()
            raw = _read(dec.stdout, frame * CHUNK)
            n = len(raw) // frame
            if n == 0:
                break
            x = torch.from_numpy(np.frombuffer(raw[:n * frame], dtype=np.uint8)
                                 .reshape(n, h, w, 3).copy()).float() / 255.0
            y = (upscale(x).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            if enc is None:
                oh, ow = y.shape[1], y.shape[2]
                cmd = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{ow}x{oh}", "-r", fps, "-i", "-"]
                if has_sound:
                    cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
                cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
                enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            enc.stdin.write(y.tobytes())
            frames += n
        if enc is None:
            raise files.ClipError(f"no frames could be read from {os.path.basename(src)}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise files.ClipError(f"ffmpeg could not encode the upscaled video: {err[-500:]}")
        return frames
    except BaseException:
        for p in (dec, enc):
            if p is not None and p.poll() is None:
                p.kill()
            if p is not None:
                p.wait()
                for pipe in (p.stdin, p.stdout, p.stderr):
                    try:
                        if pipe:
                            pipe.close()
                    except OSError:                        # a killed encoder's unflushed pipe
                        pass
        if os.path.exists(dst):
            os.remove(dst)
        raise
    finally:
        if dec.poll() is None:
            dec.kill()


def free_name(out_dir, filename, model):
    stem = os.path.splitext(os.path.basename(filename))[0]
    tag = os.path.splitext(os.path.basename(model))[0]
    n, name = 0, f"{stem}_{tag}.mp4"
    while os.path.exists(os.path.join(out_dir, name)):
        n += 1
        name = f"{stem}_{tag}_{n}.mp4"
    return name


def models() -> list:
    try:
        import folder_paths
        return list(folder_paths.get_filename_list("upscale_models"))
    except Exception:                                  # noqa: BLE001 -- headless
        return []


try:
    from comfy_api.latest import io
except Exception:                                      # noqa: BLE001 -- tests without ComfyUI
    io = None

if io is not None:
    class FunPackUpscaleVideo(io.ComfyNode):
        @classmethod
        def define_schema(cls) -> io.Schema:
            return io.Schema(
                node_id="FunPackUpscaleVideo",
                display_name="FunPack Upscale Video",
                category="FunPack/Output",
                description="Upscale a finished video file with an upscale model; keeps its fps and sound.",
                inputs=[
                    io.String.Input("filename", default=""),
                    io.String.Input("subfolder", default=""),
                    io.Combo.Input("type", options=["output", "temp"], default="output"),
                    io.Combo.Input("upscale_model", options=models() or [""],
                                   tooltip="From ComfyUI/models/upscale_models. Use .safetensors."),
                ],
                outputs=[],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, filename, subfolder, type, upscale_model) -> io.NodeOutput:
            import comfy.model_management as mm
            import folder_paths
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

            src = files.comfy_path(filename, subfolder, type)
            if not src or not os.path.isfile(src):
                raise RuntimeError(f"{filename} is not in the {type} folder (anymore).")
            model = UpscaleModelLoader.execute(upscale_model).result[0]
            out_dir = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
            os.makedirs(out_dir, exist_ok=True)
            name = free_name(out_dir, filename, upscale_model)
            frames = upscale_file(src, os.path.join(out_dir, name),
                                  lambda x: ImageUpscaleWithModel.execute(model, x).result[0],
                                  mm.throw_exception_if_processing_interrupted)
            from ..._core import log
            log.info("FunPack Upscale", f"{filename} with {upscale_model}: {frames} frames -> {SUBFOLDER}/{name}")
            return io.NodeOutput(ui={"videos": [{"filename": name, "subfolder": SUBFOLDER,
                                                 "type": "output", "format": "video/h264-mp4"}]})
