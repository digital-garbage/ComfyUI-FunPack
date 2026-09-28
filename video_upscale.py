"""Upscale a finished video with a models/upscale_models model (ESRGAN-style, via spandrel).

The Editor queues this on a render once it is ready: frames stream through ffmpeg in small
chunks (never the whole clip in RAM), each chunk goes through ComfyUI's own
ImageUpscaleWithModel (tiling, OOM back-off, device handling), and the result is encoded
with the source's frame rate and sound.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import torch

CHUNK = 8                # frames per upscale call
SUBFOLDER = "funpack_upscaled"


def _ffmpeg(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH; install ffmpeg to upscale videos.")
    return path


def probe(path):
    """-> (width, height, fps string, has sound)."""
    out = subprocess.run(
        [_ffmpeg("ffprobe"), "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout).get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError(f"no video stream in {os.path.basename(path)}")
    has_sound = any(s.get("codec_type") == "audio" for s in streams)
    return int(video["width"]), int(video["height"]), video.get("r_frame_rate") or "25", has_sound


def resolve(filename, subfolder="", type_="output"):
    """Absolute path of a ComfyUI output/temp file; refuses anything outside that folder."""
    import folder_paths
    base = folder_paths.get_output_directory() if type_ == "output" else folder_paths.get_temp_directory()
    base = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, subfolder or "", filename))
    if os.path.commonpath([base, path]) != base:
        raise ValueError("video path escapes the output folder")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{filename} is not in the {type_} folder (anymore)")
    return path


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
    -> frames written. `interrupted()` raises to stop between chunks."""
    w, h, fps, has_sound = probe(src)
    frame = w * h * 3
    dec = subprocess.Popen([_ffmpeg("ffmpeg"), "-v", "error", "-i", src, "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"],
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
                cmd = [_ffmpeg("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{ow}x{oh}", "-r", fps, "-i", "-"]
                if has_sound:
                    cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
                cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                        "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
                enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            enc.stdin.write(y.tobytes())
            frames += n
        if enc is None:
            raise RuntimeError(f"no frames could be read from {os.path.basename(src)}")
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg could not encode the upscaled video: {err[-500:]}")
        return frames
    except BaseException:
        for p in (dec, enc):
            if p is not None and p.poll() is None:
                p.kill()
        if os.path.exists(dst):
            os.remove(dst)
        raise
    finally:
        if dec.poll() is None:
            dec.kill()


class FunPackUpscaleVideo:
    """Upscale a finished video file with an upscale model; keeps its fps and sound."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "filename": ("STRING", {"default": ""}),
            "subfolder": ("STRING", {"default": ""}),
            "type": (["output", "temp"], {"default": "output"}),
            "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "FunPack/video"

    def run(self, filename, subfolder, type, upscale_model):
        import comfy.model_management as mm
        import folder_paths
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

        src = resolve(filename, subfolder, type)
        model = UpscaleModelLoader.execute(upscale_model).result[0]
        out_dir = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]
        tag = os.path.splitext(os.path.basename(upscale_model))[0]
        n, name = 0, f"{stem}_{tag}.mp4"
        while os.path.exists(os.path.join(out_dir, name)):
            n += 1
            name = f"{stem}_{tag}_{n}.mp4"
        frames = upscale_file(
            src, os.path.join(out_dir, name),
            lambda x: ImageUpscaleWithModel.execute(model, x).result[0],
            mm.throw_exception_if_processing_interrupted)
        print(f"[FunPack] upscaled {filename} with {upscale_model}: {frames} frames -> {SUBFOLDER}/{name}")
        return {"ui": {"videos": [{"filename": name, "subfolder": SUBFOLDER, "type": "output",
                                   "format": "video/h264-mp4"}]}}
