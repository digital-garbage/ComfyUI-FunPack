import torch
from comfy_api.latest import io

from ..._core import log, patching
from . import patch, projector

KEY = "funpack.identity_transfer"


def _projector_choices():
    try:
        import folder_paths
        return ["None"] + folder_paths.get_filename_list("loras")
    except Exception:                                # noqa: BLE001
        return ["None"]


def _video_part(latent):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        return samples.unbind()[0]
    return samples


class FunPackIdentityTransfer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackIdentityTransfer",
            display_name="FunPack Identity Transfer",
            category="FunPack/Conditioning",
            description="Carry a reference face into an LTX clip (Best-FaceID style): the reference goes in "
                        "as separate tokens the model attends to, plus optional ArcFace text tokens. "
                        "With no image wired, everything passes through untouched.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae", tooltip="The picture VAE, to encode the reference."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent", tooltip="The clip's starting latent: the reference is sized to its frame."),
                io.Image.Input("image", optional=True, tooltip="The reference face. None = no identity transfer."),
                io.Combo.Input("identity_projector", options=_projector_choices(), default="None", optional=True,
                               tooltip="ArcFace projector .safetensors (models/loras). None = the overlap tokens "
                                       "only (they carry most of the identity)."),
                io.Float.Input("source_id", default=2.0, min=0.0, max=8.0, step=1.0, optional=True,
                               tooltip="Source-phase id for the reference tokens (the trainer used 2). 0 turns the "
                                       "rotation off, leaving the tokens in."),
                io.Float.Input("phase_scale", default=1.0, min=0.0, max=4.0, step=0.1, optional=True),
                io.Float.Input("id_strength", default=1.0, min=0.0, max=50.0, step=0.5, optional=True,
                               tooltip="Multiplies the ArcFace tokens. A weak channel: 5-20 to test."),
                io.Combo.Input("arcface_mode", options=["auto_adjust", "as_is", "disable"], default="auto_adjust",
                               optional=True,
                               tooltip="auto_adjust retries detection zoomed out/up; as_is detects on the image only; "
                                       "disable skips ArcFace."),
            ],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.Conditioning.Output(display_name="negative"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, model, vae, positive, negative, latent, image=None, identity_projector="None",
                source_id=2.0, phase_scale=1.0, id_strength=1.0, arcface_mode="auto_adjust") -> io.NodeOutput:
        patched = patching.clone(model)
        patching.strip(patched, KEY)                       # a clone carries an earlier install forward
        if image is None:
            return io.NodeOutput(patched, positive, negative, "no reference image: untouched")

        import comfy.utils
        video = _video_part(latent)
        if video.dim() != 5:
            raise RuntimeError("identity transfer needs a video latent (frames, height, width)")
        _, w_scale, h_scale = getattr(vae, "downscale_index_formula", (8, 8, 8))
        _, _, _, lat_h, lat_w = video.shape
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), lat_w * w_scale, lat_h * h_scale,
                                            "bilinear", "center").movedim(1, -1)[:1, :, :, :3]
        ref_latent = vae.encode(pixels)
        seg = float(source_id) * float(phase_scale)
        patch.install(patched, ref_latent, seg, KEY)
        notes = [f"reference tokens appended (source phase {seg:g})"]

        if identity_projector not in (None, "", "None") and arcface_mode != "disable":
            try:
                emb = projector.arcface_embed(image, mode=arcface_mode)
                if emb is None:
                    notes.append("ArcFace found no face: overlap tokens only")
                else:
                    import folder_paths
                    path = folder_paths.get_full_path("loras", identity_projector) or identity_projector
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    net = projector.load_identity_projector(path, device)
                    with torch.no_grad():
                        pos = net(emb.to(device=device, dtype=torch.float32).unsqueeze(0)) * float(id_strength)
                        neg = net(torch.zeros(1, net.in_dim, device=device))
                    positive = projector.append_context_tokens(positive, pos)
                    negative = projector.append_context_tokens(negative, neg)
                    notes.append(f"ArcFace tokens appended (strength {id_strength:g})")
            except ImportError as exc:
                log.once("identity_transfer:insightface", log.ALERT, "FunPack Identity Transfer",
                         f"ArcFace needs insightface ({exc}); `pip install insightface`. Overlap tokens only.")
                notes.append("insightface is not installed: overlap tokens only")
        return io.NodeOutput(patched, positive, negative, "; ".join(notes))
