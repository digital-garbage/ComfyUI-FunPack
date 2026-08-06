"""Stand-ins for the ComfyUI runtime, installed once for the whole suite.

FunPack's modules are ComfyUI custom nodes: importing one pulls in `folder_paths`, `comfy.*`
and (through `templates.py`, which registers its HTTP routes at module scope) `server`. None
of those exist outside a running ComfyUI, so they have to be in `sys.modules` BEFORE the
first test module imports — a conftest is the only hook early enough.

**Why this file exists at all.** Every test module used to build its own partial `comfy`
stub and register it with `sys.modules.setdefault(...)`. First file wins, so in a full-suite
run a later file's richer stub was silently discarded and its tests failed on a missing
attribute — while passing when run alone. Two shapes of the same bug:

* `import comfy.sample` resolves through the PARENT's attribute, so a stub that sets
  `sys.modules["comfy.sample"]` without also setting `comfy.sample` leaves the import
  reading whatever the first file wired (or nothing).
* the same for `comfy.ldm.modules.attention`, which several files replace wholesale.

So the package tree is built ONCE here, with every parent attribute wired, and test modules
attach their own fakes as ATTRIBUTES on these shared module objects instead of replacing
them. Anything a single file needs only for its own tests belongs in a fixture that restores
the previous value (see `test_detailing.py` / `test_minimax_h3_sampler.py`), because these
objects are shared for the whole session.
"""
import sys
import types


def install_module(name, **attrs):
    """Register `name` in sys.modules AND on its parent, then set `attrs` on it.

    Returns the module (existing one if already registered, so callers merge rather than
    replace). Wiring the parent attribute is the part that is easy to forget and the whole
    reason imports used to resolve to a stale stub.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if "." in name:
        parent_name, _, leaf = name.rpartition(".")
        parent = install_module(parent_name)
        setattr(parent, leaf, mod)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install_folder_paths():
    install_module(
        "folder_paths",
        models_dir="",
        get_input_directory=lambda: "",
        get_output_directory=lambda: "",
        get_temp_directory=lambda: "",
        get_folder_paths=lambda _name: [],
        get_filename_list=lambda _name: [],
    )


def _install_server():
    """`server.PromptServer` — route decorators that register nothing and return the fn.

    templates.py decorates ~20 handlers with `@PromptServer.instance.routes.get(...)` at
    import time. The handlers are exercised through the Movie Editor's own aiohttp tests, so
    the decorator only has to be a no-op that leaves the function callable.
    """
    if "server" in sys.modules:
        return

    class _Routes:
        """aiohttp's RouteTableDef surface, for any verb.

        Generated rather than listed: movie_editor/server.py alone uses get/post/put/patch/
        delete, and a missing verb fails at IMPORT time (the decorators run at module scope),
        which reads as a mysterious collection error rather than a missing stub.
        """

        def __init__(self):
            self.registered = []

        def _register(self, method, path):
            def decorator(fn):
                self.registered.append((method, path, fn))
                return fn
            return decorator

        def __getattr__(self, method):
            if method.startswith("_"):
                raise AttributeError(method)
            if method in ("static", "view"):
                return lambda *_a, **_k: None
            return lambda path, **_kw: self._register(method.upper(), path)

    class _PromptServer:
        instance = None

        def __init__(self):
            self.routes = _Routes()
            self.app = types.SimpleNamespace(router=self.routes)

        def send_sync(self, *_a, **_k):
            return None

    _PromptServer.instance = _PromptServer()
    install_module("server", PromptServer=_PromptServer)


def _install_comfy():
    """The comfy package tree the suite imports, wired parent-to-child.

    Defaults are deliberately inert — a test that needs real behaviour installs it (see the
    module docstring). What matters here is that the tree EXISTS and is consistent, so no
    file can lose a race to register it.
    """
    # NOT installed here: comfy.context_windows and comfy.patcher_extension. The sampler
    # feature-detects both, and their tests swap whole modules in per test — pre-creating
    # them would leave a parent attribute that shadows the swap (the same resolution rule
    # this file exists to fix, pointed the other way).
    install_module("comfy")
    install_module("comfy.k_diffusion")
    install_module("comfy.k_diffusion.sampling")
    install_module("comfy.model_sampling")
    # KSAMPLER carries the same three attributes the real one does — FunPack builds one to
    # wrap a foreign sampler (the H3 audio clock), and the fields are what get read back.
    class _KSAMPLER:
        def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
            self.sampler_function = sampler_function
            self.extra_options = extra_options
            self.inpaint_options = inpaint_options

    install_module("comfy.samplers", KSAMPLER=_KSAMPLER)
    install_module("comfy.ldm")
    install_module("comfy.ldm.modules")
    install_module("comfy.ldm.modules.attention")
    install_module("comfy.nested_tensor", NestedTensor=object)
    install_module(
        "comfy.sample",
        prepare_noise=lambda samples, seed, noise_inds=None: None,
        sample_custom=lambda *a, **k: None,
    )
    install_module(
        "comfy.utils",
        ProgressBar=lambda total: types.SimpleNamespace(update_absolute=lambda *a, **k: None),
        common_upscale=lambda samples, width, height, method, crop: samples,
    )
    install_module(
        "comfy.model_management",
        get_torch_device=lambda: "cpu",
        intermediate_device=lambda: "cpu",
    )


def install_all():
    _install_folder_paths()
    _install_server()
    _install_comfy()
