"""Install the ComfyUI stand-ins before any test module imports.

The stubs themselves live in `_comfy_stubs.py` rather than here: test modules need
`install_module` too, and `import conftest` is ambiguous the moment a second conftest.py
exists in the run (movie_editor/tests has one), so they import a uniquely-named module
instead. See `_comfy_stubs` for what is stubbed and why.
"""
import _comfy_stubs

_comfy_stubs.install_all()
