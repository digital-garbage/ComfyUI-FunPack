"""FunPack Movie Editor — standalone web app sidecar.

A light orchestration layer (no torch/comfy import): manages projects, assembles
the Studio-format prompt from a timeline, and drives a running ComfyUI over its API
to generate. Parsing/encoding/sampling all happen inside ComfyUI via routes.
"""
