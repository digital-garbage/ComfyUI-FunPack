"""FunPack Movie Editor — runs inside ComfyUI.

Importing `movie_editor.server` registers the /funpack/movie/* routes on ComfyUI's
aiohttp PromptServer. The root package __init__ does that at custom-node load time.
This module stays import-light so `movie_editor.backend.*` can be unit-tested without
ComfyUI present.
"""
