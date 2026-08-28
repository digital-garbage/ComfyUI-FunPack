"""Fails on import. Must be absent, not fatal."""

ID = "explodes"
TITLE = "Explodes"
MOUNT = "generation.timing"

raise RuntimeError("this module cannot load")
