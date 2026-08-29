"""Declares something that is not a ComfyUI node at all."""


class NotANode:
    @classmethod
    def define_schema(cls):
        return None

    @classmethod
    def execute(cls):
        return None
    # No GET_SCHEMA: not a comfy_api io.ComfyNode.


ID = "nodes_not_a_node"
TITLE = "Not a node"
STAGE = "load"
NODES = [NotANode]
