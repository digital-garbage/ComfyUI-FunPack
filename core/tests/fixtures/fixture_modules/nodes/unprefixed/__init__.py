"""Claims a node_id outside FunPack's namespace, where it could shadow anyone."""

from ..good import GoodNode


class _Schema:
    node_id = "VAELoader"          # a real core node id
    display_name = "Hijack"


class Unprefixed(GoodNode):
    @classmethod
    def GET_SCHEMA(cls):
        return _Schema()


ID = "nodes_unprefixed"
TITLE = "Unprefixed node"
STAGE = "load"
NODES = [Unprefixed]
