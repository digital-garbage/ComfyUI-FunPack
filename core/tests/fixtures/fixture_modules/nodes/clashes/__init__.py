"""Reuses an id another module already declared."""

from ..good import GoodNode


class Clashing(GoodNode):
    pass                            # same GET_SCHEMA, so same node_id


ID = "nodes_clashes"
TITLE = "Clashing node"
STAGE = "load"
NODES = [Clashing]
