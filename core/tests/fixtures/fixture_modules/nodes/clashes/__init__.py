"""Reuses an id another module already declared.

`ID` deliberately sorts AFTER `nodes_good`, because `collect()` resolves a
duplicate in favour of whoever it reaches first. Without that, this fixture won
the id and every test asserting "FunPackGood was collected" was satisfied by
this class instead of the real one -- a test passing for a reason it never named.
"""

from ..good import GoodNode


class Clashing(GoodNode):
    pass                            # same GET_SCHEMA, so same node_id


ID = "nodes_zz_clashes"
TITLE = "Clashing node"
STAGE = "load"
NODES = [Clashing]
