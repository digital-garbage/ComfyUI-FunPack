"""A node-only module: no settings, no mount, no panel."""


class _Schema:
    node_id = "FunPackGood"
    display_name = "Good"


class GoodNode:
    @classmethod
    def define_schema(cls):
        return _Schema()

    @classmethod
    def execute(cls):
        return None

    @classmethod
    def GET_SCHEMA(cls):
        return _Schema()


ID = "nodes_good"
TITLE = "Good node"
STAGE = "load"
NODES = [GoodNode]
