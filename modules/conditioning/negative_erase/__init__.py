"""Negative erase: H3 runs at CFG 1, so the negative prompt does nothing -- this takes its direction out of
the positive prompt's words instead. A node-only module; off at strength 0.

    positive, negative -> Negative Erase -> positive
"""

from .nodes import FunPackNegativeErase

ID = "conditioning_negative_erase"
TITLE = "Negative erase"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"      # v4: the same lever as embed guidance, but whether a sentence is linear in the encoder was never judged

NODES = [FunPackNegativeErase]
