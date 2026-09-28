"""Late-branch guidance: push each step's picture away from a deliberately weakened copy.

STG-style skip guidance (the weak copy skips one block), made cheap enough for CFG=1:
both copies share every block before the branch, so the weak one only re-runs the tail.
Branching at 43 of H3's 50 blocks costs ~7 blocks per step (~15%), not a second forward.

    guided = normal + w * (normal - weak)      picture only; sound is the normal pass

The strength w is learned from ratings (rated_dial.py) or set by hand.
"""

try:
    from .rated_dial import Dial, MODES  # noqa: F401
except ImportError:
    from rated_dial import Dial, MODES  # noqa: F401

DIAL = Dial("late_guidance", start=0.5, lo=0.0, hi=1.5, explore=0.15)
DEFAULT_BLOCK = 43


def mix(normal, weak, w):
    return normal + w * (normal - weak)
