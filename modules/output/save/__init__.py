"""Save the finished video: MP4 (H.264), on the GPU's encoder when there is one, else on every CPU core.

Core's SaveVideo encodes on one CPU thread; on a rental that was most of the time after decoding.
"""

from .nodes import FunPackSaveVideo

ID = "output_save"
TITLE = "Save video"
STAGE = "post"
CATEGORY = "post"
STATUS = "proven"

NODES = [FunPackSaveVideo]
