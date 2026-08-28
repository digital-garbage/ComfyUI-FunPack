import sys
from pathlib import Path

# Tests import `core.*`; make the repo root importable however pytest was invoked.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
