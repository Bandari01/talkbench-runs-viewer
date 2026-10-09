"""These tests run in a talk-bench checkout's venv (they import talk-bench and tau2):

    ~/GitHub/ai-ds-research/talk-bench/.venv/bin/python -m pytest tau2_user_sim/tests -q -p no:cacheprovider
"""

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tau2 reads TAU2_DATA_DIR at import time; talk-bench's .env sets it relative to its
# own checkout, which does not resolve from here — point it at the vendored data.
_spec = importlib.util.find_spec("tau2")
if _spec and _spec.origin:
    _data = Path(_spec.origin).resolve().parents[2] / "data"
    if _data.is_dir():
        os.environ["TAU2_DATA_DIR"] = str(_data)
