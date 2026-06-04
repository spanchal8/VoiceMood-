"""pytest configuration: put src/ on the path so tests import voicemood."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
