# Ensure tests can import the project package when pytest runs.
# This inserts the repository root into sys.path before tests are collected.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
