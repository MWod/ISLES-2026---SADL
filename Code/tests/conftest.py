"""Pytest configuration - ensures Code/src and Code/ are importable."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
for path in (_CODE / "src", _CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
