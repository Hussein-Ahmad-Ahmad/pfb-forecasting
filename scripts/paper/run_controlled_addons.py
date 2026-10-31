#!/usr/bin/env python
"""Entry point for prospective Projected, HPO, and normalization controls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.controlled_addons.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
