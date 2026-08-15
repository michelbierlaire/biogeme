#!/usr/bin/env python3
"""Pure-estimation benchmark for the 3.3.4 b12 model on Biogeme 3.3.4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_support import run_case  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(run_case(release='3.3.4', model='b12_panel', legacy=False))

