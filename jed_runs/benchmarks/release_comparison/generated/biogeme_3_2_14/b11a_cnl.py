#!/usr/bin/env python3
"""Pure-estimation benchmark for the 3.3.4 b11 model on Biogeme 3.2.14."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_support import run_case  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(run_case(release='3.2.14', model='b11a_cnl', legacy=True))

