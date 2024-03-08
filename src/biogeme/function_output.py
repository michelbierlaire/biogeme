from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FunctionOutput:
    """Output of a function calculation"""

    function: float
    gradient: np.ndarray | None = None
    hessian: np.ndarray | None = None


@dataclass
class BiogemeFunctionOutput(FunctionOutput):
    """Output of a function calculation"""

    bhhh: np.ndarray | None = None


@dataclass
class BiogemeDisaggregateFunctionOutput:
    """Output of a function calculation"""

    functions: np.ndarray
    gradients: np.ndarray | None = None
    hessians: np.ndarray | None = None
    bhhhs: np.ndarray | None = None
